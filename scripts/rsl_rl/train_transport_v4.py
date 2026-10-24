"""PAINT transport environment, frozen v4 LL, teacher training and diagnostics."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "rsl_rl"), str(ROOT / "exts/bipedal_locomotion")]
CASES = ("zero", "forward", "backward", "left", "right", "yaw_left", "yaw_right")


def parser_for_task():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "eval", "train", "preview"), default="smoke")
    p.add_argument("--controller", choices=("hold", "reference", "teacher", "student", "probe"), default="hold")
    p.add_argument("--actor_mode", choices=("teacher", "student"), default="teacher")
    p.add_argument("--locomotion_path", required=True, help="v4 model_N.pt or locomotion_v4.pt")
    p.add_argument("--low_format", choices=("checkpoint", "torchscript"), default="checkpoint")
    p.add_argument("--low_manifest", help="Required companion contract for an exported policy; default parent/contract.json")
    p.add_argument("--teacher_checkpoint")
    p.add_argument("--student_checkpoint", help="Student weights for eval/warm-start; --resume restores optimizers")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--distill_coef", type=float, default=1e-6)
    p.add_argument("--est_learning_rate", type=float, default=1e-3)
    p.add_argument("--terrain", choices=("flat", "rough"), default="flat")
    p.add_argument("--terrain_level", type=int, default=0)
    p.add_argument("--domain", choices=("nominal", "randomized"), default="nominal")
    p.add_argument("--payload_mass_range", type=float, nargs=2, default=(.5, .5))
    p.add_argument("--payload_height_range", type=float, nargs=2,
                   help="Absolute base-frame height band; default nominal TRON2 carry height +/-0.1 m")
    p.add_argument("--max_wrench", type=float, nargs=3, default=(3., 3., .5), help="Fx/Fy component limits N, yaw Nm")
    p.add_argument("--zero_force", action="store_true", help="Disable main wrench AND periodic EE disturbance")
    p.add_argument("--disturbance_scale", type=float, default=0., help="1 = guide periodic world Fz/Tx/Ty")
    p.add_argument("--base_limits", type=float, nargs=3, default=(.35, .2, .4))
    p.add_argument("--arm_scale", type=float, default=.1)
    p.add_argument("--probe_command", type=float, nargs=3, default=(0., 0., 0.))
    p.add_argument("--num_envs", type=int)
    p.add_argument("--max_iterations", type=int, default=500)
    p.add_argument("--save_interval", type=int, default=100)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir")
    p.add_argument("--debug_vis", action="store_true")
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--video", action="store_true")
    return p


def validate_args(args):
    from bipedal_locomotion.controllers.paint_transport_v4 import validate_settings
    validate_settings(args.max_wrench, args.base_limits, args.payload_mass_range, args.arm_scale, args.disturbance_scale)
    if args.payload_height_range is not None and (not all(math.isfinite(v) for v in args.payload_height_range)
            or args.payload_height_range[0] >= args.payload_height_range[1]):
        raise ValueError("Payload height band must be finite and ordered.")
    if min(args.steps, args.episodes, args.max_iterations, args.save_interval) < 1 or (args.num_envs is not None and args.num_envs < 1):
        raise ValueError("Counts must be positive.")
    if not 0 <= args.terrain_level <= 9 or len(args.cases) != len(set(args.cases)):
        raise ValueError("Invalid terrain row or duplicate cases.")
    if not math.isfinite(args.distill_coef) or args.distill_coef < 0 or not math.isfinite(args.est_learning_rate) or args.est_learning_rate <= 0:
        raise ValueError("Invalid Student loss/estimator settings")
    student_run = args.actor_mode == "student" and (args.mode == "train" or args.controller == "student")
    if student_run and not args.teacher_checkpoint:
        raise ValueError("Student requires --teacher_checkpoint to pin its frozen Teacher")
    if args.controller == "student" and (args.actor_mode != "student" or not args.student_checkpoint):
        raise ValueError("Student evaluation requires student mode and --student_checkpoint")
    if args.student_checkpoint and not student_run:
        raise ValueError("--student_checkpoint requires Student training/controller")
    if args.resume and (not student_run or args.mode != "train" or not args.student_checkpoint):
        raise ValueError("--resume requires Student training and --student_checkpoint")
    if args.mode != "train" and args.controller == "teacher" and (args.actor_mode != "teacher" or not args.teacher_checkpoint):
        raise ValueError("Teacher controller requires teacher mode and --teacher_checkpoint.")
    if args.controller == "probe":
        if not args.zero_force or any(not math.isfinite(v) or abs(v) > limit for v, limit in zip(args.probe_command, args.base_limits)):
            raise ValueError("Probe requires zero physical force and an in-range base command.")
        xy = math.hypot(*args.probe_command[:2])
        if (0 < xy <= .02) or (0 < abs(args.probe_command[2]) <= .03):
            raise ValueError("Probe command would be removed by the v4 deadband.")
    if args.mode == "eval" and args.num_envs not in (None, 1):
        raise ValueError("Fixed-case evaluation uses one environment.")


def make_action(args, raw, obs, actor=None):
    import torch
    if args.controller in ("teacher", "student"):
        return actor(obs["policy"])
    action = torch.zeros(raw.num_envs, 9, device=raw.device)
    if args.controller == "reference":
        action[:, :3] = raw.command_manager.get_command("interaction")
    elif args.controller == "probe":
        action[:, :3] = torch.tensor(args.probe_command, device=raw.device)
    return action


def rollout(env, args, app, output, actor=None, debug=None):
    import torch
    raw = env.unwrapped
    rows, failures, timeouts, interrupted, step_count = [], 0, 0, False, 0
    cases = args.cases if args.mode == "eval" else ["random"]
    with (output / "rollout.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            raw.command_manager.get_term("interaction").cfg.case = "zero" if args.zero_force else case
            for trial in range(args.episodes if args.mode == "eval" else 1):
                raw.reset(seed=args.seed + trial)
                obs = env.get_observations()
                sums = torch.zeros(raw.num_envs, 3, device=raw.device)
                reference_sums = torch.zeros_like(sums)
                estimator_sums = torch.zeros_like(sums)
                count, peak_tilt, completed = 0, 0., False
                duration = 0.
                for i in range(raw.max_episode_length if args.mode == "eval" else args.steps):
                    if not app.is_running():
                        interrupted = True
                        break
                    started = time.monotonic()
                    estimate, estimate_target = None, None
                    with torch.no_grad():
                        if args.controller == "student":
                            estimate = actor.estimator(obs["student"]).clone()
                            estimate_target = obs["teacher"][:, -3:].clone()
                            estimator_sums += (estimate - estimate_target).square()
                        obs, reward, dones, info = env.step(make_action(args, raw, obs, actor))
                    state = info["transport_state"]
                    if not all(torch.isfinite(v).all() for v in (*obs.values(), *state.values(), reward)):
                        raise FloatingPointError("Nonfinite transport state.")
                    desired = state["reference"] if args.controller != "probe" else torch.tensor(
                        args.probe_command, device=raw.device).expand(raw.num_envs, -1)
                    error = state["base_velocity"] - desired
                    sums += error.square()
                    reference_sums += desired.square()
                    count += 1
                    step_count += 1
                    duration = (i + 1) * raw.step_dt
                    peak_tilt = max(peak_tilt, float(state["base_tilt_deg"].max()))
                    failure = info["transport_terminated"].bool()
                    failures += int(failure.sum())
                    timeouts += int((dones.bool() & info["time_outs"].bool() & ~failure).sum())
                    stream.write(json.dumps({"case": case, "trial": trial, "step": i, "time_s": duration,
                        "done_env0": bool(dones[0]), "timeout_env0": bool(info["time_outs"][0]),
                        "tracking_reference": desired[0].tolist(),
                        "estimated_c_pre_action": estimate[0].tolist() if estimate is not None else None,
                        "true_c_pre_action": estimate_target[0].tolist() if estimate_target is not None else None,
                        "terminated_env0": bool(failure[0]),
                        "termination_terms": {name: bool(value[0]) for name, value in info["transport_done_terms"].items()},
                        **{key: value[0].tolist() for key, value in state.items()},
                        "rewards": {key: float(value[0]) for key, value in info["transport_rewards"].items()}}) + "\n")
                    if args.realtime:
                        time.sleep(max(0., raw.step_dt - (time.monotonic() - started)))
                    if args.mode == "eval" and bool(dones[0]):
                        completed = bool(info["time_outs"][0]) and not bool(failure[0]) and duration >= raw.max_episode_length_s - raw.step_dt
                        break
                rmse = (sums / max(1, count)).mean(0).sqrt().tolist()
                ref_rms = (reference_sums / max(1, count)).mean(0).sqrt().tolist()
                planar_ref = math.hypot(*ref_rms[:2])
                linear_limit = min(.1, max(.01, .5 * planar_ref)) if planar_ref > .01 else .05
                yaw_limit = min(.1, max(.01, .5 * ref_rms[2])) if ref_rms[2] > .01 else .05
                rows.append({"case": case, "trial": trial, "seed": args.seed + trial,
                    "duration_s": duration, "completed": completed,
                    "reference_rms_vx_vy_wz": ref_rms,
                    "linear_rmse_limit": linear_limit, "yaw_rmse_limit": yaw_limit,
                    "rmse_vx_vy_wz": rmse, "max_base_tilt_deg": peak_tilt,
                    "estimator_rmse_normalized": (estimator_sums / max(1, count)).mean(0).sqrt().tolist() if args.controller == "student" else None,
                    "estimator_rmse_N_N_Nm": ((estimator_sums / max(1, count)).mean(0).sqrt()
                        * torch.tensor([40., 40., 10.], device=raw.device)).tolist() if args.controller == "student" else None,
                    "passed": args.mode == "eval" and completed and peak_tilt < 15.
                        and math.hypot(*rmse[:2]) < linear_limit and rmse[2] < yaw_limit})
                stream.flush()
                if interrupted:
                    break
            if interrupted:
                break
    summary = {"mode": args.mode, "controller": args.controller, "interrupted": interrupted,
        "failures": failures, "timeouts": timeouts, "control_steps": step_count, "episodes": rows,
        "all_passed": args.mode == "eval" and not interrupted and bool(rows) and all(row["passed"] for row in rows),
        "tracking_target": "probe_command" if args.controller == "probe" else "wrench_reference",
        "gate": "initial engineering gate: complete episode, tilt<15deg; moving RMSE<min(0.1,max(0.01,0.5*reference_RMS)), quiet RMS<0.05; not paper acceptance"}
    (output / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(args, app):
    import torch
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    from rsl_rl.runner import OnPolicyRunner
    import bipedal_locomotion  # noqa: F401
    from bipedal_locomotion.controllers.paint_payload import create_payload_asset
    from bipedal_locomotion.controllers.paint_transport_v4 import SCHEMA, check_teacher_contract
    from bipedal_locomotion.tasks.transport_v4.env_cfg import configure_domain
    from bipedal_locomotion.tasks.transport_v4.mdp import controller
    validate_args(args)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir or ROOT / "logs/rsl_rl/paint_transport_v4" /
                  (args.mode + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    task = f"Isaac-Limx-SFYG-TRON2A-Paint-Transport-{args.actor_mode.title()}-v4"
    cfg = parse_env_cfg(task, device=args.device,
        num_envs=args.num_envs or (256 if args.mode == "train" else 1))
    cfg.seed = args.seed
    action = cfg.actions.transport
    action.low_path, action.low_format = str(Path(args.locomotion_path).resolve()), args.low_format
    action.low_manifest = str(Path(args.low_manifest).resolve()) if args.low_manifest else ""
    action.base_limits, action.arm_scale = tuple(args.base_limits), args.arm_scale
    action.disturbance_scale = 0. if args.zero_force else args.disturbance_scale
    cfg.commands.interaction.max_wrench = (0., 0., 0.) if args.zero_force else tuple(args.max_wrench)
    configure_domain(cfg, args.terrain, args.domain == "randomized", args.payload_mass_range,
                     evaluation=args.mode != "train")
    cfg.scene.height_scanner.debug_vis = args.debug_vis
    cfg.scene.robot.spawn.usd_path = create_payload_asset(cfg.scene.robot.spawn.usd_path,
        output / "robot_payload.usdc", .5, (.20, .12, .08), (.15, 0., 0.))
    env = gym.make(task, cfg=cfg, render_mode="rgb_array" if args.video else None)
    runner, debug = None, None
    try:
        raw = env.unwrapped
        if args.terrain == "rough" and args.mode != "train":
            terrain = raw.scene.terrain
            terrain.terrain_levels[:] = args.terrain_level
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            raw.scene.env_origins[:] = terrain.env_origins
        if args.video:
            # Attach the RGB annotator and warm it up BEFORE the recorder starts.
            # Isaac returns black placeholder frames while its render graph warms up.
            raw.reset(seed=args.seed)
            frame = None
            for _ in range(40):
                frame = raw.render()
            if frame is None or not bool(frame.any()):
                raise RuntimeError("Camera is still empty/black after warmup; check the local RTX renderer before recording.")
            env = gym.wrappers.RecordVideo(env, video_folder=str(output / "videos"),
                step_trigger=lambda step: step == 0, video_length=args.steps, disable_logger=True)
        env = RslRlVecEnvWrapper(env)
        term = controller(raw)
        from isaaclab.utils.math import quat_apply_inverse
        data = raw.scene["robot"].data
        nominal_height = float(quat_apply_inverse(data.root_quat_w[0:1],
            data.body_com_pos_w[0:1, term.payload_ids[0]] - data.root_pos_w[0:1])[0, 2])
        if not math.isfinite(nominal_height):
            raise RuntimeError("Cannot calibrate nominal payload height.")
        height_band = args.payload_height_range
        if (height_band is None and args.actor_mode == "student" and args.teacher_checkpoint
                and (args.mode == "train" or args.controller == "student")):
            teacher_metadata = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
            height_band = teacher_metadata.get("transport_contract", {}).get("payload_height_range")
            if height_band is None:
                raise ValueError("Teacher has no calibrated payload height band")
        raw.cfg.payload_height_range = tuple(height_band or (nominal_height - .1, nominal_height + .1))
        before = term.low.fingerprint()
        obs = env.get_observations()
        widths = {key: list(value.shape[1:]) for key, value in obs.items()}
        if widths["teacher"] != [147] or widths["student"] != [144] or widths["critic"] != [177] or env.num_actions != 9:
            raise RuntimeError(f"Transport I/O preflight failed: {widths}")
        manifest = {"schema": SCHEMA, "task": task, "actor_mode": args.actor_mode,
            "arguments": vars(args), "observation_shapes": widths,
            "physics_hz": 200, "control_hz": 50, "episode_s": 10,
            "nominal_payload_height_m": nominal_height, "payload_height_range": raw.cfg.payload_height_range,
            "low_sha256": term.low.source_sha256, "low_contract": term.low.manifest,
            "frozen_low_fingerprint": before, "teacher_source_commit": "3136606",
            "v4_base_commit": "4f923e8", "reward_tracking_sign": "positive exponential",
            "arm_pd": [60., 6.], "wrench_frame": "base -> world -> each body local, applied every physics step"}
        agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        agent.seed, agent.device, agent.save_interval = args.seed, args.device, args.save_interval
        manifest["policy_config"] = agent.to_dict()["policy"]
        student_run = args.actor_mode == "student" and (args.mode == "train" or args.controller == "student")
        source_files = [Path(__file__).resolve(), ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers/paint_transport_v4.py",
                       *sorted((ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/transport_v4").glob("*.py"))]
        source_files += [ROOT / "rsl_rl/rsl_rl" / name for name in (
            "algorithm/ppo.py", "runner/on_policy_runner.py", "storage/rollout_storage.py",
            "modules/actor_critic.py", "modules/mlp_encoder.py", "modules/intent_estimator.py",
            "algorithm/paint_ppo.py", "storage/paint_storage.py", "runner/paint_runner.py")]
        source_files.append(ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/teacher/teacher_ppo_cfg.py")
        if student_run:
            manifest["teacher_sha256"] = hashlib.sha256(Path(args.teacher_checkpoint).read_bytes()).hexdigest()
        manifest["source_sha256"] = {}
        for source in source_files:
            relative = source.relative_to(ROOT)
            target = output / "sources" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            manifest["source_sha256"][relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
        (output / "contract.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        if args.debug_vis:
            from bipedal_locomotion.tasks.transport_v4.debug_vis import TransportDebugView
            debug = TransportDebugView(raw)
            raw._transport_debug = debug
        if student_run:
            from rsl_rl.runner.paint_runner import PaintRunner
            train_cfg = agent.to_dict()
            train_cfg["algorithm"].update(distill_coef=args.distill_coef, est_learning_rate=args.est_learning_rate)
            runner = PaintRunner(env, train_cfg, manifest, args.teacher_checkpoint, output, args.device)
            if args.student_checkpoint:
                runner.load(args.student_checkpoint, resume=args.resume)
            actor = runner.get_inference_policy(device=raw.device)
        elif args.mode == "train" or args.controller == "teacher":

            class TransportRunner(OnPolicyRunner):
                def log(self, locs, *a, **kw):
                    super().log(locs, *a, **kw)
                    done = locs["it"] + 1
                    if done % self.save_interval == 0:
                        self.current_learning_iteration = done
                        self.save(output / f"model_{done}.pt")

                def save(self, path, infos=None):
                    path = Path(path)
                    if path.stem != f"model_{self.current_learning_iteration}":
                        return
                    super().save(str(path), infos)
                    saved = torch.load(path, map_location="cpu", weights_only=False)
                    saved["transport_contract"] = manifest
                    torch.save(saved, path)

            runner = TransportRunner(env, agent.to_dict(), log_dir=str(output), device=args.device)
            # Deterministic initial mean is zero; stochastic exploration remains enabled.
            linear = [module for module in runner.alg.actor_critic.actor if isinstance(module, torch.nn.Linear)][-1]
            torch.nn.init.zeros_(linear.weight); torch.nn.init.zeros_(linear.bias)
            if args.teacher_checkpoint:
                checkpoint = torch.load(args.teacher_checkpoint, map_location=args.device, weights_only=False)
                saved = checkpoint.get("transport_contract", {})
                check_teacher_contract(saved, manifest)
                if not all(torch.isfinite(value).all() for value in checkpoint["model_state_dict"].values()):
                    raise ValueError("Nonfinite teacher checkpoint.")
                runner.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
                runner.alg.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
            actor = runner.get_inference_policy(device=raw.device)
        else:
            actor = None
        print(f"[TRANSPORT] mode={args.mode} actor={args.actor_mode} widths={widths} low_frozen=True", flush=True)
        if args.mode == "train":
            runner.learn(args.max_iterations, init_at_random_ep_len=False)
            # Export only the high-level teacher; low-level remains a separately pinned artifact.
            from copy import deepcopy
            if student_run:
                runner.export(output / "student_actor.pt")
            else:
                with (output / "teacher_actor.pt").open("wb") as stream:
                    torch.jit.save(torch.jit.script(deepcopy(runner.alg.actor_critic.actor).cpu().eval()), stream)
        else:
            result = rollout(env, args, app, output, actor, debug)
            print(f"[TRANSPORT] evaluation={result}", flush=True)
        frozen = term.low.fingerprint() == before
        completion = {"complete": True, "frozen_weights_verified": frozen, "mode": args.mode}
        if student_run:
            completion["frozen_teacher_verified"] = runner.verify_frozen_teacher()
        (output / "completion.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
        if not frozen:
            raise RuntimeError("Frozen low-level weights changed.")
    finally:
        if debug: debug.close()
        if runner is not None and runner.writer is not None: runner.writer.close()
        env.close()


if __name__ == "__main__":
    parser = parser_for_task()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.video: args.enable_cameras = True
    launcher = AppLauncher(args)
    try:
        main(args, launcher.app)
    except BaseException:
        traceback.print_exc(); sys.stderr.flush()
        raise
    finally:
        launcher.app.close()
