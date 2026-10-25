"""Train/evaluate the full-robot PAINT replacement backbone, starting from main.

No v2/v3 weights are accepted. Stage promotion uses --checkpoint_path (weights
only); --resume also restores optimizers/iteration, but resets simulator episodes.
"""
from dataclasses import replace
from datetime import datetime
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import shutil
import sys
import traceback
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "rsl_rl"), str(ROOT / "exts/bipedal_locomotion")]


def parser_for_task():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval", "probe"), default="train")
    parser.add_argument("--probe_seconds", type=float, default=5., help="Zero-policy PD-hold diagnostic duration")
    parser.add_argument("--stage", choices=("nominal", "robust", "interaction", "rough"), default="nominal")
    parser.add_argument("--design", choices=("baseline", "biped"), default="baseline",
                        help="biped: full-SFYG learning-to-walk rebuild; baseline: existing v4 ablations")
    parser.add_argument("--skill_stage", choices=("bootstrap", "omni", "transitions"), default="bootstrap")
    parser.add_argument("--promotion_report", help="Passed evaluation.json for explicit biped skill promotion")
    parser.add_argument("--checkpoint_path", help="v4 checkpoint only; omitted = fresh random initialization")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--max_iterations", type=int, default=10000, help="Additional PPO iterations")
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir")
    parser.add_argument("--velocity_limits", type=float, nargs=3, default=(.35, .2, .4))
    parser.add_argument("--gait", choices=("restored", "legacy"), default="restored",
                        help="Restored phase gait (new I/O); legacy replays pre-gait v4 checkpoints")
    parser.add_argument("--gait_weight", type=float, default=None,
                        help="Set 0 for a reward ablation with identical phase observations")
    parser.add_argument("--command_profile", choices=("mixed", "walk_first"), default=None,
                        help="Baseline-only command ablation; biped uses --skill_stage")
    parser.add_argument("--tracking_std", type=float, default=None,
                        help="Linear velocity exponential-kernel std; .5 preserves old v4")
    parser.add_argument("--payload_mass_range", type=float, nargs=2)
    parser.add_argument("--force_n", type=float)
    parser.add_argument("--yaw_torque_nm", type=float)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval_cases", nargs="+", default=["stand", "walk", "start_stop", "stop_start",
                        "slow", "forward", "left", "right", "yaw_left", "yaw_right"],
                        choices=["stand", "walk", "start_stop", "stop_start", "slow", "forward", "left", "right", "yaw_left", "yaw_right"])
    parser.add_argument("--terrain_level", type=int, default=0, help="Fixed rough-terrain evaluation row [0,9]")
    parser.add_argument("--real_time", action="store_true", help="Pace evaluation at control frequency when simulation is faster than real time")
    return parser


def settings_for_args(args):
    from bipedal_locomotion.controllers.locomotion_v4 import STAGES, validate_ranges
    if args.design == "biped":
        if args.gait != "restored" or args.command_profile is not None:
            raise ValueError("Biped design requires restored gait and uses --skill_stage, not --command_profile")
        args.command_profile = args.skill_stage
        if args.tracking_std is None:
            args.tracking_std = .2
        if args.gait_weight is None:
            args.gait_weight = 1.
        if args.mode == "train" and args.skill_stage == "bootstrap" and args.stage != "nominal":
            raise ValueError("Bootstrap starts on nominal full-SFYG physics before load/terrain disturbances")
        if args.mode == "train" and args.skill_stage != "bootstrap" and not args.checkpoint_path:
            raise ValueError("Start biped training with bootstrap, then promote an evaluated checkpoint")
    else:
        args.command_profile = args.command_profile or "mixed"
        if args.tracking_std is None:
            args.tracking_std = .5
        if args.gait_weight is None:
            args.gait_weight = .75
    if args.mode == "probe" and (args.design != "biped" or args.checkpoint_path):
        raise ValueError("Probe is a full-SFYG biped PD diagnostic, without a learned checkpoint")
    if not math.isfinite(args.probe_seconds) or args.probe_seconds <= 0:
        raise ValueError("Probe duration must be finite and positive")
    stage = STAGES[args.stage]
    stage = replace(stage, payload_mass=tuple(args.payload_mass_range) if args.payload_mass_range else stage.payload_mass,
                    force_n=args.force_n if args.force_n is not None else stage.force_n,
                    yaw_torque_nm=args.yaw_torque_nm if args.yaw_torque_nm is not None else stage.yaw_torque_nm)
    validate_ranges(stage.payload_mass, stage.force_n, stage.yaw_torque_nm, args.velocity_limits)
    if args.design == "biped" and args.mode == "train" and args.skill_stage == "bootstrap":
        if (stage.force_n or stage.yaw_torque_nm or stage.payload_mass[0] != stage.payload_mass[1]
                or stage.noise or args.velocity_limits[0] < .2):
            raise ValueError("Bootstrap requires fixed payload, zero external wrench/noise, and vx limit >=.2 m/s")
    if (not math.isfinite(args.gait_weight) or args.gait_weight < 0
            or not math.isfinite(args.tracking_std) or args.tracking_std <= 0):
        raise ValueError("Gait weight must be finite/nonnegative and tracking std positive/finite.")
    if min(args.max_iterations, args.save_interval, args.episodes) < 1 or (args.num_envs is not None and args.num_envs < 1):
        raise ValueError("Iteration, save, episode and environment counts must be positive.")
    if not 0 <= args.terrain_level <= 9 or len(args.eval_cases) != len(set(args.eval_cases)):
        raise ValueError("Invalid terrain level or duplicate evaluation cases.")
    if (args.resume or args.mode == "eval") and not args.checkpoint_path:
        raise ValueError("Evaluation/resume requires --checkpoint_path.")
    if args.mode in ("eval", "probe") and args.num_envs not in (None, 1):
        raise ValueError("Fixed-sequence evaluation requires --num_envs 1.")
    return stage


def configure_optimizer(agent, design):
    if design != "biped":
        return
    agent.policy.init_noise_std = .8
    agent.algorithm.entropy_coef = .01
    agent.algorithm.learning_rate = 1e-4
    agent.algorithm.max_learning_rate = 1e-4
    agent.algorithm.est_learning_rate = 3e-5
    agent.algorithm.update_kl_guard = True
    agent.algorithm.encoder_kl_fraction = .3


def probe_pd(env, args, output, app):
    """Measure nominal full-body support, not a test of learned walking ability."""
    import torch
    from bipedal_locomotion.controllers.locomotion_v4 import LEG_JOINTS
    raw = env.unwrapped
    command = raw.command_manager.get_term("base_velocity")
    command.cfg.evaluation_case, command.cfg.evaluation_velocity = 0, (0., 0., 0.)
    raw.reset(seed=args.seed)
    peaks = torch.zeros(10, device=raw.device)
    arm_peak, tilt_peak, contacts = 0., 0., torch.zeros(2, device=raw.device)
    contact_height_error = contact_samples = 0.
    failed = interrupted = False
    count = 0
    requested = min(round(args.probe_seconds / raw.step_dt), raw.max_episode_length)
    with (output / "pd_probe_steps.jsonl").open("w", encoding="utf-8") as stream:
        with torch.no_grad():
            for step in range(max(1, requested)):
                if not app.is_running():
                    interrupted = True
                    break
                _, _, dones, info = env.step(torch.zeros(1, 10, device=raw.device))
                state = info["standwalk_state"]
                count += 1
                peaks = torch.maximum(peaks, state["leg_torque_fraction"][0])
                arm_peak = max(arm_peak, float(state["arm_tracking_error"][0].abs().max()))
                tilt_peak = max(tilt_peak, float(state["tilt_deg"][0]))
                contacts += (state["foot_contact_force_z"][0] > 5.).float()
                in_contact = state["foot_contact_force_z"][0] > 5.
                contact_height_error += float(state["foot_sole_clearance"][0][in_contact].abs().sum())
                contact_samples += int(in_contact.sum())
                stream.write(json.dumps({"time_s": count * raw.step_dt,
                    **{key: value[0].tolist() for key, value in state.items()}}) + "\n")
                if bool(dones[0]):
                    failed = not bool(info["time_outs"][0])
                    break
    robot = raw.scene["robot"]
    report = {"kind": "zero_action_PD_hold", "is_walking_acceptance": False,
        "completed": not failed and not interrupted and count >= max(1, requested),
        "duration_s": count * raw.step_dt, "failed": failed, "interrupted": interrupted,
        "total_mass_kg": float(robot.root_physx_view.get_masses()[0].sum()),
        "leg_joints": list(LEG_JOINTS), "peak_leg_torque_fraction": peaks.tolist(),
        "peak_arm_error_rad": arm_peak, "peak_tilt_deg": tilt_peak,
        "foot_contact_fraction": (contacts / max(count, 1)).tolist(),
        "contact_sole_clearance_abs_mean_m": contact_height_error / contact_samples if contact_samples else None,
        "note": "COM versus collision-geometry XY bounds is diagnostic only, not a contact stability certificate."}
    (output / "pd_probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[PD PROBE] {report}", flush=True)


def evaluate(env, runner, args, output, app):
    import torch
    from bipedal_locomotion.controllers.standwalk_core import EVAL_CASES, EpisodeMetrics
    from bipedal_locomotion.controllers.locomotion_v4 import MotionProgress
    from bipedal_locomotion.controllers.locomotion_biped import SteppingMetrics
    is_biped = getattr(args, "design", "baseline") == "biped"
    cases = dict(EVAL_CASES)
    if is_biped:
        cases["forward"] = (1, (.25, 0., 0.))
    raw = env.unwrapped
    command = raw.command_manager.get_term("base_velocity")
    command.cfg.resampling_time_range = (1e9, 1e9)
    command.time_left[:] = 1e9
    policy = runner.get_inference_policy(device=raw.device)
    encoder = runner.get_inference_encoder(device=raw.device)
    rows, interrupted = [], False
    with (output / "evaluation_steps.jsonl").open("w", encoding="utf-8") as stream:
        for case in args.eval_cases:
            role, target = cases[case]
            if any(abs(v) > limit for v, limit in zip(target, args.velocity_limits)):
                raise ValueError(f"Evaluation target {case} exceeds the configured command envelope.")
            command.cfg.evaluation_case, command.cfg.evaluation_velocity = role, target
            for trial in range(args.episodes):
                raw.reset(seed=args.seed + trial)
                obs = env.get_observations()
                metrics, completed, steps = EpisodeMetrics(), False, 0
                progress = MotionProgress()
                stepping = SteppingMetrics() if is_biped else None
                robot = raw.scene["robot"]
                payload_id = robot.find_bodies("paint_payload_Link")[0][0]
                masses = robot.root_physx_view.get_masses()[0]
                payload_mass, total_mass = float(masses[payload_id]), float(masses.sum())
                peak_force, peak_torque = 0., 0.
                with torch.no_grad():
                    for step in range(raw.max_episode_length):
                        if not app.is_running():
                            interrupted = True
                            break
                        step_started = time.perf_counter()
                        action = policy(torch.cat((encoder(obs["obsHistory"].flatten(1)), obs["policy"], obs["commands"]), dim=-1))
                        obs, reward, dones, infos = env.step(action)
                        state = infos["standwalk_state"]
                        peak_force = max(peak_force, float(state["payload_wrench"][0, :2].norm()))
                        peak_torque = max(peak_torque, float(state["payload_wrench"][0, 2].abs()))
                        if not all(torch.isfinite(value).all() for value in (*obs.values(), *state.values(), reward)):
                            raise FloatingPointError("Nonfinite v4 evaluation state.")
                        steps = step + 1
                        t = steps * raw.step_dt
                        metrics.update(t, state["command"][0], state["linear_velocity"][0],
                            state["yaw_rate"][0], state["position"][0, :2], state["tilt_deg"][0], raw.step_dt)
                        progress.update(t, state["command"][0], state["linear_velocity"][0],
                                        state["yaw_rate"][0], raw.step_dt)
                        if stepping is not None:
                            stepping.update(t, state["command"][0], state["foot_contact_force_z"][0],
                                state["foot_sole_clearance"][0], state["foot_velocity_world"][0], raw.step_dt)
                        stream.write(json.dumps({"case": case, "trial": trial, "time_s": t,
                            **{key: value[0].tolist() for key, value in state.items()}, "done": bool(dones[0])}) + "\n")
                        if getattr(args, "real_time", False):
                            time.sleep(max(0., raw.step_dt - (time.perf_counter() - step_started)))
                        if bool(dones[0]):
                            completed = bool(infos["time_outs"][0]) and t >= raw.max_episode_length_s - raw.step_dt
                            break
                motion = progress.result(require_stop=case in ("start_stop", "stop_start"))
                result = metrics.result(steps * raw.step_dt, completed,
                    require_standing=case in ("stand", "start_stop", "stop_start"),
                    require_stop=False)
                result["stop_latency_s"] = motion["observed_stop_latency_s"]
                result["passed"] = bool(result["passed"] and motion["motion_passed"])
                steps_result = stepping.result() if stepping else {}
                if stepping:
                    result["passed"] &= steps_result["stepping_passed"]
                rows.append({"case": case, "seed": args.seed + trial,
                    "payload_mass_kg": payload_mass, "robot_total_mass_kg": total_mass,
                    "max_applied_force_n": peak_force, "max_applied_yaw_torque_nm": peak_torque,
                    **result, **motion, **steps_result})
                print(f"[EVAL] {case} trial={trial} {rows[-1]}", flush=True)
                if interrupted:
                    break
            if interrupted:
                break
    summary = {"interrupted": interrupted, "terrain_level": args.terrain_level,
        "acceptance_version": "v4_biped_stepping_v1" if is_biped else "v4_motion_progress_v1",
        "all_passed": not interrupted and len(rows) == len(args.eval_cases) * args.episodes and all(row["passed"] for row in rows),
        "episodes": rows}
    if is_biped:
        summary.update({"design": "biped", "skill_stage": args.skill_stage,
            "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint_path).read_bytes()).hexdigest()})
    (output / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(args, app):
    import torch
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    from rsl_rl.runner import OnPolicyRunner
    import bipedal_locomotion  # noqa: F401 (registers tasks)
    from bipedal_locomotion.controllers.paint_payload import create_payload_asset
    from bipedal_locomotion.controllers.locomotion_v4 import contract, check_checkpoint, ExportedPolicy
    from bipedal_locomotion.tasks.locomotion_v4.env_cfg import configure_stage, configure_experiment, configure_biped

    settings = settings_for_args(args)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir or ROOT / "logs/rsl_rl/sfyg_paint_locomotion_v4" /
                  (args.stage + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    task = "Isaac-Limx-SFYG-TRON2A-Paint-Locomotion-v4"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs or (1 if args.mode in ("eval", "probe") else 1024))
    cfg.seed = args.seed
    configure_stage(cfg, settings, args.velocity_limits, evaluation=args.mode == "eval")
    configure_experiment(cfg, args.gait, args.gait_weight, args.command_profile, args.tracking_std)
    if args.design == "biped":
        configure_biped(cfg, args.skill_stage, evaluation=args.mode != "train", gait_weight=args.gait_weight)
    if getattr(args, "quick", False):
        cfg.viewer.eye, cfg.viewer.lookat = (2.5, 2.5, 1.8), (0., 0., .65)
    if args.mode == "eval" and settings.terrain == "rough":
        cfg.scene.terrain.max_init_terrain_level = args.terrain_level
    source = cfg.scene.robot.spawn.usd_path
    cfg.scene.robot.spawn.usd_path = create_payload_asset(source, output / "robot_payload.usdc", .5,
                                                         (.20, .12, .08), (.15, 0., 0.))
    agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent.seed, agent.device = args.seed, args.device
    agent.max_iterations, agent.save_interval = args.max_iterations, args.save_interval
    configure_optimizer(agent, args.design)
    env = gym.make(task, cfg=cfg, render_mode="rgb_array" if getattr(args, "video", False) else None)
    runner = None
    try:
        raw = env.unwrapped
        if args.mode == "eval" and settings.terrain == "rough":
            # Set every evaluation env to an explicit row; max_init alone samples lower rows too.
            terrain = raw.scene.terrain
            terrain.terrain_levels[:] = args.terrain_level
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            raw.scene.env_origins[:] = terrain.env_origins
        env = RslRlVecEnvWrapper(env)
        obs = env.get_observations()
        expected_policy_width = 135 if cfg.biped else (123 if cfg.locomotion_gait is not None else 117)
        if (obs["policy"].ndim != 2 or obs["policy"].shape[1] != expected_policy_width
                or tuple(obs["obsHistory"].shape[1:]) != (10, 36) or obs["commands"].shape[1] != 3
                or env.num_actions != 10 or not all(torch.isfinite(v).all() for v in obs.values())):
            raise RuntimeError("v4 observation/action preflight failed.")
        manifest = contract(obs["policy"].shape[1], args.stage, args.velocity_limits, settings, cfg.locomotion_gait, cfg.biped)
        manifest["training_objective"] = {"gait": args.gait, "gait_weight": args.gait_weight,
            "command_profile": args.command_profile, "tracking_std": args.tracking_std}
        if cfg.biped:
            manifest["training_objective"].update({"design": "biped", "skill_stage": args.skill_stage,
                "reward_recipe": "dense_load_clearance_shortfall_v1"})
        manifest["optimizer_config"] = agent.algorithm.to_dict()
        robot = raw.scene["robot"]
        payload_ids, names = robot.find_bodies("paint_payload_Link")
        imported = robot.data.default_mass[:, payload_ids[0]]
        actual = robot.root_physx_view.get_masses()[:, payload_ids[0]]
        if len(names) != 1 or not torch.allclose(imported, torch.full_like(imported, .5), rtol=.05):
            raise RuntimeError("Payload import mass is inconsistent with the authored 0.5 kg asset.")
        if not bool(((actual >= settings.payload_mass[0] - 1e-5) & (actual <= settings.payload_mass[1] + 1e-5)).all()):
            raise RuntimeError("Runtime payload mass is outside the configured randomization range.")
        manifest.update({"seed": args.seed, "terrain_level": args.terrain_level,
            "preflight_payload_mass_kg": actual.tolist(),
            "base_commit": "457f0bba8d89fd65e29bbbb30325159f5b44daa7", "arguments": vars(args),
            "source_asset": str(source), "prepared_asset": cfg.scene.robot.spawn.usd_path,
            "arm_default_joint_pos": {key: value for key, value in cfg.scene.robot.init_state.joint_pos.items() if "arm" in key},
            "agent": agent.to_dict(), "observation_shapes": {key: list(v.shape[1:]) for key, v in obs.items()}})
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        manifest["code_commit"] = result.stdout.strip()
        source_files = [Path(__file__), ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers/locomotion_v4.py",
            ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers/locomotion_biped.py",
            *sorted((ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion_v4").glob("*.py")),
            ROOT / "rsl_rl/rsl_rl/algorithm/ppo.py",
            ROOT / "rsl_rl/rsl_rl/runner/on_policy_runner.py",
            ROOT / "rsl_rl/rsl_rl/storage/rollout_storage.py",
            ROOT / "rsl_rl/rsl_rl/modules/actor_critic.py",
            ROOT / "rsl_rl/rsl_rl/modules/mlp_encoder.py",
            ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers/standwalk_core.py",
            ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/standwalk/mdp.py",
            ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/standwalk/env.py",
            ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF_TRON2A/limx_base_env_cfg.py",
            ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/rewards.py"]
        manifest["source_files_sha256"] = {}
        for source_file in source_files:
            relative = source_file.relative_to(ROOT)
            target = output / "sources" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)
            manifest["source_files_sha256"][relative.as_posix()] = hashlib.sha256(source_file.read_bytes()).hexdigest()
        (output / "contract.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        if args.mode == "probe":
            probe_pd(env, args, output, app)
            return

        class V4Runner(OnPolicyRunner):
            def log(self, locs, *a, **kw):
                super().log(locs, *a, **kw)
                self.writer.add_scalar("Curriculum/standing_step_fraction", raw.pop_standing_step_fraction(), locs["it"])
                for key, value in raw.pop_biped_metrics().items():
                    self.writer.add_scalar(key, value, locs["it"])
                completed = locs["it"] + 1
                if completed % self.save_interval == 0:
                    self.current_learning_iteration = completed
                    self.save(output / f"model_{completed}.pt")

            def save(self, path, infos=None):
                # Ignore the generic runner's off-by-one periodic call. log()
                # saves exact intervals; the final call already has the right index.
                path = Path(path)
                if path.stem != f"model_{self.current_learning_iteration}":
                    return
                super().save(str(path), infos)
                saved = torch.load(path, map_location="cpu", weights_only=False)
                saved["v4_contract"] = manifest
                saved["learning_rate"] = self.alg.learning_rate
                torch.save(saved, path)

        runner = V4Runner(env, agent.to_dict(), log_dir=str(output), device=args.device)
        if args.checkpoint_path:
            source_checkpoint = Path(args.checkpoint_path).resolve()
            checkpoint = torch.load(source_checkpoint, map_location=args.device, weights_only=False)
            check_checkpoint(checkpoint, manifest)
            if args.design == "biped" and args.mode == "train":
                from bipedal_locomotion.controllers.locomotion_biped import validate_promotion
                validate_promotion(checkpoint["v4_contract"], args.skill_stage, args.promotion_report,
                                   hashlib.sha256(source_checkpoint.read_bytes()).hexdigest())
            if not all(torch.isfinite(tensor).all() for key in ("model_state_dict", "encoder_state_dict")
                       for tensor in checkpoint[key].values()):
                raise ValueError("v4 checkpoint contains nonfinite parameters.")
            if args.resume and (checkpoint["v4_contract"]["settings"] != manifest["settings"]
                    or checkpoint["v4_contract"]["velocity_limits"] != manifest["velocity_limits"]
                    or checkpoint["v4_contract"].get("training_objective", {
                        "gait": "legacy", "gait_weight": .75, "command_profile": "mixed", "tracking_std": .5
                    }) != manifest["training_objective"]
                    or (args.design == "biped" and checkpoint["v4_contract"].get("optimizer_config") != manifest["optimizer_config"])):
                raise ValueError("Resume requires identical physics, commands and objective; use weights-only stage promotion.")
            runner.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
            runner.alg.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
            if args.resume:
                runner.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                runner.alg.extra_optimizer.load_state_dict(checkpoint["encoder_optimizer_state_dict"])
                runner.current_learning_iteration = checkpoint["iter"]
                runner.alg.learning_rate = checkpoint["learning_rate"]
            manifest["source_checkpoint"] = str(source_checkpoint)
            manifest["source_sha256"] = hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()
            (output / "contract.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"[V4] stage={args.stage} settings={settings} shapes={manifest['observation_shapes']}", flush=True)
        if args.mode == "train":
            runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
        elif getattr(args, "quick", False):
            from play_locomotion_v4 import preview
            preview(env, runner, args, output, app)
        else:
            summary = evaluate(env, runner, args, output, app)
            print(f"[V4] evaluation all_passed={summary['all_passed']}", flush=True)
        export = ExportedPolicy(runner.alg.actor_critic.actor, runner.alg.encoder.encoder, obs["policy"].shape[1])
        with (output / "locomotion_v4.pt").open("wb") as stream:
            torch.jit.save(torch.jit.script(export), stream)
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.close()
        env.close()


if __name__ == "__main__":
    parser = parser_for_task()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    try:
        main(args, launcher.app)
    except BaseException:
        # Some Kit versions exit while closing the app; print BEFORE close.
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        launcher.app.close()
