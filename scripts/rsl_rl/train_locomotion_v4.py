"""Train/evaluate the full-robot PAINT replacement backbone, starting from main.

No v2/v3 weights are accepted. Stage promotion uses --checkpoint_path (weights
only); --resume also restores optimizers/iteration, but resets simulator episodes.
"""
from dataclasses import replace
from datetime import datetime
import argparse
import hashlib
import json
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
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--stage", choices=("nominal", "robust", "interaction", "rough"), default="nominal")
    parser.add_argument("--checkpoint_path", help="v4 checkpoint only; omitted = fresh random initialization")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--max_iterations", type=int, default=10000, help="Additional PPO iterations")
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir")
    parser.add_argument("--velocity_limits", type=float, nargs=3, default=(.35, .2, .4))
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
    stage = STAGES[args.stage]
    stage = replace(stage, payload_mass=tuple(args.payload_mass_range) if args.payload_mass_range else stage.payload_mass,
                    force_n=args.force_n if args.force_n is not None else stage.force_n,
                    yaw_torque_nm=args.yaw_torque_nm if args.yaw_torque_nm is not None else stage.yaw_torque_nm)
    validate_ranges(stage.payload_mass, stage.force_n, stage.yaw_torque_nm, args.velocity_limits)
    if min(args.max_iterations, args.save_interval, args.episodes) < 1 or (args.num_envs is not None and args.num_envs < 1):
        raise ValueError("Iteration, save, episode and environment counts must be positive.")
    if not 0 <= args.terrain_level <= 9 or len(args.eval_cases) != len(set(args.eval_cases)):
        raise ValueError("Invalid terrain level or duplicate evaluation cases.")
    if (args.resume or args.mode == "eval") and not args.checkpoint_path:
        raise ValueError("Evaluation/resume requires --checkpoint_path.")
    if args.mode == "eval" and args.num_envs not in (None, 1):
        raise ValueError("Fixed-sequence evaluation requires --num_envs 1.")
    return stage


def evaluate(env, runner, args, output, app):
    import torch
    from bipedal_locomotion.controllers.standwalk_core import EVAL_CASES, EpisodeMetrics
    raw = env.unwrapped
    command = raw.command_manager.get_term("base_velocity")
    command.cfg.resampling_time_range = (1e9, 1e9)
    command.time_left[:] = 1e9
    policy = runner.get_inference_policy(device=raw.device)
    encoder = runner.get_inference_encoder(device=raw.device)
    rows, interrupted = [], False
    with (output / "evaluation_steps.jsonl").open("w", encoding="utf-8") as stream:
        for case in args.eval_cases:
            role, target = EVAL_CASES[case]
            if any(abs(v) > limit for v, limit in zip(target, args.velocity_limits)):
                raise ValueError(f"Evaluation target {case} exceeds the configured command envelope.")
            command.cfg.evaluation_case, command.cfg.evaluation_velocity = role, target
            for trial in range(args.episodes):
                raw.reset(seed=args.seed + trial)
                obs = env.get_observations()
                metrics, completed, steps = EpisodeMetrics(), False, 0
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
                        stream.write(json.dumps({"case": case, "trial": trial, "time_s": t,
                            **{key: value[0].tolist() for key, value in state.items()}, "done": bool(dones[0])}) + "\n")
                        if getattr(args, "real_time", False):
                            time.sleep(max(0., raw.step_dt - (time.perf_counter() - step_started)))
                        if bool(dones[0]):
                            completed = bool(infos["time_outs"][0]) and t >= raw.max_episode_length_s - raw.step_dt
                            break
                rows.append({"case": case, "seed": args.seed + trial,
                    "payload_mass_kg": payload_mass, "robot_total_mass_kg": total_mass,
                    "max_applied_force_n": peak_force, "max_applied_yaw_torque_nm": peak_torque,
                    **metrics.result(steps * raw.step_dt, completed,
                        require_standing=case in ("stand", "start_stop", "stop_start"),
                        require_stop=case in ("start_stop", "stop_start"))})
                print(f"[EVAL] {case} trial={trial} {rows[-1]}", flush=True)
                if interrupted:
                    break
            if interrupted:
                break
    summary = {"interrupted": interrupted, "terrain_level": args.terrain_level,
        "all_passed": not interrupted and len(rows) == len(args.eval_cases) * args.episodes and all(row["passed"] for row in rows),
        "episodes": rows}
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
    from bipedal_locomotion.tasks.locomotion_v4.env_cfg import configure_stage

    settings = settings_for_args(args)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir or ROOT / "logs/rsl_rl/sfyg_paint_locomotion_v4" /
                  (args.stage + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    task = "Isaac-Limx-SFYG-TRON2A-Paint-Locomotion-v4"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs or (1 if args.mode == "eval" else 1024))
    cfg.seed = args.seed
    configure_stage(cfg, settings, args.velocity_limits, evaluation=args.mode == "eval")
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
        if (obs["policy"].ndim != 2 or obs["policy"].shape[1] <= 36
                or tuple(obs["obsHistory"].shape[1:]) != (10, 36) or obs["commands"].shape[1] != 3
                or env.num_actions != 10 or not all(torch.isfinite(v).all() for v in obs.values())):
            raise RuntimeError("v4 observation/action preflight failed.")
        manifest = contract(obs["policy"].shape[1], args.stage, args.velocity_limits, settings)
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
            *sorted((ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion_v4").glob("*.py")),
            ROOT / "rsl_rl/rsl_rl/algorithm/ppo.py",
            ROOT / "rsl_rl/rsl_rl/runner/on_policy_runner.py",
            ROOT / "rsl_rl/rsl_rl/storage/rollout_storage.py",
            ROOT / "rsl_rl/rsl_rl/modules/actor_critic.py",
            ROOT / "rsl_rl/rsl_rl/modules/mlp_encoder.py"]
        manifest["source_files_sha256"] = {}
        for source_file in source_files:
            relative = source_file.relative_to(ROOT)
            target = output / "sources" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)
            manifest["source_files_sha256"][relative.as_posix()] = hashlib.sha256(source_file.read_bytes()).hexdigest()
        (output / "contract.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

        class V4Runner(OnPolicyRunner):
            def log(self, locs, *a, **kw):
                super().log(locs, *a, **kw)
                self.writer.add_scalar("Curriculum/standing_step_fraction", raw.pop_standing_step_fraction(), locs["it"])
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
            if not all(torch.isfinite(tensor).all() for key in ("model_state_dict", "encoder_state_dict")
                       for tensor in checkpoint[key].values()):
                raise ValueError("v4 checkpoint contains nonfinite parameters.")
            if args.resume and (checkpoint["v4_contract"]["settings"] != manifest["settings"]
                    or checkpoint["v4_contract"]["velocity_limits"] != manifest["velocity_limits"]):
                raise ValueError("Resume requires identical physical settings and command ranges; use weights-only stage promotion.")
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
