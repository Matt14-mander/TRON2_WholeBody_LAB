"""Strict warm-start stand/walk PPO and fixed-sequence evaluation.

Prepare the asset before spawning. This is a fresh fine-tuning run, not optimizer
resume. The low-level 78/420/10 interface remains compatible with the teacher.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rsl_rl"))
sys.path.insert(0, str(ROOT / "exts/bipedal_locomotion"))


def parser_for_task():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--stage", choices=("stand", "transition", "expanded"), default="stand")
    parser.add_argument("--payload_mode", choices=("none", "attached"), default="none")
    parser.add_argument("--num_envs", type=int, help="Default: 512 train, 1 eval. Evaluation requires 1.")
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation repetitions per fixed case.")
    parser.add_argument("--eval_cases", nargs="+", default=("stand", "walk", "start_stop", "stop_start"),
                        choices=("stand", "walk", "start_stop", "stop_start", "slow", "forward", "left", "right", "yaw_left", "yaw_right"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir")
    parser.add_argument("--transition_s", type=float, default=1.)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--noise_std", type=float, default=.2)
    parser.add_argument("--train_encoder", action="store_true", help="Opt in to supervised history encoder adaptation at 1e-5.")
    return parser


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(env, runner, args, output, app):
    import torch
    from bipedal_locomotion.controllers.standwalk_core import EVAL_CASES, EpisodeMetrics
    raw = env.unwrapped
    command = raw.command_manager.get_term("base_velocity")
    policy = runner.get_inference_policy(device=raw.device)
    encoder = runner.get_inference_encoder(device=raw.device)
    results = []
    interrupted = False
    with (output / "evaluation_episodes.jsonl").open("w", encoding="utf-8") as episodes, \
            (output / "evaluation_steps.jsonl").open("w", encoding="utf-8") as telemetry:
        for case in args.eval_cases:
            command.cfg.evaluation_case, command.cfg.walk_velocity = EVAL_CASES[case]
            for trial in range(args.episodes):
                if not app.is_running():
                    interrupted = True
                    break
                # Same trial seed across cases and payload modes for reproducibility.
                raw.reset(seed=args.seed + trial)
                obs = env.get_observations()
                metrics = EpisodeMetrics()
                completed = False
                steps = 0
                with torch.no_grad():
                    for step in range(raw.max_episode_length):
                        if not app.is_running():
                            interrupted = True
                            break
                        action = policy(torch.cat((encoder(obs["obsHistory"].flatten(1)),
                                                  obs["policy"], obs["commands"]), dim=-1))
                        obs, reward, dones, infos = env.step(action)
                        state = infos["standwalk_state"]
                        values = torch.cat((obs["policy"].flatten(), state["linear_velocity"].flatten(),
                                            state["position"].flatten(), state["tilt_deg"].flatten(), reward.flatten()))
                        if not torch.isfinite(values).all():
                            raise RuntimeError("Non-finite stand/walk evaluation state.")
                        time = (step + 1) * raw.step_dt
                        metrics.update(time, state["command"][0], state["linear_velocity"][0],
                                       state["yaw_rate"][0], state["position"][0, :2], state["tilt_deg"][0], raw.step_dt)
                        steps = step + 1
                        telemetry.write(json.dumps({"case": case, "trial": trial, "time_s": time,
                            "command": state["command"][0].tolist(), "velocity": state["linear_velocity"][0].tolist(),
                            "yaw_rate": float(state["yaw_rate"][0]), "height": float(state["position"][0, 2]),
                            "tilt_deg": float(state["tilt_deg"][0]), "done": bool(dones[0])}) + "\n")
                        if bool(dones[0]):
                            completed = bool(infos["time_outs"][0]) and time >= raw.max_episode_length_s - raw.step_dt
                            break
                row = {"case": case, "trial": trial, "seed": args.seed + trial, "payload_mode": args.payload_mode,
                       **metrics.result(steps * raw.step_dt, completed,
                                        require_standing=case in ("stand", "start_stop", "stop_start"),
                                        require_stop=case in ("start_stop", "stop_start"))}
                results.append(row)
                episodes.write(json.dumps(row) + "\n")
                episodes.flush()
                print(f"[EVAL] {case} trial={trial} completed={completed} passed={row['passed']} "
                      f"duration={row['duration_s']:.2f}s tilt={row['max_tilt_deg']:.2f}deg", flush=True)
                if interrupted:
                    break
            if interrupted:
                break
    summary = {"payload_mode": args.payload_mode, "interrupted": interrupted,
               "cases": {case: {"episodes": sum(r["case"] == case for r in results),
                                "completed": sum(r["completed"] for r in results if r["case"] == case),
                                "passed": sum(r["passed"] for r in results if r["case"] == case)} for case in args.eval_cases},
               "all_passed": not interrupted and len(results) == len(args.eval_cases) * args.episodes and all(r["passed"] for r in results)}
    (output / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(args, app):
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runner import OnPolicyRunner
    import bipedal_locomotion  # noqa: F401
    from bipedal_locomotion.controllers.paint_payload import create_payload_asset, create_unloaded_robot_asset
    from bipedal_locomotion.controllers.standwalk_core import SCHEMA, load_warm_start, freeze_history_encoder
    from bipedal_locomotion.controllers.paint_teacher_core import ARM_JOINT_NAMES, LEG_JOINT_NAMES

    if any(not math.isfinite(x) or x <= 0 for x in (args.transition_s, args.learning_rate, args.noise_std)):
        raise ValueError("Transition time, learning rate and noise std must be finite and positive.")
    if args.transition_s > 2.:
        raise ValueError("Transition must be <=2 seconds to fit both transitions in the 20-second evaluation.")
    num_envs = args.num_envs if args.num_envs is not None else (512 if args.mode == "train" else 1)
    if num_envs < 1 or args.max_iterations < 1 or args.episodes < 1:
        raise ValueError("Environment, iteration and episode counts must be positive.")
    if args.mode == "eval" and num_envs != 1:
        raise ValueError("Fixed-sequence evaluation uses --num_envs 1.")
    if len(set(args.eval_cases)) != len(args.eval_cases):
        raise ValueError("Evaluation cases must be unique.")
    source = Path(args.checkpoint_path).expanduser().resolve()
    source_hash = digest_file(source)
    output = Path(args.output_dir or ROOT / "logs" / "rsl_rl" / "sfyg_wholebody_standwalk_finetune" /
                  datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    task = "Isaac-Limx-SFYG-TRON2A-WholeBody-StandWalk-Finetune-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=num_envs)
    cfg.seed = args.seed
    cfg.commands.base_velocity.stage = args.stage
    cfg.commands.base_velocity.transition_s = args.transition_s
    source_asset = cfg.scene.robot.spawn.usd_path
    if args.payload_mode == "attached":
        cfg.scene.robot.spawn.usd_path = create_payload_asset(source_asset, output / "robot_payload.usdc",
                                                             .5, (.20, .12, .08), (.15, 0., 0.))
    else:
        cfg.scene.robot.spawn.usd_path = create_unloaded_robot_asset(source_asset, output / "robot_unloaded.usdc")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed, agent_cfg.device = args.seed, args.device
    agent_cfg.algorithm.learning_rate = args.learning_rate
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.logger = "tensorboard"
    torch.manual_seed(args.seed)

    class FinetuneRunner(OnPolicyRunner):
        def log(self, locs, *positional, **kwargs):
            self.completed_iteration = locs["it"] + 1
            return super().log(locs, *positional, **kwargs)

        def save(self, path, infos=None):
            torch.save({"model_state_dict": self.alg.actor_critic.state_dict(),
                        "encoder_state_dict": self.alg.encoder.state_dict(),
                        "optimizer_state_dict": self.alg.optimizer.state_dict(),
                        "iter": getattr(self, "completed_iteration", self.current_learning_iteration),
                        "standwalk": {"schema": SCHEMA, "source_sha256": source_hash,
                                      "stage": args.stage, "payload_mode": args.payload_mode}, "infos": infos}, path)

    env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
    try:
        raw = env.unwrapped
        robot = raw.scene["robot"]
        if args.payload_mode == "attached":
            ids, names = robot.find_bodies("paint_payload_Link", preserve_order=True)
            if len(names) != 1:
                raise RuntimeError("Attached payload was not imported as one articulation link.")
            mass = robot.root_physx_view.get_masses()[:, ids[0]]
            if not torch.allclose(mass, torch.full_like(mass, .5), rtol=.05, atol=1e-5):
                raise RuntimeError("Imported payload mass mismatch.")
        runner = FinetuneRunner(env, agent_cfg.to_dict(), log_dir=str(output), device=args.device)
        obs = env.get_observations()
        if (obs["policy"].shape[1] != 72 or obs["obsHistory"].shape[1:] != (10, 42)
                or obs["commands"].shape[1] != 3 or env.num_actions != 10):
            raise RuntimeError("Fine-tuning must retain the original 78/420/10 locomotion interface.")
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        load_report = load_warm_start(runner.alg.actor_critic, runner.alg.encoder, checkpoint,
                                      allow_critic_input_reset=True)  # Only +/- one body-mass input is permitted.
        if not args.train_encoder:
            freeze_history_encoder(runner.alg)
        frozen_encoder = {k: v.clone() for k, v in runner.alg.encoder.state_dict().items()}
        if args.mode == "train":
            with torch.no_grad():
                runner.alg.actor_critic.logstd.fill_(math.log(args.noise_std))
        contract = {"schema": SCHEMA, "task": task, "args": vars(args), "source_checkpoint": str(source),
                    "source_sha256": source_hash, "load_report": load_report, "optimizer_loaded": False,
                    "encoder_frozen": not args.train_encoder, "actor_input": 78, "encoder_input": 420,
                    "leg_actions": 10, "leg_joints": list(LEG_JOINT_NAMES), "arm_joints": list(ARM_JOINT_NAMES),
                    "control_dt": raw.step_dt, "wrench_input": "zero_5x6; synthetic force disabled",
                    "phase_contract": "episode_time*frequency, continuous through standing; unchanged actor encoding",
                    "env": cfg.to_dict(), "ppo": agent_cfg.to_dict()}
        (output / "contract.json").write_text(json.dumps(contract, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"[FINETUNE] load={load_report}; encoder_frozen={not args.train_encoder}; "
              f"lr={runner.alg.learning_rate:.2e}; output={output}", flush=True)
        if args.mode == "train":
            runner.learn(args.max_iterations, init_at_random_ep_len=False)
            import copy
            for name, module in (("policy.pt", runner.alg.actor_critic.actor), ("encoder.pt", runner.alg.encoder.encoder)):
                torch.jit.script(copy.deepcopy(module).cpu().eval()).save(str(output / name))
        else:
            evaluate(env, runner, args, output, app)
        if not args.train_encoder and any(not torch.equal(v, runner.alg.encoder.state_dict()[k]) for k, v in frozen_encoder.items()):
            raise RuntimeError("Frozen history encoder changed during fine-tuning.")
        print(f"[FINETUNE] complete; output={output}", flush=True)
    finally:
        if "runner" in locals() and runner.writer is not None:
            runner.writer.close()
        env.close()


if __name__ == "__main__":
    parser = parser_for_task()
    if "--help" in sys.argv or "-h" in sys.argv:
        parser.parse_args()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        main(args, app)
    finally:
        app.close()
