"""Train/evaluate a privileged force-following teacher over frozen locomotion.

Use this entry point: it prepares a separate payload USD before gym.make and
requires the frozen low-level checkpoint independently of teacher weights.
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


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "play", "baseline", "smoke"), default="train")
    parser.add_argument("--task", default="Isaac-Limx-SFYG-TRON2A-Paint-Teacher-Flat-v0")
    parser.add_argument("--checkpoint_path", required=True, help="Frozen flat WholeBody locomotion checkpoint.")
    parser.add_argument("--teacher_checkpoint", help="High-level checkpoint, required for play.")
    parser.add_argument("--output_dir", help="New empty run directory; defaults to a timestamped logs directory.")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=1000, help="High-level steps for smoke/baseline/play.")
    parser.add_argument("--payload_mass", type=float, default=.5)
    parser.add_argument("--force_amplitude", type=float, default=5.)
    parser.add_argument("--torque_amplitude", type=float, default=.5)
    parser.add_argument("--zero_force", action="store_true", help="Support diagnostic without interaction forces.")
    parser.add_argument("--initial_force_scale", type=float, default=.2)
    parser.add_argument("--curriculum_steps", type=int, default=25000,
                        help="25 Hz teacher steps, shared across environments, not PPO iterations.")
    parser.add_argument("--velocity_scale", type=float, nargs=3, default=(.15, .08, .25))
    parser.add_argument("--arm_scale", type=float, default=.1)
    parser.add_argument("--metric_settle_s", type=float, default=4.,
                        help="Exclude this interval after EVERY reset from settled tracking metrics only.")
    return parser


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(args, app):
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runner import OnPolicyRunner
    import bipedal_locomotion  # noqa: F401
    from bipedal_locomotion.controllers.paint_payload import create_payload_asset
    from bipedal_locomotion.controllers.paint_teacher_core import SCHEMA, LEG_JOINT_NAMES, ARM_JOINT_NAMES
    from bipedal_locomotion.tasks.teacher.mdp import controller, payload_state

    class TeacherRunner(OnPolicyRunner):
        def log(self, locs, *positional, **kwargs):
            # The repository runner only updates current_learning_iteration at
            # the end of learn(). Preserve the actual progress in periodic files.
            self.completed_iteration = locs["it"] + 1
            return super().log(locs, *positional, **kwargs)

        def save(self, path, infos=None):
            torch.save({"model_state_dict": self.alg.actor_critic.state_dict(),
                "encoder_state_dict": self.alg.encoder.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": getattr(self, "completed_iteration", self.current_learning_iteration),
                "schema": SCHEMA, "locomotion_sha256": checkpoint_hash, "infos": infos}, path)

    if "Paint-Teacher-Flat" not in args.task:
        raise ValueError("Use a registered Paint-Teacher-Flat task.")
    if args.num_envs < 1 or args.steps < 1 or args.max_iterations < 1:
        raise ValueError("Environment, step and iteration counts must be positive.")
    if any(not math.isfinite(x) or x <= 0 for x in
           (args.payload_mass, args.force_amplitude, args.torque_amplitude, args.arm_scale, *args.velocity_scale)):
        raise ValueError("Payload and force/torque amplitudes must be finite and positive.")
    if not 0 < args.initial_force_scale <= 1 or args.curriculum_steps < 1:
        raise ValueError("Invalid force curriculum scale/steps")
    if not math.isfinite(args.metric_settle_s) or not 0 <= args.metric_settle_s < 20:
        raise ValueError("metric_settle_s must be finite and in [0,20)")
    if args.mode == "play" and not args.teacher_checkpoint:
        raise ValueError("--mode play requires --teacher_checkpoint.")
    if args.mode != "play" and args.teacher_checkpoint:
        raise ValueError("Teacher resume is not supported in v1; use --teacher_checkpoint only with play.")
    checkpoint = Path(args.checkpoint_path).expanduser().resolve()
    checkpoint_hash = sha256(checkpoint)
    output = Path(args.output_dir or ROOT / "logs" / "rsl_rl" / "paint_teacher_flat" /
                  datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new empty --output_dir.")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.actions.teacher.locomotion_checkpoint = str(checkpoint)
    cfg.actions.teacher.payload_mass = args.payload_mass
    cfg.actions.teacher.native_support = True
    cfg.actions.teacher.velocity_scale = tuple(args.velocity_scale)
    cfg.actions.teacher.arm_scale = args.arm_scale
    cfg.actions.teacher.slew_rate = (.3, .2, .5, .15, .15, .15, .15, .15, .15)
    cfg.commands.interaction.max_wrench = (args.force_amplitude, args.force_amplitude, args.torque_amplitude)
    cfg.commands.interaction.initial_scale = args.initial_force_scale
    cfg.commands.interaction.ramp_steps = args.curriculum_steps
    cfg.commands.interaction.velocity_limits = tuple(args.velocity_scale)
    if args.zero_force:
        cfg.commands.interaction.zero_probability = 1.
    if args.mode in ("play", "baseline"):
        cfg.commands.interaction.initial_scale = 1.
        cfg.commands.interaction.ramp_steps = 1
    cfg.scene.robot.spawn.usd_path = create_payload_asset(
        cfg.scene.robot.spawn.usd_path, output / "robot_payload.usdc", args.payload_mass,
        (.20, .12, .08), (.15, 0., 0.))
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.logger = "tensorboard"
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
    try:
        raw = env.unwrapped
        term = controller(raw)
        frozen_before = {k: v.clone() for k, v in term.locomotion.state_dict().items()}
        runner = TeacherRunner(env, agent_cfg.to_dict(), log_dir=str(output), device=args.device)
        runner.alg.encoder.requires_grad_(False)
        # The repository ActorCritic currently ignores init_noise_std.
        with torch.no_grad():
            runner.alg.actor_critic.logstd.fill_(math.log(.3))
        obs = env.get_observations()
        term.update_support()
        (output / "wrench_model_check.json").write_text(
            json.dumps(term.support_check, indent=2) + "\n", encoding="utf-8")
        if obs["policy"].shape[1] != 99 or obs["critic"].shape[1] != 106 or env.num_actions != 9:
            raise RuntimeError("Teacher observation/action contract mismatch.")
        if runner.alg.extra_optimizer is not None or any(p.requires_grad for p in term.locomotion.parameters()):
            raise RuntimeError("Only teacher PPO may be trained; no estimator/locomotion optimizer is allowed.")
        contract = {
            "schema": SCHEMA, "args": vars(args), "locomotion_checkpoint": str(checkpoint),
            "locomotion_sha256": checkpoint_hash, "leg_joints": list(LEG_JOINT_NAMES),
            "arm_joints": list(ARM_JOINT_NAMES), "physics_dt": .005, "low_dt": .02, "teacher_dt": .04,
            "teacher_policy_dim": 99, "teacher_commands_dim": 3, "teacher_actor_input_dim": 102,
            "teacher_critic_input_dim": 109, "action_dim": 9,
            "actor_input_order": ["force_scaled(3)", "base_linear_velocity(3)", "base_angular_velocity_scaled(3)",
                "base_gravity(3)", "arm_history_oldest_first_4x18(72)", "payload_relative_position_base(3)",
                "payload_local_gravity(3)", "applied_high_action_scaled(9)", "admittance_reference_payload_local(3)"],
            "arm_history_row": ["q_minus_nominal(6)", "dq_times_0.1(6)", "preceding_executed_q_target_minus_nominal(6)"],
            "force_semantics": "external_on_payload; frame=payload_local; point=payload_COM",
            "force_timing": "current known force held over the upcoming teacher interval; no future profile given",
            "support_version": "quasistatic_gravity_external_v1",
            "low_wrench_input": "added-body gravity + current known external payload wrench; held 5x6; base axes, base origin; observation only",
            "arm_gravity_compensation": {"enabled": True, "ramp_s": cfg.actions.teacher.gravity_ramp_s,
                "limit_Nm": cfg.actions.teacher.gravity_effort_limit, "external_force_compensated": False},
            "nominal_arm": term.nominal[0].tolist(),
            "initial_low_wrench_normalized": term.low_wrench_prediction[0].tolist(),
            "intent_estimator": False,
            "velocity_scale": list(cfg.actions.teacher.velocity_scale), "arm_scale": cfg.actions.teacher.arm_scale,
            "slew_rate": list(cfg.actions.teacher.slew_rate),
            "interaction": cfg.commands.interaction.to_dict(), "rewards": cfg.rewards.to_dict(),
            "ppo": agent_cfg.to_dict(),
        }
        (output / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        if args.mode == "train":
            teacher_before = {k: v.clone() for k, v in runner.alg.actor_critic.actor.state_dict().items()}
            runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
            teacher_after = runner.alg.actor_critic.actor.state_dict()
            if (not all(torch.isfinite(v).all() for v in teacher_after.values())
                    or not any(not torch.equal(v, teacher_after[k]) for k, v in teacher_before.items())):
                raise RuntimeError("Teacher actor did not update or produced non-finite weights")
            torch.jit.script(runner.alg.actor_critic.actor.cpu().eval()).save(str(output / "teacher_actor.pt"))
        else:
            if args.mode == "play":
                saved = torch.load(args.teacher_checkpoint, map_location=args.device, weights_only=False)
                source_contract = Path(args.teacher_checkpoint).resolve().parent / "contract.json"
                metadata = json.loads(source_contract.read_text(encoding="utf-8"))
                if (metadata["schema"] != SCHEMA or metadata["locomotion_sha256"] != checkpoint_hash
                        or metadata.get("support_version") != contract["support_version"]
                        or metadata["velocity_scale"] != contract["velocity_scale"]
                        or metadata["arm_scale"] != contract["arm_scale"]):
                    raise ValueError("Teacher checkpoint must use the same schema and frozen locomotion weights.")
                runner.alg.actor_critic.load_state_dict(saved["model_state_dict"], strict=True)
            runner.alg.actor_critic.eval()
            failures = timeouts = 0
            peaks = torch.zeros(3, device=raw.device)
            tracking_sum = 0.
            valid_responses = 0
            settled_tracking_sum = 0.
            settled_responses = 0
            force_active_samples = 0
            steps = 0
            with torch.no_grad(), (output / "rollout.jsonl").open("w", encoding="utf-8") as stream:
                for step in range(args.steps):
                    if not app.is_running():
                        break
                    action = (runner.alg.actor_critic.act_inference(torch.cat((obs["policy"], obs["commands"]), dim=-1))
                              if args.mode == "play" else torch.zeros(raw.num_envs, 9, device=raw.device))
                    force = raw.command_manager.get_command("interaction").clone()
                    force_active_samples += int((force.abs().amax(-1) > 1e-6).sum())
                    reference = raw.command_manager.get_term("interaction").reference.clone()
                    # Capture per-environment age BEFORE step, which may reset
                    # individual environments. A global step cutoff is incorrect.
                    settled_age = raw.episode_length_buf * raw.step_dt >= args.metric_settle_s
                    obs, reward, dones, infos = env.step(action)
                    timeout = infos["time_outs"].bool()
                    failures += int((dones.bool() & ~timeout).sum())
                    timeouts += int(timeout.sum())
                    if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
                        raise RuntimeError("Non-finite teacher rollout.")
                    lin, ang, gravity = payload_state(raw)
                    # Post-reset rows cannot be used as pre-terminal response measurements.
                    valid = ~dones.bool()
                    error = (torch.cat((lin[:, :2], ang[:, 2:3]), dim=-1) - reference).square().sum(-1)
                    settled = valid & settled_age
                    settled_tracking_sum += float(error[settled].sum())
                    settled_responses += int(settled.sum())
                    if valid.any():
                        base_angle = torch.acos((-raw.scene["robot"].data.projected_gravity_b[valid, 2]).clamp(-1, 1))
                        payload_angle = torch.acos((gravity[valid] * term.payload_gravity_reference[valid]).sum(-1).clamp(-1, 1))
                        peaks = torch.maximum(peaks, torch.stack((base_angle.max(), payload_angle.max(),
                                                                  term.delta[valid].abs().max())))
                        tracking_sum += float(error[valid].sum())
                        valid_responses += int(valid.sum())
                    stream.write(json.dumps({"step": step, "force_env0": force[0].tolist(),
                        "reference_env0": reference[0].tolist(), "action_env0": action[0].tolist(),
                        "response_valid_env0": bool(valid[0]), "reward_mean": float(reward.mean()),
                        "settled_response_env0": bool(settled[0]),
                        "tracking_error_sq_env0": float(error[0]) if bool(valid[0]) else None,
                        "failures": failures, "timeouts": timeouts}) + "\n")
                    steps += 1
            result = {"mode": args.mode, "steps": steps, "failures": failures, "timeouts": timeouts,
                      "peak_base_tilt_deg": math.degrees(float(peaks[0])),
                      "peak_payload_tilt_from_reset_deg": math.degrees(float(peaks[1])),
                      "peak_target_change": float(peaks[2]), "valid_responses": valid_responses,
                      "force_active_samples": force_active_samples,
                      "metric_settle_s": args.metric_settle_s,
                      "settled_responses": settled_responses,
                      "settled_tracking_mse": settled_tracking_sum / settled_responses if settled_responses else None,
                      "tracking_mse": tracking_sum / max(1, valid_responses)}
            (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result, indent=2), flush=True)
        if any(not torch.equal(value, term.locomotion.state_dict()[key]) for key, value in frozen_before.items()):
            raise RuntimeError("Frozen locomotion weights changed during teacher execution.")
        (output / "completion.json").write_text(json.dumps({"mode": args.mode,
            "frozen_weights_verified": True, "support_version": contract["support_version"]}) + "\n", encoding="utf-8")
        print(f"[TEACHER] complete; frozen weights verified; output={output}", flush=True)
    finally:
        if "runner" in locals() and runner.writer is not None:
            runner.writer.close()
        env.close()


if __name__ == "__main__":
    # Parse --help without requiring an installed Isaac Sim runtime.
    parser = build_parser()
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
