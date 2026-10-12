"""Collect MPC-free, matched zero/+/- planar wrench responses with a fixed payload."""
from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rsl_rl"))
sys.path.insert(0, str(ROOT / "exts/bipedal_locomotion"))
from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Limx-SFYG-TRON2A-WholeBody-Flat-Play-v0")
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--base_command", nargs=3, type=float, default=(0., 0., 0.))
parser.add_argument("--payload_mass", type=float, default=0.5)
parser.add_argument("--payload_dimensions", nargs=3, type=float, default=(0.20, 0.12, 0.08))
parser.add_argument("--payload_offset", nargs=3, type=float, default=(0.15, 0., 0.))
parser.add_argument("--arm_kp", type=float, default=60.)
parser.add_argument("--arm_kd", type=float, default=6.)
parser.add_argument("--pose_jitter", type=float, default=0.02, help="Per-group bounded hold-target jitter in rad; shared by all seven conditions.")
parser.add_argument("--force_amplitude", type=float, default=5.)
parser.add_argument("--torque_amplitude", type=float, default=0.5)
parser.add_argument("--profile", choices=("trapezoid", "sine"), default="trapezoid")
parser.add_argument("--warmup", type=float, default=2.)
parser.add_argument("--duration", type=float, default=2.)
parser.add_argument("--recovery", type=float, default=2.)
parser.add_argument("--max_tilt_deg", type=float, default=30.)
parser.add_argument("--intent_model_dir", help="Optional exported estimator for observation-only online evaluation; never changes robot commands.")
parser.add_argument("--export_locomotion_only", action="store_true", help="Strictly load checkpoint, export actor/encoder TorchScript, and exit before collecting.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.num_envs = 1
faulthandler.enable()
print("[BOOT] stage=app_launcher_start", flush=True)
app = AppLauncher(args).app
print("[BOOT] stage=app_launcher_ready", flush=True)

import gymnasium as gym
import numpy as np
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runner import OnPolicyRunner
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.controllers.paint_intent import ARM_JOINT_NAMES, CONDITIONS, SCHEMA, InteractionProtocol, IntentHistory
from bipedal_locomotion.controllers.paint_payload import create_payload_asset
from bipedal_locomotion.controllers.ocs2_reset import configure_nominal_ocs2_reset
from bipedal_locomotion.controllers.ocs2_rollout_dataset import save_episode_npz
from bipedal_locomotion.controllers.paint_run_status import PaintRunStatus

run_status = None


def row(value):
    return value[0].detach().cpu().numpy().copy()


def main():
    global run_status
    protocol = InteractionProtocol(args.warmup, args.duration, args.recovery, args.force_amplitude, args.torque_amplitude, args.profile)
    if "WholeBody" not in args.task or "Play" not in args.task:
        raise ValueError("Use a WholeBody PLAY task and its matching locomotion checkpoint.")
    numeric = np.array([*args.base_command, args.arm_kp, args.arm_kd, args.pose_jitter, args.max_tilt_deg])
    if not np.isfinite(numeric).all() or args.arm_kp <= 0 or args.arm_kd < 0 or not 0 <= args.pose_jitter <= 0.1 or not 0 < args.max_tilt_deg < 90 or args.repeats < 1:
        raise ValueError("Invalid PD, command, repetition or diagnostic bound.")
    output = Path(args.output_dir).expanduser().resolve()
    # Do not overwrite or resume partially collected pairs with a different contract.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    run_status = PaintRunStatus(output)
    run_status.mark("configure_environment", expected=args.repeats * 7)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.seed = args.seed
    cfg.episode_length_s = protocol.total_time + 2.0
    configure_nominal_ocs2_reset(cfg, SceneEntityCfg("robot"))
    for name in ("push_robot", "add_base_mass", "randomize_rigid_body_mass_inertia", "robot_physics_material", "randomize_actuator_gains", "robot_center_of_mass"):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    cfg.observations.policy.enable_corruption = False
    cfg.observations.obsHistory.enable_corruption = False
    cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_standing_envs = 0.
    cfg.commands.base_velocity.rel_heading_envs = 0.
    for actuator in cfg.scene.robot.actuators.values():
        if hasattr(actuator, "min_delay"):
            actuator.min_delay = actuator.max_delay = 0
    arm_cfg = cfg.scene.robot.actuators["arm_mpc"]
    arm_cfg.stiffness, arm_cfg.damping = args.arm_kp, args.arm_kd
    cfg.scene.robot.spawn.usd_path = create_payload_asset(
        cfg.scene.robot.spawn.usd_path, output / "robot_payload.usdc", args.payload_mass,
        args.payload_dimensions, args.payload_offset)
    run_status.mark("environment_create")
    print("[BOOT] stage=environment_create_start", flush=True)
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
    print("[BOOT] stage=environment_create_ready", flush=True)
    try:
        run_status.mark("check_payload_and_load_policy")
        raw = env.unwrapped
        robot = raw.scene["robot"]
        joint_ids, names = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
        body_ids, body_names = robot.find_bodies(["paint_payload_Link"], preserve_order=True)
        if tuple(names) != ARM_JOINT_NAMES or tuple(body_names) != ("paint_payload_Link",):
            raise RuntimeError(f"Payload/arm topology mismatch: joints={names}, bodies={body_names}. Check fixed-joint import.")
        # Verify that the payload became an articulation link, not an unattached visual.
        body_id = body_ids[0]
        sim_mass = float(robot.root_physx_view.get_masses()[0, body_id])
        if not math.isclose(sim_mass, args.payload_mass, rel_tol=0.05):
            raise RuntimeError(f"Payload mass mismatch: expected {args.payload_mass}, imported {sim_mass}.")
        agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        # The repository runner otherwise silently skips mismatched actor weights.
        # A new payload changes critic mass observations, but the deployed actor
        # and proprioception encoder must match completely.
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        expected = runner.alg.actor_critic.state_dict()
        weights = checkpoint["model_state_dict"]
        actor_keys = [key for key in expected if key.startswith("actor.")]
        if not actor_keys or any(key not in weights or weights[key].shape != expected[key].shape for key in actor_keys):
            raise ValueError("Locomotion actor checkpoint does not match this WholeBody task; do not use a partially loaded actor.")
        if "encoder_state_dict" not in checkpoint:
            raise ValueError("Checkpoint lacks the trained locomotion history encoder.")
        runner.alg.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
        runner.alg.actor_critic.load_state_dict({k: v for k, v in weights.items() if k in expected and v.shape == expected[k].shape}, strict=False)
        if args.export_locomotion_only:
            import copy
            actor = copy.deepcopy(runner.alg.actor_critic.actor).cpu().eval()
            history_encoder = copy.deepcopy(runner.alg.encoder.encoder).cpu().eval()
            with torch.inference_mode():
                if actor(torch.zeros(1, 78)).shape != (1, 10) or history_encoder(torch.zeros(1, 420)).shape != (1, 3):
                    raise ValueError("MuJoCo adapter expects the flat WholeBody 78/420 observation contract.")
            for name, network in (("policy.pt", actor), ("encoder.pt", history_encoder)):
                with (output / name).open("wb") as stream:
                    torch.jit.save(torch.jit.script(network), stream)
            (output / "export.json").write_text(json.dumps({"checkpoint": str(Path(args.checkpoint_path).resolve()), "task": args.task,
                "actor_input": 78, "encoder_input": 420, "leg_joint_names": [f"{part}_{side}_Joint" for side in ("L", "R") for part in ("proximal_pitch", "proximal_roll", "proximal_yaw", "knee", "ankle_pitch")],
                "control_dt": .02, "note": "MuJoCo walking requires separate sim-to-sim validation."}, indent=2))
            print(f"[EXPORT] {output}/policy.pt and encoder.pt", flush=True)
            run_status.mark("export_complete", complete=True, state="completed")
            return
        policy = runner.get_inference_policy(device=raw.device)
        encoder = runner.get_inference_encoder(device=raw.device)
        dt = float(raw.step_dt)
        intent_model = None
        intent_history = None
        if args.intent_model_dir:
            model_dir = Path(args.intent_model_dir).expanduser().resolve()
            model_metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
            if model_metadata["schema"] != SCHEMA or model_metadata["joint_names"] != list(names) or not math.isclose(model_metadata["step_dt"], dt, abs_tol=1e-8):
                raise ValueError("Estimator joint/rate contract does not match collection.")
            with (model_dir / "intent_estimator.ts").open("rb") as stream:
                intent_model = torch.jit.load(stream, map_location="cpu").eval()
            intent_history = IntentHistory(model_metadata["history"])
        contract = {"schema": SCHEMA, "joint_names": list(names), "feature_row": ["q_rad(6)", "dq_rad_s(6)", "previous_applied_position_rad(6)"],
                    "label": ["Fx_N", "Fy_N", "Mz_Nm"], "label_frame": "payload_local", "label_point": "payload_COM",
                    "label_semantics": "external_on_payload; wrench applied during preceding control interval",
                    "step_dt": dt, "protocol": vars(protocol), "args": vars(args), "payload_imported_mass_kg": sim_mass,
                    "control": "PD position hold; velocity target=0; feedforward=0; locomotion wrench observation=0"}
        (output / "contract.json").write_text(json.dumps(contract, indent=2, default=str) + "\n", encoding="utf-8")
        obs_dict = env.get_observations()
        template = torch.zeros((1, raw.action_manager.total_action_dim), device=raw.device)
        home = row(robot.data.default_joint_pos[:, joint_ids])
        bounds = row(robot.data.soft_joint_pos_limits[:, joint_ids])
        if np.any(home < bounds[:, 0]) or np.any(home > bounds[:, 1]):
            raise RuntimeError("Default arm pose is outside soft position limits.")
        command = torch.tensor([args.base_command], dtype=torch.float32, device=raw.device)
        zero_wrench = torch.zeros((1, 5, 6), device=raw.device)
        summary = []
        interrupted = False
        for repeat in range(args.repeats):
            pair_seed = args.seed + repeat
            jitter = np.random.default_rng(pair_seed).uniform(-args.pose_jitter, args.pose_jitter, 6)
            target = home + jitter
            if np.any(target < bounds[:, 0]) or np.any(target > bounds[:, 1]):
                raise ValueError("Jittered target violates arm bounds; reduce --pose_jitter.")
            target_tensor = torch.tensor(target[None], dtype=torch.float32, device=raw.device)
            for condition in CONDITIONS:
                if not app.is_running():
                    interrupted = True
                    break
                episode_id = f"group_{repeat:04d}_{condition}"
                run_status.mark("reset_episode", episode_id=episode_id, seed=pair_seed, step=-1)
                # Identical RNG seed and nominal reset for all seven conditions in a pair group.
                torch.manual_seed(pair_seed)
                np.random.seed(pair_seed)
                robot.set_external_force_and_torque(torch.zeros((1, 1, 3), device=raw.device), torch.zeros((1, 1, 3), device=raw.device), body_ids=body_ids)
                raw.episode_length_buf[:] = raw.max_episode_length
                obs_dict, _, dones, _ = env.step(template)
                if not bool(dones[0]):
                    raise RuntimeError("Forced timeout did not reset environment.")
                frames = []
                if intent_history is not None:
                    intent_history.reset()
                reason = "completed"
                print(f"[COLLECT] {episode_id} seed={pair_seed}", flush=True)
                run_status.mark("episode_start", step=0)
                w = torch.zeros((1, 1, 6), device=raw.device)
                for index in range(int(math.ceil(protocol.total_time / dt))):
                    if not app.is_running():
                        reason = "interrupted"
                        interrupted = True
                        break
                    t = index * dt
                    trace_step = index < 3 or index % 50 == 0
                    if trace_step:
                        run_status.mark("prepare_policy_inputs", step=index)
                    with torch.inference_mode():
                        obs = obs_dict["policy"]
                        history = obs_dict["obsHistory"].flatten(start_dim=1)
                        commands = obs_dict["commands"]
                        raw.command_manager.get_command("base_velocity").copy_(command)
                        commands.copy_(command)
                        # Explicitly disable synthetic base wrench; never feed label into actor.
                        raw.command_manager.get_term("arm_wrench").set_external_prediction(zero_wrench)
                        robot.set_joint_position_target(target_tensor, joint_ids=joint_ids)
                        robot.set_joint_velocity_target(torch.zeros_like(target_tensor), joint_ids=joint_ids)
                        robot.set_joint_effort_target(torch.zeros_like(target_tensor), joint_ids=joint_ids)
                        wrench = protocol.wrench(condition, t)
                        w = torch.tensor(wrench[None, None], dtype=torch.float32, device=raw.device)
                        robot.set_external_force_and_torque(w[..., :3], w[..., 3:], body_ids=body_ids, is_global=False)
                        if trace_step:
                            run_status.mark("policy_inference", step=index)
                        actions = policy(torch.cat((encoder(history), obs, commands), dim=-1))
                        if trace_step:
                            run_status.mark("physics_step", step=index, action_shape=list(actions.shape))
                        obs_dict, _, dones, _ = env.step(actions)
                        if trace_step:
                            run_status.mark("read_response", step=index)
                        valid = not bool(dones[0])
                        q, dq = row(robot.data.joint_pos[:, joint_ids]), row(robot.data.joint_vel[:, joint_ids])
                        tilt = math.degrees(math.acos(float(np.clip(-row(robot.data.projected_gravity_b)[2], -1., 1.))))
                        margin = float(np.min(np.minimum(q - bounds[:, 0], bounds[:, 1] - q)))
                        intent_estimate = np.zeros(3, dtype=np.float32)
                        intent_valid = False
                        if intent_history is not None and valid:
                            features = intent_history.update(q, dq, target)
                            if features is not None:
                                intent_estimate = intent_model(torch.from_numpy(features[None])).numpy()[0].copy()
                                intent_valid = bool(np.isfinite(intent_estimate).all())
                        frames.append({"time": t + dt, "interval_start": t, "arm_position": q, "arm_velocity": dq,
                                       "applied_position": target.copy(), "interaction_wrench": wrench[[0, 1, 5]],
                                       "response_valid": valid, "tilt_deg": tilt, "joint_margin": margin,
                                       "arm_applied_torque": row(robot.data.applied_torque[:, joint_ids]),
                                       "intent_estimate": intent_estimate, "intent_estimate_valid": intent_valid,
                                       "base_velocity": row(robot.data.root_lin_vel_b),
                                       "base_angular_velocity": row(robot.data.root_ang_vel_b),
                                       "payload_position_world": row(robot.data.body_pos_w[:, body_ids]),
                                       "payload_quaternion_world": row(robot.data.body_quat_w[:, body_ids])})
                        if trace_step:
                            run_status.mark("response_recorded", step=index, frames=len(frames))
                    if not valid or tilt > args.max_tilt_deg or margin < -1e-4:
                        reason = "environment_done" if not valid else "tilt_limit" if tilt > args.max_tilt_deg else "joint_limit"
                        break
                robot.set_external_force_and_torque(torch.zeros_like(w[..., :3]), torch.zeros_like(w[..., 3:]), body_ids=body_ids)
                accepted = reason == "completed"
                if frames:
                    relative = Path("episodes" if accepted else "rejected") / f"{episode_id}.npz"
                    run_status.mark("save_episode", frames=len(frames), termination_reason=reason)
                    save_episode_npz(output / relative, frames, {"schema": SCHEMA, "group_id": f"seed_{pair_seed}", "condition": condition,
                                     "accepted": accepted, "termination_reason": reason, "step_dt": dt})
                    summary.append({"episode_id": episode_id, "group_id": f"seed_{pair_seed}", "condition": condition,
                                    "accepted": accepted, "reason": reason, "file": str(relative), "frames": len(frames)})
                    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                    run_status.mark("episode_saved", accepted=sum(r["accepted"] for r in summary), total=len(summary))
                if interrupted:
                    break
            if interrupted:
                break
        run_status.mark("collection_finished", complete=not interrupted, state="interrupted" if interrupted else "completed",
                        accepted=sum(r["accepted"] for r in summary), total=len(summary), expected=args.repeats * 7)
        print(f"[RESULT] {output}: {sum(r['accepted'] for r in summary)}/{len(summary)} completed episodes", flush=True)
    except BaseException as error:
        run_status.fail(error)
        raise
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if run_status is not None and run_status.data["state"] != "failed":
            run_status.fail(error)
        elif run_status is None:
            import traceback
            traceback.print_exception(type(error), error, error.__traceback__)
        raise
    finally:
        app.close()
