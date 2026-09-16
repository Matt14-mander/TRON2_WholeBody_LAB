"""Collect aligned Isaac Lab executions of offline OCS2 expert trajectories."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "rsl_rl"))
sys.path.insert(0, os.path.join(REPO_ROOT, "exts", "bipedal_locomotion"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task", default="Isaac-Limx-SFYG-TRON2A-WholeBody-Flat-Play-v0"
)
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--trajectory_manifest", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--split_seed", type=int, default=42)
parser.add_argument("--start_index", type=int, default=0)
parser.add_argument("--max_trajectories", type=int, default=None)
parser.add_argument("--start_delay", type=float, default=1.0)
parser.add_argument("--post_motion_duration", type=float, default=2.0)
parser.add_argument(
    "--terminal_base_command", type=float, nargs=3, default=(-0.25, 0.0, 0.0),
    metavar=("VX", "VY", "WZ"),
)
parser.add_argument("--max_tilt_deg", type=float, default=45.0)
parser.add_argument(
    "--external_wrench_validation", action="store_true",
    help="Apply balanced zero/+/- Fx/Fy/Mz validation wrenches to the payload proxy.",
)
parser.add_argument(
    "--external_wrench_paired", action="store_true",
    help="Replay every selected source trajectory under every configured wrench condition.",
)
parser.add_argument(
    "--baseline_rollout_manifest",
    help="Optional prior rollout manifest; only source trajectories accepted there are collected.",
)
parser.add_argument("--external_wrench_body", default="gripper_base_Link")
parser.add_argument("--external_wrench_force_amplitude", type=float, default=5.0)
parser.add_argument("--external_wrench_torque_amplitude", type=float, default=1.0)
parser.add_argument("--external_wrench_start_time", type=float, default=1.0)
parser.add_argument("--external_wrench_duration", type=float, default=0.8)
parser.add_argument(
    "--external_wrench_profiles", nargs="+", default=("step", "ramp", "sine"),
    choices=("step", "ramp", "sine"),
)
parser.add_argument(
    "--resume_collection", action="store_true",
    help="Continue an interrupted rollout dataset and skip recorded trajectory ids.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runner import OnPolicyRunner

import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.controllers.ocs2_play_bridge import ARM_JOINT_NAMES
from bipedal_locomotion.controllers.ocs2_rollout_dataset import (
    append_manifest_row,
    completed_trajectory_ids,
    load_trajectory_manifest,
    evaluate_external_wrench,
    make_external_wrench_assignments,
    make_paired_external_wrench_excitations,
    make_split_assignments,
    save_episode_npz,
    transform_wrench_to_base_origin,
    write_json_contract,
    write_split_file,
)
from bipedal_locomotion.controllers.ocs2_trajectory_play_bridge import Ocs2TrajectoryPlayBridge
from bipedal_locomotion.tasks.locomotion.cfg.SF_TRON2A.limx_base_env_cfg import joint_order_name
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg


ROLLOUT_FIELDS = (
    "trajectory_id", "source_trajectory_id", "split", "accepted", "fall",
    "termination_reason", "frames",
    "arrival_time", "target_x", "target_y", "target_z", "terminal_vx", "terminal_vy",
    "terminal_wz", "max_base_tilt_deg", "arm_position_rmse", "base_velocity_rmse",
    "external_wrench_enabled", "external_wrench_axis", "external_wrench_sign",
    "external_wrench_profile", "external_wrench_amplitude", "episode_file",
)
FAILURE_FIELDS = (
    "trajectory_id", "source_trajectory_id", "split", "frames",
    "termination_reason", "max_base_tilt_deg",
)
PHASE_CODE = {"warmup": 0, "motion": 1, "terminal": 2}


def configure_collection_environment(
    env_cfg: ManagerBasedRLEnvCfg, deterministic_pairing: bool = False
) -> None:
    env_cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    reset_base = env_cfg.events.reset_robot_base
    reset_joints = env_cfg.events.reset_robot_joints
    if reset_base is None or reset_joints is None:
        raise RuntimeError("The rollout task must define base and joint reset events.")
    for reset_range in (reset_base.params["pose_range"], reset_base.params["velocity_range"]):
        for axis in reset_range:
            reset_range[axis] = (0.0, 0.0)
    reset_joints.params["position_range"] = (0.0, 0.0)
    reset_joints.params["velocity_range"] = (0.0, 0.0)
    if deterministic_pairing:
        # The SFYG task normally restricts joint reset to the legs so training
        # randomization cannot kick the arm.  Here the offset is exactly zero,
        # so reset the complete articulation; otherwise each paired rollout
        # inherits the previous rollout's terminal arm/gripper state.
        reset_joints.params["asset_cfg"] = SceneEntityCfg("robot")
        # PLAY already disables policy corruption, but the history group feeds
        # the encoder independently and otherwise retains its training noise.
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.observations.obsHistory.enable_corruption = False
        # DelayedImplicitActuator samples a fresh lag on every reset.  Matched
        # zero/treatment rollouts require identical actuator timing.
        for actuator_cfg in env_cfg.scene.robot.actuators.values():
            if hasattr(actuator_cfg, "min_delay") and hasattr(actuator_cfg, "max_delay"):
                actuator_cfg.min_delay = 0
                actuator_cfg.max_delay = 0


def numpy_row(tensor: torch.Tensor) -> np.ndarray:
    return tensor[0].detach().cpu().numpy().copy()


def observation_term_slice(env, group: str, term: str) -> slice:
    manager = env.observation_manager
    names = manager._group_obs_term_names[group]
    dimensions = manager._group_obs_term_dim[group]
    try:
        term_index = names.index(term)
    except ValueError as error:
        raise RuntimeError(f"Observation group {group!r} has no term {term!r}.") from error
    widths = [dimension if isinstance(dimension, int) else math.prod(dimension) for dimension in dimensions]
    start = sum(widths[:term_index])
    return slice(start, start + widths[term_index])


def robot_snapshot(
    robot, arm_joint_ids, leg_joint_ids, ee_body_id, base_body_id, payload_body_id
) -> dict[str, np.ndarray]:
    data = robot.data
    body_com_pos_w = getattr(data, "body_com_pos_w", data.body_pos_w)
    body_com_quat_w = getattr(data, "body_com_quat_w", data.body_quat_w)
    return {
        "base_position_world": numpy_row(data.root_pos_w),
        "base_quaternion_world": numpy_row(data.root_quat_w),
        "base_linear_velocity_body": numpy_row(data.root_lin_vel_b),
        "base_angular_velocity_body": numpy_row(data.root_ang_vel_b),
        "projected_gravity_body": numpy_row(data.projected_gravity_b),
        "arm_position_actual": numpy_row(data.joint_pos[:, arm_joint_ids]),
        "arm_velocity_actual": numpy_row(data.joint_vel[:, arm_joint_ids]),
        "leg_position_actual": numpy_row(data.joint_pos[:, leg_joint_ids]),
        "leg_velocity_actual": numpy_row(data.joint_vel[:, leg_joint_ids]),
        "end_effector_position_world": numpy_row(data.body_pos_w[:, ee_body_id]),
        "end_effector_quaternion_world": numpy_row(data.body_quat_w[:, ee_body_id]),
        "base_link_position_world": numpy_row(data.body_pos_w[:, base_body_id]),
        "base_link_quaternion_world": numpy_row(data.body_quat_w[:, base_body_id]),
        "payload_proxy_com_position_world": numpy_row(body_com_pos_w[:, payload_body_id]),
        "payload_proxy_com_quaternion_world": numpy_row(body_com_quat_w[:, payload_body_id]),
    }


def apply_external_wrench(robot, body_id: int, wrench_local: np.ndarray) -> None:
    wrench = torch.as_tensor(wrench_local, dtype=torch.float32, device=robot.device).reshape(1, 1, 6)
    robot.set_external_force_and_torque(
        wrench[..., :3], wrench[..., 3:], body_ids=[body_id], is_global=False
    )


def accepted_source_ids(path: str | None) -> set[str] | None:
    if path is None:
        return None
    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Baseline rollout manifest does not exist: {manifest}")
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"trajectory_id", "accepted"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Baseline rollout manifest lacks columns {sorted(required)}: {manifest}")
    return {row["trajectory_id"] for row in rows if row["accepted"] == "True"}


def extract_observations(env) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_dict = env.get_observations()
    obs = obs_dict["policy"]
    obs_history = obs_dict.get("obsHistory")
    commands = obs_dict.get("commands")
    if obs_history is None or commands is None:
        raise RuntimeError("WholeBody rollout requires obsHistory and commands observation groups.")
    return obs_dict, obs, obs_history.flatten(start_dim=1), commands, obs_history


def force_timeout_reset(env, action_template: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reset through the normal ManagerBasedRLEnv step/reset path.

    Repeated direct global ``env.reset()`` calls can tear down native PhysX
    state on some Isaac Lab/Isaac Sim builds. Triggering the configured episode
    timeout exercises the same per-environment automatic reset path used by
    training and ordinary PLAY.
    """
    env.unwrapped.episode_length_buf[:] = env.unwrapped.max_episode_length
    with torch.inference_mode():
        obs_dict, _, dones, _ = env.step(torch.zeros_like(action_template))
    if not bool(torch.all(dones).item()):
        raise RuntimeError("Forced episode timeout did not reset every rollout environment.")
    obs = obs_dict["policy"]
    obs_history = obs_dict.get("obsHistory")
    commands = obs_dict.get("commands")
    if obs_history is None or commands is None:
        raise RuntimeError("Forced reset returned incomplete observation groups.")
    return obs, obs_history.flatten(start_dim=1), commands


def main() -> None:
    if not args_cli.task or "WholeBody" not in args_cli.task or "Play" not in args_cli.task:
        raise ValueError("OCS2 rollout collection requires a WholeBody PLAY task.")
    if args_cli.start_index < 0:
        raise ValueError("--start_index must be non-negative.")
    if args_cli.max_trajectories is not None and args_cli.max_trajectories <= 0:
        raise ValueError("--max_trajectories must be positive.")
    if args_cli.start_delay < 0.0 or args_cli.post_motion_duration < 0.0:
        raise ValueError("Collection timing values must be non-negative.")
    if args_cli.max_tilt_deg <= 0.0 or not math.isfinite(args_cli.max_tilt_deg):
        raise ValueError("--max_tilt_deg must be positive and finite.")
    if args_cli.external_wrench_duration <= 0.0 or args_cli.external_wrench_start_time < 0.0:
        raise ValueError("External-wrench timing must have non-negative start and positive duration.")
    if args_cli.external_wrench_paired and not args_cli.external_wrench_validation:
        raise ValueError("--external_wrench_paired requires --external_wrench_validation.")

    specs = load_trajectory_manifest(args_cli.trajectory_manifest)
    assignments = make_split_assignments([spec.trajectory_id for spec in specs], args_cli.split_seed)
    wrench_assignments = make_external_wrench_assignments(
        [spec.trajectory_id for spec in specs],
        seed=args_cli.split_seed,
        force_amplitude=args_cli.external_wrench_force_amplitude,
        torque_amplitude=args_cli.external_wrench_torque_amplitude,
        profiles=args_cli.external_wrench_profiles,
    )
    accepted_ids = accepted_source_ids(args_cli.baseline_rollout_manifest)
    eligible_specs = specs if accepted_ids is None else [
        spec for spec in specs if spec.trajectory_id in accepted_ids
    ]
    selected = eligible_specs[args_cli.start_index:]
    if args_cli.max_trajectories is not None:
        selected = selected[:args_cli.max_trajectories]
    if not selected:
        raise ValueError("No trajectories selected for collection.")
    if args_cli.external_wrench_paired:
        paired_excitations = make_paired_external_wrench_excitations(
            args_cli.external_wrench_force_amplitude,
            args_cli.external_wrench_torque_amplitude,
            args_cli.external_wrench_profiles,
        )
        jobs = [
            (spec, f"{spec.trajectory_id}__{suffix}", excitation)
            for spec in selected
            for suffix, excitation in paired_excitations
        ]
    else:
        jobs = [
            (spec, spec.trajectory_id, wrench_assignments[spec.trajectory_id])
            for spec in selected
        ]

    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    rollout_manifest = output_dir / "rollout_manifest.csv"
    replay_failures = output_dir / "replay_failures.csv"
    if not args_cli.resume_collection and (rollout_manifest.exists() or replay_failures.exists()):
        raise FileExistsError(
            f"Output metadata already exists in {output_dir}; use --resume_collection."
        )
    completed = (
        completed_trajectory_ids(rollout_manifest, replay_failures)
        if args_cli.resume_collection else set()
    )
    write_split_file(output_dir / "split.json", assignments, args_cli.split_seed)
    write_json_contract(output_dir / "wrench_contract.json", {
        "schema_version": "2.0",
        "enabled": args_cli.external_wrench_validation,
        "paired": args_cli.external_wrench_paired,
        "deterministic_pairing": args_cli.external_wrench_paired,
        "full_articulation_reset": args_cli.external_wrench_paired,
        "seed": args_cli.split_seed,
        "component_order": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
        "application": {
            "body": args_cli.external_wrench_body,
            "frame": "body_local",
            "reference_point": "body_center_of_mass",
            "semantics": "external_on_payload_proxy",
        },
        "base_label": {
            "frame": "base_Link",
            "reference_point": "base_Link_origin",
            "semantics": "external_on_payload_proxy",
        },
        "ocs2_label": {
            "frame": "base_Link",
            "reference_point": "base_Link_origin",
            "semantics": "arm_on_base_planned_reaction",
            "prediction_times_s": [0.0, 0.2, 0.4, 0.6, 0.8],
        },
        "protocol": {
            "axes": ["zero", "+fx", "-fx", "+fy", "-fy", "+mz", "-mz"],
            "profiles": list(args_cli.external_wrench_profiles),
            "force_amplitude_n": args_cli.external_wrench_force_amplitude,
            "torque_amplitude_nm": args_cli.external_wrench_torque_amplitude,
            "start_time_s": args_cli.external_wrench_start_time,
            "duration_s": args_cli.external_wrench_duration,
        },
        "assignments": {
            trajectory_id: {
                "axis": excitation.axis,
                "sign": excitation.sign,
                "profile": excitation.profile,
                "amplitude": excitation.amplitude,
            }
            for _, trajectory_id, excitation in jobs
        },
    })

    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=1
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = args_cli.seed
    configure_collection_environment(
        env_cfg, deterministic_pairing=args_cli.external_wrench_paired
    )
    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)
    env = RslRlVecEnvWrapper(raw_env)

    print(f"[INFO] Loading model checkpoint: {args_cli.checkpoint_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    encoder = runner.get_inference_encoder(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    arm_joint_ids, arm_names = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
    leg_joint_ids, leg_names = robot.find_joints(joint_order_name, preserve_order=True)
    ee_body_ids, ee_names = robot.find_bodies(["gripper_base_Link"], preserve_order=True)
    wrench_body_ids, wrench_body_names = robot.find_bodies(
        [args_cli.external_wrench_body], preserve_order=True
    )
    base_body_ids, base_body_names = robot.find_bodies(["base_Link"], preserve_order=True)
    if tuple(arm_names) != ARM_JOINT_NAMES or tuple(leg_names) != tuple(joint_order_name):
        raise RuntimeError("Unexpected arm or leg joint ordering in rollout task.")
    if len(ee_body_ids) != 1 or tuple(ee_names) != ("gripper_base_Link",):
        raise RuntimeError(f"Unexpected end-effector body match: {ee_names}")
    if len(wrench_body_ids) != 1:
        raise RuntimeError(
            f"Expected one external-wrench body matching {args_cli.external_wrench_body!r}, "
            f"found {wrench_body_names}."
        )
    if len(base_body_ids) != 1 or tuple(base_body_names) != ("base_Link",):
        raise RuntimeError(f"Unexpected base-link body match: {base_body_names}")
    ee_body_id = ee_body_ids[0]
    wrench_body_id = wrench_body_ids[0]
    base_body_id = base_body_ids[0]
    gait_phase_slice = observation_term_slice(env.unwrapped, "policy", "gait_phase")
    last_action_slice = observation_term_slice(env.unwrapped, "policy", "last_action")
    _, obs, obs_history, commands, _ = extract_observations(env)
    previous_actions = None

    success_count = 0
    rejected_count = 0
    try:
        for ordinal, (spec, episode_id, excitation) in enumerate(jobs, start=1):
            if episode_id in completed:
                continue
            if not simulation_app.is_running():
                break
            print(
                f"[INFO] [{ordinal}/{len(jobs)}] collecting {episode_id} "
                f"from {spec.trajectory_id}: {spec.path}"
            )
            if previous_actions is not None:
                obs, obs_history, commands = force_timeout_reset(env, previous_actions)
            bridge = Ocs2TrajectoryPlayBridge(
                env.unwrapped,
                str(spec.path),
                terminal_base_command=tuple(args_cli.terminal_base_command),
                start_delay_s=args_cli.start_delay,
            )
            frames = []
            fall = False
            termination_reason = "completed"
            tilt_values = []
            arm_squared_errors = []
            base_squared_errors = []
            max_episode_time = args_cli.start_delay + bridge.trajectory_duration + args_cli.post_motion_duration
            max_steps = int(math.ceil(max_episode_time / float(env.unwrapped.step_dt))) + 2

            for frame_index in range(max_steps):
                with torch.inference_mode():
                    bridge.update(obs, commands)
                    solution = bridge.current_solution
                    if solution is None:
                        raise RuntimeError("Offline bridge did not publish a trajectory sample.")
                    state = robot_snapshot(
                        robot, arm_joint_ids, leg_joint_ids, ee_body_id,
                        base_body_id, wrench_body_id,
                    )
                    episode_time = frame_index * float(env.unwrapped.step_dt)
                    external_wrench_payload = (
                        evaluate_external_wrench(
                            excitation,
                            episode_time,
                            args_cli.external_wrench_start_time,
                            args_cli.external_wrench_duration,
                        )
                        if args_cli.external_wrench_validation else np.zeros(6, dtype=np.float64)
                    )
                    apply_external_wrench(robot, wrench_body_id, external_wrench_payload)
                    external_wrench_base = transform_wrench_to_base_origin(
                        external_wrench_payload,
                        state["payload_proxy_com_position_world"],
                        state["payload_proxy_com_quaternion_world"],
                        state["base_link_position_world"],
                        state["base_link_quaternion_world"],
                    )
                    estimate = encoder(obs_history)
                    actions = policy(torch.cat((estimate, obs, commands), dim=-1).detach())
                    obs_dict_next, rewards, dones, infos = env.step(actions)
                    next_state = robot_snapshot(
                        robot, arm_joint_ids, leg_joint_ids, ee_body_id,
                        base_body_id, wrench_body_id,
                    )

                gravity_z = float(np.clip(-state["projected_gravity_body"][2], -1.0, 1.0))
                tilt_deg = math.degrees(math.acos(gravity_z))
                arm_error = state["arm_position_actual"] - solution.arm_position
                actual_base_velocity = np.asarray([
                    state["base_linear_velocity_body"][0],
                    state["base_linear_velocity_body"][1],
                    state["base_angular_velocity_body"][2],
                ])
                base_error = actual_base_velocity - solution.base_velocity_command
                tilt_values.append(tilt_deg)
                arm_squared_errors.append(float(np.mean(np.square(arm_error))))
                base_squared_errors.append(float(np.mean(np.square(base_error))))
                frame = {
                    "episode_time": np.asarray(episode_time),
                    "playback_time": np.asarray(bridge.playback_time),
                    "phase": np.asarray(PHASE_CODE[bridge.phase], dtype=np.int8),
                    "policy_observation": numpy_row(obs),
                    "observation_history": numpy_row(obs_history),
                    "command_observation": numpy_row(commands),
                    "gait_phase": numpy_row(obs[:, gait_phase_slice]),
                    "previous_leg_action": numpy_row(obs[:, last_action_slice]),
                    "encoder_output": numpy_row(estimate),
                    "leg_action": numpy_row(actions),
                    "intent_target_position": spec.target_position.copy(),
                    "intent_target_quaternion": spec.target_quaternion.copy(),
                    "intent_arrival_time": np.asarray(spec.arrival_time),
                    **state,
                    "arm_position_reference": solution.arm_position.copy(),
                    "arm_velocity_reference": solution.arm_velocity.copy(),
                    "arm_feedforward_effort": solution.arm_feedforward_effort.copy(),
                    "base_velocity_command": solution.base_velocity_command.copy(),
                    # Deprecated compatibility alias for existing MHCT loaders.
                    "future_wrench": solution.base_wrench_prediction.copy(),
                    "ocs2_arm_on_base_wrench_plan": solution.base_wrench_prediction.copy(),
                    "external_wrench_payload_at_body_com": external_wrench_payload,
                    "external_wrench_base_at_base_origin": external_wrench_base,
                    "external_wrench_active": np.asarray(bool(np.any(external_wrench_payload))),
                    # Commanded at this simulation interval [t, t + step_dt).
                    "external_wrench_command_time": np.asarray(episode_time),
                    "reward": np.asarray(float(rewards[0].item())),
                    "done": np.asarray(bool(dones[0].item())),
                    "next_base_position_world": next_state["base_position_world"],
                    "next_base_quaternion_world": next_state["base_quaternion_world"],
                    "next_base_linear_velocity_body": next_state["base_linear_velocity_body"],
                    "next_base_angular_velocity_body": next_state["base_angular_velocity_body"],
                    "next_arm_position_actual": next_state["arm_position_actual"],
                    "next_arm_velocity_actual": next_state["arm_velocity_actual"],
                }
                frames.append(frame)

                if bool(dones[0].item()):
                    timeout_value = infos.get("time_outs") if isinstance(infos, dict) else None
                    timed_out = bool(timeout_value[0].item()) if timeout_value is not None else False
                    fall = not timed_out
                    termination_reason = "environment_timeout" if timed_out else "environment_done"
                    break
                obs = obs_dict_next["policy"]
                history_next = obs_dict_next.get("obsHistory")
                commands_next = obs_dict_next.get("commands")
                if history_next is None or commands_next is None:
                    raise RuntimeError("Missing observation group after environment step.")
                obs_history = history_next.flatten(start_dim=1)
                commands = commands_next
                if bridge.playback_time >= bridge.trajectory_duration + args_cli.post_motion_duration:
                    break

            apply_external_wrench(robot, wrench_body_id, np.zeros(6, dtype=np.float64))
            bridge.close()
            previous_actions = actions.detach().clone()
            max_tilt = max(tilt_values) if tilt_values else math.inf
            accepted = not fall and max_tilt <= args_cli.max_tilt_deg
            if not accepted and termination_reason == "completed":
                termination_reason = "tilt_limit"
            split = assignments[spec.trajectory_id]
            episode_subdir = split if accepted else "rejected"
            episode_path = output_dir / "episodes" / episode_subdir / f"{episode_id}.npz"
            save_episode_npz(
                episode_path,
                frames,
                {
                    "trajectory_id": episode_id,
                    "source_trajectory_id": spec.trajectory_id,
                    "split": split,
                    "accepted": accepted,
                    "target_position": spec.target_position,
                    "target_quaternion": spec.target_quaternion,
                    "arrival_time": spec.arrival_time,
                    "terminal_base_command": np.asarray(args_cli.terminal_base_command),
                    "step_dt": float(env.unwrapped.step_dt),
                    "phase_codes": np.asarray(["warmup", "motion", "terminal"]),
                    "wrench_schema_version": "2.0",
                    "ocs2_wrench_semantics": "arm_on_base; frame=base_Link; point=base_Link_origin",
                    "external_wrench_semantics": "external_on_payload_proxy; local_frame; point=body_COM",
                    "external_wrench_enabled": args_cli.external_wrench_validation,
                    "external_wrench_body": args_cli.external_wrench_body,
                    "external_wrench_axis": excitation.axis,
                    "external_wrench_sign": excitation.sign,
                    "external_wrench_profile": excitation.profile,
                    "external_wrench_amplitude": excitation.amplitude,
                    "external_wrench_start_time": args_cli.external_wrench_start_time,
                    "external_wrench_duration": args_cli.external_wrench_duration,
                    "wrench_component_order": np.asarray(["Fx", "Fy", "Fz", "Mx", "My", "Mz"]),
                },
            )
            manifest_row = {
                "trajectory_id": episode_id,
                "source_trajectory_id": spec.trajectory_id,
                "split": split,
                "accepted": accepted,
                "fall": fall,
                "termination_reason": termination_reason,
                "frames": len(frames),
                "arrival_time": spec.arrival_time,
                "target_x": spec.target_position[0],
                "target_y": spec.target_position[1],
                "target_z": spec.target_position[2],
                "terminal_vx": args_cli.terminal_base_command[0],
                "terminal_vy": args_cli.terminal_base_command[1],
                "terminal_wz": args_cli.terminal_base_command[2],
                "max_base_tilt_deg": max_tilt,
                "arm_position_rmse": math.sqrt(float(np.mean(arm_squared_errors))),
                "base_velocity_rmse": math.sqrt(float(np.mean(base_squared_errors))),
                "external_wrench_enabled": args_cli.external_wrench_validation,
                "external_wrench_axis": excitation.axis,
                "external_wrench_sign": excitation.sign,
                "external_wrench_profile": excitation.profile,
                "external_wrench_amplitude": excitation.amplitude,
                "episode_file": str(episode_path.relative_to(output_dir)),
            }
            append_manifest_row(rollout_manifest, ROLLOUT_FIELDS, manifest_row)
            if accepted:
                success_count += 1
            else:
                rejected_count += 1
                append_manifest_row(replay_failures, FAILURE_FIELDS, {
                    "trajectory_id": episode_id,
                    "source_trajectory_id": spec.trajectory_id,
                    "split": split,
                    "frames": len(frames),
                    "termination_reason": termination_reason,
                    "max_base_tilt_deg": max_tilt,
                })
            print(
                f"[INFO] {episode_id}: accepted={accepted}, frames={len(frames)}, "
                f"max_tilt={max_tilt:.2f}deg, arm_rmse={manifest_row['arm_position_rmse']:.4f}rad"
            )
    finally:
        env.close()
    print(
        f"[INFO] Rollout collection complete: accepted={success_count}, "
        f"rejected={rejected_count}, output={output_dir}"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
