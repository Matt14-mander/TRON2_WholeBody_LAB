"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "rsl_rl"))
sys.path.insert(0, os.path.join(REPO_ROOT, "exts", "bipedal_locomotion"))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")
ocs2_mode = parser.add_mutually_exclusive_group()
ocs2_mode.add_argument("--ocs2", action="store_true", help="Enable the live external OCS2 runtime bridge.")
ocs2_mode.add_argument(
    "--ocs2_trajectory", type=str, default=None, help="Replay an offline OCS2 trajectory CSV."
)
parser.add_argument(
    "--ocs2_trajectory_zero_base_command",
    action="store_true",
    help="Offline replay ablation: force the planned base command to zero.",
)
parser.add_argument(
    "--ocs2_trajectory_zero_wrench",
    action="store_true",
    help="Offline replay ablation: hide predicted wrench from the locomotion policy.",
)
parser.add_argument(
    "--ocs2_trajectory_zero_arm_effort",
    action="store_true",
    help="Offline replay ablation: disable arm feed-forward effort while keeping position/velocity targets.",
)
parser.add_argument(
    "--ocs2_trajectory_hold_arm",
    action="store_true",
    help="Offline replay ablation: hold the first arm pose with zero arm velocity and feed-forward effort.",
)
parser.add_argument(
    "--ocs2_trajectory_fixed_base_command",
    type=float,
    nargs=3,
    default=None,
    metavar=("VX", "VY", "WZ"),
    help="Offline replay ablation: keep this body-frame base command throughout warm-up, motion, and terminal hold.",
)
parser.add_argument(
    "--ocs2_trajectory_terminal_base_command",
    type=float,
    nargs=3,
    default=None,
    metavar=("VX", "VY", "WZ"),
    help="Smoothly continue with this body-frame base command after offline replay.",
)
parser.add_argument(
    "--ocs2_trajectory_start_delay",
    type=float,
    default=0.0,
    help="Warm-up seconds before starting offline arm motion.",
)
parser.add_argument(
    "--ocs2_trajectory_diagnostic",
    action="store_true",
    help="Run one bounded headless replay and report tilt, root height, arm error, and reset reason.",
)
parser.add_argument(
    "--ocs2_trajectory_diagnostic_post_duration",
    type=float,
    default=2.0,
    help="Additional simulation seconds to observe after trajectory completion in diagnostic mode.",
)
parser.add_argument(
    "--ocs2_deterministic_reset",
    action="store_true",
    help=(
        "OCS2 validation mode: reset the base pose/velocity and all joint "
        "position/velocity offsets deterministically."
    ),
)
parser.add_argument("--ocs2_host", type=str, default="127.0.0.1", help="OCS2 bridge IPv4 host.")
parser.add_argument("--ocs2_port", type=int, default=5555, help="OCS2 bridge TCP port.")
parser.add_argument("--ocs2_timeout", type=float, default=120.0, help="Background OCS2 socket timeout in seconds.")
parser.add_argument(
    "--ocs2_max_solution_age", type=float, default=0.5, help="Maximum accepted OCS2 solution age in simulation seconds."
)
parser.add_argument(
    "--ocs2_update_period", type=float, default=0.1, help="Simulation-time period between OCS2 submissions."
)
parser.add_argument(
    "--ocs2_target_position", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
    help="Absolute world-frame end-effector target; defaults to the initial measured pose.",
)
parser.add_argument(
    "--ocs2_target_quaternion",
    type=float,
    nargs=4,
    default=None,
    metavar=("W", "X", "Y", "Z"),
    help="Target orientation in WXYZ order; defaults to the initial measured orientation.",
)
parser.add_argument(
    "--ocs2_target_offset_world", type=float, nargs=3, default=(0.0, 0.0, 0.0),
    metavar=("DX", "DY", "DZ"), help="World-frame offset from the initial measured end-effector position.",
)
parser.add_argument(
    "--ocs2_target_arrival_time", type=float, default=1.0,
    help="Seconds from the first observation to the target; must not exceed the MPC horizon.",
)
parser.add_argument("--ocs2_target_file", help="Hot-update JSON target with id, position_world, quaternion_wxyz, arrival_s.")
parser.add_argument("--ocs2_diagnostic_log", help="Write timestamp-aligned live control/response JSONL.")
parser.add_argument("--ocs2_live_duration", type=float, default=0.0, help="Bound live diagnostic run in simulation seconds; 0 means unbounded.")
parser.add_argument("--ocs2_arm_speed", type=float, default=0.4)
parser.add_argument("--ocs2_arm_acceleration", type=float, default=1.0)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg,DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg, export_mlp_as_onnx, export_policy_as_jit, export_encoder_as_jit


def main():
    """Play with RSL-RL agent."""
    ocs2_control_enabled = args_cli.ocs2 or args_cli.ocs2_trajectory is not None
    if (args_cli.ocs2_target_file or args_cli.ocs2_diagnostic_log or args_cli.ocs2_live_duration) and not args_cli.ocs2:
        raise ValueError("Live target/diagnostic options require --ocs2.")
    if not math.isfinite(args_cli.ocs2_live_duration) or args_cli.ocs2_live_duration < 0:
        raise ValueError("Live duration must be finite and non-negative.")
    trajectory_only_option_used = (
        args_cli.ocs2_trajectory_zero_base_command
        or args_cli.ocs2_trajectory_zero_wrench
        or args_cli.ocs2_trajectory_zero_arm_effort
        or args_cli.ocs2_trajectory_hold_arm
        or args_cli.ocs2_trajectory_fixed_base_command is not None
        or args_cli.ocs2_trajectory_terminal_base_command is not None
        or args_cli.ocs2_trajectory_start_delay != 0.0
        or args_cli.ocs2_trajectory_diagnostic
    )
    if trajectory_only_option_used and args_cli.ocs2_trajectory is None:
        raise ValueError("Offline trajectory ablation flags require --ocs2_trajectory.")
    if args_cli.ocs2_deterministic_reset and not ocs2_control_enabled:
        raise ValueError("--ocs2_deterministic_reset requires --ocs2 or --ocs2_trajectory.")
    if args_cli.ocs2_trajectory_diagnostic and args_cli.ocs2_trajectory is None:
        raise ValueError("--ocs2_trajectory_diagnostic requires --ocs2_trajectory.")
    if not math.isfinite(args_cli.ocs2_trajectory_diagnostic_post_duration) or (
        args_cli.ocs2_trajectory_diagnostic_post_duration < 0.0
    ):
        raise ValueError("Diagnostic post duration must be finite and non-negative.")
    if args_cli.ocs2_trajectory_zero_base_command and (
        args_cli.ocs2_trajectory_terminal_base_command is not None
        or args_cli.ocs2_trajectory_fixed_base_command is not None
    ):
        raise ValueError(
            "--ocs2_trajectory_zero_base_command conflicts with "
            "--ocs2_trajectory_terminal_base_command and "
            "--ocs2_trajectory_fixed_base_command."
        )
    if args_cli.ocs2_trajectory_fixed_base_command is not None and (
        args_cli.ocs2_trajectory_terminal_base_command is not None
    ):
        raise ValueError(
            "--ocs2_trajectory_fixed_base_command conflicts with "
            "--ocs2_trajectory_terminal_base_command."
        )
    if ocs2_control_enabled:
        args_cli.num_envs = 1
        print("[INFO] OCS2 control enabled; forcing num_envs=1.")
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.seed = agent_cfg.seed

    if ocs2_control_enabled:
        if not args_cli.task or "WholeBody" not in args_cli.task:
            raise ValueError("OCS2 control requires a WholeBody PLAY task.")
        # The bridge owns the planar command. Prevent the random command generator
        # from resampling or applying heading control over the bridge output.
        env_cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0
        if args_cli.ocs2_deterministic_reset:
            # Isolate controller/trajectory stability from the broad training
            # reset distribution. In particular, prevent a random arm pose
            # from being pulled abruptly to the first OCS2 trajectory sample.
            reset_base = env_cfg.events.reset_robot_base
            reset_joints = env_cfg.events.reset_robot_joints
            if reset_base is None or reset_joints is None:
                raise RuntimeError("The selected OCS2 PLAY task has no robot reset events.")
            for reset_range in (
                reset_base.params["pose_range"],
                reset_base.params["velocity_range"],
            ):
                for axis in reset_range:
                    reset_range[axis] = (0.0, 0.0)
            reset_joints.params["position_range"] = (0.0, 0.0)
            reset_joints.params["velocity_range"] = (0.0, 0.0)
            print("[INFO] OCS2 deterministic reset enabled: base and joint reset offsets are zero.")

    # specify directory for logging experiments
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    ocs2_bridge = None
    if args_cli.ocs2:
        from bipedal_locomotion.controllers import Ocs2TcpClient
        from bipedal_locomotion.controllers.ocs2_play_bridge import Ocs2PlayBridge

        ocs2_bridge = Ocs2PlayBridge(
            env.unwrapped,
            Ocs2TcpClient(args_cli.ocs2_host, args_cli.ocs2_port, args_cli.ocs2_timeout),
            args_cli.ocs2_target_position,
            args_cli.ocs2_target_quaternion,
            args_cli.ocs2_max_solution_age,
            args_cli.ocs2_update_period,
            args_cli.ocs2_target_offset_world,
            args_cli.ocs2_target_arrival_time,
            args_cli.ocs2_arm_speed,
            args_cli.ocs2_arm_acceleration,
        )
        print(
            f"[INFO] OCS2 asynchronous client configured for {args_cli.ocs2_host}:{args_cli.ocs2_port}; "
            f"timeout={args_cli.ocs2_timeout:.3f}s, update_period={args_cli.ocs2_update_period:.3f}s, "
            f"max_solution_age={args_cli.ocs2_max_solution_age:.3f}s."
        )
    elif args_cli.ocs2_trajectory is not None:
        from bipedal_locomotion.controllers.ocs2_trajectory_play_bridge import Ocs2TrajectoryPlayBridge

        ocs2_bridge = Ocs2TrajectoryPlayBridge(
            env.unwrapped,
            args_cli.ocs2_trajectory,
            zero_base_command=args_cli.ocs2_trajectory_zero_base_command,
            zero_wrench=args_cli.ocs2_trajectory_zero_wrench,
            zero_arm_effort=args_cli.ocs2_trajectory_zero_arm_effort,
            hold_arm=args_cli.ocs2_trajectory_hold_arm,
            fixed_base_command=args_cli.ocs2_trajectory_fixed_base_command,
            terminal_base_command=args_cli.ocs2_trajectory_terminal_base_command,
            start_delay_s=args_cli.ocs2_trajectory_start_delay,
        )
    # load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

    # export policy to onnx
    if EXPORT_POLICY:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, export_model_dir
        )
        print("Exported policy as jit script to: ", export_model_dir)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor, 
            export_model_dir, 
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            export_model_dir,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )
        export_encoder_as_jit(ppo_runner.alg.encoder, export_model_dir)
    # reset environment
    obs_dict = env.get_observations()
    obs = obs_dict["policy"]
    obs_history = obs_dict.get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict.get("commands") 
    live_target = live_log = None
    live_steps = 0
    live_episode = 0
    if args_cli.ocs2:
        from bipedal_locomotion.controllers.ocs2_live import LiveTargetFile, JsonlDiagnostics
        live_target = LiveTargetFile(args_cli.ocs2_target_file) if args_cli.ocs2_target_file else None
        live_log = JsonlDiagnostics(args_cli.ocs2_diagnostic_log) if args_cli.ocs2_diagnostic_log else None
        robot = env.unwrapped.scene["robot"]
        if live_log is not None:
            live_log.write({"event": "session", "schema": "ocs2_live_diagnostic_v1",
                "step_dt": float(env.unwrapped.step_dt), "checkpoint": resume_path,
                "joint_names": list(robot.joint_names), "fixed_base_command": [-0.29, -0.08, 0.0],
                "ocs2_wrench_semantics": "predicted_arm_on_base; base_Link_frame; base_Link_origin",
                "external_wrench_semantics": "commanded_external_on_payload; zero_baseline; not_measured",
                "response_semantics": "control[t,t+dt); auto_reset_next_state_invalid",
                "wall_clock": "time.monotonic_seconds_on_client"})

    def live_snapshot():
        data = robot.data
        def row(value):
            return value[0].detach().cpu().numpy().copy()
        result = {
            "base_position_world": row(data.root_pos_w), "base_quaternion_world": row(data.root_quat_w),
            "base_linear_velocity_body": row(data.root_lin_vel_b), "base_angular_velocity_body": row(data.root_ang_vel_b),
            "joint_position_actual": row(data.joint_pos), "joint_velocity_actual": row(data.joint_vel),
            "arm_position_actual": row(data.joint_pos[:, ocs2_bridge._arm_joint_ids]),
            "arm_velocity_actual": row(data.joint_vel[:, ocs2_bridge._arm_joint_ids]),
            "arm_tracking_error_rad": row(data.joint_pos[:, ocs2_bridge._arm_joint_ids]) - ocs2_bridge._smoother.position,
            "end_effector_position_world": row(data.body_pos_w[:, ocs2_bridge._ee_body_id]),
            "end_effector_quaternion_world": row(data.body_quat_w[:, ocs2_bridge._ee_body_id]),
            "tilt_deg": math.degrees(math.acos(float((-data.projected_gravity_b[0, 2]).clamp(-1, 1).item()))),
        }
        sensor = env.unwrapped.scene.sensors.get("contact_forces")
        if sensor is not None:
            foot_ids, foot_names = sensor.find_bodies(["ankle_pitch_[RL]_Link"], preserve_order=True)
            result["foot_body_names"] = list(foot_names)
            result["foot_contact_force_world"] = row(sensor.data.net_forces_w[:, foot_ids])
            result["foot_contact"] = (sensor.data.net_forces_w[0, foot_ids].norm(dim=-1) > 1.0).cpu().numpy()
        return result

    if not args_cli.headless:
        from bipedal_locomotion.utils.keyboard import Keyboard
        keyboard: Keyboard = Keyboard(env.unwrapped)  # type: ignore
        gamepad_control_active = False

    diagnostic = args_cli.ocs2_trajectory_diagnostic
    if diagnostic:
        robot = env.unwrapped.scene["robot"]
        arm_joint_ids = ocs2_bridge._arm_joint_ids
        diagnostic_steps = 0
        max_tilt_deg = 0.0
        min_root_height = math.inf
        max_arm_error = 0.0
        max_arm_error_after_warmup = 0.0
        motion_actual_min = [math.inf] * 6
        motion_actual_max = [-math.inf] * 6
        motion_target_min = [math.inf] * 6
        motion_target_max = [-math.inf] * 6
        report_steps = max(1, round(1.0 / float(env.unwrapped.step_dt)))
        print("[DIAG] Starting one bounded offline replay; results use simulation time.", flush=True)
        
    # simulate environment
    while simulation_app.is_running():
        if not args_cli.headless and keyboard is not None:
            keyboard.look_at()
        # run everything in inference mode
        with torch.inference_mode():
            if live_target is not None:
                try:
                    target_update = live_target.poll()
                    if target_update is not None:
                        target_id, (position, quaternion, arrival) = target_update
                        ocs2_bridge.set_target(position, quaternion, arrival, target_id)
                        live_target.accept(target_id)
                        if live_log is not None:
                            live_log.write({"event": "target_accepted", "target_id": target_id,
                                "target_revision": ocs2_bridge.target_revision,
                                "sim_time": float(env.unwrapped.common_step_counter) * float(env.unwrapped.step_dt),
                                "wall_time": time.monotonic(), "position_world": position,
                                "quaternion_wxyz": quaternion, "arrival_s": arrival})
                except (ValueError, OSError) as error:
                    ocs2_bridge._warn(f"Rejected live target: {error}")
            ocs2_solution_ready = True
            if ocs2_bridge is not None:
                ocs2_solution_ready = ocs2_bridge.update(obs, commands)
            if not ocs2_solution_ready and not args_cli.ocs2:
                # Pump the Omniverse event/render loop so WebRTC stays alive,
                # but do not advance physics with a missing or stale MPC
                # result. This also prevents episode resets from continually
                # invalidating a slow first OCS2 solve.
                env.unwrapped.sim.render()
                continue
            if diagnostic:
                # Capture the state before env.step(): a terminal step can
                # return an already auto-reset robot state.
                gravity_z = float((-robot.data.projected_gravity_b[0, 2]).clamp(-1.0, 1.0).item())
                tilt_deg = math.degrees(math.acos(gravity_z))
                root_height = float(robot.data.root_pos_w[0, 2].item())
                arm_target = torch.as_tensor(
                    ocs2_bridge.current_solution.arm_position,
                    device=robot.data.joint_pos.device,
                    dtype=robot.data.joint_pos.dtype,
                )
                arm_error = float(
                    (robot.data.joint_pos[0, arm_joint_ids] - arm_target).abs().max().item()
                )
                max_tilt_deg = max(max_tilt_deg, tilt_deg)
                min_root_height = min(min_root_height, root_height)
                max_arm_error = max(max_arm_error, arm_error)
                if ocs2_bridge.phase != "warmup":
                    max_arm_error_after_warmup = max(max_arm_error_after_warmup, arm_error)
                if ocs2_bridge.phase == "motion":
                    actual_values = robot.data.joint_pos[0, arm_joint_ids].detach().cpu().tolist()
                    target_values = arm_target.detach().cpu().tolist()
                    for index, (actual, target) in enumerate(zip(actual_values, target_values)):
                        motion_actual_min[index] = min(motion_actual_min[index], actual)
                        motion_actual_max[index] = max(motion_actual_max[index], actual)
                        motion_target_min[index] = min(motion_target_min[index], target)
                        motion_target_max[index] = max(motion_target_max[index], target)
            # agent stepping
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            if live_log is not None:
                control_record = {
                    "event": "control_response", "episode": live_episode, "step": live_steps,
                    **ocs2_bridge.diagnostics, "state": live_snapshot(),
                    "policy_observation": obs[0].cpu().numpy().copy(),
                    "observation_history": obs_history[0].cpu().numpy().copy(),
                    "command_observation": commands[0].cpu().numpy().copy(),
                    "encoder_output": est[0].cpu().numpy().copy(), "leg_action": actions[0].cpu().numpy().copy(),
                    "arm_position_applied": ocs2_bridge._smoother.position.copy(),
                    "arm_velocity_applied": ocs2_bridge._smoother.velocity.copy(),
                    "external_wrench_payload_at_body_com": [0.0] * 6,
                    "external_wrench_enabled": False,
                }
            # env stepping
            obs_dict, _, dones, infos = env.step(actions)
            if args_cli.ocs2:
                live_steps += 1
                if live_log is not None:
                    for event in ocs2_bridge._async_client.drain_events():
                        live_log.write(event)
                    timed_out = infos.get("time_outs") if isinstance(infos, dict) else None
                    reason = None if not bool(dones[0]) else (
                        "environment_timeout" if timed_out is not None and bool(timed_out[0]) else "environment_done")
                    control_record.update({
                        "response_sim_time": control_record["control_sim_time"] + float(env.unwrapped.step_dt),
                        "response_wall_time": time.monotonic(), "next_state": live_snapshot(),
                        "response_observation_valid": not bool(dones[0]), "reset_reason": reason,
                    })
                    live_log.write(control_record)
                if bool(dones[0]):
                    live_episode += 1
                if args_cli.ocs2_live_duration and live_steps * float(env.unwrapped.step_dt) >= args_cli.ocs2_live_duration:
                    break
            if diagnostic:
                diagnostic_steps += 1
                if diagnostic_steps % report_steps == 0:
                    print(
                        f"[DIAG] phase={ocs2_bridge.phase} "
                        f"playback_t={ocs2_bridge.playback_time:.2f}s "
                        f"tilt={tilt_deg:.1f}deg root_z={root_height:.3f}m "
                        f"arm_max_error={arm_error:.3f}rad",
                        flush=True,
                    )
                if bool(torch.any(dones)):
                    timeouts = infos.get("time_outs") if isinstance(infos, dict) else None
                    timed_out = bool(timeouts[0].item()) if timeouts is not None else False
                    reason = "environment_timeout" if timed_out else "environment_done"
                elif ocs2_bridge.playback_time >= (
                    ocs2_bridge.trajectory_duration
                    + args_cli.ocs2_trajectory_diagnostic_post_duration
                ):
                    reason = "completed_without_reset"
                else:
                    reason = None
                if reason is not None:
                    print(
                        f"[DIAG] result={reason} steps={diagnostic_steps} "
                        f"max_tilt={max_tilt_deg:.1f}deg "
                        f"min_root_z={min_root_height:.3f}m "
                        f"max_arm_error={max_arm_error:.3f}rad "
                        f"max_arm_error_after_warmup={max_arm_error_after_warmup:.3f}rad",
                        flush=True,
                    )
                    if all(math.isfinite(value) for value in motion_actual_min):
                        actual_span = [
                            maximum - minimum
                            for minimum, maximum in zip(motion_actual_min, motion_actual_max)
                        ]
                        target_span = [
                            maximum - minimum
                            for minimum, maximum in zip(motion_target_min, motion_target_max)
                        ]
                        print(
                            f"[DIAG] motion_arm_span_rad actual={[round(v, 4) for v in actual_span]} "
                            f"target={[round(v, 4) for v in target_span]}",
                            flush=True,
                        )
                    break
            obs = obs_dict["policy"]
            obs_history = obs_dict.get("obsHistory")
            obs_history = obs_history.flatten(start_dim=1)
            commands = obs_dict.get("commands") 
            if ocs2_bridge is not None and torch.any(dones):
                ocs2_bridge.reset()

    # close the simulator
    if ocs2_bridge is not None:
        ocs2_bridge.close()
    if live_log is not None:
        for event in ocs2_bridge._async_client.drain_events():
            live_log.write(event)
        live_log.close()
    env.close()


if __name__ == "__main__":
    # Runtime bridge validation should reach the render/control loop quickly;
    # exporting unchanged artifacts on every OCS2 launch is unnecessary.
    EXPORT_POLICY = not (args_cli.ocs2 or args_cli.ocs2_trajectory is not None)
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
