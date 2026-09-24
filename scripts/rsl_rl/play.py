"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

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
    trajectory_only_option_used = (
        args_cli.ocs2_trajectory_zero_base_command
        or args_cli.ocs2_trajectory_zero_wrench
        or args_cli.ocs2_trajectory_terminal_base_command is not None
        or args_cli.ocs2_trajectory_start_delay != 0.0
    )
    if trajectory_only_option_used and args_cli.ocs2_trajectory is None:
        raise ValueError("Offline trajectory ablation flags require --ocs2_trajectory.")
    if args_cli.ocs2_deterministic_reset and not ocs2_control_enabled:
        raise ValueError("--ocs2_deterministic_reset requires --ocs2 or --ocs2_trajectory.")
    if args_cli.ocs2_trajectory_zero_base_command and (
        args_cli.ocs2_trajectory_terminal_base_command is not None
    ):
        raise ValueError(
            "--ocs2_trajectory_zero_base_command conflicts with "
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
        # OCS2 owns the planar command. Prevent the random command generator
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

    if not args_cli.headless:
        from bipedal_locomotion.utils.keyboard import Keyboard
        keyboard: Keyboard = Keyboard(env.unwrapped)  # type: ignore
        gamepad_control_active = False
        
    # simulate environment
    while simulation_app.is_running():
        if not args_cli.headless and keyboard is not None:
            keyboard.look_at()
        # run everything in inference mode
        with torch.inference_mode():
            ocs2_solution_ready = True
            if ocs2_bridge is not None:
                ocs2_solution_ready = ocs2_bridge.update(obs, commands)
            if not ocs2_solution_ready:
                # Pump the Omniverse event/render loop so WebRTC stays alive,
                # but do not advance physics with a missing or stale MPC
                # result. This also prevents episode resets from continually
                # invalidating a slow first OCS2 solve.
                env.unwrapped.sim.render()
                continue
            # agent stepping
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            # env stepping
            obs_dict, _, dones, infos = env.step(actions)
            obs = obs_dict["policy"]
            obs_history = obs_dict.get("obsHistory")
            obs_history = obs_history.flatten(start_dim=1)
            commands = obs_dict.get("commands") 
            if ocs2_bridge is not None and torch.any(dones):
                ocs2_bridge.reset()

    # close the simulator
    if ocs2_bridge is not None:
        ocs2_bridge.close()
    env.close()


if __name__ == "__main__":
    # Runtime bridge validation should reach the render/control loop quickly;
    # exporting unchanged artifacts on every OCS2 launch is unnecessary.
    EXPORT_POLICY = not (args_cli.ocs2 or args_cli.ocs2_trajectory is not None)
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
