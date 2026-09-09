"""Script to play a checkpoint if an RL agent from RSL-RL with gamepad teleop."""

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
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL and gamepad teleop.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")
parser.add_argument("--disable_gamepad", action="store_true", default=False, help="Disable gamepad teleoperation.")

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
import torch

import carb
from rsl_rl.runner import OnPolicyRunner

from isaaclab.devices import Se2GamepadCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg


def main():
    """Play with RSL-RL agent using gamepad for base velocity command."""
    use_gamepad = not args_cli.disable_gamepad

    if use_gamepad:
        # Enforce single environment for manual teleoperation.
        args_cli.num_envs = 1
        print("[INFO] Forcing num_envs = 1 for gamepad control.")

    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.seed = agent_cfg.seed

    cmd_ranges = None
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        cmd_ranges = env_cfg.commands.base_velocity.ranges
        print(
            "[INFO] base_velocity ranges:"
            f" lin_vel_x={cmd_ranges.lin_vel_x},"
            f" lin_vel_y={cmd_ranges.lin_vel_y},"
            f" ang_vel_z={cmd_ranges.ang_vel_z}"
        )

    if use_gamepad and hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        # Cancel command resampling; command is now directly driven by gamepad.
        env_cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
        if hasattr(env_cfg.commands.base_velocity, "heading_command"):
            env_cfg.commands.base_velocity.heading_command = False
        if hasattr(env_cfg.commands.base_velocity, "rel_standing_envs"):
            env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        if hasattr(env_cfg.commands.base_velocity, "rel_heading_envs"):
            env_cfg.commands.base_velocity.rel_heading_envs = 0.0

    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
        print("[INFO] Disabled external force (push_robot) for play.")

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
            "video_folder": os.path.join(log_dir, "videos", "play_test"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create gamepad teleop device
    gamepad = None
    if use_gamepad:
        try:
            from isaaclab.devices import Se2Gamepad

            gamepad = Se2Gamepad(
                Se2GamepadCfg(
                    v_x_sensitivity=1.0,
                    v_y_sensitivity=0.75,
                    omega_z_sensitivity=3.0,
                    dead_zone=0.02,
                    sim_device=env.unwrapped.device,
                )
            )
            gamepad.reset()
            gamepad.add_callback(carb.input.GamepadInput.RIGHT_SHOULDER, env.reset)
            print(gamepad)
            print("[INFO] Gamepad teleop enabled.")
            print("[INFO] Left stick: vx / vy, right stick: yaw rate.")
            print("[INFO] Press R1 (RIGHT_SHOULDER) to reset environment.")
        except Exception as err:
            gamepad = None
            print(f"[WARN] Failed to initialize gamepad. Fallback to policy default commands. Error: {err}")

    # load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

    # reset environment
    obs_dict = env.get_observations()
    obs = obs_dict["policy"]
    obs_history = obs_dict.get("obsHistory").flatten(start_dim=1)
    commands = obs_dict.get("commands")

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if gamepad is not None:
                base_command = gamepad.advance().to(dtype=torch.float32).unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
                # Invert y to match expected "left/right" direction convention in this project.
                base_command[:, 1] *= -1.0
                if cmd_ranges is not None:
                    base_command[:, 0].clamp_(cmd_ranges.lin_vel_x[0], cmd_ranges.lin_vel_x[1])
                    base_command[:, 1].clamp_(cmd_ranges.lin_vel_y[0], cmd_ranges.lin_vel_y[1])
                    base_command[:, 2].clamp_(cmd_ranges.ang_vel_z[0], cmd_ranges.ang_vel_z[1])
                env.unwrapped.command_manager.get_command("base_velocity")[:] = base_command

            # agent stepping
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())

            # env stepping
            obs_dict, _, _, _ = env.step(actions)
            obs = obs_dict["policy"]
            obs_history = obs_dict.get("obsHistory").flatten(start_dim=1)
            commands = obs_dict.get("commands")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
