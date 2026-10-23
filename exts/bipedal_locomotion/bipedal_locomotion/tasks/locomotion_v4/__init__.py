"""Full-robot, physically loaded PAINT replacement backbone (independent of v3)."""
import gymnasium as gym
from .env_cfg import PaintLocomotionV4EnvCfg
from .ppo_cfg import PaintLocomotionV4PPORunnerCfg

gym.register(id="Isaac-Limx-SFYG-TRON2A-Paint-Locomotion-v4",
    entry_point="bipedal_locomotion.tasks.locomotion_v4.env:PaintLocomotionV4Env",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": PaintLocomotionV4EnvCfg,
                                     "rsl_rl_cfg_entry_point": PaintLocomotionV4PPORunnerCfg})
