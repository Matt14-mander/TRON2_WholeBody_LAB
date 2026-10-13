"""Independent stand/walk registrations; original WholeBody tasks are retained."""
import gymnasium as gym

from .env_cfg import StandWalkEnvCfg
from .ppo_cfg import StandWalkPPORunnerCfg

gym.register(id="Isaac-Limx-SFYG-TRON2A-WholeBody-StandWalk-Finetune-v0",
             entry_point="bipedal_locomotion.tasks.standwalk.env:StandWalkRLEnv",
             disable_env_checker=True,
             kwargs={"env_cfg_entry_point": StandWalkEnvCfg,
                     "rsl_rl_cfg_entry_point": StandWalkPPORunnerCfg})
