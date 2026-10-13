"""Independent PAINT high-level teacher registrations."""
import gymnasium as gym

from .teacher_env_cfg import PaintTeacherFlatEnvCfg, PaintTeacherFlatEnvCfg_PLAY
from .teacher_ppo_cfg import PaintTeacherPPORunnerCfg

for task, cfg in (("Isaac-Limx-SFYG-TRON2A-Paint-Teacher-Flat-v0", PaintTeacherFlatEnvCfg),
                  ("Isaac-Limx-SFYG-TRON2A-Paint-Teacher-Flat-Play-v0", PaintTeacherFlatEnvCfg_PLAY)):
    gym.register(id=task, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": cfg, "rsl_rl_cfg_entry_point": PaintTeacherPPORunnerCfg})
