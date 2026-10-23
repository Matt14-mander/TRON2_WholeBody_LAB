"""One transport implementation with teacher/student actor observation variants."""
import gymnasium as gym
from .env_cfg import TransportV4EnvCfg, TransportV4StudentEnvCfg
from .ppo_cfg import TransportV4PPORunnerCfg

for mode, cfg in (("Teacher", TransportV4EnvCfg), ("Student", TransportV4StudentEnvCfg)):
    gym.register(id=f"Isaac-Limx-SFYG-TRON2A-Paint-Transport-{mode}-v4",
        entry_point="bipedal_locomotion.tasks.transport_v4.env:TransportV4Env", disable_env_checker=True,
        kwargs={"env_cfg_entry_point": cfg, "rsl_rl_cfg_entry_point": TransportV4PPORunnerCfg})
