import gymnasium as gym

from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg import (
    SF_TRON2AFlatPPORunnerCfg, WF_TRON2AFlatPPORunnerCfg,
    SFYG_TRON2AFlatPPORunnerCfg, WFYG_TRON2AFlatPPORunnerCfg,
    SF_TRON2ARoughPPORunnerCfg, WF_TRON2ARoughPPORunnerCfg,
    SFYG_TRON2ARoughPPORunnerCfg, WFYG_TRON2ARoughPPORunnerCfg,
    SFYG_TRON2AWholeBodyFlatPPORunnerCfg, SFYG_TRON2AWholeBodyRoughPPORunnerCfg,
)

from . import (
    limx_solefoot_tron2a_env_cfg,
    limx_wheelfoot_tron2a_env_cfg,
    limx_solefoot_yg_tron2a_env_cfg,
    limx_wheelfoot_yg_tron2a_env_cfg,
    limx_solefoot_yg_tron2a_wholebody_env_cfg,
)

##
# Create PPO runners for RSL-RL
##

limx_sf_tron2a_blind_flat_runner_cfg = SF_TRON2AFlatPPORunnerCfg()

limx_wf_tron2a_blind_flat_runner_cfg = WF_TRON2AFlatPPORunnerCfg()

limx_sfyg_tron2a_blind_flat_runner_cfg = SFYG_TRON2AFlatPPORunnerCfg()

limx_wfyg_tron2a_blind_flat_runner_cfg = WFYG_TRON2AFlatPPORunnerCfg()

limx_sf_tron2a_blind_rough_runner_cfg = SF_TRON2ARoughPPORunnerCfg()

limx_wf_tron2a_blind_rough_runner_cfg = WF_TRON2ARoughPPORunnerCfg()

limx_sfyg_tron2a_blind_rough_runner_cfg = SFYG_TRON2ARoughPPORunnerCfg()

limx_wfyg_tron2a_blind_rough_runner_cfg = WFYG_TRON2ARoughPPORunnerCfg()

limx_sfyg_tron2a_wholebody_flat_runner_cfg = SFYG_TRON2AWholeBodyFlatPPORunnerCfg()

limx_sfyg_tron2a_wholebody_rough_runner_cfg = SFYG_TRON2AWholeBodyRoughPPORunnerCfg()


##
# Register Gym environments
##

######################################
# SF_TRON2A Blind Flat Environment
######################################
gym.register(
    id="Isaac-Limx-SF-TRON2A-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-TRON2A-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_blind_flat_runner_cfg,
    },
)


######################################
# WF_TRON2A Blind Flat Environment
######################################
gym.register(
    id="Isaac-Limx-WF-TRON2A-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_tron2a_env_cfg.WF_TRON2A_BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_tron2a_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-TRON2A-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_tron2a_env_cfg.WF_TRON2A_BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_tron2a_blind_flat_runner_cfg,
    },
)


######################################
# SFYG_TRON2A Blind Flat Environment
######################################
gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_blind_flat_runner_cfg,
    },
)


######################################
# WFYG_TRON2A Blind Flat Environment
######################################
gym.register(
    id="Isaac-Limx-WFYG-TRON2A-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_yg_tron2a_env_cfg.WFYG_TRON2A_BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wfyg_tron2a_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WFYG-TRON2A-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_yg_tron2a_env_cfg.WFYG_TRON2A_BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wfyg_tron2a_blind_flat_runner_cfg,
    },
)


######################################
# SF_TRON2A Blind Rough Environment
######################################
gym.register(
    id="Isaac-Limx-SF-TRON2A-Blind-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_BlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_blind_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-TRON2A-Blind-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_BlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_blind_rough_runner_cfg,
    },
)


######################################
# WF_TRON2A Blind Rough Environment
######################################
gym.register(
    id="Isaac-Limx-WF-TRON2A-Blind-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_tron2a_env_cfg.WF_TRON2A_BlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_tron2a_blind_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-TRON2A-Blind-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_tron2a_env_cfg.WF_TRON2A_BlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_tron2a_blind_rough_runner_cfg,
    },
)


######################################
# SFYG_TRON2A Blind Rough Environment
######################################
gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Blind-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_BlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_blind_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Blind-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_BlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_blind_rough_runner_cfg,
    },
)


######################################
# WFYG_TRON2A Blind Rough Environment
######################################
gym.register(
    id="Isaac-Limx-WFYG-TRON2A-Blind-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_yg_tron2a_env_cfg.WFYG_TRON2A_BlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wfyg_tron2a_blind_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WFYG-TRON2A-Blind-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_yg_tron2a_env_cfg.WFYG_TRON2A_BlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wfyg_tron2a_blind_rough_runner_cfg,
    },
)


######################################
# SFYG_TRON2A Whole-Body Environments
######################################
gym.register(
    id="Isaac-Limx-SFYG-TRON2A-WholeBody-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_wholebody_env_cfg.SFYG_TRON2A_WholeBodyFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_wholebody_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SFYG-TRON2A-WholeBody-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_wholebody_env_cfg.SFYG_TRON2A_WholeBodyFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_wholebody_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SFYG-TRON2A-WholeBody-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_wholebody_env_cfg.SFYG_TRON2A_WholeBodyRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_wholebody_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SFYG-TRON2A-WholeBody-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_wholebody_env_cfg.SFYG_TRON2A_WholeBodyRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_wholebody_rough_runner_cfg,
    },
)
