from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
from bipedal_locomotion.utils.wrappers.rsl_rl.rl_mlp_cfg import EncoderCfg, RslRlPpoAlgorithmMlpCfg

import os
robot_type = os.getenv("ROBOT_TYPE")


#-----------------------------------------------------------------
@configclass
class SF_TRON2AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 500
    experiment_name = "sf_tron_2a_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )
    encoder = EncoderCfg(
        output_detach = True,
        num_output_dim = 3,
        hidden_dims = [256, 128],
        activation = "elu",
        orthogonal_init = False,
    )

#-----------------------------------------------------------------
@configclass
class WF_TRON2AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "wf_tron_2a_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )
    encoder = EncoderCfg(
        output_detach = True,
        num_output_dim = 3,
        hidden_dims = [256, 128],
        activation = "elu",
        orthogonal_init = False,
    )


#-----------------------------------------------------------------
@configclass
class SFYG_TRON2AFlatPPORunnerCfg(SF_TRON2AFlatPPORunnerCfg):
    experiment_name = "sfyg_tron_2a_flat"


#-----------------------------------------------------------------
@configclass
class WFYG_TRON2AFlatPPORunnerCfg(WF_TRON2AFlatPPORunnerCfg):
    experiment_name = "wfyg_tron_2a_flat"


#-----------------------------------------------------------------
# Rough 变体：与 Flat 同结构，仅 experiment_name 区分日志目录
#-----------------------------------------------------------------
@configclass
class SF_TRON2ARoughPPORunnerCfg(SF_TRON2AFlatPPORunnerCfg):
    experiment_name = "sf_tron_2a_rough"


@configclass
class WF_TRON2ARoughPPORunnerCfg(WF_TRON2AFlatPPORunnerCfg):
    experiment_name = "wf_tron_2a_rough"


@configclass
class SFYG_TRON2ARoughPPORunnerCfg(SF_TRON2AFlatPPORunnerCfg):
    experiment_name = "sfyg_tron_2a_rough"


@configclass
class WFYG_TRON2ARoughPPORunnerCfg(WF_TRON2AFlatPPORunnerCfg):
    experiment_name = "wfyg_tron_2a_rough"
