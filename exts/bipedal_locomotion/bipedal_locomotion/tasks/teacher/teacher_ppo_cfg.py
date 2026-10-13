"""Teacher-only PPO. Arm history is explicit; no estimator is optimized."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg import SF_TRON2AFlatPPORunnerCfg
from bipedal_locomotion.utils.wrappers.rsl_rl.rl_mlp_cfg import EncoderCfg


@configclass
class PaintTeacherPPORunnerCfg(SF_TRON2AFlatPPORunnerCfg):
    experiment_name = "paint_teacher_flat"
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 100
    policy = RslRlPpoActorCriticCfg(
        actor_hidden_dims=[256, 256, 128], critic_hidden_dims=[256, 256, 128],
        activation="elu", init_noise_std=.3,
    )
    # Compatibility with repository runner. num_output_dim=0 disables its
    # supervised latent regression and extra optimizer; PPO gets arm history directly.
    encoder = EncoderCfg(num_output_dim=0, hidden_dims=[1], output_detach=True)

    def __post_init__(self):
        self.algorithm.learning_rate = 3e-4
        self.algorithm.entropy_coef = .005
        self.algorithm.obs_history_len = 1
