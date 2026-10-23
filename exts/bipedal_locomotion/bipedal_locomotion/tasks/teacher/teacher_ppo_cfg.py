"""Teacher-only PPO. Arm history is explicit; no estimator is optimized."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg import SF_TRON2AFlatPPORunnerCfg
from bipedal_locomotion.utils.wrappers.rsl_rl.rl_mlp_cfg import EncoderCfg


@configclass
class TeacherPolicyCfg(RslRlPpoActorCriticCfg):
    bounded_distribution: bool = True


@configclass
class PaintTeacherPPORunnerCfg(SF_TRON2AFlatPPORunnerCfg):
    experiment_name = "paint_teacher_flat"
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 100
    policy = TeacherPolicyCfg(
        actor_hidden_dims=[256, 256, 128], critic_hidden_dims=[256, 256, 128],
        activation="elu", init_noise_std=.08,
    )
    # Compatibility with repository runner. num_output_dim=0 disables its
    # supervised latent regression and extra optimizer; PPO gets arm history directly.
    encoder = EncoderCfg(num_output_dim=0, hidden_dims=[1], output_detach=True)

    def __post_init__(self):
        self.clip_actions = None  # bounded samples go unchanged into PPO storage/environment
        self.algorithm.learning_rate = 1e-4
        self.algorithm.entropy_coef = .001
        self.algorithm.obs_history_len = 1
