"""Conservative PPO + supervised velocity estimator; fresh v4 weights."""
from isaaclab.utils import configclass
from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg import SF_TRON2AFlatPPORunnerCfg
from bipedal_locomotion.utils.wrappers.rsl_rl.rl_mlp_cfg import RslRlPpoActorCriticBoundedCfg


@configclass
class PaintLocomotionV4PPORunnerCfg(SF_TRON2AFlatPPORunnerCfg):
    def __post_init__(self):
        self.experiment_name = "sfyg_paint_locomotion_v4"
        self.num_steps_per_env = 32
        self.max_iterations = 10000
        self.save_interval = 250
        self.clip_actions = 5.
        self.policy = RslRlPpoActorCriticBoundedCfg(init_noise_std=.6,
            actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
            activation="elu", squash_action=True, squash_action_scale=5.)
        self.algorithm.learning_rate = 3e-4
        self.algorithm.min_learning_rate = 1e-5
        self.algorithm.max_learning_rate = 3e-4
        self.algorithm.early_stop = True
        self.algorithm.entropy_coef = .005
        self.algorithm.est_learning_rate = 3e-4
        self.algorithm.encoder_max_grad_norm = 1.
