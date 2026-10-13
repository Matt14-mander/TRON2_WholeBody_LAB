"""Conservative warm-start PPO, retaining the original actor/encoder widths."""
from isaaclab.utils import configclass

from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg import SFYG_TRON2AWholeBodyFlatPPORunnerCfg


@configclass
class StandWalkPPORunnerCfg(SFYG_TRON2AWholeBodyFlatPPORunnerCfg):
    experiment_name = "sfyg_wholebody_standwalk_finetune"
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 100

    def __post_init__(self):
        self.algorithm.learning_rate = 1e-4
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = .005
        self.algorithm.est_learning_rate = 1e-5
