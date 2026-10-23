"""Guide-sized teacher policy; the frozen low-level module is not optimized."""
from isaaclab.utils import configclass
from bipedal_locomotion.tasks.teacher.teacher_ppo_cfg import PaintTeacherPPORunnerCfg, TeacherPolicyCfg


@configclass
class TransportV4PPORunnerCfg(PaintTeacherPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "paint_transport_v4_teacher"
        self.policy = TeacherPolicyCfg(actor_hidden_dims=[128, 128, 128],
            critic_hidden_dims=[128, 128, 128], activation="elu", init_noise_std=.2)
        self.algorithm.min_learning_rate = 1e-5
        self.algorithm.max_learning_rate = 1e-4
        self.algorithm.early_stop = True
