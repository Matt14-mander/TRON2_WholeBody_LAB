"""Load the original S1 student checkpoint for history-only MuJoCo inference."""
from pathlib import Path
import sys

import torch

from sfyg_teacher_policy import ArmHistoryPolicy
from sfyg_policy import ROOT, sha256
from sfyg_student_core import StudentPolicy
from sfyg_teacher_core import runner_config


S1_CHECKPOINT = ROOT / 'logs/rsl_rl/sfyg_student_s1_gpu0_500_20261107_141132/train/model_500.pt'
ROBUST_BUNDLE = ROOT / 'logs/rsl_rl/sfyg_robust_5000_gpu1_20261027_131013/exported/model_5000'


class StudentCheckpointPolicy(ArmHistoryPolicy):
    input_width = 144
    # External load in MuJoCo only. Training and action limits come from the checkpoint.
    playback_force_limit_n = 16.

    def __init__(self, checkpoint, low_config=None):
        saved = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
        self.manifest = saved['student_contract']
        if (saved['iteration'] != 500 or self.manifest['schema'] != 'sfyg_student_v1'
                or self.manifest['actor_input'] != 144 or self.manifest['raw_output'] != 9):
            raise ValueError('Expected the S1 iteration-500 history144 student checkpoint')
        teacher = self.manifest['teacher_contract']
        if low_config is None:
            # State-only observation has no attached locomotion policy.
            low_config = dict(teacher['low_interface'], checkpoint_sha256=teacher['low_sha256'])
        super().__init__(teacher, low_config)
        sys.path.insert(0, str(ROOT / 'rsl_rl'))
        from rsl_rl.modules import ActorCritic
        template = ActorCritic(147, 177, 9, **runner_config()['policy'])
        self.actor = StudentPolicy(template.actor)
        self.actor.load_state_dict(saved['policy'], strict=True)
        self.actor.requires_grad_(False).eval()
        if not all(torch.isfinite(p).all() for p in self.actor.parameters()):
            raise ValueError('Nonfinite student checkpoint')
        self.sha = sha256(checkpoint)

    @torch.inference_mode()
    def command(self, arm_error, arm_velocity, previous_target_offset, tick):
        history = self.arm_history(arm_error, arm_velocity, previous_target_offset, tick)
        self.estimated_wrench = (self.actor.estimate(history)[0] * history.new_tensor([40.,40.,10.])).tolist()
        return self.execute(history, tick)
