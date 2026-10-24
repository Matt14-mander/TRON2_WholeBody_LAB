"""Paired pre-action student/teacher observations; ordinary PPO returns/GAE."""
import torch
from .rollout_storage import RolloutStorage


class PaintStorage(RolloutStorage):
    def __init__(self, num_envs, horizon, device="cpu"):
        super().__init__(num_envs, horizon, [144], [177], [1], [0], [9], device)
        self.teacher_observations = torch.zeros(horizon, num_envs, 147, device=device)

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.teacher_observations[self.step].copy_(transition.teacher_observations)
        super().add_transitions(transition)

    def batches(self, count, epochs):
        if self.step != self.num_transitions_per_env:
            raise RuntimeError("Student PPO requires a complete rollout")
        total = self.num_envs * self.num_transitions_per_env
        if not 1 <= count <= total:
            raise ValueError("Invalid student mini-batch count")
        names = ("observations", "teacher_observations", "critic_obs", "actions", "values",
                 "returns", "advantages", "actions_log_prob", "mu", "sigma")
        flat = {name: getattr(self, name).flatten(0, 1) for name in names}
        for _ in range(epochs):
            # Include all samples, also when total is not divisible by count.
            for ids in torch.randperm(total, device=self.device).tensor_split(count):
                yield {name: value[ids] for name, value in flat.items()}
