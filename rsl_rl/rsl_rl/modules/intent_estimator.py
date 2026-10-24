"""PAINT transport v4: noisy eight-frame arm history to normalized base wrench."""
import torch
from torch import nn


class IntentEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(144, 128), nn.ELU(), nn.Linear(128, 128),
                                     nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 3))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.network(history)


class StudentInference(nn.Module):
    """Deployable HL only. Input history144; output bounded action9, no privilege."""
    def __init__(self, estimator, actor):
        super().__init__()
        self.estimator = estimator
        self.actor = actor

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.actor(torch.cat((history, self.estimator(history)), dim=-1))
