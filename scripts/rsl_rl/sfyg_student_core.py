"""SFYG Student building blocks; runner lives in sfyg_student_runner.py.

History is the CURRENT T1 eight-frame, oldest-first arm history (144 values).
The estimated label is external payload-COM [Fx/40, Fy/40, Mz/10] in BASE
axes, not grasp reaction or OCS2's predicted arm-on-base wrench. T1 currently
excites only positive Fx. Historical four-frame estimator weights do not fit.
"""
from copy import deepcopy
import torch
from torch import nn


def _history(history):
    if history.dim() != 2 or history.size(1) != 144:
        raise ValueError('Student requires batched arm history144, with no true force')
    if not bool(torch.isfinite(history).all()):
        raise ValueError('Nonfinite student history')


class IntentEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(144, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 3))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _history(history)
        return self.network(history)


class StudentPolicy(nn.Module):
    """history144 -> raw physical action9, matching current T1 (no tanh).

    Caller applies the existing T1 execution mapping exactly once. Estimator
    regression is separate from actor/critic/distillation optimization. No
    clamp on estimates: log their out-of-domain values instead of hiding them.
    This module owns no history buffer; use the T1 History/reset convention.
    """
    def __init__(self, actor, estimator=None):
        super().__init__()
        # Initialization must never share weights with a frozen teacher.
        self.actor = deepcopy(actor)
        self.estimator = IntentEstimator() if estimator is None else deepcopy(estimator)

    @torch.jit.export
    def estimate(self, history: torch.Tensor) -> torch.Tensor:
        return self.estimator(history)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        estimate = self.estimator(history)
        raw = self.actor(torch.cat((history, estimate.detach()), dim=-1))
        if raw.dim() != 2 or raw.size(0) != history.size(0) or raw.size(1) != 9:
            raise ValueError('Student actor must return raw action9')
        if not bool(torch.isfinite(raw).all()):
            raise ValueError('Nonfinite student action')
        return raw


def estimator_step(estimator, optimizer, history, target_reference):
    """One S0 regression step on aligned pre-action observations.

    target_reference is teacher_obs[:, -3:] at the SAME sampling instant.
    Never pass a next-step observation or preceding-interval diagnostic.
    Privilege is a training label only and is not stored in StudentPolicy.
    """
    _history(history)
    if (target_reference.shape != (history.shape[0], 3)
            or not bool(torch.isfinite(target_reference).all())
            or history.shape[0] == 0):
        raise ValueError('Finite paired history144 and normalized reference3 required')
    expected = {id(p) for p in estimator.parameters() if p.requires_grad}
    actual = {id(p) for group in optimizer.param_groups for p in group['params']}
    if actual != expected:
        raise ValueError('Estimator optimizer must own estimator parameters exclusively')
    optimizer.zero_grad(set_to_none=True)
    predicted = estimator(history.detach())
    loss = (predicted-target_reference.detach()).square().mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite estimator loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(estimator.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def latent_normal_kl(student_mean, student_std, teacher_mean, teacher_std):
    """KL(Student || Teacher) for RAW Normal actions, before T1 clipping.

    Clipping/deadband/masking is non-invertible: this is NOT an exact KL of
    executed commands. It is unrelated to the old low-level PPO KL changes.
    """
    shape = student_mean.shape
    values = (student_mean, student_std, teacher_mean, teacher_std)
    if len(shape) != 2 or any(v.shape != shape for v in values):
        raise ValueError('Matching batched Normal parameters required')
    if any(not bool(torch.isfinite(v).all()) for v in values):
        raise ValueError('Nonfinite Normal parameters')
    if bool((student_std <= 0).any()) or bool((teacher_std <= 0).any()):
        raise ValueError('Positive Normal standard deviations required')
    tm, ts = teacher_mean.detach(), teacher_std.detach()
    return (torch.log(ts/student_std)
            + (student_std.square()+(student_mean-tm).square())/(2*ts.square())-.5).sum(-1)


def estimator_metrics(predicted, target):
    """Diagnostic errors; low regression error alone is not task acceptance."""
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 3 or not len(predicted):
        raise ValueError('Nonempty paired reference3 required')
    error = (predicted-target).detach()
    if not bool(torch.isfinite(error).all()):
        raise ValueError('Nonfinite estimator metrics')
    scale = error.new_tensor([40., 40., 10.])
    return dict(normalized_rmse=error.square().mean(0).sqrt().cpu().tolist(),
                physical_rmse=(error*scale).square().mean(0).sqrt().cpu().tolist(),
                physical_bias=(error*scale).mean(0).cpu().tolist(),
                is_acceptance=False)
