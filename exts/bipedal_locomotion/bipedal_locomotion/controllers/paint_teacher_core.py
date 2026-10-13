"""Torch-only contracts for the privileged PAINT high-level teacher.

No simulator imports: checkpoint loading, history, and admittance can be checked
on a development machine. External-on-payload is distinct from arm-on-base.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

SCHEMA = "paint_teacher_v1"
LEG_JOINT_NAMES = tuple(
    f"{part}_{side}_Joint" for side in ("L", "R")
    for part in ("proximal_pitch", "proximal_roll", "proximal_yaw", "knee", "ankle_pitch")
)
ARM_JOINT_NAMES = tuple(f"arm{i}_Joint" for i in range(1, 7))


def mlp(widths):
    layers = []
    for index, (nin, nout) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(nin, nout))
        if index < len(widths) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class FrozenLocomotion(nn.Module):
    """Strictly load the existing flat WholeBody ELU 78->10 / 420->3 pair.

    Only the critic is omitted. No partial actor/encoder load is permitted.
    """
    def __init__(self, checkpoint_path, device="cpu"):
        super().__init__()
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Locomotion checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.actor = mlp([78, 512, 512, 256, 128, 10])
        self.encoder = mlp([420, 256, 128, 3])
        try:
            actor = {k.removeprefix("actor."): v for k, v in checkpoint["model_state_dict"].items()
                     if k.startswith("actor.")}
            encoder = {k.removeprefix("encoder."): v for k, v in checkpoint["encoder_state_dict"].items()}
            self.actor.load_state_dict(actor, strict=True)
            self.encoder.load_state_dict(encoder, strict=True)
        except (KeyError, RuntimeError) as error:
            raise ValueError("Expected a complete flat WholeBody locomotion actor and history encoder.") from error
        if any(not torch.isfinite(p).all() for p in self.parameters()):
            raise ValueError("Locomotion checkpoint contains non-finite parameters.")
        self.requires_grad_(False)
        self.to(device).eval()

    @torch.no_grad()
    def forward(self, proprioception, history, base_command):
        # Original policy order: latent(3), proprioception(42), wrench(30), command(3).
        if proprioception.shape[1:] != (42,) or history.shape[1:] != (10, 42) or base_command.shape[1:] != (3,):
            raise ValueError("Frozen locomotion requires 42 / 10x42 / 3 observation widths.")
        zeros = proprioception.new_zeros((len(proprioception), 30))
        return self.actor(torch.cat((self.encoder(history.flatten(1)), proprioception, zeros, base_command), dim=-1))


class SampledHistory:
    """Oldest-first per-environment history; sample once at each physics tick.

    On reset, the first new sample fills the window, matching Isaac's history
    initialization. A reset of one environment never clears the others.
    """
    def __init__(self, num_envs, length, width, device):
        self.data = torch.zeros(num_envs, length, width, device=device)
        self.last_tick = torch.full((num_envs,), -1, device=device, dtype=torch.long)

    def reset(self, env_ids):
        self.data[env_ids] = 0
        self.last_tick[env_ids] = -1

    def append(self, row, tick):
        fresh = self.last_tick != tick
        first = self.last_tick < 0
        ids = fresh.nonzero(as_tuple=False).flatten()
        self.data[ids] = torch.cat((self.data[ids, 1:], row[ids, None]), dim=1)
        self.data[first] = row[first, None]
        self.last_tick[fresh] = tick
        return self.data


def admittance_step(velocity, wrench, dt, mass, damping, limits):
    """Exact zero-order-hold update of M dv/dt + D v = wrench.

    Velocity and wrench must use the same frame; channels are Fx/Fy/Mz.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Admittance dt must be finite and positive.")
    mass, damping, limits = (torch.as_tensor(x, device=velocity.device, dtype=velocity.dtype)
                             for x in (mass, damping, limits))
    if any(x.shape != (3,) or not torch.isfinite(x).all() or (x <= 0).any()
           for x in (mass, damping, limits)):
        raise ValueError("Admittance parameters must be positive finite three-vectors.")
    decay = torch.exp(-damping * dt / mass)
    return torch.clamp(decay * velocity + (1 - decay) * wrench / damping, -limits, limits)


def bounded_targets(actions, previous, nominal, lower, upper, velocity_scale, arm_scale, slew_rate, dt):
    """Return physical vx/vy/wz and absolute q targets with slew/soft limits."""
    desired = torch.cat((actions[:, :3].clamp(-1, 1) * velocity_scale,
                         nominal + actions[:, 3:].clamp(-1, 1) * arm_scale), dim=-1)
    limited = previous + (desired - previous).clamp(-slew_rate * dt, slew_rate * dt)
    limited[:, 3:] = torch.maximum(torch.minimum(limited[:, 3:], upper), lower)
    return limited


def force_envelope(phase, sine):
    trapezoid = torch.minimum(torch.minimum(phase / .1, (1 - phase) / .1), torch.ones_like(phase))
    envelope = torch.where(sine, torch.sin(torch.pi * phase), trapezoid).clamp_min(0)
    return torch.where((phase >= 0) & (phase < 1), envelope, torch.zeros_like(envelope))
