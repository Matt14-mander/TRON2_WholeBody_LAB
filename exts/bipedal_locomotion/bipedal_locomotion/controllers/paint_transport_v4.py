"""Torch-only PAINT transport contracts, selectively ported from teacher guide.

Source: paint-teacher-guide@3136606 (history/frame/reference logic).
The frozen backbone has the independent locomotion-v4 I/O contract.
"""
from pathlib import Path
import hashlib
import json
import math

import torch
from torch import nn

from .locomotion_v4 import SCHEMA as LOW_SCHEMA, LEG_JOINTS, contract, check_checkpoint, STAGES
from .paint_teacher_core import mlp, CheckpointScaledTanh

SCHEMA = "paint_transport_v4"
ARM_JOINTS = tuple(f"arm{i}_Joint" for i in range(1, 7))
WRENCH_SCALE = (40., 40., 10.)


class History:
    """Oldest first, once per control tick, isolated reset; optional first repeat."""
    def __init__(self, count, length, width, device, repeat_first=False):
        self.data = torch.zeros(count, length, width, device=device)
        self.tick = torch.full((count,), -1, device=device, dtype=torch.long)
        self.repeat_first = repeat_first

    def reset(self, ids):
        self.data[ids] = 0.
        self.tick[ids] = -1

    def append(self, row, tick):
        fresh = self.tick != tick
        ids = fresh.nonzero().flatten()
        first = self.tick < 0
        self.data[ids] = torch.cat((self.data[ids, 1:], row[ids, None]), 1)
        if self.repeat_first:
            self.data[first] = row[first, None]
        self.tick[fresh] = tick
        return self.data


def envelope(time, horizon):
    phase = time / horizon
    return torch.minimum(torch.minimum(phase / .1, (1 - phase) / .1), torch.ones_like(phase)).clamp(0., 1.)


def base_command(raw, limits):
    """Physical m/s, m/s, rad/s; no velocity scaling of the first three actions."""
    limits = torch.as_tensor(limits, device=raw.device, dtype=raw.dtype)
    cmd = raw[:, :3].clamp(-limits, limits).clone()
    cmd[:, :2] = torch.where(cmd[:, :2].norm(dim=-1, keepdim=True) <= .02, 0., cmd[:, :2])
    cmd[:, 2] = torch.where(cmd[:, 2].abs() <= .03, 0., cmd[:, 2])
    return cmd


def arm_targets(raw, previous, nominal, limits, scale, rate, dt):
    desired = nominal + scale * raw[:, 3:].clamp(-1., 1.)
    desired = desired.maximum(limits[..., 0]).minimum(limits[..., 1])
    return previous + (desired - previous).clamp(-rate * dt, rate * dt)


def tracking_scores(actual, reference):
    return (torch.exp(-(actual[:, :2] - reference[:, :2]).square().sum(-1) / .25),
            torch.exp(-(actual[:, 2] - reference[:, 2]).square() / .25))


def validate_low_manifest(manifest):
    expected = contract(117, "nominal", (.35, .2, .4), STAGES["nominal"])
    check_checkpoint({"v4_contract": manifest}, expected)
    if manifest.get("schema") != LOW_SCHEMA or manifest.get("arm_pd") != [60., 6.]:
        raise ValueError("Transport requires the v4 backbone with the declared arm PD 60/6.")


class FrozenBackbone(nn.Module):
    """Strict checkpoint or exported TorchScript; never a partial v3 load."""
    def __init__(self, path, device="cpu", file_format="checkpoint", manifest_path=None):
        super().__init__()
        path = Path(path).resolve()
        self.source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.file_format = file_format
        if file_format == "torchscript":
            metadata = Path(manifest_path) if manifest_path else path.with_name("contract.json")
            self.manifest = json.loads(metadata.read_text(encoding="utf-8"))
            validate_low_manifest(self.manifest)
            # Binary stream also handles Unicode paths with Windows libtorch.
            with path.open("rb") as stream:
                self.exported = torch.jit.load(stream, map_location=device)
        elif file_format == "checkpoint":
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.manifest = checkpoint.get("v4_contract", {})
            validate_low_manifest(self.manifest)
            policy_cfg = self.manifest["agent"]["policy"]
            encoder_cfg = self.manifest["agent"]["encoder"]
            if policy_cfg["activation"] != "elu" or encoder_cfg["activation"] != "elu":
                raise ValueError("This v4 adapter supports the declared ELU architecture only.")
            if not policy_cfg.get("squash_action") or policy_cfg.get("squash_action_scale") != 5.:
                raise ValueError("Unexpected v4 action-bound architecture.")
            self.actor = mlp([123, *policy_cfg["actor_hidden_dims"], 10])
            self.actor.add_module(str(len(self.actor)), CheckpointScaledTanh(5.))
            self.encoder = mlp([360, *encoder_cfg["hidden_dims"], 3])
            self.actor.load_state_dict({key.removeprefix("actor."): value
                for key, value in checkpoint["model_state_dict"].items() if key.startswith("actor.")}, strict=True)
            self.encoder.load_state_dict({key.removeprefix("encoder."): value
                for key, value in checkpoint["encoder_state_dict"].items()}, strict=True)
        else:
            raise ValueError("low format must be checkpoint or torchscript")
        self.requires_grad_(False)
        self.to(device).eval()
        if any(not torch.isfinite(value).all() for value in self.state_dict().values()):
            raise ValueError("Nonfinite frozen v4 weights.")

    @torch.no_grad()
    def forward(self, policy, history, command):
        if policy.shape[1:] != (117,) or history.shape[1:] != (10, 36) or command.shape[1:] != (3,):
            raise ValueError("Frozen v4 requires policy117 / history10x36 / command3.")
        if self.file_format == "torchscript":
            return self.exported(policy, history.flatten(1), command)
        return self.actor(torch.cat((self.encoder(history.flatten(1)), policy, command), -1)).clamp(-5., 5.)

    def fingerprint(self):
        digest = hashlib.sha256()
        for key, value in self.state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def validate_settings(max_wrench, limits, mass_range, arm_scale, disturbance_scale):
    values = [*max_wrench, *limits, *mass_range, arm_scale, disturbance_scale]
    if len(max_wrench) != 3 or len(limits) != 3 or len(mass_range) != 2 or not all(math.isfinite(v) for v in values):
        raise ValueError("Transport settings must be finite vectors of the declared dimensions.")
    if min(max_wrench) < 0 or min(limits) <= 0 or not 0 < mass_range[0] <= mass_range[1] or arm_scale <= 0 or disturbance_scale < 0:
        raise ValueError("Invalid transport settings.")
    if any(w / scale > limit + 1e-7 for w, scale, limit in zip(max_wrench, WRENCH_SCALE, limits)):
        raise ValueError("Wrench reference exceeds the frozen backbone velocity envelope.")


def check_teacher_contract(saved, current):
    for key in ("schema", "actor_mode", "low_sha256", "observation_shapes", "control_hz", "arm_pd"):
        if saved.get(key) != current[key]:
            raise ValueError(f"Teacher checkpoint contract mismatch: {key}")
    for key in ("arm_scale", "base_limits"):
        old = saved.get("arguments", {}).get(key)
        new = current["arguments"][key]
        if (tuple(old) if isinstance(old, (list, tuple)) else old) != (tuple(new) if isinstance(new, (list, tuple)) else new):
            raise ValueError(f"Teacher checkpoint action contract mismatch: {key}")
