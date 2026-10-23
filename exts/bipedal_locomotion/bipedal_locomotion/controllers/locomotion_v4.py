"""Simulator-independent contracts for the PAINT replacement locomotion task."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

SCHEMA = "paint_locomotion_v4"
LEG_JOINTS = tuple(f"{part}_{side}_Joint" for side in ("L", "R")
                   for part in ("proximal_pitch", "proximal_roll", "proximal_yaw", "knee", "ankle_pitch"))
PROPRIO_WIDTH = 36  # omega(3), gravity(3), leg q/dq(20), previous leg action(10)
HISTORY_LENGTH = 10


@dataclass(frozen=True)
class Stage:
    payload_mass: tuple
    force_n: float
    yaw_torque_nm: float
    base_mass_scale: tuple
    friction: tuple
    push_velocity: float
    noise: bool
    terrain: str = "flat"


STAGES = {
    "nominal": Stage((.5, .5), 0., 0., (1., 1.), (1., 1.), 0., False),
    "robust": Stage((.25, 1.), 5., .5, (.9, 1.1), (.6, 1.2), .05, True),
    "interaction": Stage((.1, 1.5), 15., 3., (.9, 1.1), (.6, 1.2), .1, True),
    "rough": Stage((.1, 1.5), 15., 3., (.9, 1.1), (.6, 1.2), .1, True, "rough"),
}


def validate_ranges(mass_range, force, torque, velocity_limits):
    values = [*mass_range, force, torque, *velocity_limits]
    if len(mass_range) != 2 or len(velocity_limits) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError("Expected finite mass pair, force, torque and three velocity limits.")
    if not 0 < mass_range[0] <= mass_range[1] or force < 0 or torque < 0 or min(velocity_limits) <= 0:
        raise ValueError("Mass/velocity must be positive and force/torque nonnegative.")


def sample_planar_wrench(count, force, torque, device):
    """World-horizontal disk, bounding FORCE NORM rather than each component."""
    angle = torch.rand(count, device=device) * (2 * math.pi)
    radius = torch.rand(count, device=device).sqrt() * force
    return torch.stack((radius * angle.cos(), radius * angle.sin(),
                        (2 * torch.rand(count, device=device) - 1) * torque), dim=-1)


def contract(policy_width, stage, limits, settings):
    return {"schema": SCHEMA, "stage": stage, "settings": asdict(settings),
            "leg_joints": list(LEG_JOINTS), "proprio_width": PROPRIO_WIDTH,
            "history_length": HISTORY_LENGTH, "history_width": HISTORY_LENGTH * PROPRIO_WIDTH,
            "policy_width": policy_width, "height_width": policy_width - PROPRIO_WIDTH,
            "command_width": 3, "action_width": 10, "velocity_limits": list(limits),
            "control_hz": 50, "physics_hz": 200, "action_scale_rad": .25,
            "action_clip": 5., "arm_pd": [60., 6.],
            "history_order": "oldest_first", "history_sample_hz": 50,
            "proprio_order": ["angular_velocity*.25", "projected_gravity", "leg_q-default_q",
                              "leg_dq*.05", "previous_executed_leg_action"],
            "height_observation": "base_z - terrain_z - 0.5, clipped [-1,1]",
            "wrench_frame": "world", "wrench_application": "payload_COM_each_physics_step",
            "future_wrench_input": False}


def check_checkpoint(checkpoint, expected):
    saved = checkpoint.get("v4_contract", {})
    # A stage may change physical randomization, never the policy I/O contract.
    for key in ("schema", "leg_joints", "policy_width", "history_width", "command_width",
                "action_width", "control_hz", "action_scale_rad", "action_clip",
                "proprio_order", "height_observation", "history_sample_hz"):
        if saved.get(key) != expected[key]:
            raise ValueError(f"Incompatible v4 checkpoint field: {key}")


class ExportedPolicy(nn.Module):
    """Stateless TorchScript: caller maintains per-env history and last actions."""
    def __init__(self, actor, encoder, policy_width):
        super().__init__()
        self.actor = deepcopy(actor).cpu().eval()
        self.encoder = deepcopy(encoder).cpu().eval()
        self.policy_width = int(policy_width)

    def forward(self, policy: torch.Tensor, history: torch.Tensor, command: torch.Tensor):
        torch._assert(policy.dim() == 2 and policy.size(1) == self.policy_width, "policy shape")
        torch._assert(history.dim() == 2 and history.size(1) == 360, "flattened history shape")
        torch._assert(command.dim() == 2 and command.size(1) == 3, "command shape")
        return self.actor(torch.cat((self.encoder(history), policy, command), dim=-1)).clamp(-5., 5.)
