"""Simulator-independent contracts for the PAINT replacement locomotion task."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

SCHEMA = "paint_locomotion_v4"
GAIT_SCHEMA = "paint_locomotion_v4_gait_v1"
GAIT_WIDTH = 6
DEFAULT_GAIT = {"frequency": .8, "offset": .5, "duration": .5, "swing_height": .20}
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


def gait_features(time_s, gait):
    """One episode clock for observations and rewards; continues during standing."""
    phase = torch.remainder(time_s * gait["frequency"], 1.)
    params = time_s.new_tensor([gait[key] for key in
                               ("frequency", "offset", "duration", "swing_height")])
    return torch.cat((torch.sin(2 * math.pi * phase)[:, None],
                      torch.cos(2 * math.pi * phase)[:, None],
                      params.expand(time_s.shape[0], -1)), -1)


def gait_targets(time_s, gait):
    """Legacy smoothed contacts, with per-foot cubic swing height/velocity.

    At duty=.5 this restores the old alternating .8 Hz gait. Local swing time
    also keeps height/velocity aligned if duty is changed in a later experiment.
    """
    frequency, duty = gait["frequency"], gait["duration"]
    if not frequency > 0 or not 0 < duty < 1 or not 0 <= gait["offset"] <= 1:
        raise ValueError("Invalid gait frequency, duty or offset")
    phase = torch.remainder(time_s[:, None] * frequency
                            + time_s.new_tensor([0., gait["offset"]]), 1.)
    warped = torch.where(phase < duty, phase * (.5 / duty),
                          .5 + (phase - duty) * (.5 / (1 - duty)))
    cdf = torch.distributions.Normal(0., .05).cdf
    contact = cdf(warped) * (1 - cdf(warped - .5)) + cdf(warped - 1) * (1 - cdf(warped - 1.5))
    swing = ((phase - duty) / (1 - duty)).clamp(0., 1.)
    u = torch.where(swing < .5, 2 * swing, 2 * (1 - swing))
    height = gait["swing_height"] * u.square() * (3 - 2 * u)
    velocity = (gait["swing_height"] * 6 * u * (1 - u)
                * (2 * frequency / (1 - duty)) * torch.where(swing < .5, 1., -1.))
    return contact, height, velocity


def gait_score(forces, velocities, heights, targets):
    """Original force/velocity/height scales (1, 1, -.3), before reward weight."""
    contact, height, velocity = targets
    force = ((1 - contact) * torch.exp(-forces.square() / 25.)).mean(-1)
    speed = (contact * torch.exp(-velocities.square().sum(-1) / .25)
             + (1 - contact) * torch.exp(-(velocities[..., 2] - velocity).square() / .25)).mean(-1)
    clearance = (contact * (1 - torch.exp(-heights.square() / .005))
                 + (1 - contact) * (1 - torch.exp(-(heights - height).square() / .005))).mean(-1)
    return force + speed - .3 * clearance


def contract(policy_width, stage, limits, settings, gait=None):
    gait_width = GAIT_WIDTH if gait is not None else 0
    result = {"schema": GAIT_SCHEMA if gait is not None else SCHEMA,
            "stage": stage, "settings": asdict(settings),
            "leg_joints": list(LEG_JOINTS), "proprio_width": PROPRIO_WIDTH,
            "history_length": HISTORY_LENGTH, "history_width": HISTORY_LENGTH * PROPRIO_WIDTH,
            "policy_width": policy_width, "height_width": policy_width - PROPRIO_WIDTH - gait_width,
            "command_width": 3, "action_width": 10, "velocity_limits": list(limits),
            "control_hz": 50, "physics_hz": 200, "action_scale_rad": .25,
            "action_clip": 5., "arm_pd": [60., 6.],
            "history_order": "oldest_first", "history_sample_hz": 50,
            "proprio_order": ["angular_velocity*.25", "projected_gravity", "leg_q-default_q",
                              "leg_dq*.05", "previous_executed_leg_action"],
            "height_observation": "base_z - terrain_z - 0.5, clipped [-1,1]",
            "wrench_frame": "world", "wrench_application": "payload_COM_each_physics_step",
            "future_wrench_input": False}
    if gait is not None:
        result.update({"gait": dict(gait), "gait_width": GAIT_WIDTH,
            "gait_order": ["sin_phase", "cos_phase", "frequency", "offset", "duration", "swing_height"],
            "gait_clock": "episode_time_continuous_in_standing",
            "policy_order": ["proprio36", "heights81", "gait6"]})
    return result


def check_checkpoint(checkpoint, expected):
    saved = checkpoint.get("v4_contract", {})
    # A stage may change physical randomization, never the policy I/O contract.
    for key in ("schema", "leg_joints", "policy_width", "history_width", "command_width",
                "action_width", "control_hz", "action_scale_rad", "action_clip",
                "proprio_order", "height_observation", "history_sample_hz"):
        if saved.get(key) != expected[key]:
            raise ValueError(f"Incompatible v4 checkpoint field: {key}")
    if expected["schema"] == GAIT_SCHEMA:
        for key in ("gait", "gait_width", "gait_order", "gait_clock", "policy_order"):
            if saved.get(key) != expected[key]:
                raise ValueError(f"Incompatible v4 gait checkpoint field: {key}")


class MotionProgress:
    """Reject standing under moving commands, including the .05 m/s slow case.

    Measure signed progress in the commanded body direction after a 1 s
    settling window per moving segment. Yaw uses the commanded turn direction.
    """
    def __init__(self):
        self.moving_since = None
        self.expected_linear = self.actual_linear = 0.
        self.expected_yaw = self.actual_yaw = 0.
        self.linear_requested = self.yaw_requested = False
        self.actual_motion_seen = False
        self.quiet_since = None
        self.stop_latency = None

    def update(self, time_s, command, velocity, yaw_rate, dt):
        speed = float(command[:2].norm())
        turn = abs(float(command[2]))
        moving = speed > .02 or turn > .03
        if not moving:
            self.moving_since = None
            if self.quiet_since is None:
                self.quiet_since = time_s - dt
            if (self.actual_motion_seen and self.stop_latency is None
                    and float(velocity[:2].norm()) < .05 and abs(float(yaw_rate)) < .05):
                self.stop_latency = time_s - self.quiet_since
            return
        self.quiet_since = None
        self.linear_requested |= speed > .02
        self.yaw_requested |= turn > .03
        if self.moving_since is None:
            self.moving_since = time_s - dt
        if float(velocity[:2].norm()) > .025 or abs(float(yaw_rate)) > .05:
            self.actual_motion_seen = True
        if time_s - self.moving_since < 1.:
            return
        if speed > .02:
            self.expected_linear += speed * dt
            self.actual_linear += float((velocity[:2] * command[:2]).sum()) / speed * dt
        if turn > .03:
            self.expected_yaw += turn * dt
            self.actual_yaw += math.copysign(1., float(command[2])) * float(yaw_rate) * dt

    def result(self, require_stop=False):
        linear = self.actual_linear / self.expected_linear if self.expected_linear > .02 else None
        yaw = self.actual_yaw / self.expected_yaw if self.expected_yaw > .02 else None
        passed = ((not self.linear_requested or (linear is not None and .7 <= linear <= 1.3))
                  and (not self.yaw_requested or (yaw is not None and .7 <= yaw <= 1.3)))
        if require_stop:
            passed &= self.stop_latency is not None and self.stop_latency < 2.
        return {"motion_passed": bool(passed), "linear_progress_ratio": linear,
                "yaw_progress_ratio": yaw, "expected_linear_distance_m": self.expected_linear,
                "actual_linear_progress_m": self.actual_linear,
                "expected_yaw_travel_rad": self.expected_yaw,
                "actual_yaw_progress_rad": self.actual_yaw,
                "observed_stop_latency_s": self.stop_latency}


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
