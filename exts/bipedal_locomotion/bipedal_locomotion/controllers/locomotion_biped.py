"""Full-SFYG learning-to-walk recipe. No simulator imports or synthetic arm wrench."""
from pathlib import Path
import math
import re
import xml.etree.ElementTree as ET
import json

import torch

SCHEMA = "paint_locomotion_v4_biped_v1"
GAIT = {"frequency": .8, "offset": .5, "duration": .6, "swing_height": .08,
        "phase_start": .05, "startup_s": .25}
SKILLS = ("bootstrap", "omni", "transitions")
ARM_JOINTS = tuple(f"arm{i}_Joint" for i in range(1, 7))
POLICY_WIDTH = 135  # proprio36, terrain81, gait6, measured arm q/dq12


def validate_promotion(saved, skill, report_path, checkpoint_sha):
    old_skill = saved.get("training_objective", {}).get("skill_stage")
    if old_skill == skill:
        return
    if old_skill not in SKILLS or SKILLS.index(skill) != SKILLS.index(old_skill) + 1:
        raise ValueError("Promote one biped skill at a time: bootstrap -> omni -> transitions")
    if not report_path:
        raise ValueError("Skill promotion requires --promotion_report from the source checkpoint")
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if (report.get("checkpoint_sha256") != checkpoint_sha or report.get("interrupted")
            or report.get("acceptance_version") != "v4_biped_stepping_v1"
            or report.get("skill_stage") != old_skill):
        raise ValueError("Promotion report must be a complete biped evaluation of this checkpoint and skill")
    cases = ("walk", "forward") if old_skill == "bootstrap" else (
        "walk", "forward", "left", "right", "yaw_left", "yaw_right")
    for case in cases:
        rows = [row for row in report.get("episodes", []) if row.get("case") == case]
        if len(rows) < 10 or sum(bool(row.get("passed")) for row in rows) / len(rows) < .9:
            raise ValueError(f"Promotion requires >=10 trials and >=90% pass rate for {case}")


def command_roles(count, skill, device):
    """Fixed roles: bootstrap 5/95/0/0%, omni 10/80/5/5%, transitions 20/40/20/20%."""
    if skill not in SKILLS:
        raise ValueError(f"Unknown biped skill: {skill}")
    role = torch.arange(count, device=device) % 20
    thresholds = {"bootstrap": (1, 20, 20), "omni": (2, 18, 19), "transitions": (4, 12, 16)}[skill]
    return torch.where(role < thresholds[0], 0, torch.where(role < thresholds[1], 1,
                       torch.where(role < thresholds[2], 2, 3)))


def movement_targets(count, skill, limits, device):
    """Bounded nonzero commands; low speeds are introduced only after learning to step."""
    if skill not in SKILLS:
        raise ValueError(f"Unknown biped skill: {skill}")
    limits = torch.as_tensor(limits, device=device)
    if skill == "bootstrap":
        target = torch.zeros(count, 3, device=device)
        minimum = min(.2, float(limits[0]))
        target[:, 0] = minimum + torch.rand(count, device=device) * (limits[0] - minimum)
        # Learn both signs, with more samples devoted to forward transport.
        target[:, 0] *= torch.where(torch.rand(count, device=device) < .25, -1., 1.)
        return target
    minimum = torch.minimum(limits, limits.new_tensor((.15, .10, .20)))
    target = minimum + torch.rand(count, 3, device=device) * (limits - minimum)
    target *= torch.where(torch.rand_like(target) < .5, -1., 1.)
    axes = torch.nn.functional.one_hot(torch.randint(0, 3, (count,), device=device), 3).bool()
    target = torch.where((torch.rand(count, 1, device=device) < .5) | axes, target, 0.)
    if skill == "transitions":
        # A quarter of targets cover small but actionable commands, including pure yaw.
        slow = torch.minimum(limits, limits.new_tensor((.05, .05, .08)))
        small = (slow + torch.rand_like(target) * (torch.minimum(2 * slow, limits) - slow)) * target.sign()
        target = torch.where(torch.rand(count, 1, device=device) < .25, small, target)
    return target


def load_tracking_cost(force_z, total_mass, desired_contact):
    """Dense unloading signal, normalized by weight instead of a saturated 5 N kernel."""
    load = (force_z / (total_mass[:, None] * 9.81)).clamp(0., 2.)
    desired = desired_contact / desired_contact.sum(-1, keepdim=True).clamp_min(1e-6)
    return (load - desired).square().mean(-1)


def motion_shortfall(command, velocity, yaw_rate):
    """Standing costs one for an active axis; matching the command costs zero."""
    speed = command[:, :2].norm(dim=-1)
    turning = command[:, 2].abs()
    linear = speed > .02
    angular = turning > .03
    along = (velocity[:, :2] * command[:, :2]).sum(-1) / speed.clamp_min(.02)
    linear_cost = (1 - along / speed.clamp_min(.02)).clamp(0., 2.) * linear
    angular_cost = (1 - yaw_rate * command[:, 2].sign() / turning.clamp_min(.03)).clamp(0., 2.) * angular
    return (linear_cost + angular_cost) / (linear.float() + angular.float()).clamp_min(1.)


def velocity_score(command, velocity, tracking_std=.2, cross_std=.3):
    """Track along the command tightly, allowing some lateral balance motion."""
    speed = command[:, :2].norm(dim=-1, keepdim=True)
    direction = command[:, :2] / speed.clamp_min(.02)
    direction = torch.where(speed > .02, direction, direction.new_tensor([1., 0.]))
    error = velocity[:, :2] - command[:, :2]
    along = (error * direction).sum(-1)
    cross = error[:, 0] * direction[:, 1] - error[:, 1] * direction[:, 0]
    return torch.exp(-along.square() / tracking_std**2 - cross.square() / cross_std**2)


def sole_geometry(urdf):
    """Collision box corners and sphere centers/radii in each foot-link frame.

    Spheres keep their radius in world z after transforming their CENTER. This
    avoids confusing link height (~7 cm at contact) with actual sole clearance.
    The source hash and geometry are saved in the run contract for inspection.
    """
    root = ET.parse(Path(urdf)).getroot()
    result = []
    for side in ("L", "R"):
        link = root.find(f"link[@name='ankle_pitch_{side}_Link']")
        points, radii = [], []
        if link is None:
            raise ValueError(f"Missing {side} foot collision geometry")
        for collision in link.findall("collision"):
            origin = collision.find("origin")
            xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()] if origin is not None else [0., 0., 0.]
            rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()] if origin is not None else [0., 0., 0.]
            if any(abs(v) > 1e-8 for v in rpy):
                raise ValueError("Foot collision rotation needs explicit geometry support")
            box, sphere = collision.find("geometry/box"), collision.find("geometry/sphere")
            if box is not None:
                half = [float(x) / 2 for x in box.get("size").split()]
                for x in (-1, 1):
                    for y in (-1, 1):
                        for z in (-1, 1):
                            points.append([xyz[i] + sign * half[i] for i, sign in enumerate((x, y, z))])
                            radii.append(0.)
            elif sphere is not None:
                points.append(xyz)
                radii.append(float(sphere.get("radius")))
            else:
                raise ValueError("Expected authored SFYG box/sphere foot colliders; inspect changed asset")
        if not points:
            raise ValueError("Empty foot collision geometry")
        result.append({"points": points, "radii": radii})
    if len(result[0]["points"]) != len(result[1]["points"]):
        raise ValueError("Asymmetric foot collider counts need explicit padding")
    return result


def nominal_root_height(urdf, joint_positions, clearance=.01):
    """Forward kinematics of authored joints/colliders; place the sole near the floor.

    SFYG inherited root z=.85 while its straight-leg soles were ~.122 m above
    the floor. Remove that unintended drop without freezing any base DoF.
    """
    root = ET.parse(Path(urdf)).getroot()
    transforms = {"base_Link": torch.eye(4, dtype=torch.float64)}
    remaining = list(root.findall("joint"))
    def rotation(axis, angle):
        axis = torch.tensor(axis, dtype=torch.float64)
        axis /= axis.norm()
        x, y, z = axis
        skew = torch.tensor([[0., -z, y], [z, 0., -x], [-y, x, 0.]], dtype=torch.float64)
        return torch.eye(3, dtype=torch.float64) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)
    while remaining:
        pending = []
        for joint in remaining:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in transforms:
                pending.append(joint)
                continue
            origin = joint.find("origin")
            xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0., 0., 0.]
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0., 0., 0.]
            local = torch.eye(4, dtype=torch.float64)
            local[:3, 3] = torch.tensor(xyz)
            local[:3, :3] = rotation((0,0,1), rpy[2]) @ rotation((0,1,0), rpy[1]) @ rotation((1,0,0), rpy[0])
            q = next((float(value) for name, value in joint_positions.items() if re.fullmatch(name, joint.get("name"))), 0.)
            axis_tag = joint.find("axis")
            axis = [float(v) for v in axis_tag.get("xyz", "1 0 0").split()] if axis_tag is not None else [1., 0., 0.]
            motion = torch.eye(4, dtype=torch.float64)
            if joint.get("type") in ("revolute", "continuous"):
                motion[:3, :3] = rotation(axis, q)
            elif joint.get("type") == "prismatic":
                motion[:3, 3] = torch.tensor(axis) * q
            transforms[child] = transforms[parent] @ local @ motion
        if len(pending) == len(remaining):
            raise ValueError("URDF is not rooted at the expected base_Link")
        remaining = pending
    geometry = sole_geometry(urdf)
    minimum = float("inf")
    for name, foot in zip(("ankle_pitch_L_Link", "ankle_pitch_R_Link"), geometry):
        tf = transforms[name]
        for point, radius in zip(foot["points"], foot["radii"]):
            p = tf[:3, :3] @ torch.tensor(point, dtype=torch.float64) + tf[:3, 3]
            minimum = min(minimum, float(p[2]) - radius)
    return -minimum + clearance


class SteppingMetrics:
    """Actual contact transitions and clearance, independent of phase reward magnitude."""
    def __init__(self):
        self.previous = None
        self.air_start = [None, None]
        self.air_peak = [0., 0.]
        self.steps = [0, 0]
        self.last_landing = None
        self.alternating = 0
        self.moving_time = self.single_time = self.slip_integral = self.contact_time = 0.

    def update(self, time_s, command, force_z, clearance, foot_velocity, dt):
        contact = [bool(v > 5.) for v in force_z]
        moving = float(command[:2].norm()) > .02 or abs(float(command[2])) > .03
        if not moving:
            self.previous = contact
            self.air_start, self.air_peak = [None, None], [0., 0.]
            self.last_landing = None
            return
        self.moving_time += dt
        self.single_time += dt * (sum(contact) == 1)
        for i in range(2):
            if contact[i]:
                self.slip_integral += float(foot_velocity[i, :2].norm()) * dt
                self.contact_time += dt
            else:
                if self.air_start[i] is None:
                    self.air_start[i] = time_s - dt
                self.air_peak[i] = max(self.air_peak[i], float(clearance[i]))
            if self.previous is not None and contact[i] and not self.previous[i]:
                if (self.air_start[i] is not None and .08 <= time_s - self.air_start[i] <= 1.
                        and self.air_peak[i] >= .015):
                    self.steps[i] += 1
                    if self.last_landing is not None and self.last_landing != i:
                        self.alternating += 1
                    self.last_landing = i
                self.air_start[i], self.air_peak[i] = None, 0.
        self.previous = contact

    def result(self):
        fraction = self.single_time / self.moving_time if self.moving_time else None
        slip = self.slip_integral / self.contact_time if self.contact_time else None
        enough = (min(self.steps) >= 2 and self.alternating >= 3
                  and fraction is not None and .1 <= fraction <= .9
                  and slip is not None and slip < .1)
        # Quiet standing is a separate valid skill; no stepping requirement there.
        return {"stepping_passed": bool(not self.moving_time or enough),
                "valid_touchdowns_left": self.steps[0], "valid_touchdowns_right": self.steps[1],
                "alternating_touchdowns": self.alternating,
                "moving_single_support_fraction": fraction, "stance_foot_speed_mean_m_s": slip}
