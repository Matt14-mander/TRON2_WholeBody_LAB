"""Scripted inverse-dynamics preview, without base acceleration or an MPC solver.

Sum Newton-Euler balances for bodies added to the SF base. Positive wrench is
the added subsystem acting ON the base. All moments are about base_Link origin.
URDF link inertias include their inertial-frame rotations. No simulator imports.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

TIMES = np.array([0., .2, .4, .6, .8])


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*k + (1-np.cos(angle))*(k@k)


def origin(element):
    out = np.eye(4)
    if element is not None:
        out[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        r, p, y = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        out[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
    return out


def vee(skew):
    return np.array([skew[2, 1]-skew[1, 2], skew[0, 2]-skew[2, 0], skew[1, 0]-skew[0, 1]])*.5


def body_reaction(mass, com, gravity, acceleration, inertia, omega, alpha):
    force = mass * (gravity-acceleration)
    return np.r_[force, np.cross(com, force)-inertia@alpha-np.cross(omega, inertia@omega)]


class AddedBodyModel:
    def __init__(self, urdf, reference_urdf, payload_mass=0., dimensions=(.2, .12, .08), offset=(.15, 0, 0)):
        self.source = str(Path(urdf).resolve())
        root = ET.parse(urdf).getroot()
        reference = {l.get("name") for l in ET.parse(reference_urdf).getroot().findall("link")}
        self.bodies = {}
        for link in root.findall("link"):
            inertial = link.find("inertial")
            if link.get("name") in reference or inertial is None:
                continue
            mass = float(inertial.find("mass").get("value"))
            i = inertial.find("inertia")
            inertia = np.array([[float(i.get(f"i{a}{b}" if a <= b else f"i{b}{a}"))
                                 for b in "xyz"] for a in "xyz"])
            self.bodies[link.get("name")] = (mass, origin(inertial.find("origin")), inertia)
        self.joints = []
        for j in root.findall("joint"):
            axis = j.find("axis")
            self.joints.append((j.get("name"), j.get("type"), j.find("parent").get("link"),
                j.find("child").get("link"), origin(j.find("origin")),
                np.fromstring(axis.get("xyz") if axis is not None else "1 0 0", sep=" ")))
        if payload_mass > 0:
            pose = np.eye(4); pose[:3, 3] = offset
            x, y, z = dimensions
            self.bodies["paint_payload_Link"] = (payload_mass, np.eye(4),
                np.diag([y*y+z*z, x*x+z*z, x*x+y*y])*payload_mass/12)
            self.joints.append(("paint_payload_fixed", "fixed", "gripper_base_Link", "paint_payload_Link", pose, np.array([1., 0, 0])))

    def states(self, positions):
        transforms = {"base_Link": np.eye(4)}
        pending = list(self.joints)
        while pending:
            remaining = []
            for name, kind, parent, child, joint_pose, axis in pending:
                if parent not in transforms:
                    remaining.append((name, kind, parent, child, joint_pose, axis)); continue
                motion = np.eye(4)
                value = positions.get(name, 0.)
                if kind in ("revolute", "continuous"):
                    motion[:3, :3] = rotation(axis, value)
                elif kind == "prismatic":
                    motion[:3, 3] = axis * value
                elif kind != "fixed":
                    raise ValueError(f"Unsupported joint type: {kind}")
                transforms[child] = transforms[parent] @ joint_pose @ motion
            if len(remaining) == len(pending):
                raise ValueError("URDF is not rooted at base_Link")
            pending = remaining
        return {name: transforms[name] @ body[1] for name, body in self.bodies.items()}

    def gravity_wrench(self, positions, gravity):
        states = self.states(positions)
        return sum((body_reaction(m, states[name][:3, 3], gravity, np.zeros(3),
                    np.zeros((3, 3)), np.zeros(3), np.zeros(3))
                    for name, (m, _, _) in self.bodies.items()), np.zeros(6))

    def preview(self, position_at, time_s, gravity, h=.002):
        result = []
        for offset in TIMES:
            t = time_s + offset
            minus, now, plus = [self.states(position_at(t+d)) for d in (-h, 0., h)]
            wrench = np.zeros(6)
            for name, (mass, _, inertia_local) in self.bodies.items():
                a, b, c = minus[name], now[name], plus[name]
                rotation_now = b[:3, :3]
                rd = (c[:3, :3]-a[:3, :3])/(2*h)
                rdd = (c[:3, :3]-2*rotation_now+a[:3, :3])/(h*h)
                acceleration = (c[:3, 3]-2*b[:3, 3]+a[:3, 3])/(h*h)
                omega, alpha = vee(rd @ rotation_now.T), vee(rdd @ rotation_now.T)
                inertia = rotation_now @ inertia_local @ rotation_now.T
                wrench += body_reaction(mass, b[:3, 3], gravity, acceleration, inertia, omega, alpha)
            result.append(wrench)
        result = np.asarray(result)
        if not np.isfinite(result).all():
            raise ValueError("Non-finite inverse-dynamics preview")
        return result
