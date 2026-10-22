"""Scripted inverse-dynamics preview, without base acceleration or an MPC solver.

Sum Newton-Euler balances for bodies added to the SF base. Positive wrench is
the added subsystem acting ON the base. All moments are about base_Link origin.
URDF link inertias include their inertial-frame rotations. No simulator imports.
"""
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET
import numpy as np

TIMES = np.array([0., .2, .4, .6, .8])


class LatestPreviewWorker:
    """One worker owns the private model; one pending request, no unbounded queue."""
    def __init__(self, compute, seed, seed_request):
        self.compute = compute
        self.condition = threading.Condition()
        self.pending = None
        self.result = (self._checked(seed), seed_request, 0.)
        self.error = None
        self.stopping = False
        self.thread = threading.Thread(target=self._run, name="wrench-preview", daemon=True)
        self.thread.start()

    @staticmethod
    def _checked(value):
        value = np.asarray(value, dtype=float)
        if value.shape != (5, 6) or not np.isfinite(value).all():
            raise ValueError("Preview worker requires a finite 5x6 wrench")
        return value.copy()

    def submit(self, request):
        with self.condition:
            if self.stopping:
                raise RuntimeError("Preview worker has stopped")
            self.pending = request
            self.condition.notify()

    def read(self, now, max_age):
        with self.condition:
            if self.error is not None:
                raise RuntimeError(f"Preview worker failed: {self.error}")
            result = self.result
        age = now - result[1]["stamp"]
        if age < 0 or age > max_age:
            raise RuntimeError(f"Stale wrench preview: age={age:.3f}s limit={max_age:.3f}s")
        return result, age

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopping or self.pending is not None)
                if self.stopping:
                    return
                request, self.pending = self.pending, None
            started = time.monotonic()
            try:
                prediction = self._checked(self.compute(request))
            except Exception as error:
                with self.condition:
                    self.error = f"{type(error).__name__}: {error}"
                    self.condition.notify_all()
                return
            with self.condition:
                self.result = (prediction, request, (time.monotonic()-started)*1000.)
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify()
        self.thread.join(timeout=.2)


def current_frame_preview(prediction, source_quaternion, current_quaternion, age):
    """Shift the horizon by its age; rotate vectors into current base axes.

    The model neglects base acceleration and expresses moments about a co-moving
    base origin. The last predicted sample is held when the shifted tail ends.
    """
    if not np.isfinite(age) or age < 0:
        raise ValueError("Invalid prediction age")
    def matrix(q):
        q = np.asarray(q, dtype=float)
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
            raise ValueError("Invalid preview frame quaternion")
        w, x, y, z = q / np.linalg.norm(q)
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    prediction = LatestPreviewWorker._checked(prediction)
    shifted = np.stack([np.interp(TIMES+age, TIMES, prediction[:, i]) for i in range(6)], axis=1)
    transform = matrix(source_quaternion).T @ matrix(current_quaternion)
    return np.c_[shifted[:, :3] @ transform, shifted[:, 3:] @ transform]


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
    return np.stack([skew[..., 2, 1]-skew[..., 1, 2], skew[..., 0, 2]-skew[..., 2, 0],
                     skew[..., 1, 0]-skew[..., 0, 1]], axis=-1)*.5


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
        if not self.bodies:
            return np.zeros(6)
        force = np.array([b[0] for b in self.bodies.values()])[:, None] * np.asarray(gravity)
        com = np.array([states[name][:3, 3] for name in self.bodies])
        return np.r_[force.sum(axis=0), np.cross(com, force).sum(axis=0)]

    def preview(self, position_at, time_s, gravity, h=.002):
        positions = [[dict(position_at(time_s+offset+d)) for d in (-h, 0., h)] for offset in TIMES]
        first = positions[0][0]
        if all(q == first for triple in positions for q in triple):
            return np.repeat(self.gravity_wrench(first, gravity)[None], len(TIMES), axis=0)
        if not self.bodies:
            return np.zeros((len(TIMES), 6))
        masses = np.array([b[0] for b in self.bodies.values()])[:, None]
        inertias = np.array([b[2] for b in self.bodies.values()])
        result = []
        for triple in positions:
            state = [self.states(q) for q in triple]
            a, b, c = [np.array([s[name] for name in self.bodies]) for s in state]
            r = b[:, :3, :3]
            rt = r.swapaxes(-1, -2)
            rd = (c[:, :3, :3]-a[:, :3, :3])/(2*h)
            rdd = (c[:, :3, :3]-2*r+a[:, :3, :3])/(h*h)
            acceleration = (c[:, :3, 3]-2*b[:, :3, 3]+a[:, :3, 3])/(h*h)
            omega, alpha = vee(rd @ rt), vee(rdd @ rt)
            inertia = r @ inertias @ rt
            force = masses * (gravity-acceleration)
            moment = (np.cross(b[:, :3, 3], force)-np.einsum('nij,nj->ni', inertia, alpha)
                      -np.cross(omega, np.einsum('nij,nj->ni', inertia, omega)))
            result.append(np.r_[force.sum(axis=0), moment.sum(axis=0)])
        result = np.asarray(result)
        if not np.isfinite(result).all():
            raise ValueError("Non-finite inverse-dynamics preview")
        return result


class MuJoCoAddedBodyModel(AddedBodyModel):
    """Use compiled MuJoCo masses, inertias and FK for the same preview contract.

    This private model/data pair never advances or modifies the live simulator.
    An inertial-frame pose at each COM makes diagonal MuJoCo inertia compatible
    with the Newton-Euler summation in AddedBodyModel.preview.
    """
    def __init__(self, model, reference_urdf):
        import mujoco
        self.mj = mujoco
        self.model, self.data = model, mujoco.MjData(model)
        reference = {l.get("name") for l in ET.parse(reference_urdf).getroot().findall("link")}
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
        if self.base_id < 0:
            raise ValueError("MuJoCo asset has no base_Link")
        self.ids, self.bodies = {}, {}
        for i in range(1, model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name in reference or model.body_mass[i] == 0:
                continue
            if name is None:
                raise ValueError("Unnamed massive MuJoCo body cannot be classified")
            self.ids[name] = i
            self.bodies[name] = (float(model.body_mass[i]), np.eye(4), np.diag(model.body_inertia[i]))
        self.addresses = {}
        for i in range(model.njnt):
            # Infer scalar joints from native coordinate spans. Comparing numpy
            # jnt_type scalars against pybind enum objects is version-dependent.
            start = int(model.jnt_qposadr[i])
            end = int(model.jnt_qposadr[i+1]) if i+1 < model.njnt else model.nq
            dof_start = int(model.jnt_dofadr[i])
            dof_end = int(model.jnt_dofadr[i+1]) if i+1 < model.njnt else model.nv
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name is not None and end-start == 1 and dof_end-dof_start == 1:
                self.addresses[name] = start
        if not self.addresses:
            raise ValueError(f"MuJoCo preview has no named scalar joints: njnt={model.njnt}, "
                             f"nq={model.nq}, nv={model.nv}, types={model.jnt_type.tolist()}")

    def resolve_joint_names(self, names):
        """Resolve exact names first, then the explicit SDK '_Joint' suffix alias."""
        result = {}
        for name in names:
            if name in self.addresses:
                result[name] = name
                continue
            canonical = name.removesuffix("_Joint")
            matches = [actual for actual in self.addresses if actual and actual.removesuffix("_Joint") == canonical]
            if len(matches) != 1:
                raise ValueError(f"MuJoCo preview joint not found or ambiguous: {name}; "
                                 f"available={list(self.addresses)}")
            result[name] = matches[0]
        if len(set(result.values())) != len(result):
            raise ValueError("Two supplied joint names resolve to the same MuJoCo joint")
        return result

    def states(self, positions):
        resolved = self.resolve_joint_names(positions)
        self.mj.mj_resetData(self.model, self.data)
        for name, value in positions.items():
            self.data.qpos[self.addresses[resolved[name]]] = value
        # Only poses/COM/inertial-frame rotations are needed. Full mj_forward
        # also runs dynamics, collision and constraints in this private model.
        self.mj.mj_kinematics(self.model, self.data)
        r = self.data.xmat[self.base_id].reshape(3, 3).T
        p = self.data.xpos[self.base_id]
        out = {}
        for name, i in self.ids.items():
            pose = np.eye(4)
            pose[:3, :3] = r @ self.data.ximat[i].reshape(3, 3)
            pose[:3, 3] = r @ (self.data.xipos[i]-p)
            out[name] = pose
        return out
