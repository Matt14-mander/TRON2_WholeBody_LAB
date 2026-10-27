"""Name-based single SFYG MuJoCo bridge. No training or Isaac imports."""
from collections import deque
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sfyg_asset_reference import ROOT, load_reference
from sfyg_policy import ObservationHistory, make_observation, sha256

XML = ROOT / "robot_description/tron2/SFYG_TRON2A/xml/robot.xml"


def numbers(value):
    return " ".join(f"{float(v):.12g}" for v in np.asarray(value).ravel())


def rotation(quaternion):
    matrix = np.empty(9)
    quaternion = np.asarray(quaternion, float)
    quaternion /= np.linalg.norm(quaternion)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def named_id(model, kind, name):
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise ValueError(f"Missing model object: {name}")
    return result


def build_model(cfg, mass, friction):
    reference = load_reference()
    if cfg["usd_sources"] != reference["sources"]:
        raise ValueError("Export was created against a different USD asset")
    root = ET.fromstring(XML.read_bytes())
    root.find("compiler").set("meshdir", ".")
    root.find("compiler").set("autolimits", "true")
    root.find("compiler").set("fusestatic", "false")
    root.find("option").attrib.update(timestep=str(cfg["physics_dt"]), integrator="implicitfast",
                                      cone="elliptic", iterations="100", gravity="0 0 -9.81")
    if root.find("visual") is None:
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", offwidth="960", offheight="720")
    bodies = {b.get("name"): b for b in root.iter("body")}
    # Reconstruct every zero-position parent/child transform and joint axis from
    # the authored USD joint frames, rather than relying on an old MJCF pose.
    for name, joint in reference["joints"].items():
        parent, child = joint["parent"], joint["child"]
        if child not in bodies:
            bodies[child] = ET.SubElement(bodies[parent], "body", name=child)
        body = bodies[child]
        if body not in list(bodies[parent]):
            raise ValueError(f"USD/MJCF parent mismatch for {child}")
        r0, r1 = rotation(joint["rot0"]), rotation(joint["rot1"])
        r = r0 @ r1.T
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, r.ravel())
        body.set("pos", numbers(np.asarray(joint["pos0"]) - r @ np.asarray(joint["pos1"])))
        for key in ("euler", "axisangle", "xyaxes", "zaxis"):
            body.attrib.pop(key, None)
        body.set("quat", numbers(q))
        if joint["type"] != "PhysicsFixedJoint":
            node = body.find(f"joint[@name='{name}']")
            if node is None:
                raise ValueError(f"Missing moving joint {name}")
            axis = np.eye(3)["XYZ".index(joint["axis"])]
            node.set("axis", numbers(r1 @ axis))
            node.set("pos", numbers(joint["pos1"]))
            limits = np.array([joint["lower"], joint["upper"]])
            if joint["type"] == "PhysicsRevoluteJoint":
                limits = np.deg2rad(limits)
            node.set("range", numbers(limits))
    for name, source in reference["bodies"].items():
        body = bodies[name]
        inertial = body.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(body, "inertial")
        inertial.attrib.clear()
        inertial.attrib.update(mass=str(source["mass"]), pos=numbers(source["com"]),
                              diaginertia=numbers(source["inertia"]), quat=numbers(source["principal_axes"]))
    bodies["base_Link"].set("pos", numbers(cfg["root_pos"]))
    bodies["base_Link"].set("quat", numbers(cfg["root_quat"]))
    free = bodies["base_Link"].find("joint[@type='free']")
    free.attrib.clear()
    free.attrib.update(type="free", name="root", damping="0", stiffness="0", armature="0", frictionloss="0")
    for geom in root.iter("geom"):
        if geom.get("class") != "visual":
            # No invented foot torsional/rolling friction. PhysX and MuJoCo
            # contact solvers still differ; this is a documented mapping.
            geom.set("condim", "3")
            geom.set("friction", numbers([friction, 0, 0]))
            geom.set("priority", "1")
            geom.set("margin", "0")
    contacts = root.find("contact")
    if contacts is None:
        contacts = ET.SubElement(root, "contact")
    if mass > 0:
        payload = ET.SubElement(bodies["gripper_base_Link"], "body", name="paint_payload_Link", pos=".15 0 0")
        inertia = mass / 12 * np.array([.12**2 + .08**2, .20**2 + .08**2, .20**2 + .12**2])
        ET.SubElement(payload, "inertial", mass=str(mass), pos="0 0 0", diaginertia=numbers(inertia))
        ET.SubElement(payload, "geom", name="payload_box", type="box", size=".10 .06 .04", density="0",
                      rgba=".15 .55 .85 1", friction=numbers([friction, 0, 0]), condim="3")
        for name in ("gripper_base_Link", "gripper1_Link", "gripper2_Link"):
            ET.SubElement(contacts, "exclude", body1="paint_payload_Link", body2=name)
    actuators = root.find("actuator")
    actuators.clear()
    for name, params in zip(cfg["names"], cfg["actuators"]):
        joint = root.find(f".//joint[@name='{name}']")
        joint.set("armature", str(params["armature"]))
        joint.set("damping", "0")
        joint.set("frictionloss", str(params["friction"]))
        joint.set("actuatorfrcrange", numbers([-params["effort"], params["effort"]]))
        joint.set("actuatorfrclimited", "true")
        # Equivalent kp*(target-q)-kd*dq, integrated as an affine actuator.
        ET.SubElement(actuators, "general", name="pd/" + name, joint=name, dyntype="none", gaintype="fixed",
                      biastype="affine", gainprm=str(params["kp"]),
                      biasprm=numbers([0, -params["kp"], -params["kd"]]),
                      forcelimited="true", forcerange=numbers([-params["effort"], params["effort"]]),
                      ctrllimited="false", gear="1")
    assets = {}
    meshes = XML.parent.parent / "meshes"
    for mesh in root.findall("./asset/mesh"):
        path = meshes / mesh.get("file")
        mesh.set("file", path.name)
        assets[path.name] = path.read_bytes()
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets)
    return model, reference, sha256(XML)


class SFYGBridge:
    def __init__(self, cfg, mass=None, friction=None, delay_steps=None, seed=42):
        self.cfg = cfg
        self.mass = cfg["payload_mass"] if mass is None else mass
        self.friction = cfg["friction_coefficient"] if friction is None else friction
        if (not np.isfinite([self.mass, self.friction]).all() or self.mass < 0 or self.friction <= 0
                or (cfg["phase"] != "source" and self.mass <= 0)
                or (cfg["phase"] == "source" and self.mass != 0)):
            raise ValueError("Invalid mass/friction for the exported phase")
        if delay_steps is not None and delay_steps not in range(5):
            raise ValueError("Delay must be 0--4 training physics steps")
        self.model, self.reference, self.xml_hash = build_model(cfg, self.mass, self.friction)
        self.data = mujoco.MjData(self.model)
        for value in (self.model.body_mass, self.model.body_ipos, self.model.body_inertia,
                      self.model.body_iquat, self.model.jnt_axis):
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite physical model parameters")
        self.base = named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
        self.payload = (named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "paint_payload_Link") if self.mass > 0 else None)
        self.joints = np.array([named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg["names"]])
        self.qpos = self.model.jnt_qposadr[self.joints]
        self.qvel = self.model.jnt_dofadr[self.joints]
        self.motors = np.array([named_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pd/" + n) for n in cfg["names"]])
        self.defaults = np.asarray(cfg["defaults"], float)
        self.history = ObservationHistory()
        rng = np.random.default_rng(seed)
        # The two delayed actuator groups sample independently, as in Isaac.
        sampled = {}
        self.delays = []
        for a in cfg["actuators"]:
            key = (a["kp"], a["min_delay"], a["max_delay"])
            if key not in sampled:
                sampled[key] = (0 if a["max_delay"] == 0 else delay_steps if delay_steps is not None
                                else int(rng.integers(a["min_delay"], a["max_delay"] + 1)))
            self.delays.append(sampled[key])
        self.reset()
        expected = self.reference["total_mass_kg"] + self.mass
        if not np.isclose(self.model.body_mass.sum(), expected, atol=1e-5):
            raise ValueError("Model mass does not match authored USD plus payload")

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos] = self.defaults
        self.data.ctrl[self.motors] = self.defaults
        self.previous_action = np.zeros(10, np.float32)
        self.target = self.defaults.copy()
        self.delay_queues = [deque(maxlen=d + 1) for d in self.delays]
        self.control_steps = 0
        self.history.reset()
        mujoco.mj_forward(self.model, self.data)

    def base_state(self):
        r = self.data.xmat[self.base].reshape(3, 3)
        world = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base, world, 0)
        # Isaac root_lin_vel_b is CoM velocity expressed in the regular link axes.
        return r.T @ world[3:], r.T @ world[:3], r.T @ [0., 0., -1.]

    def observation(self):
        linear, angular, gravity = self.base_state()
        obs = make_observation(angular, gravity, self.data.qpos[self.qpos[:10]],
            self.data.qvel[self.qvel[:10]], self.previous_action, self.control_steps, self.cfg)
        return obs, self.history.push(obs)

    def set_action(self, action):
        action = np.asarray(action, np.float32)
        if action.shape != (10,) or not np.isfinite(action).all():
            raise ValueError("Expected finite ten leg actions")
        self.previous_action = action.copy()  # The source has no action clamp.
        self.target = self.defaults.copy()
        self.target[:10] += .25 * action

    def physics_step(self, force_world=None):
        for index, queue in enumerate(self.delay_queues):
            if not queue:
                queue.extend([self.target[index]] * queue.maxlen)
            else:
                queue.append(self.target[index])
            self.data.ctrl[self.motors[index]] = queue[0]
        self.data.xfrc_applied[:] = 0
        if force_world is not None:
            if self.payload is None:
                raise ValueError("Payload force requires a carry/robust asset")
            force = np.asarray(force_world, float)
            if force.shape != (3,) or not np.isfinite(force).all() or force[2] != 0:
                raise ValueError("Only finite world horizontal payload force is supported")
            self.data.xfrc_applied[self.payload, :3] = force
        mujoco.mj_step(self.model, self.data)

    def audit(self):
        mass_errors, com_errors, inertia_errors = [], [], []
        for name, source in self.reference["bodies"].items():
            body = named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            mass_errors.append(abs(self.model.body_mass[body] - source["mass"]))
            com_errors.append(np.max(abs(self.model.body_ipos[body] - source["com"])))
            r = rotation(source["principal_axes"])
            expected = r @ np.diag(source["inertia"]) @ r.T
            actual_r = rotation(self.model.body_iquat[body])
            actual = actual_r @ np.diag(self.model.body_inertia[body]) @ actual_r.T
            inertia_errors.append(np.max(abs(expected - actual)))
        return dict(total_mass_kg=float(self.model.body_mass.sum()), payload_mass_kg=self.mass,
            authored_mass_error_kg=max(mass_errors), authored_com_error_m=max(com_errors),
            authored_inertia_error=max(inertia_errors), original_mjcf_sha256=self.xml_hash,
            automatic_com_assumptions=[n for n,b in self.reference["bodies"].items()
                                       if b["com_source"] != "authored_usd"],
            friction_coefficient=self.friction, delay_steps=self.delays,
            actuators=self.cfg["actuators"], self_collisions=self.cfg["self_collisions"],
            root_free_joints=int(np.count_nonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)),
            warnings=self.cfg["limitations"] + [self.cfg["contact_model"]])
