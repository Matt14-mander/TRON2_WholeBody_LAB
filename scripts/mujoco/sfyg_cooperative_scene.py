"""Two free SFYG bases, one free shared box and two ideal closed grasp welds."""
from copy import deepcopy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sfyg_bridge import SFYGBridge, build_xml, named_id, numbers


def build_scene_xml(configs, spacing=1., payload_mass=1., friction=.5):
    if len(configs) != 2 or any(c["phase"] not in ("carry","robust") for c in configs):
        raise ValueError("Exactly two carry/robust deployment bundles are required")
    if (not np.isfinite([spacing,payload_mass,friction]).all()
            or spacing < .7 or payload_mass <= 0 or friction <= 0):
        raise ValueError("Positive mass/friction and base spacing >=0.7 m required")
    for key in ("physics_dt","decimation","names","usd_sources","defaults","root_quat"):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"The two bundles must share physical timing/pose/interface: {key}")
    root = ET.Element("mujoco",model="sfyg_shared_payload")
    first, assets, reference, xml_hash = build_xml(configs[0],0.,friction)
    for tag in ("compiler","option","visual","size"):
        original = first.find(tag)
        if original is not None:
            root.append(deepcopy(original))
    root.find("option").set("jacobian","dense")
    sections = {tag:ET.SubElement(root,tag) for tag in
                ("default","asset","worldbody","actuator","sensor","contact","equality")}
    references = {"name","class","childclass","mesh","material","texture","joint","joint1","joint2",
                  "body","body1","body2","site","site1","site2","objname","target","actuator","tendon"}
    offsets = [-spacing/2,spacing/2]
    for index,cfg in enumerate(configs):
        original,other_assets,_,_ = build_xml(cfg,0.,friction) if index else (first,assets,reference,xml_hash)
        assets.update(other_assets)
        clone = deepcopy(original)
        names = {node.get("name") for node in clone.iter() if node.get("name")}
        names.update(node.get("class") for node in clone.iter("default") if node.get("class"))
        prefix = f"r{index}/"
        for node in clone.iter():
            for key,value in list(node.attrib.items()):
                if key in references and value in names:
                    node.set(key,prefix+value)
        base = clone.find(f"./worldbody/body[@name='{prefix}base_Link']")
        pos = np.array(cfg["root_pos"],float)
        pos[1] += offsets[index]
        base.set("pos",numbers(pos))
        grip = clone.find(f".//body[@name='{prefix}gripper_base_Link']")
        frame = ET.SubElement(grip,"body",name=prefix+"grasp_frame",pos=".15 0 0")
        ET.SubElement(frame,"site",name=prefix+"grasp_site",size=".015",rgba="1 .3 .1 1")
        # Explicit, massless ideal handle extension; no world support or collision.
        ET.SubElement(grip,"geom",name=prefix+"grip_extension",type="capsule",fromto="0 0 0 .15 0 0",
                      size=".008",density="0",contype="0",conaffinity="0",rgba=".3 .3 .3 1")
        for tag,destination in sections.items():
            source = clone.find(tag)
            if source is None:
                continue
            if tag == "worldbody":
                destination.append(base)
                if index == 0:
                    destination.extend(n for n in source if n is not base)
            elif tag == "default" and index:
                destination.extend(n for n in source if n.tag == "default")
            else:
                destination.extend(list(source))
    # Obtain real nominal FK before adding the object. This is construction only.
    probe = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"),assets)
    state = mujoco.MjData(probe)
    for index,cfg in enumerate(configs):
        ids = [named_id(probe,mujoco.mjtObj.mjOBJ_JOINT,f"r{index}/"+n) for n in cfg["names"]]
        state.qpos[probe.jnt_qposadr[ids]] = cfg["defaults"]
    mujoco.mj_forward(probe,state)
    points = np.array([state.xpos[probe.body(f"r{i}/grasp_frame").id].copy() for i in range(2)])
    orientations = [state.xquat[probe.body(f"r{i}/grasp_frame").id].copy() for i in range(2)]
    if abs(points[1,1]-points[0,1]) < .6 or np.linalg.norm((points[1]-points[0])[[0,2]]) > 1e-5:
        raise ValueError("Nominal grasp frames must align in X/Z with sufficient lateral spacing")
    center = points.mean(0)
    dimensions = np.array([.20,spacing+.12,.08])
    box = ET.SubElement(sections["worldbody"],"body",name="payload",pos=numbers(center))
    ET.SubElement(box,"freejoint",name="payload/free")
    inertia = payload_mass / 12 * np.array([dimensions[1]**2+dimensions[2]**2,
                                          dimensions[0]**2+dimensions[2]**2,
                                          dimensions[0]**2+dimensions[1]**2])
    ET.SubElement(box,"inertial",mass=str(payload_mass),pos="0 0 0",diaginertia=numbers(inertia))
    ET.SubElement(box,"geom",name="payload/box",type="box",size=numbers(dimensions/2),density="0",
                  rgba=".15 .55 .85 1",friction=numbers([friction,0,0]),condim="3",margin="0")
    for index,point in enumerate(points):
        ET.SubElement(box,"site",name=f"payload/handle_{index}",pos=numbers(point-center),
                      quat=numbers(orientations[index]),size=".02",rgba="1 .5 .1 1")
        ET.SubElement(sections["equality"],"weld",name=f"grasp_{index}",site1=f"r{index}/grasp_site",site2=f"payload/handle_{index}",
                      active="false",solref=".02 1",solimp=".99 .999 .001",torquescale=".2")
        for name in ("gripper_base_Link","gripper1_Link","gripper2_Link"):
            ET.SubElement(sections["contact"],"exclude",body1=f"r{index}/"+name,body2="payload")
    # Non-colliding 1 m progress marks, retained even with a following camera.
    for meter in range(-5,21):
        ET.SubElement(sections["worldbody"],"geom",name=f"road/{meter}",type="box",pos=f"{meter} 0 .001",
                      size=".01 3 .0005",density="0",contype="0",conaffinity="0",rgba=".8 .65 .2 .5")
    return root,assets,reference,xml_hash


def initialize(model,data,configs):
    mujoco.mj_resetData(model,data)
    for index,cfg in enumerate(configs):
        ids = [model.joint(f"r{index}/"+n).id for n in cfg["names"]]
        data.qpos[model.jnt_qposadr[ids]] = cfg["defaults"]
        motors = [model.actuator(f"r{index}/pd/"+n).id for n in cfg["names"]]
        data.ctrl[motors] = cfg["defaults"]
    mujoco.mj_forward(model,data)
    payload = model.body("payload").id
    for i in range(2):
        eq = model.equality(f"grasp_{i}").id
        data.eq_active[eq] = True
    mujoco.mj_forward(model,data)
    return payload


class SharedRobot(SFYGBridge):
    def __init__(self,cfg,model,data,index,delay_steps,seed):
        self.cfg,self.model,self.data = cfg,model,data
        self.bind_robot(f"r{index}/",delay_steps,seed)
        self.reset_controller()
        self.frame = model.body(f"r{index}/grasp_frame").id
        self.site = model.site(f"payload/handle_{index}").id
        self.eq = model.equality(f"grasp_{index}").id

    def physics_step(self,*args,**kwargs):
        raise RuntimeError("Shared robots write controls; only CooperativeScene steps the world")

    def reset(self):
        raise RuntimeError("Use CooperativeScene.reset() to reset the shared world")


class CooperativeScene:
    def __init__(self,configs,spacing=1.,payload_mass=1.,friction=.5,delay_steps=None,seed=42):
        if delay_steps is not None and delay_steps not in range(5):
            raise ValueError("Delay must be within 0--4 training physics steps")
        root,assets,self.reference,self.xml_hash = build_scene_xml(configs,spacing,payload_mass,friction)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"),assets)
        self.data = mujoco.MjData(self.model)
        self.configs,self.spacing,self.mass = configs,spacing,payload_mass
        self.payload = initialize(self.model,self.data,configs)
        self.robots = [SharedRobot(c,self.model,self.data,i,delay_steps,seed+i) for i,c in enumerate(configs)]
        if (np.count_nonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) != 3
                or not np.isclose(self.model.body_mass.sum(),2*self.reference["total_mass_kg"]+payload_mass,atol=1e-5)):
            raise ValueError("Expected two free complete robots plus exactly one free shared load")
        if not all(np.isfinite(v).all() for v in (self.model.body_mass,self.model.body_ipos,self.model.body_inertia)):
            raise ValueError("Nonfinite shared physical model")

    def reset(self):
        initialize(self.model,self.data,self.configs)
        for robot in self.robots:
            robot.reset_controller()

    def step(self,wrench_world):
        wrench = np.asarray(wrench_world,float)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise ValueError("Finite WORLD [Fx,Fy,Fz,Tx,Ty,Tz] required")
        for robot in self.robots:
            robot.write_controls()
        self.data.xfrc_applied[:] = 0
        self.data.xfrc_applied[self.payload] = wrench
        mujoco.mj_step(self.model,self.data)

    def grasp_errors(self):
        return [float(np.linalg.norm(self.data.xpos[r.frame]-self.data.site_xpos[r.site])) for r in self.robots]

    def grasp_angles(self):
        result=[]
        for r in self.robots:
            first=self.data.xmat[r.frame].reshape(3,3)
            second=self.data.site_xmat[r.site].reshape(3,3)
            result.append(float(np.rad2deg(np.arccos(np.clip((np.trace(first.T@second)-1)/2,-1,1)))))
        return result


def grasp_reaction(model,data,eq,payload):
    """Payload->robot WORLD [F,T] at its grasp frame, from that weld only."""
    rows = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id == eq)
    if np.count_nonzero(rows) != 6:
        raise ValueError("Expected six active grasp-weld constraint rows")
    jac = data.efc_J.reshape(data.nefc,model.nv)
    generalized = jac[rows].T @ data.efc_force[rows]
    start = model.jnt_dofadr[model.body_jntadr[payload]]
    dofs = np.arange(start,start+6)
    jp,jr = np.zeros((3,model.nv)),np.zeros((3,model.nv))
    obj = model.eq_obj1id[eq]
    point = data.site_xpos[obj] if model.eq_objtype[eq] == mujoco.mjtObj.mjOBJ_SITE else data.xpos[obj]
    mujoco.mj_jac(model,data,jp,jr,point,payload)
    wrench = -np.linalg.solve(np.vstack((jp[:,dofs],jr[:,dofs])).T,generalized[dofs])
    if not np.isfinite(wrench).all():
        raise FloatingPointError("Nonfinite grasp reaction")
    return wrench


def audit_reactions(scene,reactions):
    """Read-only generalized-force reconstruction and full payload dynamics."""
    model,data = scene.model,scene.data
    jac = data.efc_J.reshape(data.nefc,model.nv)
    expected,reconstructed = np.zeros(model.nv),np.zeros(model.nv)
    wrench_at_com = np.zeros(6)
    for robot,wrench in zip(scene.robots,reactions):
        rows = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id == robot.eq)
        expected += jac[rows].T @ data.efc_force[rows]
        jp,jr = np.zeros((3,model.nv)),np.zeros((3,model.nv))
        point = data.xpos[robot.frame]
        mujoco.mj_jac(model,data,jp,jr,point,scene.payload)
        reconstructed -= jp.T @ wrench[:3] + jr.T @ wrench[3:]
        wrench_at_com -= np.r_[wrench[:3],wrench[3:]+np.cross(point-data.xipos[scene.payload],wrench[:3])]
    start = model.jnt_dofadr[model.body_jntadr[scene.payload]]
    dofs = np.arange(start,start+6)
    ma = np.zeros(model.nv)
    mujoco.mj_mulM(model,data,ma,data.qacc)
    residual = ma-data.qfrc_smooth-(data.qfrc_constraint-expected)-reconstructed
    return dict(extraction_max=float(np.max(abs((expected-reconstructed)[dofs]))),
                dynamics_max=float(np.max(abs(residual[dofs]))),payload_grasp_world=wrench_at_com.tolist())
