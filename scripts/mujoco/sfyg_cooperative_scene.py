"""Free SFYG teams with a single shared payload and explicit grasp constraints."""
from copy import deepcopy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sfyg_bridge import SFYGBridge, build_xml, named_id, numbers


def build_scene_xml(configs, spacing=1., payload_mass=1., friction=.5, grasp_model="weld", layout="side-by-side"):
    if grasp_model not in ("weld","ball"):
        raise ValueError("Grasp model must be weld or ball")
    counts = {"side-by-side":2,"leader-follow":2,"sofa":3}
    if layout not in counts or len(configs) != counts[layout] or any(c["phase"] not in ("carry","robust") for c in configs):
        raise ValueError("Expected two carry/robust robots for box/tandem, three for sofa")
    if (not np.isfinite([spacing,payload_mass,friction]).all()
            or spacing < .7 or payload_mass <= 0 or friction <= 0):
        raise ValueError("Positive mass/friction and base spacing >=0.7 m required")
    for key in ("physics_dt","decimation","names","usd_sources","defaults","root_quat"):
        if any(configs[0][key] != c[key] for c in configs[1:]):
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
    offsets = [[0,-spacing/2],[0,spacing/2]]
    if layout != "side-by-side":
        from sfyg_team_scene import placements
        offsets = placements(layout,spacing)
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
        pos[:2] += offsets[index]
        base.set("pos",numbers(pos))
        grip = clone.find(f".//body[@name='{prefix}gripper_base_Link']")
        leader = layout != "side-by-side" and index == 0
        attachment = base if leader else grip
        offset = "-.22 0 .45" if leader else ".15 0 0"
        frame = ET.SubElement(attachment,"body",name=prefix+"grasp_frame",pos=offset)
        ET.SubElement(frame,"site",name=prefix+"grasp_site",size=".015",rgba="1 .3 .1 1")
        # Explicit, massless ideal handle extension; no world support or collision.
        ET.SubElement(attachment,"geom",name=prefix+"grip_extension",type="capsule",fromto="0 0 0 "+offset,
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
    if layout != "side-by-side":
        from sfyg_team_scene import add_payload
        add_payload(root,probe,state,len(configs),payload_mass,friction,grasp_model,layout,spacing)
        return root,assets,reference,xml_hash
    points = np.array([state.xpos[probe.body(f"r{i}/grasp_frame").id].copy() for i in range(2)])
    orientations = [state.xquat[probe.body(f"r{i}/grasp_frame").id].copy() for i in range(2)]
    if abs(points[1,1]-points[0,1]) < .6 or np.linalg.norm((points[1]-points[0])[[0,2]]) > 1e-5:
        raise ValueError("Nominal grasp frames must align in X/Z with sufficient lateral spacing")
    center = points.mean(0)
    if grasp_model == "ball":
        # Two ball grips leave rotation about their connecting line free.
        # Put the CoM below the handles, giving that axis gravity restoration.
        center[2] -= .04
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
        constraint=ET.SubElement(sections["equality"],"weld" if grasp_model=="weld" else "connect",
                      name=f"grasp_{index}",site1=f"r{index}/grasp_site",site2=f"payload/handle_{index}",
                      active="false",solref=".02 1",solimp=".99 .999 .001")
        if grasp_model=="weld":
            constraint.set("torquescale",".2")
        else:
            # Constant impedance avoids a state-dependent stiffness transition
            # in the ball-grip diagnostic's closed chain of two held arms.
            constraint.set("solref",".04 1")
            constraint.set("solimp",".99 .99 .001")
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
    for i in range(len(configs)):
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
    def __init__(self,configs,spacing=1.,payload_mass=1.,friction=.5,delay_steps=None,seed=42,
                 grasp_model="weld",physics_substeps=1,layout="side-by-side"):
        if delay_steps is not None and delay_steps not in range(5):
            raise ValueError("Delay must be within 0--4 training physics steps")
        if physics_substeps not in (1,2,4,8):
            raise ValueError("Solver substeps must be 1,2,4 or 8 per training physics tick")
        root,assets,self.reference,self.xml_hash = build_scene_xml(configs,spacing,payload_mass,friction,grasp_model,layout)
        self.layout = layout
        self.grasp_model,self.physics_substeps=grasp_model,physics_substeps
        root.find("option").set("timestep",str(configs[0]["physics_dt"]/physics_substeps))
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"),assets)
        self.data = mujoco.MjData(self.model)
        self.configs,self.spacing,self.mass = configs,spacing,payload_mass
        self.payload = initialize(self.model,self.data,configs)
        self.robots = [SharedRobot(c,self.model,self.data,i,delay_steps,seed+i) for i,c in enumerate(configs)]
        if (np.count_nonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) != len(configs)+1
                or not np.isclose(self.model.body_mass.sum(),len(configs)*self.reference["total_mass_kg"]+payload_mass,atol=1e-5)):
            raise ValueError("Expected complete free robots plus exactly one free shared load")
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
        # Keep actuator queues at 200 Hz, including their 0--4 tick delays;
        # only the integrator/constraint solver runs at the smaller time step.
        for _ in range(self.physics_substeps):
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


def _constraint_generalized(model,data,eq):
    rows = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id == eq)
    expected=6 if model.eq_type[eq]==mujoco.mjtEq.mjEQ_WELD else 3
    if np.count_nonzero(rows) != expected:
        raise ValueError(f"Expected {expected} active grasp constraint rows")
    jac = data.efc_J.reshape(data.nefc,model.nv)
    return jac[rows].T @ data.efc_force[rows]


def _grasp_point(model,data,eq):
    obj=model.eq_obj1id[eq]
    if model.eq_objtype[eq]==mujoco.mjtObj.mjOBJ_SITE:
        return data.site_xpos[obj],model.site_bodyid[obj]
    return data.xpos[obj],obj


def _root_wrench(model,data,point,body,generalized):
    root=model.body_rootid[body]
    joint=model.body_jntadr[root]
    if joint<0 or model.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Wrench reconstruction requires the free root of this body tree")
    start = model.jnt_dofadr[joint]
    dofs = np.arange(start,start+6)
    jp,jr = np.zeros((3,model.nv)),np.zeros((3,model.nv))
    mujoco.mj_jac(model,data,jp,jr,point,body)
    wrench = np.linalg.solve(np.vstack((jp[:,dofs],jr[:,dofs])).T,generalized[dofs])
    if not np.isfinite(wrench).all():
        raise FloatingPointError("Nonfinite grasp reaction")
    return wrench


def grasp_reaction(model,data,eq,payload):
    """Own robot-side WORLD [F,T] at its actual grasp point, from this grasp only.

    Read the generalized force on the robot's free root, not the negative
    payload-side moment shifted across a possibly violated soft constraint.
    """
    point,body=_grasp_point(model,data,eq)
    return _root_wrench(model,data,point,body,_constraint_generalized(model,data,eq))


def payload_side_reaction(model,data,eq,payload):
    """Legacy negative object-side wrench at robot point, for diagnostics only."""
    point,_=_grasp_point(model,data,eq)
    return -_root_wrench(model,data,point,payload,_constraint_generalized(model,data,eq))


def audit_reactions(scene,reactions):
    """Read-only generalized-force reconstruction and full payload dynamics."""
    model,data = scene.model,scene.data
    jac = data.efc_J.reshape(data.nefc,model.nv)
    expected,reconstructed = np.zeros(model.nv),np.zeros(model.nv)
    wrench_at_com = np.zeros(6)
    robot_errors,moment_differences,payload_estimates=[],[],[]
    for robot,wrench in zip(scene.robots,reactions):
        rows = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id == robot.eq)
        generalized=jac[rows].T @ data.efc_force[rows]
        expected += generalized
        jp,jr = np.zeros((3,model.nv)),np.zeros((3,model.nv))
        point = data.xpos[robot.frame]
        mujoco.mj_jac(model,data,jp,jr,point,robot.frame)
        root=model.body_rootid[robot.frame]
        start=model.jnt_dofadr[model.body_jntadr[root]]
        robot_errors.append(float(np.max(abs((jp.T@wrench[:3]+jr.T@wrench[3:]-generalized)[start:start+6]))))
        opposite=payload_side_reaction(model,data,robot.eq,scene.payload)
        payload_estimates.append(opposite.tolist())
        moment_differences.append(float(np.linalg.norm(wrench[3:]-opposite[3:])))
        mujoco.mj_jac(model,data,jp,jr,point,scene.payload)
        reconstructed -= jp.T @ opposite[:3] + jr.T @ opposite[3:]
        wrench_at_com -= np.r_[opposite[:3],opposite[3:]+np.cross(point-data.xipos[scene.payload],opposite[:3])]
    start = model.jnt_dofadr[model.body_jntadr[scene.payload]]
    dofs = np.arange(start,start+6)
    ma = np.zeros(model.nv)
    mujoco.mj_mulM(model,data,ma,data.qacc)
    residual = ma-data.qfrc_smooth-(data.qfrc_constraint-expected)-reconstructed
    return dict(extraction_max=max([*robot_errors,float(np.max(abs((expected-reconstructed)[dofs])))]),
                robot_payload_moment_difference_nm=moment_differences,
                legacy_payload_estimates_world=payload_estimates,
                dynamics_max=float(np.max(abs(residual[dofs]))),payload_grasp_world=wrench_at_com.tolist())
