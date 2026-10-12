"""MPC-free MuJoCo PD pairs and observation-only intent deployment.

Default fixed-base mode isolates estimator transfer. --locomotion_dir loads
the existing exported policy.pt/encoder.pt for floating-base sim-to-sim checks.
The estimator NEVER supplies robot commands.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers"))
from paint_intent import ARM_JOINT_NAMES, CONDITIONS, SCHEMA, IntentHistory, InteractionProtocol, collection_conditions
from ocs2_rollout_dataset import save_episode_npz

LEG_NAMES = tuple(f"{part}_{side}_Joint" for side in ("L", "R") for part in ("proximal_pitch", "proximal_roll", "proximal_yaw", "knee", "ankle_pitch"))
LEG_HOME = np.array([0., 0., -3.14159, 0., 0., 0., 0., 3.14159, 0., 0.])


def payload_xml(source, mass, dimensions, offset, floating):
    source = Path(source).resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((source.parent / compiler.get("meshdir", ".")).resolve()))
    base = root.find(".//body[@name='base_Link']")
    base.set("pos", "0 0 0.85")
    if not floating:
        for joint in list(base.findall("joint")):
            if joint.get("type") == "free":
                base.remove(joint)
    attachment = root.find(".//body[@name='gripper_base_Link']")
    if attachment is None:
        raise ValueError("No gripper_base_Link in MJCF.")
    body = ET.SubElement(attachment, "body", name="paint_payload_Link", pos=" ".join(map(str, offset)))
    x, y, z = dimensions
    inertia = (mass*(y*y+z*z)/12, mass*(x*x+z*z)/12, mass*(x*x+y*y)/12)
    ET.SubElement(body, "inertial", pos="0 0 0", mass=str(mass), diaginertia=" ".join(map(str, inertia)))
    ET.SubElement(body, "geom", type="box", size=" ".join(str(d/2) for d in dimensions), density="0", rgba="0.15 0.55 0.85 1")
    return ET.tostring(root, encoding="unicode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=str(ROOT / "robot_description/tron2/SFYG_TRON2A/xml/robot.xml"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_dir", help="Trained intent model; omission collects baseline data only.")
    parser.add_argument("--locomotion_dir", help="Exported policy.pt + encoder.pt; enables floating-base mode.")
    parser.add_argument("--base_command", nargs=3, type=float, default=(0., 0., 0.))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, help="Diagnostic subset, e.g. zero; default runs all seven conditions.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pose_jitter", type=float, default=0.02)
    parser.add_argument("--payload_mass", type=float, default=0.5)
    parser.add_argument("--payload_dimensions", nargs=3, type=float, default=(0.20, 0.12, 0.08))
    parser.add_argument("--payload_offset", nargs=3, type=float, default=(0.15, 0., 0.))
    parser.add_argument("--force_amplitude", type=float, default=5.)
    parser.add_argument("--torque_amplitude", type=float, default=0.5)
    parser.add_argument("--arm_kp", type=float, default=60.)
    parser.add_argument("--arm_kd", type=float, default=6.)
    parser.add_argument("--warmup", type=float, default=2.)
    parser.add_argument("--duration", type=float, default=2.)
    parser.add_argument("--recovery", type=float, default=2.)
    parser.add_argument("--profile", choices=("trapezoid", "sine"), default="trapezoid")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    conditions = collection_conditions("attached", args.conditions)
    values = np.array([args.payload_mass, *args.payload_dimensions, *args.payload_offset, args.arm_kp, args.arm_kd, args.pose_jitter, *args.base_command])
    if not np.isfinite(values).all() or args.payload_mass <= 0 or min(args.payload_dimensions) <= 0 or args.arm_kp <= 0 or args.arm_kd < 0 or not 0 <= args.pose_jitter <= .1 or args.repeats < 1:
        raise ValueError("Invalid diagnostic settings.")
    if not args.locomotion_dir and any(args.base_command):
        raise ValueError("Fixed-base mode cannot execute velocity commands; supply --locomotion_dir.")
    protocol = InteractionProtocol(args.warmup, args.duration, args.recovery, args.force_amplitude, args.torque_amplitude, args.profile)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory.")
    output.mkdir(parents=True, exist_ok=True)
    xml = payload_xml(args.xml, args.payload_mass, args.payload_dimensions, args.payload_offset, bool(args.locomotion_dir))
    (output / "robot_payload.xml").write_text(xml, encoding="utf-8")
    # Read mesh bytes with Python to support Windows workspaces with Unicode paths.
    compile_root = ET.fromstring(xml)
    meshdir = Path(compile_root.find("compiler").get("meshdir"))
    assets = {mesh.get("file"): (meshdir / mesh.get("file")).read_bytes() for mesh in compile_root.findall("./asset/mesh")}
    compile_root.find("compiler").set("meshdir", "")
    model = mujoco.MjModel.from_xml_string(ET.tostring(compile_root, encoding="unicode"), assets=assets)
    data = mujoco.MjData(model)
    dt = .02
    substeps = round(dt / model.opt.timestep)
    if not np.isclose(substeps * model.opt.timestep, dt):
        raise ValueError("Physics timestep must divide .02 seconds.")
    arm_ids = np.array([model.joint(n).id for n in ARM_JOINT_NAMES])
    leg_ids = np.array([model.joint(n).id for n in LEG_NAMES])
    arm_q, arm_v = model.jnt_qposadr[arm_ids], model.jnt_dofadr[arm_ids]
    leg_q, leg_v = model.jnt_qposadr[leg_ids], model.jnt_dofadr[leg_ids]
    home = np.array([0., math.pi/2, -1.4835299, 0., 0., 0.])
    leg_home = LEG_HOME.copy()
    # Match the Isaac asset armature and PD gains for the locomotion adapter.
    for joint_id, name in zip(leg_ids, LEG_NAMES):
        model.dof_armature[model.jnt_dofadr[joint_id]] = .053923687 if "yaw" in name or "ankle" in name else .161777558
    model.dof_damping[arm_v] = 0.  # explicit PD derivative term, not double damping
    model.dof_damping[leg_v] = 0.
    base_id = model.body("base_Link").id
    payload_id = model.body("paint_payload_Link").id
    motor_by_joint = {int(joint): i for i, joint in enumerate(model.actuator_trnid[:, 0])}
    arm_motor = np.array([motor_by_joint[i] for i in arm_ids])
    leg_motor = np.array([motor_by_joint[i] for i in leg_ids])
    kp = np.array([53.22 if "yaw" in n or "ankle" in n else 159.67 for n in LEG_NAMES])
    kd = np.array([3.39 if "yaw" in n or "ankle" in n else 10.16 for n in LEG_NAMES])
    effort = np.array([40. if "yaw" in n or "ankle" in n else 140. for n in LEG_NAMES])
    intent_model = intent_history = None
    if args.model_dir:
        directory = Path(args.model_dir)
        meta = json.loads((directory / "metadata.json").read_text())
        if meta["schema"] != SCHEMA or meta["joint_names"] != list(ARM_JOINT_NAMES) or not np.isclose(meta["step_dt"], dt):
            raise ValueError("Intent model contract mismatch.")
        trained_versions = {item.get("runtime", {}).get("mujoco") for item in meta.get("training_backends", []) if item.get("backend") == "mujoco"}
        trained_versions.discard(None)
        if trained_versions and mujoco.__version__ not in trained_versions:
            print(f"[WARNING] Estimator trained with MuJoCo {sorted(trained_versions)}, current runtime={mujoco.__version__}. "
                  "Zero-force responses may differ; this run is a transfer diagnostic, not a validated demo.", flush=True)
        with (directory / "intent_estimator.ts").open("rb") as stream:
            intent_model = torch.jit.load(stream, map_location="cpu").eval()
        intent_history = IntentHistory(meta["history"])
    actor = encoder = None
    if args.locomotion_dir:
        directory = Path(args.locomotion_dir)
        with (directory / "policy.pt").open("rb") as stream:
            actor = torch.jit.load(stream, map_location="cpu").eval()
        with (directory / "encoder.pt").open("rb") as stream:
            encoder = torch.jit.load(stream, map_location="cpu").eval()
        with torch.inference_mode():
            latent = encoder(torch.zeros((1, 420)))
            if latent.shape != (1, 3) or actor(torch.zeros((1, 78))).shape != (1, 10):
                raise ValueError("Expected WholeBody 420-input/3-output encoder and 78-input/10-output actor.")
    contract = {"schema": SCHEMA, "joint_names": list(ARM_JOINT_NAMES), "step_dt": dt, "label_frame": "payload_local", "label_point": "payload_COM",
                "backend": "mujoco", "runtime": {"mujoco": mujoco.__version__, "torch": torch.__version__, "numpy": np.__version__},
                "base_mode": "floating_policy" if actor else "fixed", "conditions": list(conditions), "args": vars(args), "protocol": vars(protocol)}
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    viewer = None
    if args.viewer:
        from mujoco import viewer as mujoco_viewer
        viewer = mujoco_viewer.launch_passive(model, data)
    manifest = []
    try:
        for repeat in range(args.repeats):
            target = home + np.random.default_rng(args.seed + repeat).uniform(-args.pose_jitter, args.pose_jitter, 6)
            limits = model.jnt_range[arm_ids]
            if np.any(target <= limits[:, 0]) or np.any(target >= limits[:, 1]):
                raise ValueError("Arm target violates MuJoCo joint limits.")
            for condition in conditions:
                mujoco.mj_resetData(model, data)
                data.qpos[arm_q] = home
                data.qpos[leg_q] = leg_home
                for name, position in (("gripper1_Joint", .05), ("gripper2_Joint", -.05)):
                    data.qpos[model.jnt_qposadr[model.joint(name).id]] = position
                if actor:
                    data.qpos[:3] = [0., 0., .85]
                    data.qpos[3:7] = [1., 0., 0., 0.]
                mujoco.mj_forward(model, data)
                initial_base_position = data.xpos[base_id].copy()
                history = np.zeros((10, 42), dtype=np.float32)
                last_action = np.zeros(10)
                leg_target = leg_home.copy()
                if intent_history:
                    intent_history.reset()
                frames, reason = [], "completed"
                for index in range(math.ceil(protocol.total_time / dt)):
                    wall_start = time.monotonic()
                    rotation = data.xmat[base_id].reshape(3, 3)
                    gravity = rotation.T @ np.array([0., 0., -1.])
                    if actor:
                        velocity = np.zeros(6)
                        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, velocity, 1)
                        phase = index * dt * .8 * 2 * math.pi
                        obs = np.concatenate([np.clip(velocity[:3], -100, 100) * .25, gravity,
                                              np.clip(data.qpos[leg_q]-leg_home, -100, 100), np.clip(data.qvel[leg_v], -100, 100)*.05,
                                              last_action, [math.sin(phase), math.cos(phase)], [.8, .5, .5, .2]]).astype(np.float32)
                        if index == 0:
                            # Isaac Lab's CircularBuffer fills initial history
                            # with the first observation rather than zeros.
                            history[:] = obs
                        else:
                            history[:-1] = history[1:]
                            history[-1] = obs
                        with torch.inference_mode():
                            latent = encoder(torch.from_numpy(history.reshape(1, -1)))
                            actor_input = torch.cat([latent, torch.from_numpy(np.concatenate([obs, np.zeros(30), args.base_command]).astype(np.float32)[None])], dim=1)
                            last_action = actor(actor_input).numpy()[0]
                        if not np.isfinite(last_action).all():
                            raise RuntimeError("Nonfinite locomotion action.")
                        last_action = np.clip(last_action, -100., 100.)
                        leg_target = leg_home + .25 * last_action
                    wrench = protocol.wrench(condition, index * dt)
                    torque_used = np.zeros(6)
                    for _ in range(substeps):
                        data.xfrc_applied[:] = 0.
                        payload_rotation = data.xmat[payload_id].reshape(3, 3)
                        data.xfrc_applied[payload_id, :3] = payload_rotation @ wrench[:3]
                        data.xfrc_applied[payload_id, 3:] = payload_rotation @ wrench[3:]
                        torque_used = np.clip(args.arm_kp * (target-data.qpos[arm_q]) - args.arm_kd*data.qvel[arm_v], -100., 100.)
                        data.ctrl[arm_motor] = torque_used
                        data.ctrl[leg_motor] = np.clip(kp*(leg_target-data.qpos[leg_q])-kd*data.qvel[leg_v], -effort, effort)
                        for name, position in (("gripper1_Joint", .05), ("gripper2_Joint", -.05)):
                            joint = model.joint(name).id
                            data.ctrl[motor_by_joint[joint]] = np.clip(100*(position-data.qpos[model.jnt_qposadr[joint]])-5*data.qvel[model.jnt_dofadr[joint]], -10, 10)
                        mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                    q, dq = data.qpos[arm_q].copy(), data.qvel[arm_v].copy()
                    measured_base_velocity = np.zeros(6)
                    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, measured_base_velocity, 1)
                    tilt = math.degrees(math.acos(float(np.clip(data.xmat[base_id].reshape(3, 3)[2, 2], -1, 1))))
                    margin = float(np.minimum(q-limits[:, 0], limits[:, 1]-q).min())
                    prediction, prediction_valid = np.zeros(3), False
                    if intent_history:
                        features = intent_history.update(q, dq, target)
                        if features is not None:
                            with torch.inference_mode():
                                prediction = intent_model(torch.from_numpy(features[None])).numpy()[0]
                            prediction_valid = bool(np.isfinite(prediction).all())
                    frames.append({"time": (index+1)*dt, "interval_start": index*dt, "arm_position": q, "arm_velocity": dq,
                                   "applied_position": target.copy(), "interaction_wrench": wrench[[0, 1, 5]], "response_valid": True,
                                   "tilt_deg": tilt, "joint_margin": margin, "arm_applied_torque": torque_used.copy(),
                                   "base_position_world": data.xpos[base_id].copy(),
                                   "base_velocity": measured_base_velocity[3:].copy(),
                                   "base_angular_velocity": measured_base_velocity[:3].copy(),
                                   "locomotion_action": last_action.copy(),
                                   "intent_estimate": prediction.copy(), "intent_estimate_valid": prediction_valid})
                    if viewer:
                        if not viewer.is_running():
                            reason = "interrupted"
                            break
                        viewer.sync()
                        time.sleep(max(0., dt-(time.monotonic()-wall_start)))
                    if tilt > 30 or margin < -.01:
                        reason = "tilt_limit" if tilt > 30 else "joint_limit"
                        break
                data.xfrc_applied[:] = 0.
                accepted = reason == "completed"
                episode_id = f"group_{repeat:04d}_{condition}"
                filename = Path("episodes" if accepted else "rejected") / (episode_id + ".npz")
                save_episode_npz(output / filename, frames, {"schema": SCHEMA, "group_id": f"seed_{args.seed+repeat}", "condition": condition,
                                 "accepted": accepted, "termination_reason": reason, "step_dt": dt})
                manifest.append({"episode_id": episode_id, "group_id": f"seed_{args.seed+repeat}", "condition": condition,
                                 "accepted": accepted, "reason": reason, "file": str(filename), "frames": len(frames),
                                 "duration_s": len(frames)*dt, "max_tilt_deg": max(f["tilt_deg"] for f in frames),
                                 "min_joint_margin_rad": min(f["joint_margin"] for f in frames),
                                 "base_displacement_world_m": (data.xpos[base_id]-initial_base_position).tolist(),
                                 "mean_base_velocity_body_m_s": np.mean([f["base_velocity"] for f in frames], axis=0).tolist()})
                (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
                print(f"[MUJOCO] {episode_id} {reason} max_tilt_deg={manifest[-1]['max_tilt_deg']:.2f}", flush=True)
                if reason == "interrupted":
                    return
        (output / "run_status.json").write_text(json.dumps({"complete": True, "accepted": sum(r["accepted"] for r in manifest), "total": len(manifest),
                                                           "expected": args.repeats*len(conditions), "base_mode": "floating_policy" if actor else "fixed"}))
    finally:
        if viewer:
            viewer.close()


if __name__ == "__main__":
    main()
