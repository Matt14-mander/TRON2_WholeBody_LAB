"""Two SFYG robots follow keyboard forces on one real shared box using local FT admittance."""
import argparse
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import time

import torch
import mujoco
import numpy as np

from sfyg_bridge import ROOT
from sfyg_policy import sha256
from sfyg_cooperative_control import LocalAdmittance,PayloadKeyboard,slew
from sfyg_cooperative_scene import CooperativeScene,grasp_reaction,audit_reactions
from sfyg_cooperative_visuals import draw


def load_bundle(path):
    cfg = json.loads((path / "deployment.json").read_text(encoding="utf-8"))
    if cfg.get("schema") != "sfyg_mujoco_v1" or cfg.get("phase") not in ("carry","robust"):
        raise ValueError("Use a verified carry/robust export_sfyg.py bundle")
    if sha256(path / "policy.pt") != cfg["policy_sha256"]:
        raise ValueError("Policy SHA256 differs from deployment.json")
    with (path / "policy.pt").open("rb") as stream:
        policy = torch.jit.load(stream,map_location="cpu").eval()
    with torch.inference_mode():
        result = policy(torch.zeros(1,42),torch.zeros(1,10,42),torch.zeros(1,3))
    if result.shape != (1,10) or not torch.isfinite(result).all():
        raise ValueError("Expected finite 42+10x42+3 -> 10 policy")
    return cfg,policy


def robot_state(scene,robot):
    linear,angular,gravity = robot.base_state()
    return dict(velocity_vx_vy_wz=[float(linear[0]),float(linear[1]),float(angular[2])],
        position=scene.data.xpos[robot.base].tolist(),
        tilt_deg=float(np.rad2deg(np.arccos(np.clip(-gravity[2],-1,1)))),
        peak_arm_error_rad=float(np.max(abs(scene.data.qpos[robot.qpos[10:16]]-robot.defaults[10:16]))))


def run(args):
    if len(args.bundle_dir) not in (1,2):
        raise ValueError("Pass one shared bundle, or two bundles in robot order")
    if args.keyboard and args.headless:
        raise ValueError("Keyboard control needs the visible MuJoCo window")
    values = [args.duration,args.settle_seconds,args.calibration_seconds,args.release_tail,args.force_slew,
              args.torque_slew,args.max_speed,args.max_yaw,args.friction]
    if not np.isfinite(values).all() or min(values) <= 0 or args.fps < 1:
        raise ValueError("Positive finite timing, slew, speed and friction parameters required")
    if (args.calibration_seconds > args.settle_seconds or args.max_speed > .5 or args.max_yaw > 1.5
            or not args.check_only and args.duration <= args.settle_seconds+args.release_tail):
        raise ValueError("Check calibration/duration or source command limits")
    torch.set_num_threads(1)
    loaded = [load_bundle(path) for path in args.bundle_dir]
    if len(loaded) == 1:
        loaded *= 2  # Stateless network weights shared; controller histories remain separate.
    configs,policies = zip(*loaded)
    keyboard = PayloadKeyboard([*args.force_world,args.yaw_torque],args.force_step,args.torque_step,args.max_force,args.max_torque)
    controllers = [LocalAdmittance(mass=args.virtual_mass,damping=args.damping,filter_tau=args.filter_tau,
                                   max_speed=args.max_speed,max_yaw=args.max_yaw) for _ in range(2)]
    scene = CooperativeScene(configs,args.spacing,args.payload_mass,args.friction,args.delay_steps,args.seed)
    output = args.output_dir or ROOT / "logs/mujoco" / ("sfyg_cooperate_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory")
    output.mkdir(parents=True,exist_ok=True)
    initial = scene.data.xpos[scene.payload].copy()
    errors = scene.grasp_errors()
    if max(errors) > 1e-6:
        raise ValueError(f"Initial grasp FK alignment failed: {errors}")
    grasp_rows = scene.data.efc_pos[(scene.data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)]
    if np.max(abs(grasp_rows)) > 1e-6:
        raise ValueError("Nonzero initial weld residual")
    manifest = dict(kind="sfyg_two_robot_ft_admittance",is_acceptance=False,
        policies=[dict(path=str(p.resolve()),policy_sha256=c["policy_sha256"],checkpoint_sha256=c["checkpoint_sha256"])
                  for p,c in zip(args.bundle_dir*2 if len(args.bundle_dir)==1 else args.bundle_dir,configs)],
        controller=args.controller,controller_inputs="each robot own grasp WORLD wrench and own base axes only",
        payload_mass_kg=args.payload_mass,total_mass_kg=float(scene.model.body_mass.sum()),
        individual_payloads_removed=True,free_roots=3,attachment_model="two ideal closed grasp welds with massless .15m extensions",
        initial_grasp_errors_m=errors,initial_weld_residual_max=float(np.max(abs(grasp_rows))),
        physics_dt=configs[0]["physics_dt"],decimation=configs[0]["decimation"],
        robot_delay_steps=[r.delays for r in scene.robots],
        admittance=dict(virtual_mass_kg=args.virtual_mass,damping_ns_m=args.damping,filter_tau_s=args.filter_tau,
                        yaw_inertia=.2,yaw_damping=1.,force_deadband_n=.15,torque_deadband_nm=.03,
                        acceleration_m_s2=.10,yaw_acceleration_rad_s2=.3,max_speed=args.max_speed,max_yaw=args.max_yaw),
        source_mjcf_sha256=scene.xml_hash,mujoco_version=mujoco.__version__,
        arguments={k:str(v) if isinstance(v,Path) else [str(p) for p in v] if k=="bundle_dir" else v for k,v in vars(args).items()},
        notes=["No learned PAINT teacher/student; reference FT admittance",
               "No object force share, common velocity, base pose target, formation feedback or world weld",
               "Shared-load interaction torque and reaction peaks can exceed single-robot training conditions"])
    (output / "interface.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"[COOP] two SFYG, free shared load {args.payload_mass}kg; total={manifest['total_mass_kg']:.4f}kg; output={output}",flush=True)
    if args.check_only:
        print("[CHECK] 3 free roots, 2 aligned grasp welds, both policy interfaces loaded; no rollout",flush=True)
        return output
    print("[KEYS] Focus MuJoCo window: W/S=Fx +/-; A/D=Fy +/-; Q/E=Tz +/-; X=release force; R=initial; P=pause",flush=True)
    print(f"[COOP] settle {args.settle_seconds}s, calibrate last {args.calibration_seconds}s; "
          "red arrow=applied object force, green/blue=measured horizontal grasp reactions",flush=True)
    context = nullcontext(None)
    if not args.headless:
        from mujoco import viewer as mj_viewer
        context = mj_viewer.launch_passive(scene.model,scene.data,key_callback=keyboard.on_key if args.keyboard else None)
    camera = mujoco.MjvCamera()
    camera.distance,camera.elevation,camera.azimuth = 4.,-18.,135.
    dt = configs[0]["physics_dt"]
    control_dt = dt*configs[0]["decimation"]
    renderer = writer = None
    applied = np.zeros(3)
    next_frame,physical_steps = 0.,0
    rows,trace = [],[]
    reason,error = "completed",None
    try:
        if args.video:
            import imageio.v2 as imageio
            renderer = mujoco.Renderer(scene.model,height=720,width=960)
            writer = imageio.get_writer(str(output / "preview.mp4"),fps=args.fps,codec="libx264")
        with (output / "steps.jsonl").open("w",encoding="utf-8") as stream, context as viewer:
            while scene.data.time < args.duration-dt/2:
                started = time.monotonic()
                requested,paused = keyboard.snapshot()
                if viewer is not None and not viewer.is_running():
                    reason = "viewer_closed"
                    break
                if paused:
                    viewer.sync()
                    time.sleep(.01)
                    continue
                t = float(scene.data.time)
                settling = t < args.settle_seconds
                releasing = t >= args.duration-args.release_tail
                phase = "SETTLE" if settling else "RELEASE" if releasing else args.controller.upper()
                target_force = np.zeros(3) if settling or releasing else requested
                reactions = [grasp_reaction(scene.model,scene.data,r.eq,scene.payload) for r in scene.robots]
                audit = audit_reactions(scene,reactions)
                commands,suggestions,states = [],[],[]
                for robot,policy,controller,reaction in zip(scene.robots,policies,controllers,reactions):
                    r = scene.data.xmat[robot.base].reshape(3,3)
                    suggestion = controller.update(reaction,r,control_dt,settling=settling,
                        calibrating=settling and t>=args.settle_seconds-args.calibration_seconds)
                    command = np.zeros(3) if args.controller=="hold" else suggestion
                    # No requested/applied force, partner state or object pose is passed to the controller.
                    obs,history = robot.observation()
                    with torch.inference_mode():
                        action = policy(torch.from_numpy(obs[None]),torch.from_numpy(history[None]),
                                        torch.tensor(command[None],dtype=torch.float32))[0].cpu().numpy().copy()
                    robot.set_action(action)
                    commands.append(command.copy())
                    suggestions.append(suggestion.copy())
                    states.append(robot_state(scene,robot))
                    if args.trace:
                        trace.append(dict(time_s=t,robot=len(commands)-1,observation=obs,history=history,
                                          command=command.copy(),action=action.copy()))
                wrench = np.array([applied[0],applied[1],0,0,0,applied[2]])
                row = dict(time_s=t,phase=phase,requested_fx_fy_tz=requested.tolist(),applied_world_wrench=wrench.tolist(),
                    payload_position=scene.data.xpos[scene.payload].tolist(),grasp_errors_m=scene.grasp_errors(),
                    grasp_angles_deg=scene.grasp_angles(),
                    extraction_error=audit["extraction_max"],dynamics_residual=audit["dynamics_max"],
                    payload_grasp_world=audit["payload_grasp_world"],
                    robots=[dict(**state,command=command.tolist(),admittance_suggestion=suggestion.tolist(),
                                 grasp_world=reaction.tolist(),bias_world=controller.bias.tolist(),
                                 filtered_local_fx_fy_tz=controller.corrected_local.tolist())
                            for state,command,suggestion,reaction,controller in zip(states,commands,suggestions,reactions,controllers)])
                stream.write(json.dumps(row)+"\n")
                rows.append(row)
                if len(rows)%50==1:
                    print(f"[COOP] t={t:.2f} {phase} object Fx/Fy/Tz={applied.round(2)} "
                          f"cmd={[np.round(c,3).tolist() for c in commands]} "
                          f"actual={[np.round(s['velocity_vx_vy_wz'],3).tolist() for s in states]} "
                          f"grasp_err={np.round(row['grasp_errors_m'],4)}",flush=True)
                for _ in range(configs[0]["decimation"]):
                    applied = slew(applied,target_force,dt,[args.force_slew,args.torque_slew])
                    scene.step([applied[0],applied[1],0,0,0,applied[2]])
                    physical_steps += 1
                for robot in scene.robots:
                    robot.control_steps += 1
                mujoco.mj_forward(scene.model,scene.data)
                camera.lookat[:] = scene.data.xpos[scene.payload]+[0,0,-.3]
                positions = [scene.data.xpos[r.base] for r in scene.robots]
                wrench = np.array([applied[0],applied[1],0,0,0,applied[2]])
                if writer is not None and scene.data.time >= next_frame:
                    renderer.update_scene(scene.data,camera)
                    draw(renderer.scene,scene.data.xpos[scene.payload],wrench,positions,commands,reactions,phase)
                    writer.append_data(renderer.render())
                    next_frame += 1./args.fps
                if viewer is not None:
                    with viewer.lock():
                        viewer.cam.lookat[:] = camera.lookat
                        if len(rows)==1:
                            viewer.cam.distance,viewer.cam.elevation,viewer.cam.azimuth = camera.distance,camera.elevation,camera.azimuth
                        viewer.user_scn.ngeom = 0
                        draw(viewer.user_scn,scene.data.xpos[scene.payload],wrench,positions,commands,reactions,phase)
                    viewer.sync()
                    time.sleep(max(0.,control_dt-(time.monotonic()-started)))
                if not np.isfinite(scene.data.qpos).all() or not np.isfinite(scene.data.qvel).all():
                    reason="nonfinite_state"
                    break
                if scene.data.time < physical_steps*dt-1e-6:
                    reason="simulator_reset"
                    break
                terminal = [robot_state(scene,r) for r in scene.robots]
                if any(s["tilt_deg"]>45 or s["position"][2]<.45 for s in terminal):
                    reason="fall"
                    break
                if max(scene.grasp_errors())>.1:
                    reason="grasp_error"
                    break
    except Exception as exc:
        reason,error="error",repr(exc)
        raise
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        terminal = [robot_state(scene,r) for r in scene.robots]
        report = dict(kind="sfyg_two_robot_ft_admittance_diagnostic",is_acceptance=False,reason=reason,error=error,
            completed=reason=="completed" and scene.data.time>=args.duration-dt,
            duration_s=float(scene.data.time),controller=args.controller,
            payload_displacement_world_m=(scene.data.xpos[scene.payload]-initial).tolist(),
            max_grasp_error_m=max([*scene.grasp_errors(),*[max(r["grasp_errors_m"]) for r in rows]]),
            max_grasp_angle_deg=max([*scene.grasp_angles(),*[max(r["grasp_angles_deg"]) for r in rows]]),
            max_extraction_error=max((r["extraction_error"] for r in rows),default=0),
            max_dynamics_residual=max((r["dynamics_residual"] for r in rows),default=0),
            max_applied_horizontal_force_n=max((np.linalg.norm(r["applied_world_wrench"][:2]) for r in rows),default=0.),
            calibration_samples=[c.bias_samples for c in controllers],bias_world=[c.bias.tolist() for c in controllers],
            robots=[])
        for index in range(2):
            robot_rows=[r["robots"][index] for r in rows]
            moving=[r["robots"][index] for r in rows if r["time_s"]>=args.settle_seconds]
            final_hold=[r["robots"][index] for r in rows if r["time_s"]>=args.duration-1]
            mean=lambda seq,key:np.mean([r[key] for r in seq],axis=0).tolist() if seq else None
            report["robots"].append(dict(terminal=terminal[index],
                max_tilt_deg=max([terminal[index]["tilt_deg"],*[r["tilt_deg"] for r in robot_rows]]),
                peak_arm_error_rad=max([terminal[index]["peak_arm_error_rad"],*[r["peak_arm_error_rad"] for r in robot_rows]]),
                post_settle_mean_velocity=mean(moving,"velocity_vx_vy_wz"),
                last_second_mean_velocity=mean(final_hold,"velocity_vx_vy_wz"),
                peak_raw_horizontal_grasp_force_n=max((float(np.linalg.norm(r["grasp_world"][:2])) for r in robot_rows),default=0),
                fraction_corrected_horizontal_force_above_training_2n=float(np.mean([
                    np.linalg.norm(np.array(r["grasp_world"][:2])-np.array(r["bias_world"][:2]))>2 for r in moving])) if moving else None))
        report["interface"]=manifest
        (output / "evaluation.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        if trace:
            np.savez_compressed(output / "trace.npz",**{key:np.array([r[key] for r in trace]) for key in trace[0]})
        print(f"[REPORT] {reason} at {scene.data.time:.2f}s; {output / 'evaluation.json'}",flush=True)
    return output


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle_dir",type=Path,nargs="+",required=True)
    p.add_argument("--keyboard",action="store_true")
    p.add_argument("--headless",action="store_true")
    p.add_argument("--video",action="store_true")
    p.add_argument("--fps",type=int,default=30)
    p.add_argument("--trace",action="store_true")
    p.add_argument("--check_only",action="store_true")
    p.add_argument("--controller",choices=("admittance","hold"),default="admittance")
    p.add_argument("--duration",type=float,default=120.)
    p.add_argument("--settle_seconds",type=float,default=3.)
    p.add_argument("--calibration_seconds",type=float,default=1.)
    p.add_argument("--release_tail",type=float,default=3.)
    p.add_argument("--payload_mass",type=float,default=1.)
    p.add_argument("--spacing",type=float,default=1.)
    p.add_argument("--friction",type=float,default=.5)
    p.add_argument("--delay_steps",type=int,choices=range(5))
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--force_world",type=float,nargs=2,default=[0.,0.],metavar=("FX","FY"))
    p.add_argument("--yaw_torque",type=float,default=0.)
    p.add_argument("--force_step",type=float,default=.5)
    p.add_argument("--torque_step",type=float,default=.05)
    p.add_argument("--max_force",type=float,default=4.)
    p.add_argument("--max_torque",type=float,default=.2)
    p.add_argument("--force_slew",type=float,default=2.)
    p.add_argument("--torque_slew",type=float,default=.1)
    p.add_argument("--virtual_mass",type=float,default=4.)
    p.add_argument("--damping",type=float,default=8.)
    p.add_argument("--filter_tau",type=float,default=.3)
    p.add_argument("--max_speed",type=float,default=.15)
    p.add_argument("--max_yaw",type=float,default=.2)
    p.add_argument("--output_dir",type=Path)
    return p


if __name__=="__main__":
    run(parser().parse_args())
