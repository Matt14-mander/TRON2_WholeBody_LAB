"""Two-SFYG tandem / three-SFYG sofa: velocity leader, local FT followers."""
import argparse
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from sfyg_bridge import ROOT
from run_sfyg_cooperate import load_bundle, robot_state
from sfyg_cooperative_scene import CooperativeScene, grasp_reaction, audit_reactions
from sfyg_cooperative_control import calibration_quality
from sfyg_team_control import own_input, make_follower, checked_action, LeaderKeyboard, leader_command
from sfyg_team_visuals import draw_team


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def payload_floor_contact(scene):
    for contact in scene.data.contact:
        bodies=scene.model.geom_bodyid[[contact.geom1,contact.geom2]]
        if scene.payload in bodies and 0 in bodies: return True
    return False


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',choices=('leader-follow','sofa'),required=True)
    p.add_argument('--bundle_dir',type=Path,nargs='+',required=True)
    p.add_argument('--follower',choices=('admittance','hold','teacher','student'),default='admittance')
    p.add_argument('--teacher_bundle',type=Path,nargs='+',help='One shared teacher export or one per follower')
    p.add_argument('--teacher_input',choices=('grasp-proxy',),help='Required experimental input adaptation for shared teacher')
    p.add_argument('--follower_axes',choices=('forward','planar'),default='forward')
    p.add_argument('--hold_all',action='store_true',help='Zero commands to ALL robots for physical diagnosis')
    p.add_argument('--keyboard',action='store_true')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--video',action='store_true')
    p.add_argument('--trace',action='store_true')
    p.add_argument('--check_only',action='store_true')
    p.add_argument('--camera',choices=('follow','fixed'),default='follow')
    p.add_argument('--duration',type=float,default=30.)
    p.add_argument('--settle_seconds',type=float,default=3.)
    p.add_argument('--calibration_seconds',type=float,default=1.)
    p.add_argument('--stop_seconds',type=float,default=5.)
    p.add_argument('--leader_vx',type=float,default=.10)
    p.add_argument('--leader_max_speed',type=float,default=.15)
    p.add_argument('--leader_acceleration',type=float,default=.05)
    p.add_argument('--max_speed',type=float,default=.20,help='Follower planar speed limit')
    p.add_argument('--max_yaw',type=float,default=.20)
    p.add_argument('--virtual_mass',type=float,default=4.)
    p.add_argument('--damping',type=float,default=8.)
    p.add_argument('--filter_tau',type=float,default=.3)
    p.add_argument('--payload_mass',type=float,default=1.)
    p.add_argument('--spacing',type=float,help='Tandem longitudinal / sofa follower lateral separation')
    p.add_argument('--grasp_model',choices=('ball','weld'),default='ball',help='Follower connection; leader always ball')
    p.add_argument('--physics_substeps',type=int,choices=(1,2,4,8),default=4)
    p.add_argument('--delay_steps',type=int,choices=range(5),default=2)
    p.add_argument('--friction',type=float,default=.5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--fps',type=int,default=30)
    p.add_argument('--output_dir',type=Path)
    return p.parse_args(argv)


def run(args):
    count=2 if args.scenario=='leader-follow' else 3
    if len(args.bundle_dir) not in (1,count): raise ValueError('Provide one shared bundle or one per robot in role order')
    if args.keyboard and args.headless: raise ValueError('Keyboard needs the MuJoCo window')
    positive=[args.duration,args.settle_seconds,args.calibration_seconds,args.stop_seconds,
        args.leader_acceleration,args.leader_max_speed,args.max_speed,args.max_yaw,args.payload_mass,args.fps]
    if not np.isfinite(positive).all() or min(positive)<=0: raise ValueError('Positive finite timing/limits/mass required')
    if args.max_speed>.5 or args.leader_max_speed>.5 or args.max_yaw>1.5: raise ValueError('Commands exceed source bounds')
    if args.calibration_seconds>args.settle_seconds: raise ValueError('Calibration must fit inside settle interval')
    if not (args.check_only or args.hold_all) and (
            args.duration<=args.settle_seconds+args.stop_seconds+2
            or args.stop_seconds<args.leader_max_speed/args.leader_acceleration+1):
        raise ValueError('Allow motion after settling, and deceleration plus >=1s stationary leader tail')
    spacing=args.spacing if args.spacing is not None else (1.8 if count==2 else 1.1)
    keyboard=LeaderKeyboard(args.leader_vx,args.leader_max_speed)
    torch.set_num_threads(1)
    paths=args.bundle_dir*count if len(args.bundle_dir)==1 else args.bundle_dir
    loaded=[load_bundle(path) for path in args.bundle_dir]
    if len(loaded)==1: loaded*=count
    configs,policies=zip(*loaded)
    teacher_paths=args.teacher_bundle
    if args.follower=='teacher':
        if not teacher_paths or len(teacher_paths) not in (1,count-1) or args.teacher_input is None:
            raise ValueError('Teacher requires 1 or N-followers --teacher_bundle paths and --teacher_input grasp-proxy')
        if len(teacher_paths)==1: teacher_paths=teacher_paths*(count-1)
    elif teacher_paths or args.teacher_input:
        raise ValueError('Teacher options require --follower teacher')
    followers=[make_follower(args.follower,axes=args.follower_axes,mass=args.virtual_mass,
        damping=args.damping,filter_tau=args.filter_tau,max_speed=args.max_speed,max_yaw=args.max_yaw,
        teacher_bundle=teacher_paths[i] if teacher_paths else None,low_config=configs[i+1],teacher_input=args.teacher_input)
        for i in range(count-1)]
    if args.follower=='teacher':
        from sfyg_teacher_shared import validate_proxy_timing
        validate_proxy_timing(followers,args.settle_seconds,args.delay_steps)
    scene=CooperativeScene(configs,spacing,args.payload_mass,args.friction,args.delay_steps,args.seed,
        args.grasp_model,args.physics_substeps,layout=args.scenario)
    output=args.output_dir or ROOT/'logs/mujoco'/('sfyg_'+args.scenario+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    if max(scene.grasp_errors())>1e-6: raise ValueError('Initial FK alignment failed')
    residual=scene.data.efc_pos[scene.data.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY]
    if np.max(abs(residual))>1e-6: raise ValueError('Initial grasp constraint residual is not zero')
    dt=configs[0]['physics_dt']
    control_dt=dt*configs[0]['decimation']
    source_files=['sfyg_cooperative_scene.py','sfyg_cooperative_control.py','sfyg_team_scene.py',
        'sfyg_team_control.py','sfyg_team_visuals.py','run_sfyg_team.py','run_sfyg_cooperate.py','sfyg_bridge.py','sfyg_policy.py',
        'sfyg_teacher_shared.py','sfyg_teacher_policy.py']
    manifest=dict(schema='sfyg_team_demo_v1',scenario=args.scenario,is_acceptance=False,
        roles=['leader']+['follower']*(count-1),follower=args.follower,follower_axes=args.follower_axes,
        policies=[dict(path=str(p.resolve()),policy_sha256=c['policy_sha256'],checkpoint_sha256=c['checkpoint_sha256']) for p,c in zip(paths,configs)],
        total_mass_kg=float(scene.model.body_mass.sum()),payload_mass_kg=float(scene.model.body_mass[scene.payload]),
        payload_com_local=scene.model.body_ipos[scene.payload].tolist(),payload_inertia=scene.model.body_inertia[scene.payload].tolist(),
        free_roots=count+1,attachments=['rear_ball_hitch']+[args.grasp_model+'_arm_grip']*(count-1),
        massless_ideal_handles=True,world_support=False,external_applied_force=False,
        physics_dt=dt,solver_dt=scene.model.opt.timestep,control_dt=control_dt,
        robot_delay_steps=[r.delays for r in scene.robots],initial_grasp_errors_m=scene.grasp_errors(),
        high_interface=dict(version=1,input_semantics='own object-to-robot grasp wrench at own point, world axes; own base axes and arm state',
            output_semantics='own-base velocity3 m/s,rad/s; arm_offset6 rad must be zero',
            reserved_policies=['student'],learned_adapters_implemented=args.follower=='teacher',
            no_partner_state=True,no_leader_command=True,no_object_pose=True),
        teacher_proxy=[p.metadata() for p in followers] if args.follower=='teacher' else None,
        arguments={k:str(v) if isinstance(v,Path) else [str(p) for p in v] if k in ('bundle_dir','teacher_bundle') and v else v for k,v in vars(args).items()},
        sources={name:hashlib.sha256((ROOT/'scripts/mujoco'/name).read_bytes()).hexdigest() for name in source_files},
        mujoco_version=mujoco.__version__)
    write_json(output/'interface.json',manifest)
    print(f'[TEAM] {args.scenario}: {count} robots; free {args.payload_mass:g}kg load; '
          f'leader velocity / followers {args.follower}; output={output}',flush=True)
    if args.follower=='teacher': print('[TEACHER PROXY] EXPERIMENTAL: own grasp reaction replaces training external force; forward only; input clipped to 0--8N',flush=True)
    if args.check_only: return output
    print('[KEYS] Leader ONLY: W/S increase/decrease forward speed; X stop; R initial speed; P pause',flush=True)
    context=nullcontext(None)
    if not args.headless:
        from mujoco import viewer as mj_viewer
        context=mj_viewer.launch_passive(scene.model,scene.data,key_callback=keyboard.on_key if args.keyboard else None)
    camera=mujoco.MjvCamera()
    camera.distance,camera.elevation,camera.azimuth=6.,-25.,125.
    camera.lookat[:]=scene.data.xpos[scene.payload]+[0,0,-.25]
    renderer=writer=None
    rows,trace=[],[]
    leader=np.zeros(3)
    calibration=None
    initial=scene.data.xpos[scene.payload].copy()
    reason,error='completed',None
    tick,next_frame=0,0.
    try:
        if args.video:
            import imageio.v2 as imageio
            renderer=mujoco.Renderer(scene.model,height=720,width=960)
            writer=imageio.get_writer(str(output/'preview.mp4'),fps=args.fps,codec='libx264')
        with (output/'steps.jsonl').open('w',encoding='utf-8') as stream,context as viewer:
            while scene.data.time<args.duration-dt/2:
                start=time.monotonic()
                requested,paused=keyboard.snapshot()
                if viewer is not None and not viewer.is_running(): reason='viewer_closed';break
                if paused:
                    viewer.sync();time.sleep(.01);continue
                t=float(scene.data.time)
                settling=t<args.settle_seconds
                if not settling and calibration is None:
                    calibration=calibration_quality(rows,args.settle_seconds,args.calibration_seconds,control_dt)
                    print(f'[CALIBRATION] {calibration}',flush=True)
                    if not calibration['passed'] and not args.hold_all:
                        reason='unsettled_calibration';break
                stopping=t>=args.duration-args.stop_seconds
                phase='HOLD' if args.hold_all else 'SETTLE' if settling else 'STOP' if stopping else 'FOLLOW'
                reactions=[grasp_reaction(scene.model,scene.data,r.eq,scene.payload) for r in scene.robots]
                audit=audit_reactions(scene,reactions)
                leader=leader_command(leader,requested,control_dt,args.leader_acceleration,
                    settling or stopping or args.hold_all)
                commands=[leader.copy()]
                diagnostic=[dict(kind='leader_velocity',requested_vx=requested)]
                for robot,policy,reaction in zip(scene.robots[1:],followers,reactions[1:]):
                    observation=own_input(scene,robot,reaction,control_dt,settling,
                        settling and t>=args.settle_seconds-args.calibration_seconds)
                    action=policy.update(observation)
                    command=checked_action(action,args.max_speed,args.max_yaw)
                    commands.append(np.zeros(3) if args.hold_all else command)
                    diagnostic.append(policy.diagnostics())
                states=[robot_state(scene,r) for r in scene.robots]
                for i,(robot,policy,command) in enumerate(zip(scene.robots,policies,commands)):
                    obs,history=robot.observation()
                    with torch.inference_mode():
                        action=policy(torch.from_numpy(obs[None]),torch.from_numpy(history[None]),
                            torch.tensor(command[None],dtype=torch.float32))[0].cpu().numpy().copy()
                    robot.set_action(action)
                    if args.trace:
                        trace.append(dict(time_s=t,robot=i,observation=obs.copy(),history=history.copy(),command=command.copy(),action=action.copy(),
                            teacher_observation=followers[i-1].teacher.observation.copy() if args.follower=='teacher' and i else np.zeros(147,np.float32),
                            teacher_raw=followers[i-1].teacher.raw.copy() if args.follower=='teacher' and i else np.zeros(9,np.float32)))
                payload_rotation=scene.data.xmat[scene.payload].reshape(3,3)
                row=dict(time_s=t,phase=phase,leader_requested_vx=requested,
                    payload_position=scene.data.xpos[scene.payload].tolist(),payload_tilt_deg=float(np.rad2deg(np.arccos(np.clip(payload_rotation[2,2],-1,1)))),
                    payload_floor_contact=payload_floor_contact(scene),grasp_errors_m=scene.grasp_errors(),
                    grasp_angles_deg=scene.grasp_angles(),applied_world_wrench=[0.]*6,
                    extraction_error=audit['extraction_max'],dynamics_residual=audit['dynamics_max'],
                    robots=[dict(**state,command=cmd.tolist(),grasp_world=wrench.tolist(),controller=diag)
                        for state,cmd,wrench,diag in zip(states,commands,reactions,diagnostic)])
                stream.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
                if len(rows)%50==1:
                    print(f'[TEAM] t={t:.2f} {phase} cmd={np.round(commands,3).tolist()} '
                        f'actual={[np.round(s["velocity_vx_vy_wz"],3).tolist() for s in states]}',flush=True)
                for _ in range(configs[0]['decimation']):
                    scene.step(np.zeros(6));tick+=1
                for robot in scene.robots: robot.control_steps+=1
                mujoco.mj_forward(scene.model,scene.data)
                if args.camera=='follow': camera.lookat[:]=scene.data.xpos[scene.payload]+[0,0,-.25]
                positions=[scene.data.xpos[r.base] for r in scene.robots]
                display_controller='T-PROXY EXP' if args.follower=='teacher' else args.follower
                draw_args=(positions,scene.data.xpos[scene.payload],commands,reactions,phase,args.payload_mass,display_controller)
                if writer is not None and scene.data.time>=next_frame:
                    renderer.update_scene(scene.data,camera)
                    draw_team(renderer.scene,*draw_args)
                    frame=renderer.render()
                    if frame.std()<1: raise RuntimeError('Empty video frame')
                    writer.append_data(frame);next_frame+=1/args.fps
                if viewer is not None:
                    with viewer.lock():
                        if args.camera=='follow': viewer.cam.lookat[:]=camera.lookat
                        if len(rows)==1:
                            viewer.cam.lookat[:]=camera.lookat
                            viewer.cam.distance,viewer.cam.elevation,viewer.cam.azimuth=camera.distance,camera.elevation,camera.azimuth
                        viewer.user_scn.ngeom=0
                        draw_team(viewer.user_scn,*draw_args)
                    viewer.sync()
                    time.sleep(max(0.,control_dt-(time.monotonic()-start)))
                if not all(np.isfinite(v).all() for v in (scene.data.qpos,scene.data.qvel,scene.data.qacc)):
                    reason='nonfinite_state';break
                if scene.data.time<tick*dt-1e-6: reason='simulator_reset';break
                terminal=[robot_state(scene,r) for r in scene.robots]
                if any(s['tilt_deg']>45 or s['position'][2]<.45 for s in terminal): reason='fall';break
                if max(scene.grasp_errors())>.1: reason='grasp_error';break
                if payload_floor_contact(scene): reason='payload_floor_contact';break
    except BaseException as exc:
        reason,error='error',repr(exc)
        raise
    finally:
        if writer is not None: writer.close()
        if renderer is not None: renderer.close()
        terminal=[robot_state(scene,r) for r in scene.robots]
        report=dict(schema='sfyg_team_report_v1',is_acceptance=False,reason=reason,error=error,
            completed=reason=='completed' and scene.data.time>=args.duration-dt,
            duration_s=float(scene.data.time),calibration=calibration,interface=manifest,
            payload_displacement_world_m=(scene.data.xpos[scene.payload]-initial).tolist(),
            payload_floor_contact=payload_floor_contact(scene) or any(r['payload_floor_contact'] for r in rows),
            max_grasp_error_m=max([*scene.grasp_errors(),*[max(r['grasp_errors_m']) for r in rows]]),
            max_extraction_error=max((r['extraction_error'] for r in rows),default=0.),
            max_dynamics_residual=max((r['dynamics_residual'] for r in rows),default=0.),robots=[])
        terminal_payload_tilt=float(np.rad2deg(np.arccos(np.clip(scene.data.xmat[scene.payload].reshape(3,3)[2,2],-1,1))))
        report['max_payload_tilt_deg']=max([terminal_payload_tilt,*[r['payload_tilt_deg'] for r in rows]])
        for i in range(count):
            samples=[r['robots'][i] for r in rows]
            tail=[r['robots'][i] for r in rows if r['time_s']>=args.duration-1]
            moving=[r['robots'][i] for r in rows if args.settle_seconds<=r['time_s']<args.duration-args.stop_seconds]
            distances=[float(np.linalg.norm(np.array(r['robots'][i]['position'])[:2]-np.array(r['robots'][0]['position'])[:2])) for r in rows]
            report['robots'].append(dict(role=manifest['roles'][i],terminal=terminal[i],
                max_tilt_deg=max([terminal[i]['tilt_deg'],*[s['tilt_deg'] for s in samples]]),
                peak_arm_error_rad=max([terminal[i]['peak_arm_error_rad'],*[s['peak_arm_error_rad'] for s in samples]]),
                peak_horizontal_grasp_force_n=max((float(np.linalg.norm(s['grasp_world'][:2])) for s in samples),default=0.),
                leader_distance_range_m=[min(distances),max(distances)] if distances and i else None,
                moving_mean_velocity=np.mean([s['velocity_vx_vy_wz'] for s in moving],axis=0).tolist() if moving else None,
                tail_velocity_rms=np.sqrt(np.mean(np.square([s['velocity_vx_vy_wz'] for s in tail]),axis=0)).tolist() if tail else None))
            if args.follower=='teacher' and i: report['robots'][-1]['teacher_proxy']=followers[i-1].diagnostics()
        write_json(output/'evaluation.json',report)
        if trace: np.savez_compressed(output/'trace.npz',**{k:np.array([r[k] for r in trace]) for k in trace[0]})
        print(f'[REPORT] {reason} at {scene.data.time:.2f}s: {output / "evaluation.json"}',flush=True)
    return output


if __name__=='__main__':
    output=run(parse_args())
    report_path=output/'evaluation.json'
    if report_path.exists() and not json.loads(report_path.read_text(encoding='utf-8'))['completed']:
        raise SystemExit(2)
