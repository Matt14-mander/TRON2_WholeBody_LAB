"""Single-robot MuJoCo playback of the current frozen SFYG + privileged T1 teacher."""
import argparse
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from sfyg_bridge import ROOT,SFYGBridge
from run_sfyg_cooperate import load_bundle
from sfyg_teacher_policy import TeacherPolicy
from sfyg_teacher_core import reward_terms
from sfyg_teacher_evaluation import summarize


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def physics_tick(bridge,force_base):
    """One unchanged 5ms PD/queue tick; force at payload COM rotates with BASE."""
    force=np.asarray(force_base,float)
    if force.shape!=(3,) or not np.isfinite(force).all(): raise ValueError('Finite applied force3 required')
    bridge.write_controls()
    bridge.data.xfrc_applied[:]=0
    world=bridge.data.xmat[bridge.base].reshape(3,3)@force
    bridge.data.xfrc_applied[bridge.payload,:3]=world
    mujoco.mj_step(bridge.model,bridge.data)
    mujoco.mj_forward(bridge.model,bridge.data)
    return world


def contact_failure(bridge):
    # Sum contact forces per body, then use the teacher's 1N body threshold.
    net=np.zeros((bridge.model.nbody,3))
    for c in range(bridge.data.ncon):
        contact=bridge.data.contact[c]
        force=np.zeros(6)
        mujoco.mj_contactForce(bridge.model,bridge.data,c,force)
        world=contact.frame.reshape(3,3).T@force[:3]
        a,b=bridge.model.geom_bodyid[[contact.geom1,contact.geom2]]
        net[a]-=world;net[b]+=world
    base=['base_Link','proximal_pitch_L_Link','proximal_pitch_R_Link']
    arm=[*[f'arm{i}_Link' for i in range(1,7)],'gripper_base_Link','gripper1_Link','gripper2_Link','paint_payload_Link']
    for label,names in (('base_contact',base),('arm_payload_contact',arm)):
        if any(np.linalg.norm(net[bridge.model.body(name).id])>1 for name in names): return label
    return None


def draw(scene,position,force,world_force,force_point,command,velocity,time_s):
    def geom(kind,pos,color):
        if scene.ngeom>=scene.maxgeom: return None
        g=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g,kind,np.zeros(3),np.asarray(pos,float),np.eye(3).ravel(),np.array(color,np.float32))
        scene.ngeom+=1
        return g
    pos=np.asarray(position)+[0,0,.9]
    g=geom(mujoco.mjtGeom.mjGEOM_LABEL,pos,[1,.85,.4,1])
    if g is not None: g.label=f'TEACHER t={time_s:.1f}s F={force[0]:.1f}N cmd={command[0]:.2f} vx={velocity[0]:.2f}'
    if force[0]>.01:
        start=np.asarray(force_point)
        g=geom(mujoco.mjtGeom.mjGEOM_ARROW,start,[1,.3,.1,1])
        direction=np.asarray(world_force)/np.linalg.norm(world_force)
        if g is not None: mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,.018,start,start+direction*min(.6,force[0]*.06))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle_dir',type=Path,required=True,help='Existing frozen SFYG low-level export')
    p.add_argument('--teacher_bundle',type=Path,required=True,help='Existing teacher exported/model_5000 directory')
    p.add_argument('--force_n',type=float,default=8.)
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--video',action='store_true',help='Optional MuJoCo recording; does not use Isaac rendering')
    p.add_argument('--trace',action='store_true')
    p.add_argument('--fps',type=int,default=30)
    p.add_argument('--check_only',action='store_true')
    p.add_argument('--smoke_steps',type=int,help='Development only, <=100 steps; never a complete episode')
    p.add_argument('--output_dir',type=Path)
    return p.parse_args(argv)


def run(args):
    if args.episodes<1 or args.fps not in range(1,51): raise ValueError('Positive episodes and fps 1--50 required')
    if args.smoke_steps is not None and args.smoke_steps not in range(1,101): raise ValueError('Smoke is limited to 1--100 steps')
    torch.set_num_threads(1)
    cfg,low=load_bundle(args.bundle_dir)
    teacher=TeacherPolicy(args.teacher_bundle,cfg)
    settings=teacher.settings
    teacher.force(0,args.force_n)  # Validate requested force before creating a world.
    bridge=SFYGBridge(cfg,settings.payload_mass,cfg['friction_coefficient'],settings.delay_steps,args.seed)
    audit=bridge.audit()
    if max(audit[k] for k in ('authored_mass_error_kg','authored_com_error_m','authored_inertia_error'))>1e-6:
        raise ValueError('Original robot mass/COM/inertia audit failed')
    output=args.output_dir or ROOT/'logs/mujoco'/('sfyg_teacher_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    interface=dict(schema='sfyg_teacher_mujoco_v1',is_acceptance=False,teacher_contract=teacher.manifest,
        teacher_policy_sha256=teacher.sha,low_policy_sha256=cfg['policy_sha256'],low_checkpoint_sha256=cfg['checkpoint_sha256'],
        teacher_bundle=str(args.teacher_bundle.resolve()),low_bundle=str(args.bundle_dir.resolve()),
        force_input='known external force at own payload COM, BASE axes /40; NOT measured grasp reaction',
        physical_force='amplitude held per control interval; re-rotated from BASE every physics tick; zero torque',
        teacher_hz=50,low_hz=50,physics_hz=200,bridge=audit,mujoco_version=mujoco.__version__,
        smoke_only=args.smoke_steps is not None,force_n=args.force_n,
        limitations=['Single robot privileged teacher only; no student or shared-load controller',
            'PhysX/MuJoCo contact and implicit PD differ; use as migration diagnostic'])
    write_json(output/'interface.json',interface)
    print(f'[MJ TEACHER] actor147 -> raw9 -> vx -> frozen SFYG; F={args.force_n:g}N; '
        f'payload={settings.payload_mass:g}kg; output={output}',flush=True)
    if args.check_only: return output
    context=nullcontext(None)
    if not args.headless:
        from mujoco import viewer as mj_viewer
        context=mj_viewer.launch_passive(bridge.model,bridge.data)
    camera=mujoco.MjvCamera()
    camera.distance,camera.elevation,camera.azimuth=3.5,-20.,135.
    renderer=writer=None
    reports,trace=[],[]
    reason,error='completed',None
    previous_raw=np.zeros(9,np.float32)
    global_step=0
    next_frame=0.
    try:
        if args.video:
            import imageio.v2 as imageio
            renderer=mujoco.Renderer(bridge.model,height=720,width=960)
            writer=imageio.get_writer(str(output/'preview.mp4'),fps=args.fps,codec='libx264')
        with (output/'steps.jsonl').open('w',encoding='utf-8') as stream,context as viewer,torch.inference_mode():
            for episode in range(args.episodes):
                bridge.reset();teacher.reset();previous_raw[:]=0
                initial=bridge.data.xpos[bridge.base].copy()
                rows=[]
                horizon=round(settings.horizon/settings.dt)
                steps=args.smoke_steps or horizon
                for tick in range(steps):
                    start=time.monotonic()
                    if viewer is not None and not viewer.is_running(): reason='viewer_closed';break
                    force=teacher.force(tick,args.force_n)
                    arm=bridge.data.qpos[bridge.qpos[10:16]]-bridge.defaults[10:16]
                    dq=bridge.data.qvel[bridge.qvel[10:16]]
                    previous_target=bridge.target[10:16]-bridge.defaults[10:16]
                    command=teacher.command(arm,dq,previous_target,force,tick)
                    obs,history=bridge.observation()
                    action=low(torch.from_numpy(obs[None]),torch.from_numpy(history[None]),
                        torch.from_numpy(command[None]))[0].numpy().copy()
                    bridge.set_action(action)
                    for _ in range(cfg['decimation']): world_force=physics_tick(bridge,force)
                    bridge.control_steps+=1
                    global_step+=1
                    finite=all(np.isfinite(v).all() for v in (bridge.data.qpos,bridge.data.qvel,bridge.data.qacc))
                    if not finite: raise FloatingPointError('Nonfinite MuJoCo state')
                    linear,angular,gravity=bridge.base_state()
                    actual=np.r_[linear[:2],angular[2]].astype(np.float32)
                    position=bridge.data.xpos[bridge.base].copy()
                    tilt=float(np.rad2deg(np.arccos(np.clip(-gravity[2],-1,1))))
                    terminal=None
                    if bridge.data.time<bridge.control_steps*settings.dt-1e-6: terminal='simulator_reset'
                    elif tilt>35 or position[2]<.45: terminal='fall'
                    else: terminal=contact_failure(bridge)
                    failed=terminal is not None
                    timeout=tick+1==horizon and not failed
                    reference=force/40.
                    terms=reward_terms(torch.from_numpy(actual[None]),torch.from_numpy(reference[None]),
                        torch.from_numpy(command[None]),torch.from_numpy(teacher.raw[None]),
                        torch.from_numpy(previous_raw[None]),torch.tensor(gravity[None],dtype=torch.float32))
                    costs={k:float(v[0])*settings.dt for k,v in terms.items()}
                    costs['failure']=-50. if failed else 0.
                    row=dict(episode=episode,environment=0,controller='teacher',force_n=args.force_n,
                        time_s=float(bridge.data.time),velocity=actual.tolist(),command=command.tolist(),reference=reference.tolist(),
                        raw_action=teacher.raw.tolist(),arm_error=(bridge.data.qpos[bridge.qpos[10:16]]-bridge.defaults[10:16]).tolist(),
                        tilt_deg=tilt,position=position.tolist(),force_base=force.tolist(),last_applied_world_force=world_force.tolist(),
                        failed=failed,timeout=timeout,termination=terminal,rewards=costs,reward=sum(costs.values()))
                    rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                    if args.trace: trace.append(dict(episode=episode,time_s=row['time_s'],teacher_observation=teacher.observation.copy(),
                        low_observation=obs.copy(),low_history=history.copy(),command=command.copy(),raw=teacher.raw.copy(),leg_action=action.copy()))
                    previous_raw[:]=teacher.raw
                    if tick%50==0:
                        print(f'[MJ TEACHER] ep={episode} t={row["time_s"]:.2f} F={force[0]:.2f} '
                            f'cmd={command[0]:+.3f} actual={np.round(actual,3)} tilt={tilt:.2f}',flush=True)
                    camera.lookat[:]=position+[0,0,.15]
                    draw_args=(position,force,world_force,bridge.data.xipos[bridge.payload],command,actual,row['time_s'])
                    if writer is not None and global_step*settings.dt>=next_frame:
                        renderer.update_scene(bridge.data,camera);draw(renderer.scene,*draw_args)
                        frame=renderer.render()
                        if frame.std()<1: raise RuntimeError('Empty MuJoCo frame')
                        writer.append_data(frame);next_frame+=1/args.fps
                    if viewer is not None:
                        with viewer.lock():
                            viewer.cam.lookat[:]=camera.lookat
                            if tick==0 and episode==0:
                                viewer.cam.distance,viewer.cam.elevation,viewer.cam.azimuth=camera.distance,camera.elevation,camera.azimuth
                            viewer.user_scn.ngeom=0;draw(viewer.user_scn,*draw_args)
                        viewer.sync();time.sleep(max(0.,settings.dt-(time.monotonic()-start)))
                    if failed: reason=terminal;break
                result=summarize(rows,settings,args.force_n,'teacher')
                result.update(episode=episode,environment=0,displacement_world_m=(bridge.data.xpos[bridge.base]-initial).tolist())
                reports.append(result)
                if reason!='completed': break
    except BaseException as exc:
        reason,error='error',repr(exc)
        raise
    finally:
        if writer is not None: writer.close()
        if renderer is not None: renderer.close()
        complete=bool(len(reports)==args.episodes and all(r['completed'] for r in reports) and reason=='completed')
        write_json(output/'evaluation.json',dict(schema='sfyg_teacher_mujoco_report_v1',is_acceptance=False,
            completed=complete,reason='smoke_only' if args.smoke_steps and reason=='completed' else reason,
            error=error,interface=interface,reports=reports))
        if trace: np.savez_compressed(output/'trace.npz',**{k:np.array([r[k] for r in trace]) for k in trace[0]})
        print(f'[REPORT] {reason}; {output / "evaluation.json"}',flush=True)
    return output


if __name__=='__main__':
    output=run(parse_args())
    report=output/'evaluation.json'
    if report.exists():
        result=json.loads(report.read_text(encoding='utf-8'))
        if not result['completed'] and result['reason']!='smoke_only': raise SystemExit(2)
