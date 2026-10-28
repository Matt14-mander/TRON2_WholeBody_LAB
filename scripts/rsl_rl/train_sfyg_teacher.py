"""Independent SFYG T0/T1 teacher; original locomotion/PPO sources stay intact."""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'rsl_rl'),str(ROOT/'exts/bipedal_locomotion'),str(ROOT/'scripts/mujoco')]


def parser_for_task():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('smoke','baseline','train','eval'),default='baseline')
    p.add_argument('--low_checkpoint',required=True)
    p.add_argument('--teacher_checkpoint',help='Strict evaluation checkpoint; T1 training starts fresh')
    p.add_argument('--baseline_report',help='Passing T0 baseline evaluation.json from this exact contract')
    p.add_argument('--development',action='store_true',help='Allow at most 2 PPO updates / 16 envs without T0 acceptance')
    p.add_argument('--controller',choices=('hold','reference','probe','teacher'),default='reference')
    p.add_argument('--force_n',type=float,default=8.,help='Fixed eval force on SINGLE robot payload, 0--8 N')
    p.add_argument('--probe_vx',type=float,default=.1)
    p.add_argument('--num_envs',type=int)
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--steps',type=int,default=128,help='Smoke only; never used to accept T0')
    p.add_argument('--max_iterations',type=int,default=5000)
    p.add_argument('--save_interval',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--delay_steps',type=int,choices=range(5),default=2)
    p.add_argument('--output_dir')
    p.add_argument('--video',action='store_true',help='One-env eval only')
    return p


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=str,allow_nan=False),encoding='utf-8')


def warm_camera(env):
    frame = None
    for _ in range(60):
        frame = env.render(recompute=True)
    if frame is None or float(frame.mean())<1. or float(frame.std())<1.:
        raise RuntimeError('Camera is empty/black after warmup; inspect renderer before recording')
    return frame


def evaluate(env,args,output,app,settings):
    import torch
    from sfyg_teacher_env import controller
    from sfyg_teacher_core import check_teacher
    from sfyg_teacher_evaluation import summarize,baseline_gate
    term = controller(env)
    actor = None
    if args.mode=='eval' and args.controller=='teacher':
        from rsl_rl.modules import ActorCritic
        saved = torch.load(args.teacher_checkpoint,map_location=env.device,weights_only=False)
        check_teacher(saved,args.contract)
        actor = ActorCritic(147,177,9,actor_hidden_dims=[128]*3,critic_hidden_dims=[128]*3,activation='elu').to(env.device)
        actor.load_state_dict(saved['model_state_dict'],strict=True)
        actor.eval()
    if args.mode=='baseline':
        cases = [(mode,force,0.) for force in (0.,2.,4.,6.,8.) for mode in ('hold','reference')]
        cases += [('probe',0.,v) for v in (.05,.1,.15,.2)]
    else:
        cases = [(args.controller,args.force_n,args.probe_vx)]
    reports = []
    video = None
    if args.video:
        import imageio.v2 as imageio
        warm_camera(env)
        video = imageio.get_writer(str(output/'preview.mp4'),fps=50,codec='libx264')
    try:
        with (output/'steps.jsonl').open('w',encoding='utf-8') as stream,torch.inference_mode():
            for mode,force,probe in cases:
                env.cfg.fixed_force = force
                for episode in range(args.episodes):
                    obs,_ = env.reset(seed=args.seed+episode)
                    if video is not None: warm_camera(env)
                    if env.cfg.fixed_force != force or not bool((term.target_force==force).all()):
                        raise RuntimeError('Fixed force was not reset correctly')
                    alive = torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
                    rows = [[] for _ in range(env.num_envs)]
                    steps = args.steps if args.mode=='smoke' else env.max_episode_length
                    for step in range(steps):
                        if not app.is_running(): raise RuntimeError('Simulator interrupted before evaluation completed')
                        action = torch.zeros(env.num_envs,9,device=env.device)
                        reference = term.reference()
                        if mode=='reference': action[:,:3] = reference
                        elif mode=='probe':
                            action[:,0] = probe*__import__('sfyg_teacher_core').envelope(term.time(),settings)
                        elif mode=='teacher': action = actor.act_inference(obs['policy'])
                        obs,reward,failed,timeout,info = env.step(action)
                        if not all(torch.isfinite(v).all() for v in (*obs.values(),reward,*info['teacher_state'].values())):
                            raise FloatingPointError('Nonfinite teacher evaluation')
                        states = {k:v.cpu().tolist() for k,v in info['teacher_state'].items()}
                        costs = {k:v.cpu().tolist() for k,v in info['teacher_rewards'].items()}
                        for index in alive.nonzero().flatten().tolist():
                            row = {k:v[index] for k,v in states.items()}
                            row.update(controller=mode,force_n=force,probe_vx=probe,episode=episode,environment=index,
                                reward=float(reward[index]),failed=bool(failed[index]),timeout=bool(timeout[index] and not failed[index]),
                                rewards={k:v[index] for k,v in costs.items()})
                            rows[index].append(row)
                            stream.write(json.dumps(row,allow_nan=False)+'\n')
                        if video is not None: video.append_data(env.render())
                        if step%250==0:
                            print(f'[T0] {mode} F={force:g}N t={(step+1)*settings.dt:.2f} '
                                  f'v={states["velocity"][0]} cmd={states["command"][0]}',flush=True)
                        alive &= ~(failed|timeout)
                        if not alive.any(): break
                    for index,data in enumerate(rows):
                        result = summarize(data,settings,force,mode,probe)
                        result.update(episode=episode,environment=index)
                        reports.append(result)
                        print(f'[T0 CASE] {mode} F={force:g}N env={index} '
                              f'complete={result["completed"]} gate={result["gate"]}',flush=True)
    finally:
        if video is not None: video.close()
    gate = baseline_gate(reports) if args.mode=='baseline' else None
    result = dict(schema='sfyg_t0_report_v1',mode=args.mode,contract=args.contract,
        seed=args.seed,is_acceptance=args.mode=='baseline',gate=gate,reports=reports)
    write_json(output/'evaluation.json',result)
    print(f'[T0 REPORT] {output / "evaluation.json"}; gate={gate}',flush=True)
    return result


def validate_baseline(path,contract):
    from sfyg_teacher_evaluation import baseline_gate
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    if (report.get('schema')!='sfyg_t0_report_v1' or report.get('mode')!='baseline'
            or report.get('contract')!=contract or not report.get('frozen_weights_verified')
            or not baseline_gate(report.get('reports',[]))['passed']):
        raise ValueError('T0 baseline missing, failed, incomplete, or bound to another low-level/settings contract')


def main(args,app):
    import torch
    from sfyg_teacher_core import (Settings,runner_config,execution_contract,fingerprint,
        initialize_teacher,export_teacher)
    from sfyg_teacher_env import TeacherEnv,TeacherEnvCfg,controller
    from bipedal_locomotion.controllers.sfyg_reset import verify_source
    from bipedal_locomotion.controllers.paint_payload import create_payload_asset
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runner import OnPolicyRunner
    settings = Settings(delay_steps=args.delay_steps).validate()
    count = args.num_envs or (512 if args.mode=='train' else 1 if args.mode=='eval' else 4)
    if min(count,args.episodes,args.steps,args.max_iterations,args.save_interval)<1:
        raise ValueError('Positive counts required')
    if not 0<=args.force_n<=8 or not 0<args.probe_vx<=.3:
        raise ValueError('Finite forward force 0--8 N and probe velocity (0,.3] required')
    if args.controller=='probe' and args.force_n!=0 and args.mode in ('eval','smoke'):
        raise ValueError('Probe requires --force_n 0')
    if args.teacher_checkpoint and not (args.mode=='eval' and args.controller=='teacher'):
        raise ValueError('Teacher checkpoint is evaluation-only; T1 training starts from scratch')
    if args.mode=='eval' and args.controller=='teacher' and not args.teacher_checkpoint:
        raise ValueError('Teacher evaluation requires --teacher_checkpoint')
    if args.mode=='smoke' and args.controller=='teacher':
        raise ValueError('Use eval for a trained teacher')
    if args.development and (args.mode!='train' or count>16 or args.max_iterations>2):
        raise ValueError('Development bypass is limited to train <=2 updates / <=16 envs')
    if args.mode=='train' and not args.development and not args.baseline_report:
        raise ValueError('Production training requires the matching passing --baseline_report')
    if args.video and (args.mode!='eval' or count!=1):
        raise ValueError('Video requires one-env eval')
    if args.mode=='smoke' and count<2:
        raise ValueError('Smoke requires >=2 environments to test partial reset isolation')
    core = verify_source(ROOT)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir or ROOT/'logs/rsl_rl'/('sfyg_teacher_'+args.mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    cfg = TeacherEnvCfg()
    cfg.teacher_settings = asdict(settings)
    for actuator in cfg.scene.robot.actuators.values():
        if hasattr(actuator,'min_delay'): actuator.min_delay=actuator.max_delay=args.delay_steps
    cfg.scene.num_envs,cfg.sim.device,cfg.seed = count,args.device,args.seed
    cfg.actions.transport.low_path = str(Path(args.low_checkpoint).resolve())
    cfg.scene.robot.spawn.usd_path = create_payload_asset(cfg.scene.robot.spawn.usd_path,
        output/'robot_payload.usdc',mass=settings.payload_mass,dimensions=(.20,.12,.08),offset=(.15,0.,0.))
    env = TeacherEnv(cfg=cfg,render_mode='rgb_array' if args.video else None)
    runner = None
    try:
        term = controller(env)
        low_before = fingerprint(term.low)
        contract = execution_contract(term.low_sha,term.low_cfg,settings)
        args.contract = contract
        obs,_ = env.reset(seed=args.seed)
        shapes = {k:list(v.shape[1:]) for k,v in obs.items()}
        if shapes != dict(policy=[147],critic=[177],commands=[0],obsHistory=[147]):
            raise ValueError(f'Unexpected observation interface {shapes}')
        actual_mass = term._asset.root_physx_view.get_masses()[:,term.payload_id]
        if not torch.allclose(actual_mass,torch.full_like(actual_mass,.5),atol=1e-5):
            raise ValueError('Nominal physical payload is not 0.5 kg')
        write_json(output/'contract.json',contract)
        manifest = dict(contract=contract,arguments={k:v for k,v in vars(args).items() if k!='contract'},
            agent=runner_config(),environment=cfg.to_dict(),core_files=core,shapes=shapes,
            total_mass_kg=term._asset.root_physx_view.get_masses().sum(-1).tolist(),sources={})
        sources = [*core,*[f'scripts/rsl_rl/{name}' for name in ('train_sfyg_teacher.py','sfyg_teacher_core.py',
            'sfyg_teacher_env.py','sfyg_teacher_evaluation.py','probe_sfyg_teacher.py','probe_sfyg_robust.py',
            'check_sfyg_teacher_run.py','train_sfyg_teacher_t1.sh')],
            'scripts/mujoco/sfyg_policy.py',
            'exts/bipedal_locomotion/bipedal_locomotion/controllers/paint_payload.py',
            'exts/bipedal_locomotion/bipedal_locomotion/controllers/sfyg_reset.py',
            'exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_sfyg_carry_env_cfg.py']
        for name in sources:
            target = output/'sources'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
            manifest['sources'][name] = hashlib.sha256(target.read_bytes()).hexdigest()
        write_json(output/'run.json',manifest)
        print(f'[TEACHER] T1 forward, held arm, 0.5kg; shapes={shapes}; low_sha={term.low_sha}',flush=True)
        if args.mode=='train':
            if not args.development: validate_baseline(args.baseline_report,contract)
            wrapped = RslRlVecEnvWrapper(env,clip_actions=None)
            class TeacherRunner(OnPolicyRunner):
                def log(self,locs,*a,**kw):
                    self.completed_updates = locs['it']+1
                    super().log(locs,*a,**kw)
                    for key,value in env.pop_metrics().items(): self.writer.add_scalar(key,value,locs['it'])
                def save(self,path,infos=None):
                    if fingerprint(term.low)!=low_before: raise RuntimeError('Frozen low-level weights changed')
                    torch.save(dict(model_state_dict=self.alg.actor_critic.state_dict(),
                        optimizer_state_dict=self.alg.optimizer.state_dict(),
                        iter=getattr(self,'completed_updates',0),sfyg_teacher_contract=contract,
                        learning_rate=self.alg.learning_rate,development_only=args.development),path)
            rcfg = runner_config()
            rcfg['save_interval'] = args.save_interval
            runner = TeacherRunner(wrapped,deepcopy(rcfg),log_dir=str(output),device=args.device)
            initialize_teacher(runner.alg.actor_critic)
            if runner.alg.extra_optimizer is not None or runner.alg.early_stop:
                raise RuntimeError('Teacher must use plain PPO without estimator or early stopping')
            low_ids = {id(p) for p in term.low.parameters()}
            if any(id(p) in low_ids for group in runner.alg.optimizer.param_groups for p in group['params']):
                raise RuntimeError('Low-level parameters entered the teacher optimizer')
            print(f'[TEACHER PPO] std={runner.alg.actor_critic.logstd.exp().tolist()} '
                  f'lr={runner.alg.learning_rate} early_stop={runner.alg.early_stop}',flush=True)
            runner.learn(args.max_iterations,init_at_random_ep_len=False)
            export_teacher(runner.alg.actor_critic,output/'exported'/f'model_{args.max_iterations}',contract)
        else:
            if args.mode=='smoke':
                from probe_sfyg_teacher import audit
                write_json(output/'interface_audit.json',audit(env))
            result = evaluate(env,args,output,app,settings)
            result['frozen_weights_verified'] = fingerprint(term.low)==low_before
            write_json(output/'evaluation.json',result)
        unchanged = fingerprint(term.low)==low_before
        if not unchanged: raise RuntimeError('Frozen low-level weights changed')
        verify_source(ROOT)
        write_json(output/'completion.json',dict(completed=True,mode=args.mode,frozen_weights_verified=True,
            source_core_verified=True,development_only=args.development,
            baseline_gate=result.get('gate') if args.mode!='train' else None))
        if args.mode=='baseline' and not result['gate']['passed']:
            raise RuntimeError('T0 physical/reward gates failed. Inspect evaluation.json; no teacher training authorized by this report.')
    finally:
        if runner is not None and runner.writer is not None: runner.writer.close()
        env.close()


if __name__=='__main__':
    from isaaclab.app import AppLauncher
    parser = parser_for_task()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.video: args.enable_cameras=True
    launcher = AppLauncher(args)
    status = 0
    try:
        main(args,launcher.app)
    except BaseException:
        traceback.print_exc()
        status = 1
    finally:
        # Kit fast shutdown can terminate Python before exception propagation.
        # Explicitly set its return code; the Bash pipeline also checks artifacts.
        launcher.app.app.post_quit(status)
        launcher.app.close()
    raise SystemExit(status)
