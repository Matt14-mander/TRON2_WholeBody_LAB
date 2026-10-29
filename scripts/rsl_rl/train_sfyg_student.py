"""Independent SFYG Student training/evaluation over the unchanged T1 physics."""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'rsl_rl'),str(ROOT/'exts/bipedal_locomotion'),str(ROOT/'scripts/mujoco')]


def parser_for_task():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('train','eval'),default='train')
    p.add_argument('--low_checkpoint',required=True)
    p.add_argument('--teacher_checkpoint',required=True)
    p.add_argument('--student_checkpoint')
    p.add_argument('--resume',action='store_true',help='Restore both optimizers; iterations below are ADDITIONAL')
    p.add_argument('--development',action='store_true',help='Train <=2 updates and <=16 envs')
    p.add_argument('--num_envs',type=int,default=256)
    p.add_argument('--max_iterations',type=int,default=2000)
    p.add_argument('--save_interval',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--forces',type=float,nargs='+',default=[0.,2.,4.,6.,8.])
    p.add_argument('--controllers',nargs='+',choices=('student','teacher','reference','hold'),default=['student','teacher'])
    p.add_argument('--distill_coef',type=float,default=.001)
    p.add_argument('--learning_rate',type=float,default=1e-4)
    p.add_argument('--estimator_learning_rate',type=float,default=1e-3)
    p.add_argument('--output_dir',required=True)
    return p


def validate(args):
    if min(args.num_envs,args.max_iterations,args.save_interval,args.episodes) <= 0:
        raise ValueError('Positive counts required')
    if args.resume and (args.mode!='train' or not args.student_checkpoint):
        raise ValueError('Resume requires train and student checkpoint')
    if args.mode=='eval' and not args.student_checkpoint:
        raise ValueError('Evaluation requires student checkpoint')
    if args.development and (args.mode!='train' or args.num_envs>16 or args.max_iterations>2):
        raise ValueError('Development requires train <=2 updates / <=16 envs')
    if not args.forces or any(not 0<=f<=8 for f in args.forces):
        raise ValueError('Evaluation forces must be within 0--8 N')
    if len(set(args.forces))!=len(args.forces) or len(set(args.controllers))!=len(args.controllers):
        raise ValueError('Duplicate evaluation cases')


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=str),encoding='utf-8')


def evaluate(env, runner, args, app, settings, output):
    import torch
    from sfyg_teacher_core import process_action
    from sfyg_student_core import estimator_metrics
    from sfyg_teacher_evaluation import summarize
    reports = []
    runner.policy.eval()
    with (output/'steps.jsonl').open('w',encoding='utf-8') as stream,torch.inference_mode():
        for mode in args.controllers:
            for force in args.forces:
                env.cfg.fixed_force = force
                for episode in range(args.episodes):
                    obs,_ = env.reset(seed=args.seed+episode)
                    alive = torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
                    rows = [[] for _ in range(env.num_envs)]
                    for step in range(env.max_episode_length):
                        if not app.is_running(): raise RuntimeError('Evaluation interrupted')
                        pre = obs['policy'].clone()
                        estimate = runner.policy.estimate(pre[:,:144])
                        teacher_raw = runner.teacher.actor(pre)
                        student_raw = runner.policy(pre[:,:144])
                        action = {'student':student_raw,'teacher':teacher_raw,
                                  'reference':torch.cat((pre[:,-3:],torch.zeros(env.num_envs,6,device=env.device)),-1),
                                  'hold':torch.zeros_like(student_raw)}[mode]
                        # These counterfactual actions use the SAME state; paired
                        # teacher rollouts are evaluated separately above.
                        times = env.episode_length_buf*env.step_dt
                        teacher_cmd = process_action(teacher_raw,times,settings)
                        student_cmd = process_action(student_raw,times,settings)
                        obs,reward,failed,timeout,info = env.step(action)
                        states = {k:v.cpu().tolist() for k,v in info['teacher_state'].items()}
                        costs = {k:v.cpu().tolist() for k,v in info['teacher_rewards'].items()}
                        for i in alive.nonzero().flatten().tolist():
                            row = {k:v[i] for k,v in states.items()}
                            row.update(controller=mode,force_n=force,probe_vx=0.,episode=episode,environment=i,
                                       reward=float(reward[i]),failed=bool(failed[i]),timeout=bool(timeout[i] and not failed[i]),
                                       rewards={k:v[i] for k,v in costs.items()},
                                       pre_action_reference=pre[i,-3:].cpu().tolist(),
                                       pre_action_estimate=estimate[i].cpu().tolist(),
                                       teacher_command_same_state=teacher_cmd[i].cpu().tolist(),
                                       student_command_same_state=student_cmd[i].cpu().tolist())
                            rows[i].append(row)
                            stream.write(json.dumps(row,allow_nan=False)+'\n')
                        alive &= ~(failed|timeout)
                        if not bool(alive.any()): break
                    for i,data in enumerate(rows):
                        result = summarize(data,settings,force,mode)
                        prediction = torch.tensor([r['pre_action_estimate'] for r in data])
                        target = torch.tensor([r['pre_action_reference'] for r in data])
                        result.update(episode=episode,environment=i,estimator=estimator_metrics(prediction,target))
                        reports.append(result)
                        print(f'[STUDENT EVAL] {mode} F={force:g} env={i} gate={result["gate"]}',flush=True)
    selected = [r for r in reports if r['controller']=='student' and r['force_n'] in (0,4,6,8)]
    full = all(any(r['force_n']==f for r in selected) for f in (0,4,6,8))
    result = dict(schema='sfyg_student_eval_v1',contract=runner.contract,reports=reports,
                  required_task_cases_passed=full and all(r['gate']['passed'] for r in selected),
                  is_acceptance=False,scope='nominal forward single robot; estimator/generalization acceptance not established')
    write_json(output/'evaluation.json',result)
    return result


def main(args,app):
    import torch
    from sfyg_student_runner import StudentRunner,Config,sha256
    from sfyg_teacher_core import Settings,execution_contract,fingerprint
    from sfyg_teacher_env import TeacherEnv,TeacherEnvCfg,controller,critic_observation
    from bipedal_locomotion.controllers.sfyg_reset import verify_source
    from bipedal_locomotion.controllers.paint_payload import create_payload_asset
    validate(args)
    config = Config(learning_rate=args.learning_rate,estimator_learning_rate=args.estimator_learning_rate,
                    distill_coef=args.distill_coef).validate()
    saved_teacher = torch.load(args.teacher_checkpoint,map_location='cpu',weights_only=False)
    settings = Settings(**saved_teacher['sfyg_teacher_contract']['settings']).validate()
    source_core = verify_source(ROOT)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True,exist_ok=True)

    class StudentEnv(TeacherEnv):
        def __init__(self,*a,**kw):
            self.terminal_critics = []
            super().__init__(*a,**kw)

        def _reset_idx(self,ids):
            active = ids[self.episode_length_buf[ids]>0]
            if len(active):
                self.terminal_critics.append((active.clone(),critic_observation(self)[active].clone()))
            super()._reset_idx(ids)

        def step(self,action):
            self.terminal_critics = []
            obs,reward,failed,timeout,info = super().step(action)
            terminal = torch.zeros_like(obs['critic'])
            for ids,values in self.terminal_critics: terminal[ids] = values
            info['student_terminal_critic'] = terminal
            return obs,reward,failed,timeout,info

    cfg = TeacherEnvCfg()
    cfg.teacher_settings = asdict(settings)
    cfg.scene.num_envs,cfg.sim.device,cfg.seed = args.num_envs,args.device,args.seed
    for actuator in cfg.scene.robot.actuators.values():
        if hasattr(actuator,'min_delay'): actuator.min_delay=actuator.max_delay=settings.delay_steps
    cfg.actions.transport.low_path = str(Path(args.low_checkpoint).resolve())
    cfg.scene.robot.spawn.usd_path = create_payload_asset(cfg.scene.robot.spawn.usd_path,
        output/'robot_payload.usdc',mass=settings.payload_mass,dimensions=(.20,.12,.08),offset=(.15,0.,0.))
    env = StudentEnv(cfg=cfg)
    writer = None
    try:
        term = controller(env)
        low_hash = fingerprint(term.low)
        contract = execution_contract(term.low_sha,term.low_cfg,settings)
        runner = StudentRunner(args.teacher_checkpoint,contract,args.device,config)
        if args.student_checkpoint:
            loaded_iteration = runner.load(args.student_checkpoint,args.resume)
            if args.mode=='eval': runner.iteration = loaded_iteration
        obs,_ = env.reset(seed=args.seed)
        if obs['policy'].shape[1]!=147 or obs['critic'].shape[1]!=177:
            raise ValueError('Wrong Student environment observations')
        mass = term._asset.root_physx_view.get_masses()[:,term.payload_id]
        if not torch.allclose(mass,torch.full_like(mass,.5),atol=1e-5):
            raise ValueError('Expected physical 0.5 kg payload')
        write_json(output/'contract.json',runner.contract)
        sources = [*source_core,*[f'scripts/rsl_rl/{name}' for name in
                   ('train_sfyg_student.py','sfyg_student_runner.py','sfyg_student_core.py',
                    'sfyg_teacher_core.py','sfyg_teacher_env.py','sfyg_teacher_evaluation.py')],
                   'scripts/mujoco/sfyg_policy.py',
                   'exts/bipedal_locomotion/bipedal_locomotion/controllers/paint_payload.py',
                   'exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_sfyg_carry_env_cfg.py']
        hashes = {}
        for name in sources:
            target = output/'sources'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
            hashes[name] = sha256(target)
        write_json(output/'run.json',dict(arguments=vars(args),config=asdict(config),sources=hashes,
                    environment=cfg.to_dict(),teacher_development_only=saved_teacher.get('development_only',False),
                    started=datetime.now().isoformat(),is_acceptance=False))
        if args.mode=='train':
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(output))
            runner.save(output/f'model_{runner.iteration}.pt')
            end = runner.iteration+args.max_iterations
            with (output/'training.jsonl').open('w',encoding='utf-8') as stream:
                while runner.iteration < end:
                    data,obs = runner.collect(env,obs,app)
                    stats = runner.update(data)
                    stats.update({k.replace('Teacher/','Task/'):v for k,v in env.pop_metrics().items()})
                    stream.write(json.dumps(stats,allow_nan=False)+'\n')
                    stream.flush()
                    for key,value in stats.items():
                        if isinstance(value,(int,float)): writer.add_scalar('Student/'+key,value,runner.iteration)
                        elif isinstance(value,list):
                            for axis,v in zip(('fx','fy','mz'),value): writer.add_scalar('Student/'+key+'/'+axis,v,runner.iteration)
                    print(f'[STUDENT] {runner.iteration}/{end} reward={stats["mean_reward"]:.5f} '
                          f'est_rmse={stats["physical_rmse"]} teacher_KL={stats["teacher_kl"]:.5f}',flush=True)
                    if runner.iteration%args.save_interval==0 or runner.iteration==end:
                        if fingerprint(term.low)!=low_hash: raise RuntimeError('Frozen low-level changed')
                        runner.save(output/f'model_{runner.iteration}.pt')
            runner.export(output/'exported'/f'model_{runner.iteration}')
        else:
            evaluate(env,runner,args,app,settings,output)
        runner.assert_teacher_frozen()
        if fingerprint(term.low)!=low_hash: raise RuntimeError('Frozen low-level changed')
        verify_source(ROOT)
        write_json(output/'completion.json',dict(completed=True,mode=args.mode,iteration=runner.iteration,
                   frozen_weights_verified=True,source_core_verified=True,is_acceptance=False))
    finally:
        if writer is not None: writer.close()
        env.close()


if __name__=='__main__':
    from isaaclab.app import AppLauncher
    parser = parser_for_task()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    validate(args)
    launcher = AppLauncher(args)
    status = 0
    try: main(args,launcher.app)
    except BaseException:
        traceback.print_exc()
        status = 1
    finally:
        launcher.app.app.post_quit(status)
        launcher.app.close()
    raise SystemExit(status)
