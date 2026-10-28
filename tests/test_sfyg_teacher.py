"""T0/T1 contracts, real source PPO and frozen SFYG inference regressions."""
import ast
import contextlib
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/rsl_rl'),str(ROOT/'scripts/mujoco'),str(ROOT/'rsl_rl')]
import sfyg_teacher_core as core
from sfyg_teacher_evaluation import summarize,baseline_gate
from sfyg_policy import load_checkpoint,make_observation


class TeacherTests(unittest.TestCase):
    def test_source_core_unchanged(self):
        spec = importlib.util.spec_from_file_location('source_reset',ROOT/'exts/bipedal_locomotion/bipedal_locomotion/controllers/sfyg_reset.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(len(module.verify_source(ROOT)),16)

    def test_profile_has_settle_and_unforced_release_tail(self):
        s = core.Settings().validate()
        time = torch.tensor([0.,2.99,3.,4.,5.,13.,14.,15.,17.99,18.])
        torch.testing.assert_close(core.envelope(time,s),torch.tensor([0.,0.,0.,.5,1.,1.,.5,0.,0.,0.]))
        # Force is zero in the tail, but actor output is NOT overridden there.
        command = core.process_action(torch.full((len(time),9),.2),time,s)
        self.assertEqual(float(command[-1,0]),float(torch.tensor(.2)))
        self.assertEqual(float(command[0,0]),0.)
        self.assertFalse(command[:,1:].any())

    def test_sampling_and_settings(self):
        torch.manual_seed(123)
        f = core.sample_force(20000,'cpu',core.Settings())
        self.assertTrue(bool(((f>=0)&(f<=8)).all()))
        self.assertAlmostEqual(float((f==0).float().mean()),.2,delta=.015)
        self.assertAlmostEqual(float(((f>0)&(f<2)).float().mean()),.1,delta=.015)
        self.assertFalse(bool(((f>2)&(f<4)).any()))
        with self.assertRaises(ValueError): core.Settings(force_max=40).validate()

    def test_history_isolation_order_and_read_idempotency(self):
        high = core.History(3,8,18,'cpu')
        low = core.History(3,10,42,'cpu',repeat_first=True)
        high.append(torch.ones(3,18),0,valid=torch.zeros(3,dtype=torch.bool))
        self.assertFalse(high.data.any())
        high.append(torch.ones(3,18),1)
        self.assertFalse(high.data[:,:7].any())
        before = high.data.clone()
        high.append(torch.full((3,18),99.),1)
        torch.testing.assert_close(high.data,before)
        low.append(torch.ones(3,42),1)
        self.assertTrue(bool((low.data==1).all()))
        high.reset(torch.tensor([1]))
        low.reset(torch.tensor([1]))
        torch.testing.assert_close(high.data[0],before[0])
        low.append(torch.full((3,42),2.),2)
        self.assertTrue(bool((low.data[1]==2).all()))
        self.assertTrue(bool((low.data[0,:-1]==1).all()))

    def test_low_observation_matches_existing_single_robot_bridge(self):
        # The small native smoke checkpoint contains the actual SFYG contract;
        # keep this test portable by using the same contract stored in the asset
        # bridge test fixture when available, otherwise synthesize exact terms.
        cfg = dict(physics_dt=.005,decimation=4,gait=[.8,.5,.5,.2],defaults=[0.]*18,
            terms=[dict(clip=(-100.,100.),scale=x) for x in (.25,1.,1.,.05,1.)]
                  +[dict(clip=None,scale=None),dict(clip=None,scale=None)])
        for term,width in zip(cfg['terms'],(3,3,10,10,10,2,4)): term['width']=width
        rng = np.random.default_rng(4)
        values = [rng.normal(size=(4,n)).astype(np.float32)*scale
                  for n,scale in ((3,200),(3,1),(10,1),(10,200),(10,5))]
        steps = torch.tensor([0,1,125,900])
        actual = core.low_observation(*(torch.tensor(v) for v in values),steps,cfg)
        expected = np.stack([make_observation(*(v[i] for v in values),int(steps[i]),cfg) for i in range(4)])
        torch.testing.assert_close(actual,torch.tensor(expected),atol=3e-6,rtol=1e-5)

    def test_rewards_prefer_walking_and_penalize_unclipped_actions(self):
        target = torch.tensor([[.2,0.,0.]])
        zero = torch.zeros_like(target)
        raw = torch.zeros(1,9)
        gravity = torch.tensor([[0.,0.,-1.]])
        standing = sum(core.reward_terms(zero,target,zero,raw,raw,gravity).values())
        raw[:,0] = .2
        walking = sum(core.reward_terms(target,target,target,raw,raw,gravity).values())
        self.assertGreater(float(walking),float(standing))
        a,b = torch.full((1,9),1.),torch.full((1,9),100.)
        ca = core.reward_terms(zero,zero,zero,a,torch.zeros_like(a),gravity)['raw_action']
        cb = core.reward_terms(zero,zero,zero,b,torch.zeros_like(b),gravity)['raw_action']
        torch.testing.assert_close(cb,ca*10000)
        out = core.process_action(b,torch.tensor([8.]),core.Settings())
        torch.testing.assert_close(out,torch.tensor([[.3,0.,0.]]))

    def test_actual_source_ppo_without_encoder_updates_and_correct_std(self):
        from rsl_rl.modules import ActorCritic,MLP_Encoder
        from rsl_rl.algorithm import PPO
        with contextlib.redirect_stdout(io.StringIO()):
            actor = ActorCritic(147,177,9,actor_hidden_dims=[16],critic_hidden_dims=[16],init_noise_std=.1)
            encoder = MLP_Encoder(147,0,hidden_dims=[1],output_detach=True)
        core.initialize_teacher(actor)
        actor.update_distribution(torch.zeros(4,147))
        torch.testing.assert_close(actor.action_std,torch.full((4,9),.1))
        cfg = core.runner_config()['algorithm']
        cfg.pop('class_name');cfg.pop('obs_history_len')
        cfg.update(num_learning_epochs=2,num_mini_batches=2)
        ppo = PPO(4,encoder,actor,device='cpu',**cfg)
        self.assertIsNone(ppo.extra_optimizer)
        self.assertFalse(ppo.early_stop)
        ppo.init_storage(4,4,[147],[177],[147],[0],[9])
        before = core.fingerprint(actor)
        for _ in range(4):
            with torch.no_grad():
                obs = torch.randn(4,147)
                ppo.act(obs,torch.zeros(4,147),torch.zeros(4,0),torch.randn(4,177))
                ppo.process_env_step(torch.randn(4),torch.zeros(4,dtype=torch.bool),{},obs)
        ppo.compute_returns(torch.randn(4,177))
        losses = ppo.update()
        self.assertTrue(all(np.isfinite(v) for v in losses))
        self.assertNotEqual(before,core.fingerprint(actor))
        self.assertEqual(int(next(iter(ppo.optimizer.state.values()))['step']),4)

    def test_contract_and_export_reload(self):
        from rsl_rl.modules import ActorCritic
        with contextlib.redirect_stdout(io.StringIO()):
            actor = ActorCritic(147,177,9,actor_hidden_dims=[16],critic_hidden_dims=[16])
        contract = core.execution_contract('abc',dict(gait=(.8,.5,.5,.2),clip=(-100,100)),core.Settings())
        self.assertEqual(contract,json.loads(json.dumps(contract)))
        saved = dict(sfyg_teacher_contract=contract,model_state_dict=actor.state_dict())
        core.check_teacher(saved,contract)
        other = deepcopy(contract);other['low_sha256']='different'
        with self.assertRaises(ValueError): core.check_teacher(saved,other)
        parent = ROOT/'.test-temp';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            core.export_teacher(actor,Path(tmp)/'export',contract)
            meta = json.loads((Path(tmp)/'export/manifest.json').read_text())
            self.assertTrue(meta['freeze_arm'])
            self.assertEqual(meta['active_axes'],['vx'])

    def test_step_force_at_payload_only_rotates_each_substep_and_reset_clears(self):
        # Execute the real action methods against a recording articulation.
        tree = ast.parse((ROOT/'scripts/rsl_rl/sfyg_teacher_env.py').read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TeacherAction')
        methods = [n for n in cls.body if getattr(n,'name',None) in ('apply_actions','reset')]
        module = ast.Module(body=methods,type_ignores=[])
        calls=[]
        def rotate(q,v):
            # Test stand-in with a 90 degree yaw selected by q[0]; this checks
            # that the live base/body transform is reread on every application.
            return torch.stack((-v[:,1],v[:,0],v[:,2]),-1) if float(q[0,0])==0 else v
        ns=dict(torch=torch,quat_apply=rotate,quat_apply_inverse=lambda q,v:rotate(q,v),
                sample_force=core.sample_force)
        exec(compile(module,'action_methods','exec'),ns)
        robot=types.SimpleNamespace(data=types.SimpleNamespace(root_quat_w=torch.tensor([[1.,0,0,0]]),
            body_quat_w=torch.tensor([[[1.,0,0,0]]]),default_joint_pos=torch.zeros(1,18)),
            set_joint_position_target=lambda *a,**k:None,set_joint_velocity_target=lambda *a,**k:None,
            set_joint_effort_target=lambda *a,**k:None,
            set_external_force_and_torque=lambda f,t,**kw:calls.append((f.clone(),t.clone(),kw)))
        a=types.SimpleNamespace(_asset=robot,interval_reference=torch.tensor([[.2,0,0]]),
            leg_target=torch.zeros(1,10),leg_ids=list(range(10)),hold_ids=list(range(10,18)),
            nominal=torch.zeros(1,8),world_force=torch.zeros(1,1,3),local_force=torch.zeros(1,1,3),
            zero_torque=torch.zeros(1,1,3),payload_id=0)
        ns['apply_actions'](a)
        torch.testing.assert_close(calls[-1][0],torch.tensor([[[8.,0,0]]]))
        robot.data.root_quat_w[0,0]=0
        ns['apply_actions'](a)
        torch.testing.assert_close(calls[-1][0],torch.tensor([[[0.,8,0]]]))
        self.assertEqual(calls[-1][2]['body_ids'],[0])
        self.assertFalse(calls[-1][1].any())
        a.raw=torch.ones(1,9);a.previous=a.raw.clone();a.command=torch.ones(1,3)
        a.last_leg=torch.ones(1,10);a.target_force=torch.ones(1);a.num_envs=1;a.device='cpu'
        a.settings=core.Settings();a._env=types.SimpleNamespace(cfg=types.SimpleNamespace(fixed_force=0.))
        a.low_history=core.History(1,10,42,'cpu');a.arm_history=core.History(1,8,18,'cpu')
        ns['reset'](a,torch.tensor([0]))
        self.assertFalse(calls[-1][0].any())
        self.assertFalse(a.command.any())

    def rows(self,force=8.,moving=True,failed=False):
        s=core.Settings()
        rows=[]
        for i in range(900):
            t=(i+1)*s.dt
            ref=float(core.envelope(torch.tensor(i*s.dt),s))*force/40
            rows.append(dict(time_s=t,velocity=[ref if moving else 0.,0.,0.],reference=[ref,0.,0.],
                raw_action=[ref]+[0.]*8,arm_error=[0.]*6,tilt_deg=1.,reward=0.,rewards={'test':0.},
                timeout=i==899,failed=failed and i==899))
        return rows

    def test_evaluation_rejects_stationary_and_timeout_failure(self):
        good=summarize(self.rows(),core.Settings(),8.,'reference')
        self.assertTrue(good['gate']['passed'])
        bad=summarize(self.rows(moving=False),core.Settings(),8.,'reference')
        self.assertFalse(bad['gate']['passed'])
        bad=summarize(self.rows(failed=True),core.Settings(),8.,'reference')
        self.assertFalse(bad['completed'])
        self.assertFalse(baseline_gate([])['passed'])
        self.assertFalse(baseline_gate([dict(good,episode=0,environment=0)])['passed'])

    def test_artifact_checker_rejects_failed_gate_even_if_process_completed(self):
        from check_sfyg_teacher_run import check
        with tempfile.TemporaryDirectory(dir=ROOT/'.test-temp') as folder:
            p=Path(folder)
            done=dict(mode='baseline',completed=True,frozen_weights_verified=True,source_core_verified=True)
            (p/'completion.json').write_text(json.dumps(done))
            (p/'evaluation.json').write_text(json.dumps(dict(mode='baseline',frozen_weights_verified=True,reports=[])))
            with self.assertRaises(ValueError): check(p,'baseline')
            done['mode']='smoke'
            (p/'completion.json').write_text(json.dumps(done))
            with self.assertRaises(FileNotFoundError): check(p,'smoke')
            (p/'interface_audit.json').write_text(json.dumps(dict(passed=True)))
            self.assertTrue(check(p,'smoke')['completed'])
            done['frozen_weights_verified']=False
            (p/'completion.json').write_text(json.dumps(done))
            with self.assertRaises(ValueError): check(p,'smoke')

    def test_full_gate_and_train_report_binding(self):
        from train_sfyg_teacher import validate_baseline
        reports=[]
        for force in (0.,2.,4.,6.,8.):
            for mode in ('hold','reference'):
                rows=self.rows(force=force,moving=mode=='reference')
                # Exercise total-return comparison separately from velocity gates.
                for row in rows: row['reward']=-.001 if mode=='reference' else -.01
                result=summarize(rows,core.Settings(),force,mode)
                result.update(episode=0,environment=0)
                reports.append(result)
        self.assertTrue(baseline_gate(reports)['passed'])
        c=core.execution_contract('frozen-sha',{'defaults':(0,1)},core.Settings())
        record=dict(schema='sfyg_t0_report_v1',mode='baseline',contract=c,
                    frozen_weights_verified=True,reports=reports)
        with tempfile.TemporaryDirectory(dir=ROOT/'.test-temp') as folder:
            p=Path(folder)/'evaluation.json';p.write_text(json.dumps(record))
            validate_baseline(p,c)
            other=deepcopy(c);other['low_sha256']='wrong'
            with self.assertRaises(ValueError): validate_baseline(p,other)
            record['reports'][0]['completed']=False
            record['reports'][0]['gate']['passed']=False
            p.write_text(json.dumps(record))
            with self.assertRaises(ValueError): validate_baseline(p,c)


if __name__=='__main__': unittest.main()
