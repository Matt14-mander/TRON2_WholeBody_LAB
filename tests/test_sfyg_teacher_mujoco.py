"""Real exported actor/low-level and MuJoCo checks; fixture policies do not prove walking."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/mujoco'),str(ROOT/'scripts/rsl_rl'),str(ROOT/'tests'),str(ROOT/'rsl_rl')]
from test_sfyg_mujoco import fixture_contract
from sfyg_policy import deployment_config
from sfyg_bridge import SFYGBridge
from sfyg_teacher_policy import TeacherPolicy
import sfyg_teacher_core as core
from export_sfyg import export
import run_sfyg_teacher as cli


class TeacherMujocoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rsl_rl.modules import ActorCritic,MLP_Encoder
        torch.set_num_threads(1)
        (ROOT/'.test-temp').mkdir(exist_ok=True)
        cls.temporary=tempfile.TemporaryDirectory(dir=ROOT/'.test-temp')
        cls.folder=Path(cls.temporary.name)
        with contextlib.redirect_stdout(io.StringIO()):
            low=ActorCritic(48,126,10,actor_hidden_dims=[16],critic_hidden_dims=[16])
            encoder=MLP_Encoder(420,3,hidden_dims=[16])
            for p in low.parameters(): p.data.zero_()
            checkpoint=cls.folder/'low.pt'
            with checkpoint.open('wb') as stream:
                torch.save(dict(sfyg_reset_contract=fixture_contract(),model_state_dict=low.state_dict(),encoder_state_dict=encoder.state_dict()),stream)
            cls.cfg=export(checkpoint,cls.folder/'low')
            cls.actor=ActorCritic(147,177,9,actor_hidden_dims=[16],critic_hidden_dims=[16])
            for p in cls.actor.parameters(): p.data.zero_()
            cls.actor.actor[-1].bias.data.fill_(.4)
            cls.actor.actor[-1].bias.data[0]=.12
            contract=core.execution_contract(cls.cfg['checkpoint_sha256'],deployment_config(fixture_contract()),core.Settings())
            core.export_teacher(cls.actor,cls.folder/'teacher',contract)

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def test_exact_training_history_input_and_execution_parity(self):
        policy=TeacherPolicy(self.folder/'teacher',self.cfg)
        history=core.History(1,8,18,'cpu')
        rng=np.random.default_rng(42)
        for tick in range(180):
            arm=rng.normal(0,.01,6).astype(np.float32)
            dq=rng.normal(0,.1,6).astype(np.float32)
            force=policy.force(tick,8.)
            command=policy.command(arm,dq,np.zeros(6),force,tick)
            frame=torch.from_numpy(np.r_[arm,dq,np.zeros(6,np.float32)])[None]
            h=history.append(frame,tick,valid=torch.tensor([tick>0]))
            observation=torch.cat((h.flatten(1),torch.from_numpy(force/40.)[None]),-1)
            with torch.no_grad():
                raw=self.actor.act_inference(observation)
                expected=core.process_action(raw,torch.tensor([tick])*core.Settings().dt,core.Settings())[0]
            np.testing.assert_array_equal(policy.observation,observation[0].numpy())
            np.testing.assert_allclose(command,expected.numpy(),atol=1e-7)
        policy.reset()
        self.assertFalse(policy.history.data.any())
        self.assertFalse(policy.observation.any())

    def test_no_hidden_stop_and_arm_outputs_are_masked(self):
        policy=TeacherPolicy(self.folder/'teacher',self.cfg)
        # Even with no force after release, the raw teacher controls stopping.
        cmd=policy.command(np.zeros(6),np.zeros(6),np.zeros(6),np.zeros(3),800)
        np.testing.assert_allclose(cmd,[.12,0,0])
        self.assertTrue(np.all(policy.raw[1:]>.3))
        for force in (-1.,9.,float('nan')):
            with self.assertRaises(ValueError): policy.force(0,force)
        with self.assertRaises(ValueError): policy.command(np.zeros(6),np.zeros(6),np.ones(6),np.zeros(3),0)
        with self.assertRaises(ValueError): policy.command(np.zeros(6),np.zeros(6),np.zeros(6),[0,1,0],0)

    def test_wrong_low_contract_and_corrupt_actor_rejected(self):
        cfg=deepcopy(self.cfg);cfg['checkpoint_sha256']='other'
        with self.assertRaises(ValueError): TeacherPolicy(self.folder/'teacher',cfg)
        cfg=deepcopy(self.cfg);cfg['defaults'][0]+=.1
        with self.assertRaises(ValueError): TeacherPolicy(self.folder/'teacher',cfg)
        with tempfile.TemporaryDirectory(dir=self.folder) as tmp:
            p=Path(tmp)
            shutil.copy2(self.folder/'teacher/manifest.json',p/'manifest.json')
            (p/'policy.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'SHA256'): TeacherPolicy(p,self.cfg)
            record=json.loads((p/'manifest.json').read_text())
            record['privilege']='grasp reaction'
            (p/'manifest.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'contract'): TeacherPolicy(p,self.cfg)

    def test_native_force_rotates_with_base_at_payload_com_and_zero_torque(self):
        bridge=SFYGBridge(self.cfg,.5,.5,2,42)
        root=bridge.model.jnt_qposadr[bridge.model.body_jntadr[bridge.base]]
        bridge.data.qpos[root+3:root+7]=[np.cos(.2),0,np.sin(.2),0]
        mujoco.mj_forward(bridge.model,bridge.data)
        expected=bridge.data.xmat[bridge.base].reshape(3,3)@np.array([8,0,0])
        force=cli.physics_tick(bridge,[8,0,0])
        np.testing.assert_allclose(force,expected)
        np.testing.assert_allclose(bridge.data.xfrc_applied[bridge.payload,:3],expected)
        self.assertGreater(abs(force[2]),1.)
        self.assertFalse(bridge.data.xfrc_applied[:,:][np.arange(bridge.model.nbody)!=bridge.payload].any())
        self.assertFalse(bridge.data.xfrc_applied[:,3:].any())
        self.assertAlmostEqual(bridge.data.time,.005)
        bridge.reset()
        self.assertFalse(bridge.data.xfrc_applied.any())

    def test_pd_queues_and_held_arm_unchanged(self):
        bridge=SFYGBridge(self.cfg,.5,.5,2,42)
        cli.physics_tick(bridge,[0,0,0])
        bridge.set_action(np.ones(10))
        for _ in range(2):
            cli.physics_tick(bridge,[0,0,0])
            np.testing.assert_allclose(bridge.data.ctrl[bridge.motors[:10]],bridge.defaults[:10])
        cli.physics_tick(bridge,[0,0,0])
        np.testing.assert_allclose(bridge.data.ctrl[bridge.motors[:10]],bridge.defaults[:10]+.25)
        np.testing.assert_allclose(bridge.data.ctrl[bridge.motors[10:]],bridge.defaults[10:])

    def test_force_overlay_is_read_only(self):
        bridge=SFYGBridge(self.cfg,.5,.5,2,42)
        before=[x.copy() for x in (bridge.data.qpos,bridge.data.qvel,bridge.data.xfrc_applied)]
        scene=mujoco.MjvScene(bridge.model,maxgeom=100)
        cli.draw(scene,bridge.data.xpos[bridge.base],[8,0,0],[0,8,0],bridge.data.xipos[bridge.payload],
            [.12,0,0],[.1,0,0],5.)
        self.assertEqual(scene.ngeom,2)
        for a,b in zip(before,(bridge.data.qpos,bridge.data.qvel,bridge.data.xfrc_applied)):
            np.testing.assert_array_equal(a,b)

    def test_actual_cli_smoke_resets_history_across_episodes(self):
        with tempfile.TemporaryDirectory(dir=self.folder) as tmp,contextlib.redirect_stdout(io.StringIO()):
            out=Path(tmp)/'run'
            args=cli.parse_args(['--bundle_dir',str(self.folder/'low'),'--teacher_bundle',str(self.folder/'teacher'),
                '--headless','--trace','--smoke_steps','20','--episodes','2','--output_dir',str(out)])
            cli.run(args)
            report=json.loads((out/'evaluation.json').read_text())
            self.assertFalse(report['completed'])
            self.assertEqual(report['reason'],'smoke_only')
            self.assertEqual(len(report['reports']),2)
            with np.load(out/'trace.npz') as trace:
                self.assertEqual(trace['teacher_observation'].shape,(40,147))
                self.assertEqual(trace['low_history'].shape,(40,10,42))
                for episode in (0,1):
                    first=np.flatnonzero(trace['episode']==episode)[0]
                    self.assertFalse(trace['teacher_observation'][first].any())
                    self.assertFalse(trace['command'][first].any())

    def test_native_complete_short_fixture_applies_and_releases_force(self):
        # A short fixture contract exercises the WHOLE force/command pipeline.
        # It does not claim nominal 18s walking performance.
        with tempfile.TemporaryDirectory(dir=self.folder) as tmp,contextlib.redirect_stdout(io.StringIO()):
            folder=Path(tmp)
            settings=core.Settings(settle=.04,ramp=.04,hold=.12,release_hold=.06)
            contract=core.execution_contract(self.cfg['checkpoint_sha256'],deployment_config(fixture_contract()),settings)
            core.export_teacher(self.actor,folder/'teacher',contract)
            args=cli.parse_args(['--bundle_dir',str(self.folder/'low'),'--teacher_bundle',str(folder/'teacher'),
                '--headless','--force_n','8','--trace','--output_dir',str(folder/'run')])
            cli.run(args)
            record=json.loads((folder/'run/evaluation.json').read_text())
            self.assertTrue(record['completed'])
            rows=[json.loads(s) for s in (folder/'run/steps.jsonl').read_text().splitlines()]
            self.assertTrue(any(r['force_base'][0]==8 for r in rows))
            self.assertEqual(rows[-1]['force_base'][0],0.)
            self.assertGreater(rows[-1]['command'][0],.1)
            self.assertTrue(rows[-1]['timeout'])
            self.assertFalse(record['is_acceptance'])

    def test_student_shadow_and_direct_speed_sweep(self):
        import diagnose_sfyg_student as diag
        from rsl_rl.modules import ActorCritic
        from sfyg_student_core import StudentPolicy

        with tempfile.TemporaryDirectory(dir=self.folder) as tmp,contextlib.redirect_stdout(io.StringIO()):
            folder=Path(tmp)
            settings=core.Settings(settle=.04,ramp=.04,hold=.12,release_hold=.06)
            bound=core.execution_contract(self.cfg['checkpoint_sha256'],deployment_config(fixture_contract()),settings)
            contract=dict(schema='sfyg_student_v1',teacher_contract=bound,actor_input=144,raw_output=9)
            student=StudentPolicy(ActorCritic(147,177,9,**core.runner_config()['policy']).actor)
            # Analytically distinguish all three conditions: raw vx = Fx_norm + Fy_norm.
            with torch.no_grad():
                for param in student.parameters():
                    param.zero_()
                student.estimator.network[-1].bias.copy_(torch.tensor([-.10,.03,0.]))
                student.actor[0].weight[0,144:146]=1.
                student.actor[0].bias[0]=1.
                for layer in (student.actor[2],student.actor[4],student.actor[6]):
                    layer.weight[0,0]=1.
                student.actor[-1].bias[0]=-1.
            checkpoint=folder/'model_500.pt'
            torch.save(dict(policy=student.state_dict(),student_contract=contract,iteration=500),checkpoint)
            output=folder/'controls'
            with patch.object(diag,'S1_CHECKPOINT',checkpoint):
                report=diag.run(output,self.folder/'low',42)
            cases=report['cases']
            self.assertEqual(len(cases),15)
            shadow=cases['student_8N']['shadow']
            for mode,raw,cmd in [('estimated',-.07,0.),('true_fx',.23,.23),('true_wrench',.2,.2)]:
                self.assertAlmostEqual(shadow[mode]['raw_vx'],raw,places=5)
                self.assertAlmostEqual(shadow[mode]['cmd_vx'],cmd,places=5)
            for name,case in cases.items():
                rows=[json.loads(s) for s in (output/f'{name}.jsonl').read_text().splitlines()]
                self.assertTrue(case['completed'])
                self.assertEqual(case['plateau_frames'],round(settings.hold/settings.dt))
                self.assertAlmostEqual(max(np.linalg.norm(r['applied_world_force']) for r in rows),case['force_n'])
                self.assertEqual(rows[0]['cmd_vx'],0.)
                self.assertEqual(rows[-1]['cmd_vx'],0.)
                if case['direct_vx'] is not None:
                    self.assertAlmostEqual(case['cmd_vx'],case['direct_vx'])
                    self.assertFalse(any('estimated_fx_n' in r for r in rows))
            # The native fixture always commands zero. The two independent 8 N
            # rollouts must coincide even though one also computes the shadows.
            for key in ('vx','dx_world_m','foot_releases_lr'):
                self.assertEqual(cases['student_8N'][key],cases['low_8N_vx0.00'][key])
            self.assertEqual(cases['low_16N_vx0.00']['cmd_vx'],0.)
            with patch.object(diag,'S1_CHECKPOINT',checkpoint):
                paired=diag.run(folder/'transfer',self.folder/'low',42,arm_transfer=True)['cases']
            self.assertEqual(len(paired),5)
            self.assertAlmostEqual(paired['student_8N_true_fx_closed_loop']['cmd_vx'],.23,places=5)
            self.assertEqual(paired['student_8N']['cmd_vx'],0.)
            oracle=[json.loads(s) for s in (folder/'transfer/student_8N_true_fx_closed_loop.jsonl').read_text().splitlines()]
            for row in oracle:
                self.assertAlmostEqual(row['cmd_vx'],row['shadow_raw_cmd_vx']['true_fx'][1])
            # With force removed the fixture still commands +.03 due to estimated Fy.
            self.assertAlmostEqual(oracle[-1]['cmd_vx'],.03,places=5)
            from compare_sfyg_arm_hold import statistics
            for f in (0,4,8):
                held=[json.loads(s) for s in (folder/f'transfer/low_{f}N_vx0.00.jsonl').read_text().splitlines()]
                stat=statistics(held,f,False,vars(settings))
                self.assertEqual(stat['frames'],6)
                self.assertAlmostEqual(stat['est'],-4.,places=5)
                self.assertEqual(len(stat['q']),6)

    def test_closed_loop_motion_and_release_metrics(self):
        from diagnose_sfyg_student import motion_metrics
        settings=core.Settings()
        rows=[]
        for tick in range(round(settings.horizon/settings.dt)):
            t=tick*settings.dt
            vx=.10 if 4<=t<6 or 7<=t<15 else 0.
            rows.append(dict(tick=tick,action_time_s=t,vx=vx,vy=0.,wz=0.,cmd_vx=vx,termination=None))
        metrics=motion_metrics(rows,settings)
        self.assertEqual([e['event'] for e in metrics['motion_events']],['start','stop','start'])
        self.assertTrue(metrics['release_stable'])
        self.assertIsNone(motion_metrics(rows[:-1],settings)['release_stable'])
        for r in rows[-75:]: r['wz']=.12
        self.assertFalse(motion_metrics(rows,settings)['release_stable'])

    def test_single_student_checkpoint_playback_matches_training_policy(self):
        from rsl_rl.modules import ActorCritic
        from sfyg_student_core import StudentPolicy
        from sfyg_student_policy import StudentCheckpointPolicy
        with tempfile.TemporaryDirectory(dir=self.folder) as tmp,contextlib.redirect_stdout(io.StringIO()):
            folder=Path(tmp)
            settings=core.Settings(settle=.04,ramp=.04,hold=.12,release_hold=.06)
            bound=core.execution_contract(self.cfg['checkpoint_sha256'],deployment_config(fixture_contract()),settings)
            contract=dict(schema='sfyg_student_v1',teacher_contract=bound,actor_input=144,raw_output=9)
            student=StudentPolicy(ActorCritic(147,177,9,**core.runner_config()['policy']).actor).eval()
            saved=dict(policy=student.state_dict(),student_contract=contract,iteration=500)
            checkpoint=folder/'model_500.pt'
            torch.save(saved,checkpoint)
            args=cli.parse_args(['--bundle_dir',str(self.folder/'low'),'--output_dir',str(folder/'run'),
                                 '--headless','--force_n','16','--episodes','2','--trace'],student=True)
            with patch('sfyg_student_policy.S1_CHECKPOINT',checkpoint):
                out=cli.run(args)
            report=json.loads((out/'evaluation.json').read_text())
            self.assertTrue(report['completed'])
            self.assertEqual(report['interface']['iteration'],500)
            self.assertTrue(report['interface']['outside_training_force_range'])
            self.assertEqual(report['interface']['training_force_max_n'],8.)
            rows=[json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines()]
            self.assertEqual(max(r['force_base'][0] for r in rows),16.)
            np.testing.assert_allclose([np.linalg.norm(r['last_applied_world_force']) for r in rows],
                                       [r['force_base'][0] for r in rows],atol=1e-5)
            self.assertEqual(rows[-1]['force_base'][0],0.)
            with np.load(out/'trace.npz') as trace,torch.no_grad():
                h=torch.from_numpy(trace['student_observation'])
                self.assertEqual(h.shape,(len(rows),144))
                for ep in (0,1):
                    self.assertFalse(h[np.flatnonzero(trace['episode']==ep)[0]].any())
                raw=student(h)
                times=torch.tensor([r['action_time_s'] for r in rows])
                np.testing.assert_allclose(trace['raw'],raw.numpy(),atol=1e-6)
                np.testing.assert_allclose(trace['command'],core.process_action(raw,times,settings).numpy(),atol=1e-6)
                np.testing.assert_allclose([r['estimated_base_fx_fy_mz'] for r in rows],
                    (student.estimate(h)*torch.tensor([40.,40.,10.])).numpy(),atol=1e-5)
            torch.save(dict(saved,iteration=200),checkpoint)
            with self.assertRaisesRegex(ValueError,'iteration-500'):
                StudentCheckpointPolicy(checkpoint,self.cfg)


if __name__=='__main__': unittest.main()
