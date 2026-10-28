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


if __name__=='__main__': unittest.main()
