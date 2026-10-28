"""Explicit grasp-proxy semantics and actual multi-demo wiring, not walking acceptance."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/mujoco'),str(ROOT/'scripts/rsl_rl'),str(ROOT/'tests'),str(ROOT/'rsl_rl')]
from test_sfyg_mujoco import fixture_contract
from export_sfyg import export
from sfyg_policy import deployment_config
from sfyg_team_control import FollowerInput,make_follower
from sfyg_teacher_shared import validate_proxy_timing
import sfyg_teacher_core as core
import run_sfyg_team as team
import run_sfyg_cooperate as coop
import sfyg_cooperative_scene as scenes


def obs(tick,force=(0,0,0,0,0,0),rotation=None,settling=True,calibrating=True):
    return FollowerInput(tick*.02,.02,settling,calibrating,np.array(force,float),
        np.eye(3) if rotation is None else rotation,np.zeros(6),np.zeros(6),np.zeros(6))


class SharedTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rsl_rl.modules import ActorCritic,MLP_Encoder
        torch.set_num_threads(1)
        (ROOT/'.test-temp').mkdir(exist_ok=True)
        cls.tmp=tempfile.TemporaryDirectory(dir=ROOT/'.test-temp')
        cls.folder=Path(cls.tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            low=ActorCritic(48,126,10,actor_hidden_dims=[16],critic_hidden_dims=[16])
            encoder=MLP_Encoder(420,3,hidden_dims=[16])
            for p in low.parameters(): p.data.zero_()
            with (cls.folder/'low.pt').open('wb') as stream:
                torch.save(dict(sfyg_reset_contract=fixture_contract(),model_state_dict=low.state_dict(),encoder_state_dict=encoder.state_dict()),stream)
            cls.cfg=export(cls.folder/'low.pt',cls.folder/'low')
        actor=torch.nn.Linear(147,9)
        with torch.no_grad():
            actor.weight.zero_();actor.bias.fill_(.4);actor.bias[0]=.12
        wrapper=type('Actor',(),{'actor':actor})()
        contract=core.execution_contract(cls.cfg['checkpoint_sha256'],deployment_config(fixture_contract()),core.Settings())
        core.export_teacher(wrapper,cls.folder/'teacher',contract)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def policy(self,**kwargs):
        options=dict(teacher_bundle=self.folder/'teacher',low_config=self.cfg,teacher_input='grasp-proxy',axes='forward',max_speed=.3)
        options.update(kwargs)
        return make_follower('teacher',**options)

    def calibrate(self,p,bias=(5,1,2,.1,.2,.3)):
        for tick in range(100,150): p.update(obs(tick,bias))

    def test_world_bias_base_projection_clip_and_logged_ood(self):
        p=self.policy();self.calibrate(p)
        result=p.update(obs(150,(11,1,2,.1,.2,.3),settling=False,calibrating=False))
        self.assertAlmostEqual(p.proxy_force,6.)
        np.testing.assert_allclose(p.teacher.observation[-3:],[.15,0,0])
        np.testing.assert_allclose(result.velocity,[.12,0,0])
        self.assertFalse(result.arm_offset.any())
        p.update(obs(151,(3,1,2,.1,.2,.3),settling=False,calibrating=False))
        self.assertEqual(p.proxy_force,0.)
        p.update(obs(152,(25,1,2,.1,.2,.3),settling=False,calibrating=False))
        self.assertEqual(p.proxy_force,8.)
        diagnostic=p.diagnostics()
        self.assertEqual(diagnostic['negative_fx_samples'],1)
        self.assertEqual(diagnostic['above_8n_samples'],1)
        self.assertFalse(diagnostic['input_equivalent_to_training'])
        # Rotate current world force, not a stale local bias, at every update.
        rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        p.update(obs(153,(5,7,2,.1,.2,.3),rotation,False,False))
        self.assertAlmostEqual(p.proxy_force,6.)

    def test_no_shared_state_no_forced_tail_stop_and_reset(self):
        a,b=self.policy(),self.policy()
        self.calibrate(a);self.calibrate(b,(0,0,0,0,0,0))
        a.update(obs(150,(11,1,2,.1,.2,.3),settling=False,calibrating=False))
        b.update(obs(150,settling=False,calibrating=False))
        self.assertEqual(a.proxy_force,6.)
        self.assertEqual(b.proxy_force,0.)
        self.assertIsNot(a.teacher.history,b.teacher.history)
        # Actor deliberately outputs .12 even at zero force: the adapter must
        # retain this bad stopping behavior rather than hide it.
        self.assertAlmostEqual(float(b.suggestion[0]),.12,places=6)
        a.reset()
        self.assertFalse(a.bias.any());self.assertFalse(a.teacher.history.data.any())
        self.assertEqual(a.last_tick,-1);self.assertEqual(a.samples,0)
        self.assertEqual(b.bias_samples,50)

    def test_explicit_mode_and_contract_guards(self):
        with self.assertRaises(NotImplementedError): make_follower('teacher')
        with self.assertRaises(NotImplementedError): make_follower('student')
        with self.assertRaises(ValueError): self.policy(teacher_input='external-share')
        with self.assertRaises(ValueError): self.policy(axes='planar')
        with self.assertRaises(ValueError): self.policy(max_speed=.2)
        cfg=deepcopy(self.cfg);cfg['checkpoint_sha256']='wrong'
        with self.assertRaises(ValueError): self.policy(low_config=cfg)
        p=self.policy()
        with self.assertRaises(ValueError): validate_proxy_timing([p],2.,2)
        with self.assertRaises(ValueError): validate_proxy_timing([p],3.,0)
        with self.assertRaises(ValueError): p.update(obs(150,settling=False,calibrating=False))
        p.reset();self.calibrate(p)
        with self.assertRaises(ValueError): p.update(obs(149))

    def test_native_all_three_demo_model_checks_and_actor_trace(self):
        with tempfile.TemporaryDirectory(dir=self.folder) as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            for scenario in ('leader-follow','sofa'):
                args=team.parse_args(['--scenario',scenario,'--bundle_dir',str(self.folder/'low'),
                    '--follower','teacher','--teacher_bundle',str(self.folder/'teacher'),'--teacher_input','grasp-proxy',
                    '--max_speed','.3','--headless','--hold_all','--duration','.3','--trace','--output_dir',str(root/scenario)])
                team.run(args)
                report=json.loads((root/scenario/'evaluation.json').read_text())
                self.assertTrue(report['completed'])
                n=1 if scenario=='leader-follow' else 2
                self.assertEqual(len(report['interface']['teacher_proxy']),n)
                self.assertFalse(report['interface']['teacher_proxy'][0]['input_equivalent_to_training'])
                with np.load(root/scenario/'trace.npz') as trace:
                    self.assertEqual(trace['teacher_observation'].shape[1:],(147,))
                    self.assertEqual(trace['teacher_raw'].shape[1:],(9,))
                    self.assertTrue(np.any(trace['teacher_raw'][trace['robot']>0,0]>.1))
                    self.assertFalse(trace['command'].any())  # still initial settle
            args=coop.parser().parse_args(['--bundle_dir',str(self.folder/'low'),'--controller','teacher',
                '--teacher_bundle',str(self.folder/'teacher'),'--teacher_input','grasp-proxy','--max_speed','.3',
                '--delay_steps','2','--grasp_model','ball','--physics_substeps','4','--headless',
                '--duration','12','--trace','--output_dir',str(root/'side')])
            coop.run(args)
            report=json.loads((root/'side/evaluation.json').read_text())
            self.assertEqual(report['controller'],'teacher')
            self.assertEqual(len(report['interface']['teacher_proxy']),2)
            with np.load(root/'side/trace.npz') as trace:
                self.assertTrue(np.any(trace['teacher_raw'][:,0]>.1))
                self.assertEqual(trace['teacher_observation'].shape[1:],(147,))
                self.assertEqual(set(trace['robot']),{0,1})
            # Fixed supports + a mocked gate ONLY for control-routing coverage.
            # The fixture has soft-grip sag, so it is NOT a static acceptance
            # test. Native gate rejection is separately tested without mocks.
            original=scenes.build_scene_xml
            def supported(*args,**kwargs):
                model,assets,ref,sha=original(*args,**kwargs)
                for i in range(len(args[0])):
                    ET.SubElement(model.find('equality'),'weld',name=f'test_support{i}',body1=f'r{i}/base_Link',solref='.01 1')
                return model,assets,ref,sha
            args=team.parse_args(['--scenario','sofa','--bundle_dir',str(self.folder/'low'),
                '--follower','teacher','--teacher_bundle',str(self.folder/'teacher'),'--teacher_input','grasp-proxy',
                '--max_speed','.3','--headless','--duration','12','--trace','--output_dir',str(root/'supported_sofa')])
            with patch.object(scenes,'build_scene_xml',side_effect=supported),patch.object(team,'calibration_quality',
                    return_value={'passed':True,'test_only_forced_gate':True}):
                team.run(args)
            report=json.loads((root/'supported_sofa/evaluation.json').read_text())
            self.assertTrue(report['completed'],(report['reason'],report['calibration'],report['max_grasp_error_m']))
            self.assertTrue(report['calibration']['test_only_forced_gate'])
            with np.load(root/'supported_sofa/trace.npz') as trace:
                active=(trace['time_s']>3.02)&(trace['robot']>0)
                self.assertGreater(active.sum(),0)
                np.testing.assert_allclose(trace['command'][active,0],.12)


if __name__=='__main__': unittest.main()
