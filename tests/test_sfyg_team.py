"""Team dynamics/control contracts, using fixtures rather than server-trained policies."""
import contextlib
from copy import deepcopy
from dataclasses import fields
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/mujoco'),str(ROOT/'tests'),str(ROOT/'rsl_rl')]
from test_sfyg_mujoco import fixture_config,fixture_contract
from sfyg_cooperative_scene import CooperativeScene,SharedRobot,build_scene_xml,initialize,grasp_reaction,audit_reactions
from sfyg_cooperative_control import calibration_quality
from sfyg_team_control import FollowerInput,HighAction,make_follower,checked_action,LeaderKeyboard,leader_command
from sfyg_team_visuals import draw_team
from export_sfyg import export
import run_sfyg_team as cli


def observation(force=0.,time=4.,settling=False):
    return FollowerInput(time,.02,settling,settling,np.array([force,0,0,0,0,0.]),
        np.eye(3),np.zeros(6),np.zeros(6),np.zeros(6))


class TeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        (ROOT/'.test-temp').mkdir(exist_ok=True)

    def test_layout_physical_mass_free_roots_alignment_and_original_bodies(self):
        c=fixture_config()
        for layout,n,spacing in (('leader-follow',2,1.8),('sofa',3,1.1)):
            scene=CooperativeScene([c]*n,spacing=spacing,grasp_model='ball',physics_substeps=4,layout=layout)
            self.assertEqual(np.count_nonzero(scene.model.jnt_type==mujoco.mjtJoint.mjJNT_FREE),n+1)
            self.assertAlmostEqual(scene.model.body_mass.sum(),n*c['usd_mass_kg']+1.,places=5)
            self.assertAlmostEqual(scene.model.body_mass[scene.payload],1.)
            self.assertLess(max(scene.grasp_errors()),1e-10)
            self.assertEqual(scene.model.neq,n)
            self.assertTrue(np.all(scene.model.eq_type==mujoco.mjtEq.mjEQ_CONNECT))
            for i in range(n):
                for name,body in scene.reference['bodies'].items():
                    self.assertAlmostEqual(scene.model.body_mass[scene.model.body(f'r{i}/{name}').id],body['mass'])
            self.assertGreater(scene.data.xpos[scene.robots[0].base,0],scene.data.xpos[scene.robots[1].base,0])
            # Rear leader hitch is on the base tree, not a relocated arm.
            self.assertEqual(scene.model.body_parentid[scene.robots[0].frame],scene.robots[0].base)
            scene.step(np.zeros(6))
            self.assertAlmostEqual(scene.data.time,.005)
            self.assertFalse(scene.data.xfrc_applied.any())
            scene.robots[-1].observation()
            self.assertIsNone(scene.robots[0].history.frames)
            scene.reset()
            self.assertFalse(scene.data.xfrc_applied.any())
            self.assertLess(max(scene.grasp_errors()),1e-10)

    def test_all_three_contracts_checked_and_invalid_spacing_rejected(self):
        c=fixture_config();other=deepcopy(c);other['defaults'][0]+=.1
        with self.assertRaises(ValueError): build_scene_xml([c,c,other],layout='sofa',spacing=1.1)
        with self.assertRaises(ValueError): build_scene_xml([c,c],layout='leader-follow',spacing=1.)
        with self.assertRaises(ValueError): build_scene_xml([c,c],layout='sofa')

    def test_static_supported_fixture_three_way_reaction_balance(self):
        # Fixed supports exist ONLY in this analytical load-cell fixture.
        c=fixture_config()
        root,assets,_,_=build_scene_xml([c]*3,layout='sofa',spacing=1.1,grasp_model='ball')
        root.find('option').set('timestep','.00125')
        for i in range(3):
            ET.SubElement(root.find('equality'),'weld',name=f'test_support{i}',body1=f'r{i}/base_Link',solref='.01 1')
        model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'),assets)
        data=mujoco.MjData(model);payload=initialize(model,data,[c]*3)
        robots=[SharedRobot(c,model,data,i,0,42+i) for i in range(3)]
        scene=types.SimpleNamespace(model=model,data=data,payload=payload,robots=robots)
        for _ in range(1200):
            for robot in robots: robot.write_controls()
            data.xfrc_applied[payload]=[8,0,0,0,0,0]
            mujoco.mj_step(model,data)
        mujoco.mj_forward(model,data)
        measured=[grasp_reaction(model,data,r.eq,payload) for r in robots]
        audit=audit_reactions(scene,measured)
        self.assertLess(audit['extraction_max'],1e-8)
        self.assertLess(audit['dynamics_max'],1e-7)
        np.testing.assert_allclose(audit['payload_grasp_world'][:3],[-8,0,9.81],atol=.2)
        self.assertTrue(all(np.linalg.norm(w[3:])<1e-8 for w in measured))
        self.assertGreater(sum(w[0] for w in measured),7.8)

    def test_follower_information_boundary_force_response_and_stop(self):
        names={f.name for f in fields(FollowerInput)}
        self.assertFalse(names & {'scene','leader','leader_command','payload_position','external_force'})
        follower=make_follower('admittance')
        for _ in range(100):
            result=follower.update(observation())
            np.testing.assert_array_equal(result.velocity,[0,0,0])
        for _ in range(100): result=follower.update(observation(4.))
        self.assertGreater(result.velocity[0],.1)
        self.assertTrue(np.all(result.velocity[1:]==0))
        # Stopping the leader is not an input: follower keeps responding to force.
        self.assertGreater(follower.update(observation(4.,time=28.)).velocity[0],.1)
        for _ in range(300): result=follower.update(observation())
        self.assertLess(abs(result.velocity[0]),.001)
        follower.reset()
        self.assertEqual(follower.diagnostics()['bias_samples'],0)
        for _ in range(100): result=follower.update(observation(-4.))
        self.assertLess(result.velocity[0],-.1)

    def test_reserved_policies_rejected_and_arm_execution_guard(self):
        for name in ('teacher','student'):
            with self.assertRaises(NotImplementedError): make_follower(name)
        with self.assertRaises(ValueError): checked_action(HighAction(np.zeros(3),np.ones(6)),.2,.2)
        with self.assertRaises(ValueError): checked_action(HighAction(np.ones(3),np.zeros(6)),.2,.2)
        hold=make_follower('hold')
        for _ in range(100): output=hold.update(observation(4.))
        self.assertEqual(output.velocity.sum(),0.)
        self.assertGreater(hold.diagnostics()['suggestion'][0],.1)

    def test_keyboard_and_leader_slew_only_forward(self):
        keyboard=LeaderKeyboard(.1,.15)
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(10): keyboard.on_key(ord('W'))
            self.assertEqual(keyboard.snapshot()[0],.15)
            keyboard.on_key(ord('X'));self.assertEqual(keyboard.snapshot()[0],0.)
            keyboard.on_key(ord('R'));self.assertEqual(keyboard.snapshot()[0],.1)
            keyboard.on_key(ord('P'));self.assertTrue(keyboard.snapshot()[1])
        np.testing.assert_allclose(leader_command(np.zeros(3),.1,.02,.05,False),[.001,0,0])
        np.testing.assert_allclose(leader_command(np.array([.1,0,0]),.1,.02,.05,True),[.099,0,0])

    def test_third_robot_calibration_is_not_ignored(self):
        rows=[dict(time_s=2+i*.02,grasp_errors_m=[0,0,0],robots=[
            dict(velocity_vx_vy_wz=[0,0,0]),dict(velocity_vx_vy_wz=[0,0,0]),
            dict(velocity_vx_vy_wz=[.2,0,0])]) for i in range(50)]
        self.assertFalse(calibration_quality(rows,3.,1.,.02)['passed'])
        rows[-1]['robots'][2]['velocity_vx_vy_wz']=[0,0,0]
        self.assertEqual(len(calibration_quality(rows,3.,1.,.02)['robots']),3)

    def test_cli_rollouts_and_actual_command_routing(self):
        from rsl_rl.modules import ActorCritic,MLP_Encoder
        with tempfile.TemporaryDirectory(dir=ROOT/'.test-temp') as folder,contextlib.redirect_stdout(io.StringIO()):
            folder=Path(folder)
            actor=ActorCritic(48,126,10,actor_hidden_dims=[16],critic_hidden_dims=[16])
            encoder=MLP_Encoder(420,3,hidden_dims=[16])
            for p in actor.parameters(): p.data.zero_()
            with (folder/'fixture.pt').open('wb') as stream:
                torch.save(dict(sfyg_reset_contract=fixture_contract(),model_state_dict=actor.state_dict(),
                    encoder_state_dict=encoder.state_dict()),stream)
            export(folder/'fixture.pt',folder/'bundle')
            for layout,n in (('leader-follow',2),('sofa',3)):
                out=folder/layout
                args=cli.parse_args(['--scenario',layout,'--bundle_dir',str(folder/'bundle'),
                    '--headless','--hold_all','--duration','.3','--trace','--output_dir',str(out)])
                cli.run(args)
                report=json.loads((out/'evaluation.json').read_text())
                self.assertTrue(report['completed'])
                self.assertEqual(len(report['robots']),n)
                self.assertFalse(report['payload_floor_contact'])
                self.assertLess(report['max_extraction_error'],1e-8)
                with np.load(out/'trace.npz') as trace:
                    self.assertEqual(set(trace['robot']),set(range(n)))
                    self.assertEqual(trace['history'].shape[1:],(10,42))
                    self.assertFalse(trace['command'].any())
            args=cli.parse_args(['--scenario','sofa','--bundle_dir',str(folder/'bundle'),
                '--headless','--settle_seconds','.1','--calibration_seconds','.04','--output_dir',str(folder/'reject')])
            with patch.object(cli,'calibration_quality',return_value={'passed':False}): cli.run(args)
            report=json.loads((folder/'reject/evaluation.json').read_text())
            self.assertEqual(report['reason'],'unsettled_calibration')
            rows=[json.loads(line) for line in (folder/'reject/steps.jsonl').read_text().splitlines()]
            self.assertTrue(all(not np.any(r['command']) for row in rows for r in row['robots']))
            # Inject deterministic high-level adapters ONLY in this routing test.
            # Physics still advances; these fixture weights do not prove walking.
            class Spy:
                def __init__(self,value): self.value=value;self.seen=[]
                def update(self,o):
                    self.seen.append(o)
                    return HighAction(np.array([0. if o.settling else self.value,0,0]),np.zeros(6))
                def diagnostics(self): return dict(kind='test_spy')
            spies=[Spy(.03),Spy(.07)]
            args=cli.parse_args(['--scenario','sofa','--bundle_dir',str(folder/'bundle'),
                '--headless','--duration','8','--settle_seconds','.1','--calibration_seconds','.04',
                '--trace','--output_dir',str(folder/'routing')])
            with patch.object(cli,'make_follower',side_effect=spies),patch.object(cli,'calibration_quality',return_value={'passed':True}):
                cli.run(args)
            with np.load(folder/'routing/trace.npz') as trace:
                active=trace['time_s']>.1
                for i,value in ((1,.03),(2,.07)):
                    selected=trace['command'][active & (trace['robot']==i)]
                    self.assertGreater(len(selected),0)
                    np.testing.assert_allclose(selected[:,0],value)
                self.assertTrue(np.any(trace['command'][active & (trace['robot']==0),0]>0))
            self.assertEqual(len(spies[0].seen),len(spies[1].seen))
            self.assertIsNot(spies[0].seen[0].arm_position_error,spies[1].seen[0].arm_position_error)

    def test_visual_overlay_is_read_only(self):
        c=fixture_config();s=CooperativeScene([c]*3,spacing=1.1,layout='sofa',grasp_model='ball')
        before=[a.copy() for a in (s.data.qpos,s.data.qvel,s.data.ctrl,s.data.xfrc_applied)]
        visual=mujoco.MjvScene(s.model,maxgeom=100)
        draw_team(visual,[s.data.xpos[r.base] for r in s.robots],s.data.xpos[s.payload],
            [np.zeros(3)]*3,[np.ones(6)]*3,'FOLLOW',1.,'admittance')
        for a,b in zip(before,(s.data.qpos,s.data.qvel,s.data.ctrl,s.data.xfrc_applied)): np.testing.assert_array_equal(a,b)
        self.assertGreater(visual.ngeom,0)


if __name__=='__main__': unittest.main()
