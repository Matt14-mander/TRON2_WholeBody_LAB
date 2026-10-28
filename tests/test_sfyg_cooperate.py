"""Real MuJoCo grasp physics and FT control wiring; fixture policies do not prove walking."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET

import torch
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts/mujoco"),str(ROOT/"tests"),str(ROOT/"rsl_rl")]
from test_sfyg_mujoco import fixture_config,fixture_contract
from sfyg_cooperative_control import PayloadKeyboard,LocalAdmittance
from sfyg_cooperative_scene import CooperativeScene,SharedRobot,build_scene_xml,initialize,grasp_reaction,audit_reactions
from sfyg_cooperative_visuals import draw
from export_sfyg import export


def supported_fixture():
    """Fixed base supports ONLY in this independent load-cell test, never the demo."""
    cfg=fixture_config()
    root,assets,_,_=build_scene_xml([cfg,cfg])
    for i in range(2):
        ET.SubElement(root.find("equality"),"weld",name=f"test_support_{i}",body1=f"r{i}/base_Link",solref=".01 1")
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"),assets)
    data=mujoco.MjData(model)
    payload=initialize(model,data,[cfg,cfg])
    robots=[SharedRobot(cfg,model,data,i,0,42+i) for i in range(2)]
    return types.SimpleNamespace(model=model,data=data,payload=payload,robots=robots)


def hold_and_step(scene,wrench,steps=400):
    for _ in range(steps):
        for r in scene.robots: r.write_controls()
        scene.data.xfrc_applied[:]=0
        scene.data.xfrc_applied[scene.payload]=wrench
        mujoco.mj_step(scene.model,scene.data)
    mujoco.mj_forward(scene.model,scene.data)
    return [grasp_reaction(scene.model,scene.data,r.eq,scene.payload) for r in scene.robots]


class CooperativeTests(unittest.TestCase):
    def test_three_free_roots_single_mass_and_no_world_support(self):
        cfg=fixture_config()
        scene=CooperativeScene([cfg,cfg])
        self.assertEqual(np.count_nonzero(scene.model.jnt_type==mujoco.mjtJoint.mjJNT_FREE),3)
        self.assertEqual(scene.model.neq,2)
        self.assertAlmostEqual(scene.model.body_mass.sum(),80.046,places=5)
        self.assertEqual(scene.model.body("payload").parentid,0)
        self.assertLess(max(scene.grasp_errors()),1e-8)
        self.assertTrue(np.all(scene.model.eq_obj1id>0))
        self.assertTrue(np.all(scene.model.eq_objtype==mujoco.mjtObj.mjOBJ_SITE))
        self.assertTrue(np.all(scene.model.site_bodyid[scene.model.eq_obj2id]==scene.payload))
        self.assertEqual(mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_BODY,"r0/paint_payload_Link"),-1)
        r0,r1=scene.robots
        r0.set_action(np.ones(10))
        self.assertEqual(np.count_nonzero(r1.previous_action),0)
        r0.observation()
        self.assertIsNone(r1.history.frames)
        with self.assertRaises(RuntimeError): r0.physics_step()
        scene.step([2,-1,0,0,0,.1])
        self.assertEqual(np.count_nonzero(scene.data.xfrc_applied),3)
        np.testing.assert_array_equal(scene.data.xfrc_applied[scene.payload],[2,-1,0,0,0,.1])
        scene.reset()
        self.assertEqual(np.count_nonzero(scene.data.xfrc_applied),0)
        self.assertIsNone(r0.history.frames)
        self.assertEqual(r0.control_steps,0)

    def test_static_weight_and_signed_horizontal_force_torque(self):
        s=supported_fixture()
        for applied in ([0,0,0,0,0,0],[2,-1,0,0,0,.1],[-2,1,0,0,0,-.1]):
            reactions=hold_and_step(s,applied,500)
            audit=audit_reactions(s,reactions)
            self.assertLess(audit["extraction_max"],1e-8)
            self.assertLess(audit["dynamics_max"],1e-5)
            support=np.array(audit["payload_grasp_world"])
            np.testing.assert_allclose(support[:3],[-applied[0],-applied[1],9.81],atol=.08)
            self.assertAlmostEqual(support[5],-applied[5],delta=.02)
            self.assertTrue(all(w[2]<0 for w in reactions))  # Box pushes down on each robot.
        faulty=audit_reactions(s,[-r for r in reactions])
        self.assertGreater(faulty["extraction_max"],1)

    def test_released_box_falls_with_gravity(self):
        s=supported_fixture()
        hold_and_step(s,[0,0,0,0,0,0],100)
        for r in s.robots: s.data.eq_active[r.eq]=False
        s.data.xfrc_applied[:]=0
        mujoco.mj_forward(s.model,s.data)
        start=s.model.jnt_dofadr[s.model.body_jntadr[s.payload]]
        self.assertAlmostEqual(s.data.qacc[start+2],-9.81,places=6)
        with self.assertRaises(ValueError): grasp_reaction(s.model,s.data,s.robots[0].eq,s.payload)

    def test_keyboard_axes_force_norm_and_release(self):
        k=PayloadKeyboard()
        k.on_key(ord("W")); k.on_key(ord("A")); k.on_key(ord("Q"))
        np.testing.assert_allclose(k.snapshot()[0],[.5,.5,.05])
        for _ in range(20): k.on_key(ord("W"))
        self.assertLessEqual(np.linalg.norm(k.snapshot()[0][:2]),4+1e-12)
        k.on_key(ord("X"))
        np.testing.assert_array_equal(k.snapshot()[0],[0,0,0])
        k.on_key(ord("S")); k.on_key(ord("D")); k.on_key(ord("E"))
        np.testing.assert_allclose(k.snapshot()[0],[-.5,-.5,-.05])
        k.on_key(ord("P")); self.assertTrue(k.snapshot()[1])
        k.on_key(ord("R")); np.testing.assert_array_equal(k.snapshot()[0],[0,0,0])

    def test_admittance_bias_own_frame_response_and_release_decay(self):
        a=LocalAdmittance(mass=10,damping=20,force_deadband=0,torque_deadband=0,max_speed=.5)
        b=LocalAdmittance(mass=10,damping=20,force_deadband=0,torque_deadband=0,max_speed=.5)
        bias=np.array([.3,-.2,-4.905,.1,.2,.03])
        for _ in range(50):
            a.update(bias,np.eye(3),.02,settling=True,calibrating=True)
            b.update(bias,np.eye(3),.02,settling=True,calibrating=True)
        np.testing.assert_allclose(a.bias,bias)
        for _ in range(400):
            command=a.update(bias+[2,0,0,0,0,0],np.eye(3),.02)
            b.update(bias,np.eye(3),.02)
        self.assertAlmostEqual(command[0],.1,delta=1e-4)
        np.testing.assert_allclose(b.command,0,atol=1e-12)
        # World +X should be body -Y after a +90deg turn. No partner pose enters.
        r=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        for _ in range(100): command=a.update(bias+[2,0,0,0,0,0],r,.02)
        np.testing.assert_allclose(command,[0,-.1,0],atol=1e-4)
        for _ in range(400): command=a.update(bias,r,.02)
        self.assertLess(np.linalg.norm(command),1e-5)

    def test_admittance_saturation_acceleration_and_invalid_inputs(self):
        a=LocalAdmittance()
        previous=np.zeros(3)
        for _ in range(100):
            current=a.update([1000,1000,-10,0,0,20],np.eye(3),.02)
            self.assertLessEqual(np.linalg.norm(current[:2]),.15+1e-12)
            self.assertLessEqual(abs(current[2]),.2+1e-12)
            self.assertLessEqual(np.linalg.norm(current[:2]-previous[:2]),.002+1e-12)
            previous=current
        with self.assertRaises(ValueError): a.update([np.nan]*6,np.eye(3),.02)
        with self.assertRaises(ValueError): LocalAdmittance(damping=0)

    def test_visuals_are_read_only(self):
        c=fixture_config()
        s=CooperativeScene([c,c])
        before=(s.data.qpos.copy(),s.data.qvel.copy(),s.data.xfrc_applied.copy())
        visual=mujoco.MjvScene(s.model,maxgeom=32)
        draw(visual,s.data.xpos[s.payload],[2,1,0,0,0,.1],
             [s.data.xpos[r.base] for r in s.robots],[np.zeros(3)]*2,[np.ones(6)]*2,"TEST")
        self.assertGreater(visual.ngeom,0)
        for expected,actual in zip(before,(s.data.qpos,s.data.qvel,s.data.xfrc_applied)):
            np.testing.assert_array_equal(expected,actual)

    def test_real_cli_admittance_and_hold_outputs(self):
        from rsl_rl.modules import ActorCritic,MLP_Encoder
        (ROOT/".test-temp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/".test-temp") as folder,contextlib.redirect_stdout(io.StringIO()):
            folder=Path(folder)
            actor=ActorCritic(48,126,10,actor_hidden_dims=[16],critic_hidden_dims=[16])
            encoder=MLP_Encoder(420,3,hidden_dims=[16])
            for p in actor.parameters(): p.data.zero_()
            checkpoint=folder/"fixture.pt"
            with checkpoint.open("wb") as stream:
                torch.save(dict(sfyg_reset_contract=fixture_contract(),model_state_dict=actor.state_dict(),
                                encoder_state_dict=encoder.state_dict()),stream)
            export(checkpoint,folder/"bundle")
            for controller in ("admittance","hold"):
                result=subprocess.run([sys.executable,str(ROOT/"scripts/mujoco/run_sfyg_cooperate.py"),
                    "--bundle_dir",str(folder/"bundle"),"--headless","--duration",".3",
                    "--settle_seconds",".06","--calibration_seconds",".04","--release_tail",".04",
                    "--force_world","2","0","--force_slew","100","--controller",controller,"--trace",
                    "--output_dir",str(folder/controller)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr+result.stdout)
                report=json.loads((folder/controller/"evaluation.json").read_text())
                self.assertFalse(report["is_acceptance"])
                self.assertEqual(report["interface"]["free_roots"],3)
                self.assertEqual(report["interface"]["controller"],controller)
                rows=[json.loads(line) for line in (folder/controller/"steps.jsonl").read_text().splitlines()]
                self.assertTrue(any(r["applied_world_wrench"][0]>0 for r in rows))
                self.assertLess(report["max_extraction_error"],1e-8)
                if controller=="hold":
                    self.assertTrue(all(np.count_nonzero(bot["command"])==0 for row in rows for bot in row["robots"]))
                else:
                    self.assertTrue(any(np.linalg.norm(bot["command"])>0 for row in rows for bot in row["robots"]))
                with np.load(folder/controller/"trace.npz") as trace:
                    self.assertEqual(trace["history"].shape[1:],(10,42))
                    self.assertEqual(set(trace["robot"]),{0,1})


if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main()
