"""Actual TorchScript/MuJoCo checks, independent of Isaac or trained server files."""
import ast
import contextlib
from copy import deepcopy
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

import torch
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/mujoco"), str(ROOT / "rsl_rl")]
from sfyg_policy import LEG_NAMES, HOLD_NAMES, TERM_NAMES, deployment_config, make_observation, ObservationHistory
from sfyg_bridge import SFYGBridge, rotation
from sfyg_asset_reference import load_reference
from export_sfyg import export


def fixture_contract():
    defaults = {name:0. for name in LEG_NAMES + HOLD_NAMES}
    defaults.update(proximal_yaw_L_Joint=-3.14159, proximal_yaw_R_Joint=3.14159,
                    arm2_Joint=1.5707963, arm3_Joint=-1.4835299, gripper1_Joint=.05, gripper2_Joint=-.05)
    terms = {}
    for name, scale in zip(TERM_NAMES, [.25,1.,1.,.05,1.,None,None]):
        params = {"asset_cfg":{"joint_names":LEG_NAMES}} if name in ("joint_pos","joint_vel") else {}
        terms[name] = dict(func=name,params=params,scale=scale,clip=[-100.,100.] if scale is not None else None)
    history = dict(deepcopy(terms), history_length=10, flatten_history_dim=False)
    def actuator(pattern, kp, kd, effort, armature, delay=4):
        return dict(joint_names_expr=pattern, stiffness=kp, damping=kd, effort_limit=effort,
                    velocity_limit=12.57, armature=armature, min_delay=0,max_delay=delay)
    robot = dict(init_state=dict(joint_pos=defaults,pos=[0,0,.85],rot=[1,0,0,0]),
        spawn=dict(articulation_props=dict(enabled_self_collisions=True)),
        actuators=dict(heavy=actuator(["proximal_pitch_[LR]_Joint","proximal_roll_[LR]_Joint","knee_[LR]_Joint"],159.67,10.16,140,.161777558),
                       light=actuator(["proximal_yaw_[LR]_Joint","ankle_pitch_[LR]_Joint"],53.22,3.39,40,.053923687),
                       hold=actuator(["arm[1-6]_Joint","gripper[12]_Joint"],800,40,200,0,0)))
    return dict(schema="sfyg_source_reset_v1",source_commit="f441d67",phase="robust",
        observation_shapes=dict(policy=[42],obsHistory=[10,42],commands=[3]),arguments=dict(payload_mass=.5),
        agent=dict(empirical_normalization=False,policy=dict(actor_hidden_dims=[16],critic_hidden_dims=[16],activation="elu"),
                   encoder=dict(hidden_dims=[16],activation="elu")),
        environment=dict(observations=dict(policy=terms,obsHistory=history),scene=dict(robot=robot),
            actions=dict(joint_pos=dict(joint_names=LEG_NAMES,clip=None,use_default_offset=True,scale=.25)),
            commands=dict(gait_command=dict(ranges=dict(frequencies=[.8,.8],offsets=[.5,.5],durations=[.5,.5],swing_height=[.2,.2]))),
            sim=dict(dt=.005),decimation=4))


def fixture_config():
    cfg = deployment_config(fixture_contract())
    ref = load_reference()
    cfg.update(usd_sources=ref["sources"],usd_mass_kg=ref["total_mass_kg"])
    return cfg


class SFYGMujocoTests(unittest.TestCase):
    def test_source_contract_rejects_changed_semantics(self):
        c = fixture_contract()
        c["environment"]["actions"]["joint_pos"]["clip"] = [-4,4]
        with self.assertRaises(ValueError): deployment_config(c)
        c = fixture_contract()
        c["environment"]["observations"]["obsHistory"]["joint_vel"]["scale"] = .1
        with self.assertRaises(ValueError): deployment_config(c)
        c = fixture_contract()
        c["phase"] = "v4"
        with self.assertRaises(ValueError): deployment_config(c)

    def test_observation_order_scaling_clipping_and_episode_phase(self):
        cfg = fixture_config()
        obs = make_observation([4,8,12],[0,0,-1],np.array(cfg["defaults"][:10])+np.arange(10),
                               [200]*10,np.arange(10),25,cfg)
        np.testing.assert_allclose(obs[:6],[1,2,3,0,0,-1])
        np.testing.assert_allclose(obs[6:16],np.arange(10),atol=1e-6)
        np.testing.assert_allclose(obs[16:26],[5]*10)  # clip before scaling
        np.testing.assert_allclose(obs[26:36],np.arange(10))
        np.testing.assert_allclose(obs[36:38],[np.sin(.8*np.pi),np.cos(.8*np.pi)],atol=1e-6)
        np.testing.assert_allclose(obs[38:],[.8,.5,.5,.2])

    def test_history_padding_order_and_independent_reset(self):
        first, second = ObservationHistory(), ObservationHistory()
        np.testing.assert_array_equal(first.push(np.ones(42)),np.ones((10,42)))
        for i in range(2,13): first.push(np.full(42,i))
        np.testing.assert_array_equal(first.frames[:,0],np.arange(3,13))
        second.push(np.zeros(42))
        first.reset()
        np.testing.assert_array_equal(first.push(np.full(42,7)),np.full((10,42),7))
        np.testing.assert_array_equal(second.frames,np.zeros((10,42)))

    def test_usd_mass_inertia_and_joint_frame_forward_kinematics(self):
        bridge = SFYGBridge(fixture_config(), delay_steps=0)
        audit = bridge.audit()
        self.assertAlmostEqual(audit["total_mass_kg"],40.023,places=5)
        self.assertLess(audit["authored_inertia_error"],1e-9)
        self.assertEqual(audit["root_free_joints"],1)
        self.assertTrue(np.isfinite(bridge.model.body_ipos).all())
        bridge.data.qpos[bridge.qpos] = 0
        mujoco.mj_forward(bridge.model,bridge.data)
        for name,j in bridge.reference["joints"].items():
            parent,child = bridge.model.body(j["parent"]).id,bridge.model.body(j["child"]).id
            rp = bridge.data.xmat[parent].reshape(3,3)
            rc = bridge.data.xmat[child].reshape(3,3)
            r = rotation(j["rot0"]) @ rotation(j["rot1"]).T
            np.testing.assert_allclose(rp.T @ rc,r,atol=1e-9,err_msg=name)
            expected = np.array(j["pos0"])-r @ np.array(j["pos1"])
            np.testing.assert_allclose(rp.T @ (bridge.data.xpos[child]-bridge.data.xpos[parent]),expected,atol=1e-9)
        payload = bridge.payload
        self.assertAlmostEqual(bridge.model.body_mass[payload],.5)
        np.testing.assert_allclose(bridge.model.body_pos[payload],[.15,0,0])

    def test_original_pd_force_caps_no_action_clip_and_upperbody_hold(self):
        bridge = SFYGBridge(fixture_config(), delay_steps=0)
        action = np.zeros(10,np.float32)
        action[0] = 10  # Must remain 10, not a Go2-style clipped 4.
        bridge.set_action(action)
        bridge.physics_step()
        self.assertEqual(bridge.previous_action[0],10)
        self.assertAlmostEqual(bridge.target[0]-bridge.defaults[0],2.5)
        np.testing.assert_array_equal(bridge.data.ctrl[bridge.motors[10:]],bridge.defaults[10:])
        self.assertLessEqual(abs(bridge.data.actuator_force[bridge.motors[0]]),140+1e-6)
        bridge.reset()
        bridge.set_action(np.r_[.1,np.zeros(9)])
        bridge.data.ctrl[bridge.motors] = bridge.target
        bridge.data.qvel[bridge.qvel[0]] = .1
        mujoco.mj_forward(bridge.model,bridge.data)
        self.assertAlmostEqual(bridge.data.actuator_force[bridge.motors[0]],159.67*.025-10.16*.1,places=6)

    def test_delays_force_application_and_reset(self):
        bridge = SFYGBridge(fixture_config(),delay_steps=2)
        bridge.set_action(np.zeros(10))
        bridge.physics_step([1.,-1.,0.])
        bridge.set_action(np.ones(10))
        bridge.physics_step([1.,-1.,0.])
        np.testing.assert_allclose(bridge.data.ctrl[bridge.motors[:10]],bridge.defaults[:10])
        bridge.physics_step()
        np.testing.assert_allclose(bridge.data.ctrl[bridge.motors[:10]],bridge.defaults[:10])
        bridge.physics_step()
        np.testing.assert_allclose(bridge.data.ctrl[bridge.motors[:10]],bridge.defaults[:10]+.25)
        bridge.physics_step([1.,-1.,0.])
        np.testing.assert_array_equal(bridge.data.xfrc_applied[bridge.payload],[1,-1,0,0,0,0])
        self.assertEqual(np.count_nonzero(bridge.data.xfrc_applied),2)
        bridge.reset()
        self.assertEqual(np.count_nonzero(bridge.data.xfrc_applied),0)
        self.assertIsNone(bridge.history.frames)

    def test_export_actual_torchscript_and_cli_report(self):
        from rsl_rl.modules import ActorCritic, MLP_Encoder
        (ROOT / ".test-temp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sfyg_test_",dir=ROOT / ".test-temp") as folder, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(folder)
            actor = ActorCritic(48,126,10,actor_hidden_dims=[16],critic_hidden_dims=[16])
            encoder = MLP_Encoder(420,3,hidden_dims=[16])
            checkpoint = folder / "model.pt"
            with checkpoint.open("wb") as stream:
                torch.save(dict(sfyg_reset_contract=fixture_contract(),model_state_dict=actor.state_dict(),
                                encoder_state_dict=encoder.state_dict()),stream)
            cfg = export(checkpoint,folder / "bundle")
            self.assertEqual(cfg["export_max_abs_error"],0.)
            with (folder / "bundle/policy.pt").open("rb") as stream:
                deployed = torch.jit.load(stream)
            obs,hist,cmd = torch.randn(3,42),torch.randn(3,10,42),torch.randn(3,3)
            torch.testing.assert_close(deployed(obs,hist,cmd),actor.act_inference(torch.cat((encoder(hist.flatten(1)),obs,cmd),-1)))
            result = subprocess.run([sys.executable,str(ROOT / "scripts/mujoco/run_sfyg.py"),
                "--bundle_dir",str(folder / "bundle"),"--headless","--duration","0.1","--trace",
                "--output_dir",str(folder / "rollout")],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            report = json.loads((folder / "rollout/evaluation.json").read_text())
            self.assertTrue(report["completed"])
            self.assertFalse(report["is_acceptance"])
            with (folder / "rollout/steps.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),5)
            with np.load(folder / "rollout/trace.npz") as trace:
                self.assertEqual(trace["history"].shape,(5,10,42))

    def test_multiple_isaac_eval_resets_share_inference_scope(self):
        path = ROOT / "scripts/rsl_rl/train_sfyg_reset.py"
        function = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="evaluate")
        class Env:
            max_episode_length,step_dt = 1,.02
            def __init__(self): self.unwrapped=self; self.resets=0
            def reset(self,seed):
                if self.resets and not torch.is_inference_mode_enabled():
                    raise RuntimeError("Inference cache reset outside inference mode")
                self.resets+=1
            def get_observations(self):
                return dict(policy=torch.zeros(1,42),obsHistory=torch.zeros(1,10,42),commands=torch.zeros(1,3))
            def step(self,action):
                state = dict(force_z=torch.ones(1,2),velocity=torch.zeros(1,3),tilt_deg=torch.zeros(1),
                             arm_error=torch.zeros(1,6),command=torch.zeros(1,3))
                return self.get_observations(),torch.zeros(1),torch.ones(1,dtype=torch.bool),dict(reset_state=state,time_outs=torch.zeros(1,dtype=torch.bool))
        scope={"json":json}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),"exec"),scope)
        runner=types.SimpleNamespace(get_inference_policy=lambda:lambda x:torch.zeros(1,10),
                                     get_inference_encoder=lambda:lambda x:torch.zeros(1,3))
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-temp") as folder, contextlib.redirect_stdout(io.StringIO()):
            env=Env()
            scope["evaluate"](env,runner,types.SimpleNamespace(video=False,episodes=3,seed=42,command=[.25,0,0],phase="robust"),
                              Path(folder),types.SimpleNamespace(is_running=lambda:True))
            self.assertEqual(env.resets,3)
            self.assertEqual(len(json.loads((Path(folder)/"evaluation.json").read_text())["episodes"]),3)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
