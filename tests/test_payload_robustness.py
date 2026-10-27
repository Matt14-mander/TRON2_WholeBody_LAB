"""Bounds, noncumulative randomization, frame conversion and reset isolation."""
import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("payload_rules",
    ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers/payload_robustness.py")
rules = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rules
spec.loader.exec_module(rules)


def native_functions():
    path = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_sfyg_robust_env_cfg.py"
    tree = ast.parse(path.read_text())
    names = {"RobustState","robust_state","randomize_payload_mass","randomize_friction","actual_masses","RobustCarryLegAction"}
    class Base:
        def apply_actions(self): self.base_calls += 1
        def reset(self,ids=None): self.base_reset = ids
    def inverse(q,v):
        xyz = -q[:,1:]
        return v + 2*(q[:,:1]*torch.cross(xyz,v,dim=-1) + torch.cross(xyz,torch.cross(xyz,v,dim=-1),dim=-1))
    namespace = {"torch":torch,"RobustSettings":rules.RobustSettings,"ForceSchedule":rules.ForceSchedule,
        "nominal_mask":rules.nominal_mask,"scale_payload_inertia":rules.scale_payload_inertia,
        "CarryLegAction":Base,"quat_apply_inverse":inverse}
    exec(compile(ast.Module(body=[n for n in tree.body if getattr(n,"name",None) in names],type_ignores=[]),str(path),"exec"),namespace)
    return namespace


def fake_env(namespace):
    default_mass = torch.tensor([[10.,.5,2.]]).repeat(12,1)
    default_inertia = torch.ones(12,3,9)
    masses, inertias = default_mass.clone(), default_inertia.clone()
    material = torch.tensor([.5,.4,.2]).repeat(12,4,1)
    class View:
        def get_masses(self): return masses.clone()
        def get_inertias(self): return inertias.clone()
        def get_material_properties(self): return material.clone()
        def set_masses(self,value,ids): masses[ids] = value[ids]
        def set_inertias(self,value,ids): inertias[ids] = value[ids]
        def set_material_properties(self,value,ids): material[ids] = value[ids]
    robot = types.SimpleNamespace(root_physx_view=View(),find_bodies=lambda *a,**k:([1],["paint_payload_Link"]),
        data=types.SimpleNamespace(default_mass=default_mass,default_inertia=default_inertia,
            root_lin_vel_b=torch.zeros(12,3)))
    from dataclasses import asdict
    env = types.SimpleNamespace(scene={"robot":robot},num_envs=12,device="cpu",
        cfg=types.SimpleNamespace(robustness=asdict(rules.RobustSettings())))
    return env, masses, inertias, material


class RobustTests(unittest.TestCase):
    def test_force_is_a_horizontal_disk_with_release_intervals(self):
        torch.manual_seed(2)
        force = rules.sample_force(20000,2.,.25,"cpu")
        self.assertLessEqual(float(force.norm(dim=-1).max()),2.)
        self.assertEqual(float(force[:,2].abs().max()),0.)
        self.assertAlmostEqual(float((force.norm(dim=-1)==0).float().mean()),.25,delta=.02)
        self.assertGreater(float(force[:,0].max()),1.9)
        self.assertLess(float(force[:,0].min()),-1.9)

    def test_force_schedule_warmup_nominal_and_partial_reset(self):
        mask = torch.tensor([True,False,False,False])
        cfg = rules.RobustSettings(zero_probability=0.)
        schedule = rules.ForceSchedule(mask,cfg)
        for _ in range(95):
            torch.testing.assert_close(schedule.advance(.005),torch.zeros(4,3))
        for _ in range(120):
            value = schedule.advance(.005)
            self.assertLessEqual(float(value.norm(dim=-1).max()),2.+1e-6)
            torch.testing.assert_close(value[0],torch.zeros(3))
        self.assertGreater(float(value[1].norm()),0.)
        before = value.clone()
        schedule.reset(torch.tensor([1]))
        torch.testing.assert_close(schedule.current[1],torch.zeros(3))
        torch.testing.assert_close(schedule.current[2:],before[2:])

    def test_mass_and_inertia_scale_from_authored_values_on_each_reset(self):
        n = native_functions()
        env,masses,inertias,_ = fake_env(n)
        ids = torch.arange(env.num_envs)
        for _ in range(3):
            n["randomize_payload_mass"](env,ids)
            s = env._sfyg_robust
            self.assertTrue(bool(((masses[:,1]>=.25)&(masses[:,1]<=.75)).all()))
            torch.testing.assert_close(inertias[:,1],(masses[:,1]/.5)[:,None].expand(-1,9))
            torch.testing.assert_close(masses[s.nominal,1],torch.full((int(s.nominal.sum()),),.5))
            torch.testing.assert_close(n["actual_masses"](env),masses)
            torch.testing.assert_close(masses[:,[0,2]],torch.tensor([10.,2.]).repeat(12,1))

    def test_material_buckets_preserve_nominal_and_restitution_without_compounding(self):
        n = native_functions()
        env,_,_,materials = fake_env(n)
        for _ in range(3):
            n["randomize_friction"](env,None)
            s = env._sfyg_robust
            torch.testing.assert_close(materials[...,:2],s.materials[...,:2]*s.frictions[:,None,None])
            torch.testing.assert_close(materials[...,2],torch.full((12,4),.2))
            torch.testing.assert_close(s.frictions[s.nominal],torch.ones(int(s.nominal.sum())))
            self.assertTrue(bool(((s.frictions>=.8)&(s.frictions<=1.2)).all()))

    def test_force_reapplied_in_local_frame_each_substep_and_reset_clears_only_selected(self):
        n = native_functions()
        env,_,_,_ = fake_env(n)
        s = n["robust_state"](env)
        s.schedule.nominal[:] = False
        s.schedule.ttl[:] = 5.
        s.schedule.current[:] = torch.tensor([2.,0.,0.])
        s.schedule.start.copy_(s.schedule.current)
        s.schedule.target.copy_(s.schedule.current)
        calls = []
        q = torch.tensor([1.,0.,0.,0.]).repeat(12,3,1)
        s.robot.data.body_quat_w = q
        s.robot.set_external_force_and_torque = lambda f,t,**k:calls.append((f.clone(),t.clone(),k))
        env.physics_dt = .005
        action = object.__new__(n["RobustCarryLegAction"])
        action.state,action._env,action._asset,action.base_calls = s,env,s.robot,0
        action.apply_actions()
        q[:,1] = torch.tensor([2**-.5,0.,0.,2**-.5])
        action.apply_actions()
        torch.testing.assert_close(calls[0][0][:,0],torch.tensor([2.,0.,0.]).repeat(12,1))
        torch.testing.assert_close(calls[1][0][:,0],torch.tensor([0.,-2.,0.]).repeat(12,1),atol=1e-6,rtol=0)
        q[:,1] = torch.tensor([2**-.5,2**-.5,0.,0.])
        for value in (s.schedule.current,s.schedule.start,s.schedule.target):
            value[:] = torch.tensor([0.,2.,0.])
        action.apply_actions()
        torch.testing.assert_close(calls[2][0][:,0],torch.tensor([0.,0.,-2.]).repeat(12,1),atol=1e-6,rtol=0)
        self.assertEqual(action.base_calls,3)
        action.reset(torch.tensor([1]))
        torch.testing.assert_close(calls[-1][0],torch.zeros(1,1,3))
        self.assertEqual(calls[-1][2]["env_ids"].tolist(),[1])
        self.assertGreater(float(s.schedule.current[0].norm()),0.)
        self.assertEqual(float(s.schedule.current[1].norm()),0.)

    def test_invalid_ranges_are_rejected(self):
        for args in ({"mass_range":(0.,1.)},{"max_force_n":float("nan")},
                     {"friction_scale_range":(1.2,.8)},{"nominal_fraction":1.1}):
            with self.assertRaises(ValueError): rules.RobustSettings(**args).validate()

    def test_rollout_metrics_can_be_cleared_after_inference_mode(self):
        n = native_functions()
        env,_,_,_ = fake_env(n)
        s = n["robust_state"](env)
        with torch.inference_mode():
            s.observe(torch.tensor([.25,0.,0.]).repeat(12,1))
        values = s.pop_metrics()
        self.assertEqual(values["Motion/linear_progress_ratio"],0.)
        self.assertEqual(s.samples,0)


if __name__ == "__main__":
    unittest.main()
