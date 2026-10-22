"""Independent inverse dynamics and actual publication-path checks."""
import ast
from dataclasses import dataclass, replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import time
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("arm_gravity_compensation",
    ROOT / "scripts/mujoco/arm_gravity_compensation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GravityTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string('''<mujoco>
          <worldbody><body name="base_Link"><body name="arm1" pos="0 0 1">
          <joint name="arm1_Joint" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 .4 0 0" size=".02" mass="2" contype="0" conaffinity="0"/>
          <body name="payload" pos=".4 0 0"><joint name="arm2_Joint" axis="0 1 0"/>
          <geom type="box" pos=".1 0 0" size=".1 .05 .05" mass=".5" contype="0" conaffinity="0"/>
          </body></body></body></worldbody>
          <actuator><motor joint="arm1_Joint" ctrllimited="true" ctrlrange="-7 7"/>
          <motor joint="arm2_Joint" ctrllimited="true" ctrlrange="-100 100"/></actuator>
          </mujoco>''')
        self.mapping = {n: n for n in ("arm1_Joint", "arm2_Joint")}
        self.comp = module.ArmGravityCompensation(self.model, self.mapping, list(self.mapping))

    def test_matches_independent_rne_for_changed_pose_tilt_and_payload(self):
        for q, gravity in (([0., 0.], [0., 0., -9.81]),
                           ([.3, -.2], [-2., 1., -9.5])):
            self.model.opt.gravity[:] = gravity
            data = mujoco.MjData(self.model)
            data.qpos[:] = q
            mujoco.mj_forward(self.model, data)
            expected = np.empty(self.model.nv)
            mujoco.mj_rne(self.model, data, 0, expected)
            actual = self.comp.compute(dict(zip(self.mapping, q)), gravity)
            np.testing.assert_allclose(actual, expected, atol=1e-10)
            data.ctrl[:] = actual
            mujoco.mj_forward(self.model, data)
            np.testing.assert_allclose(data.qacc, 0., atol=1e-9)
        # Analytic unloaded+payload torque at the horizontal pose, holding sign.
        actual = self.comp.compute(dict.fromkeys(self.mapping, 0.), [0, 0, -9.81])
        np.testing.assert_allclose(actual, [-9.81*(2*.2+.5*.5), -9.81*.5*.1])

    def test_ramp_cap_and_private_state(self):
        positions = dict.fromkeys(self.mapping, 0.)
        before = self.model.qpos0.copy()
        raw = self.comp.compute(positions, [0, 0, -9.81])
        np.testing.assert_array_equal(self.model.qpos0, before)
        np.testing.assert_allclose(self.comp.command(raw, 0.)[0], 0.)
        applied, weight = self.comp.command([-20, 40], 1.)
        self.assertEqual(weight, .5)
        np.testing.assert_allclose(applied, [-3.5, 15.])
        with self.assertRaises(ValueError):
            self.comp.compute({}, [0, 0, -9.81])

    def test_gravity_reaches_publication_without_mutating_input_solution(self):
        # Execute the real wrapper method with a fake SDK parent, rather than
        # testing an inference-only replacement which never reaches the motor.
        tree = ast.parse((ROOT / "scripts/mujoco/run_wholebody_v3.py").read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_publish_policy_targets")
        class Parent:
            def _publish_policy_targets(self, targets, solution):
                self.sent = solution.arm_effort.copy()
        @dataclass
        class Solution:
            arm_effort: np.ndarray
            time: float = 0.
        class Comp:
            model = SimpleNamespace(opt=SimpleNamespace(gravity=np.array([0, 0, -9.81])))
            limits = np.ones(6)*30
            def compute(self, positions, gravity):
                return np.arange(6, dtype=float)
            def command(self, raw, elapsed):
                return raw, 1.
        wrapper = ast.ClassDef(name="Wrapper", bases=[ast.Name(id="Parent", ctx=ast.Load())],
                               keywords=[], body=[method], decorator_list=[])
        namespace = dict(Parent=Parent, arm_gravity=Comp(), arm_requested=False,
                         np=np, replace=replace, time=time,
                         projected_gravity_from_wxyz=lambda _: np.array([0, 0, -1.]))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                     "publication_test", "exec"), namespace)
        controller = namespace["Wrapper"]()
        controller.arm_test = controller.trajectory = None
        controller.config = SimpleNamespace(wire_joint_names=list(range(18)))
        controller.rl_started = time.monotonic()-3.
        controller._snapshot = lambda: (np.zeros(18), np.zeros(18), [1, 0, 0, 0], np.zeros(3))
        solution = Solution(np.zeros(6))
        controller._publish_policy_targets(np.zeros(18), solution)
        np.testing.assert_array_equal(controller.sent, np.arange(6))
        np.testing.assert_array_equal(solution.arm_effort, 0.)
        self.assertFalse(controller._record_policy_publish)


if __name__ == "__main__":
    unittest.main()
