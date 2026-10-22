"""Analytical force/moment checks independent of the Isaac runtime."""
import sys
from pathlib import Path
import tempfile
import unittest
import importlib.util
import threading
import time
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers"))
from paint_wrench_preview import (AddedBodyModel, MuJoCoAddedBodyModel, TIMES,
                                 LatestPreviewWorker, current_frame_preview)


class WrenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name)
        (path / "base.urdf").write_text('<robot name="base"><link name="base_Link"/></robot>')
        (path / "arm.urdf").write_text('''<robot name="test"><link name="base_Link"/>
          <link name="arm"><inertial><origin xyz="1 0 0"/>
          <mass value="2"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
          </inertial></link><joint name="q" type="revolute"><parent link="base_Link"/>
          <child link="arm"/><axis xyz="0 0 1"/></joint></robot>''')
        self.model = AddedBodyModel(path / "arm.urdf", path / "base.urdf")

    def tearDown(self):
        self.temp.cleanup()

    def test_weight_sign_and_moment_about_base_origin(self):
        gravity = np.array([0., 0., -9.81])
        np.testing.assert_allclose(self.model.gravity_wrench({"q": 0.}, gravity), [0, 0, -19.62, 0, 19.62, 0])
        np.testing.assert_allclose(self.model.gravity_wrench({"q": np.pi/2}, gravity),
                                   [0, 0, -19.62, -19.62, 0, 0], atol=1e-10)
        static = self.model.preview(lambda t: {"q": 0.}, 0., gravity)
        np.testing.assert_allclose(static, np.tile([0, 0, -19.62, 0, 19.62, 0], (5, 1)))

    def test_centripetal_reaction(self):
        preview = self.model.preview(lambda t: {"q": 2*t}, 0., np.zeros(3))
        np.testing.assert_allclose(preview[0], [8, 0, 0, 0, 0, 0], atol=2e-5)

    def test_angular_acceleration_includes_link_rotational_inertia(self):
        preview = self.model.preview(lambda t: {"q": 1.5*t*t}, 0., np.zeros(3))
        np.testing.assert_allclose(preview[0], [0, -6, 0, 0, 0, -6.3], atol=3e-5)
        self.assertGreater(np.linalg.norm(preview[-1]-preview[0]), 1.)

    def test_actual_robot_added_mass_and_static_horizon(self):
        model = AddedBodyModel(ROOT / "robot_description/tron2/SFYG_TRON2A/urdf/robot_rl.urdf",
            ROOT / "robot_description/tron2/SF_TRON2A/urdf/robot_rl.urdf", .5)
        q = {"arm2_Joint": np.pi/2, "arm3_Joint": np.deg2rad(-85)}
        w = model.gravity_wrench(q, np.array([0, 0, -9.81]))
        self.assertAlmostEqual(w[2], -sum(x[0] for x in model.bodies.values())*9.81)
        np.testing.assert_allclose(model.preview(lambda t:q, 0., np.array([0, 0, -9.81])), np.tile(w, (5, 1)))

    def test_simulation_gripper_sign_matches_negative_joint_coordinates(self):
        root = ROOT / "robot_description/tron2"
        reference = root / "SF_TRON2A/urdf/robot_rl.urdf"
        simulation = AddedBodyModel(root / "SFYG_TRON2A/urdf/robot_rl.urdf", reference)
        original = AddedBodyModel(root / "SFYG_TRON2A/urdf/robot.urdf", reference)
        q = {"arm2_Joint": np.pi/2, "arm3_Joint": np.deg2rad(-85),
             "gripper1_Joint": .04875, "gripper2_Joint": -.04875}
        actual = simulation.states(q)["gripper2_Link"]
        converted = dict(q, gripper2_Joint=.04875)
        np.testing.assert_allclose(actual, original.states(converted)["gripper2_Link"], atol=1e-10)
        wrong = original.states(q)["gripper2_Link"]
        self.assertAlmostEqual(np.linalg.norm(actual[:3, 3]-wrong[:3, 3]), .0975)

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo not installed")
    def test_mujoco_preview_uses_native_inertia_and_private_state(self):
        import mujoco
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="base_Link" pos="0 0 0.85"><freejoint name="floating_base"/>
          <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
          <body name="arm"><joint name="q" axis="0 0 1"/>
          <inertial pos="1 0 0" mass="2" diaginertia=".1 .1 .1"/>
          </body></body></worldbody></mujoco>''')
        live = mujoco.MjData(model)
        live.qpos[-1] = .73
        before = live.qpos.copy()
        # Emulate bindings whose enum objects do not compare to numpy ints.
        with patch.object(mujoco, "mjtJoint", SimpleNamespace(mjJNT_HINGE=object(), mjJNT_SLIDE=object())):
            preview = MuJoCoAddedBodyModel(model, Path(self.temp.name)/"base.urdf")
        self.assertEqual(set(preview.addresses), {"q"})
        gravity = np.array([0, 0, -9.81])
        np.testing.assert_allclose(preview.gravity_wrench({"q": 0}, gravity), [0, 0, -19.62, 0, 19.62, 0])
        dynamic = preview.preview(lambda t:{"q":1.5*t*t}, 0., np.zeros(3))
        np.testing.assert_allclose(dynamic[0], [0, -6, 0, 0, 0, -6.3], atol=3e-5)
        np.testing.assert_array_equal(live.qpos, before)
        with self.assertRaises(ValueError):
            preview.states({"unknown": 0})
        np.testing.assert_allclose(preview.gravity_wrench({"q_Joint": 0}, gravity),
                                   preview.gravity_wrench({"q": 0}, gravity))
        with self.assertRaises(ValueError):
            preview.states({"q": 0, "q_Joint": 0})
        # Independent MuJoCo inverse dynamics oracle at nonzero q, dq, ddq.
        dynamic = preview.preview(lambda t:{"q":.3+.2*t+.2*t*t}, 0., gravity)[0]
        live.qpos[-1] = .3
        live.qvel[:] = 0
        live.qvel[-1] = .2
        mujoco.mj_forward(model, live)
        live.qacc[:] = 0
        live.qacc[-1] = .4
        required = np.zeros(model.nv)
        mujoco.mj_rne(model, live, 1, required)
        expected = -required[:6] - np.r_[gravity, np.zeros(3)]  # remove 1kg base weight
        np.testing.assert_allclose(dynamic, expected, atol=3e-5)

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo not installed")
    def test_full_sdk_joint_names_resolve_both_mujoco_asset_conventions(self):
        import mujoco
        names = [f"{part}_{side}_Joint" for side in ("L", "R")
                 for part in ("proximal_pitch", "proximal_roll", "proximal_yaw", "knee", "ankle_pitch")]
        names += [f"arm{i}_Joint" for i in range(1, 7)]+["gripper1_Joint", "gripper2_Joint"]
        results = []
        for short in (False, True):
            bodies = []
            for i, name in enumerate(names):
                actual = name.removesuffix("_Joint") if short else name
                bodies.append(f'<body name="part{i}" pos="{i*.01} 0 0"><joint name="{actual}"/>'
                              '<inertial pos=".1 0 0" mass=".1" diaginertia=".01 .01 .01"/></body>')
            model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="base_Link">'
                + ''.join(bodies)+'</body></worldbody></mujoco>')
            preview = MuJoCoAddedBodyModel(model, Path(self.temp.name)/"base.urdf")
            self.assertEqual(len(preview.resolve_joint_names(names)), 18)
            results.append(preview.gravity_wrench(dict.fromkeys(names, .1), np.array([0, 0, -9.81])))
        np.testing.assert_allclose(results[0], results[1])


class WorkerTests(unittest.TestCase):
    def test_nonblocking_read_and_latest_request_replaces_backlog(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def compute(request):
            calls.append(request["id"])
            if request["id"] == 1:
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("test release timed out")
            return np.full((5, 6), request["id"])
        worker = LatestPreviewWorker(compute, np.zeros((5, 6)), dict(id=0, stamp=time.monotonic()))
        try:
            worker.submit(dict(id=1, stamp=time.monotonic()))
            self.assertTrue(entered.wait(1))
            # A slow model does not block this read of the seeded prediction.
            result, _ = worker.read(time.monotonic(), 2)
            self.assertEqual(result[1]["id"], 0)
            worker.submit(dict(id=2, stamp=time.monotonic()))
            worker.submit(dict(id=3, stamp=time.monotonic()))
            release.set()
            with worker.condition:
                self.assertTrue(worker.condition.wait_for(lambda: worker.result[1]["id"] == 3, timeout=2))
            result, _ = worker.read(time.monotonic(), 2)
            np.testing.assert_array_equal(result[0], np.full((5, 6), 3))
            self.assertEqual(calls, [1, 3])
        finally:
            release.set()
            worker.close()

    def test_stale_prediction_and_worker_failure_are_rejected(self):
        def fail(_):
            raise ValueError("bad dynamics")
        stamp = time.monotonic()
        worker = LatestPreviewWorker(fail, np.zeros((5, 6)), dict(stamp=stamp))
        try:
            with self.assertRaisesRegex(RuntimeError, "Stale"):
                worker.read(stamp+1, .1)
            worker.submit(dict(stamp=time.monotonic()))
            with worker.condition:
                self.assertTrue(worker.condition.wait_for(lambda: worker.error is not None, timeout=2))
            with self.assertRaisesRegex(RuntimeError, "bad dynamics"):
                worker.read(time.monotonic(), 2)
        finally:
            worker.close()

    def test_delayed_preview_is_retimed_and_rebased(self):
        prediction = np.tile([1., 0, 0, 2., 0, 0], (5, 1))
        prediction[:, 2] = TIMES
        out = current_frame_preview(prediction, [1, 0, 0, 0],
                                    [np.sqrt(.5), 0, 0, np.sqrt(.5)], .1)
        np.testing.assert_allclose(out[:, :2], np.tile([0., -1.], (5, 1)), atol=1e-12)
        np.testing.assert_allclose(out[:, 3:5], np.tile([0., -2.], (5, 1)), atol=1e-12)
        np.testing.assert_allclose(out[:, 2], [.1, .3, .5, .7, .8], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
