"""Analytical force/moment checks independent of the Isaac runtime."""
import sys
from pathlib import Path
import tempfile
import unittest
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers"))
from paint_wrench_preview import AddedBodyModel, MuJoCoAddedBodyModel, TIMES


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
          <body name="base_Link" pos="0 0 0.85"><freejoint/>
          <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
          <body name="arm"><joint name="q" axis="0 0 1"/>
          <inertial pos="1 0 0" mass="2" diaginertia=".1 .1 .1"/>
          </body></body></worldbody></mujoco>''')
        live = mujoco.MjData(model)
        live.qpos[-1] = .73
        before = live.qpos.copy()
        preview = MuJoCoAddedBodyModel(model, Path(self.temp.name)/"base.urdf")
        gravity = np.array([0, 0, -9.81])
        np.testing.assert_allclose(preview.gravity_wrench({"q": 0}, gravity), [0, 0, -19.62, 0, 19.62, 0])
        dynamic = preview.preview(lambda t:{"q":1.5*t*t}, 0., np.zeros(3))
        np.testing.assert_allclose(dynamic[0], [0, -6, 0, 0, 0, -6.3], atol=3e-5)
        np.testing.assert_array_equal(live.qpos, before)
        with self.assertRaises(ValueError):
            preview.states({"unknown": 0})


if __name__ == "__main__":
    unittest.main()
