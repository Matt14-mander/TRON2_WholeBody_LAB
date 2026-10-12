import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


REPO = Path(__file__).resolve().parents[1]
path = REPO / "exts/bipedal_locomotion/bipedal_locomotion/controllers/ocs2_reset.py"
spec = importlib.util.spec_from_file_location("ocs2_reset_under_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NominalResetTest(unittest.TestCase):
    def config(self):
        return SimpleNamespace(
            events=SimpleNamespace(
                reset_robot_base=SimpleNamespace(params={
                    "pose_range": {"x": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {"x": (-0.5, 0.5)},
                }),
                reset_robot_joints=SimpleNamespace(params={
                    "position_range": (-0.5, 0.5), "velocity_range": (-0.1, 0.1),
                    "asset_cfg": SimpleNamespace(name="robot", joint_names=["leg.*"]),
                }),
            ),
            scene=SimpleNamespace(robot=SimpleNamespace(
                init_state={"arm2_Joint": 1.5707963},
                actuators={"legs": {"min_delay": 0, "max_delay": 4}},
            )),
        )

    def test_nominal_reset_includes_arm_and_preserves_asset_and_delay(self):
        cfg = self.config()
        scene = cfg.scene
        original_base = cfg.events.reset_robot_base
        original_joints = cfg.events.reset_robot_joints
        all_joints = SimpleNamespace(name="robot", joint_names=None)
        module.configure_nominal_ocs2_reset(cfg, all_joints)
        self.assertEqual(cfg.events.reset_robot_joints.params["asset_cfg"].joint_names, None)
        self.assertEqual(cfg.events.reset_robot_joints.params["position_range"], (0, 0))
        self.assertEqual(cfg.events.reset_robot_joints.params["velocity_range"], (0, 0))
        self.assertTrue(all(value == (0, 0) for value in cfg.events.reset_robot_base.params["pose_range"].values()))
        self.assertEqual(original_joints.params["position_range"], (-0.5, 0.5))
        self.assertEqual(original_base.params["pose_range"]["x"], (-0.5, 0.5))
        self.assertIs(cfg.scene, scene)
        self.assertEqual(scene.robot.init_state["arm2_Joint"], 1.5707963)
        self.assertEqual(scene.robot.actuators["legs"]["max_delay"], 4)

    def test_missing_event_rejected_without_partial_changes(self):
        cfg = self.config()
        original_base = cfg.events.reset_robot_base
        cfg.events.reset_robot_joints = None
        with self.assertRaises(RuntimeError):
            module.configure_nominal_ocs2_reset(cfg, SimpleNamespace(name="robot"))
        self.assertIs(cfg.events.reset_robot_base, original_base)


if __name__ == "__main__":
    unittest.main()
