"""Separate horizontal-normal turning from actual inclination/tracking errors."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("task_heading",
    Path(__file__).resolve().parents[1]/"scripts/mujoco/analyze_task_heading.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def ry(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rows(yaws, base_pitches=None, errors=None):
    import mujoco
    out = []
    for i, yaw in enumerate(yaws):
        pitch = base_pitches[i] if base_pitches is not None else 0.
        error = errors[i] if errors is not None else 0.
        base = rz(yaw)@ry(pitch)
        target = base@ry(np.pi/2)
        actual = base@ry(np.pi/2+error)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(target.reshape(-1)))
        out.append(dict(time_s=i*.1, sim_time_s=i*.1, phase="hold", segment=0,
            base_rotation_world=base.reshape(-1).tolist(), base_position_world_m=[i*.01, 0, 0],
            payload=dict(up_world=actual[:, 2].tolist(), target_quaternion_world_wxyz=quat.tolist(),
                         acceleration_world_m_s2=None)))
    return out


class HeadingTests(unittest.TestCase):
    def test_horizontal_payload_turn_is_not_inclination_wobble(self):
        source = rows(np.linspace(0, np.pi/3, 101))
        data = module.series(source, settle_s=0.)
        self.assertAlmostEqual(data["normal_from_settled_deg"][-1], 60.)
        np.testing.assert_allclose(data["heading_removed_normal_from_settled_deg"], 0., atol=2e-6)
        np.testing.assert_allclose(data["normal_error_to_command_deg"], 0., atol=2e-6)
        self.assertAlmostEqual(np.std(data["tilt_world_deg"]), 0.)
        report = module.summarize(data, np.ones(101, dtype=bool))
        self.assertAlmostEqual(report["heading_net_change_deg"], 60.)
        self.assertAlmostEqual(report["heading_linear_slope_deg_s"], 6.)

    def test_roll_pitch_remains_after_yaw_removal_but_not_base_removal(self):
        pitch = .05*np.sin(np.linspace(0, 4*np.pi, 101))
        data = module.series(rows(np.linspace(0, 1., 101), pitch), settle_s=0.)
        self.assertGreater(np.max(data["heading_removed_normal_from_settled_deg"]), 2.8)
        np.testing.assert_allclose(data["base_relative_normal_from_settled_deg"], 0., atol=2e-6)
        np.testing.assert_allclose(data["normal_error_to_command_deg"], 0., atol=2e-6)

    def test_yaw_wrap_and_actual_arm_normal_error(self):
        source = rows(np.radians(np.linspace(179, 185, 61)), errors=np.ones(61)*.03)
        data = module.series(source, settle_s=0.)
        self.assertAlmostEqual(data["heading_delta_deg"][-1], 6.)
        np.testing.assert_allclose(data["heading_rate_deg_s"], 1., atol=1e-10)
        np.testing.assert_allclose(data["normal_error_to_command_deg"], np.degrees(.03), atol=1e-10)
        np.testing.assert_allclose(data["tilt_error_to_command_deg"], np.degrees(.03), atol=1e-10)

    def test_reject_reset_and_write_strict_json_for_missing_acceleration(self):
        source = rows(np.linspace(0, .2, 10))
        source[-1]["segment"] = 1
        with self.assertRaisesRegex(ValueError, "reset"):
            module.series(source, settle_s=0.)
        source[-1]["segment"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            steps = root/"task_aligned.jsonl"
            steps.write_text('\n'.join(map(json.dumps, source)))
            (root/"task_report.json").write_text(json.dumps(dict(definitions=dict(payload_up_axis=[0, 0, 1]))))
            module.analyze(steps, root/"out", settle_s=0.)
            text = (root/"out/heading_steps.jsonl").read_text()
            self.assertNotIn("NaN", text)
            self.assertIsNone(json.loads(text.splitlines()[0])["acceleration_norm_m_s2"])


if __name__ == "__main__":
    unittest.main()
