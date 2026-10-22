"""Exercise diagnostic torque sign, phase isolation and incomplete-run persistence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("arm_telemetry",
    Path(__file__).resolve().parents[1] / "scripts/mujoco/arm_telemetry.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ArmTelemetryTests(unittest.TestCase):
    def test_load_error_pd_and_feedback_remain_distinct_on_aborted_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            record = module.ArmTelemetry(directory, dict(joints=[f"arm{i}" for i in range(1, 7)]))
            for t, phase, feedback in ((1., "outbound", None), (4., "aborted", np.ones(6)*9)):
                record.append(time_s=t, phase=phase, q=np.ones(6)*.2, dq=np.ones(6)*.1,
                    target=np.zeros(6), velocity_target=np.zeros(6), effort=np.ones(6)*3,
                    kp=np.ones(6)*60, kd=np.ones(6)*6, feedback=feedback,
                    state_age_s=.003, state_count=int(t))
            report = record.save(dict(completed=False, error="guard abort"))
            np.testing.assert_allclose(record.rows[0]["position_error"], .2)
            np.testing.assert_allclose(record.rows[0]["estimated_pd_effort"], -12.6)
            np.testing.assert_allclose(record.rows[0]["estimated_total_effort"], -9.6)
            self.assertIsNone(report["phases"]["outbound"]["actuator_feedback"])
            self.assertEqual(report["phases"]["aborted"]["actuator_feedback"]["mean"], [9.]*6)
            self.assertEqual(report["after_2s"]["samples"], 1)
            self.assertFalse(json.loads((directory / "arm_report.json").read_text())["outcome"]["completed"])
            self.assertEqual(len((directory / "arm_steps.jsonl").read_text().splitlines()), 2)

    def test_empty_run_and_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            record = module.ArmTelemetry(directory, dict(joints=[]))
            report = record.save(dict(completed=False))
            self.assertEqual(report["all"]["samples"], 0)
            with self.assertRaises(FileExistsError):
                module.ArmTelemetry(directory, dict(joints=[]))


if __name__ == "__main__":
    unittest.main()
