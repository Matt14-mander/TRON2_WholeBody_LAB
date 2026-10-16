"""Check scripted disturbance timing and restoration metrics without Isaac Sim."""
import importlib.util
from pathlib import Path
import unittest
import contextlib
import io
import json
import tempfile
from types import SimpleNamespace

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/validate_paint_support.py"
spec = importlib.util.spec_from_file_location("paint_support", SCRIPT)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


class AcceptanceTests(unittest.TestCase):
    def test_original_failure_survives_shutdown_exit_zero(self):
        with tempfile.TemporaryDirectory() as output:
            args = SimpleNamespace(_acceptance_output=output, _diagnostic_stage="COM check")
            def fail(*_):
                raise ValueError("inertial mismatch")
            def shutdown():
                # Isaac shutdown must not turn an evaluation failure into success.
                raise SystemExit(0)
            with contextlib.redirect_stderr(io.StringIO()) as log:
                status = support.run_with_diagnostics(args, SimpleNamespace(close=shutdown), run=fail)
            self.assertEqual(status, 1)
            result = json.loads(Path(output, "failure.json").read_text())
            self.assertEqual(result["stage"], "COM check")
            self.assertEqual(result["exception"], "ValueError")
            self.assertIn("inertial mismatch", result["traceback"])
            self.assertIn("PAINT ERROR", log.getvalue())

    def test_disturbance_is_bounded_and_released(self):
        self.assertEqual(support.pulse(0), 0)
        self.assertEqual(support.pulse(10), 0)
        self.assertEqual(support.pulse(11), 1)
        self.assertEqual(support.pulse(12), 0)
        for t in np.linspace(0, 30, 6001):
            self.assertTrue(0 <= support.pulse(t) <= 1)

    def test_hold_has_no_scripted_motion_and_start_stop_releases(self):
        for t in (0, 5, 10, 20):
            v, arm = support.targets(t, "hold", [.15, 0, 0], [.1]*6)
            self.assertEqual(v, [0, 0, 0])
            self.assertEqual(arm, [0]*6)
        self.assertEqual(support.targets(8, "start_stop", [.15, 0, 0], [.1]*6)[0], [.15, 0, 0])
        self.assertEqual(support.targets(14, "start_stop", [.15, 0, 0], [.1]*6)[0], [0, 0, 0])
        self.assertEqual(support.targets(11, "arm_hold", [.15, 0, 0], [.1]*6)[1], [.1]*6)
        self.assertEqual(support.targets(16, "arm_hold", [.15, 0, 0], [.1]*6)[1], [0]*6)

    def test_recovery_requires_restored_acceleration_and_velocity(self):
        rows = []
        for t in np.arange(0, 22, .005):
            active = 10 < t < 13
            rows.append(dict(time_s=float(t), payload_tilt_from_reset_deg=4 if active else 1,
                payload_tilt_world_deg=4 if active else 1, base_tilt_deg=1,
                payload_acc_world=[3 if active else .2, 0, 0],
                base_velocity=[.2 if active else 0, 0, 0], base_command=[0, 0, 0],
                arm_q=[0]*6, arm_target=[0]*6, force_payload_local=[support.pulse(t)*5, 0, 0],
                low_action_abs_max=.5))
        result = support.summarize(rows, 12, 1.25)
        self.assertTrue(result["recovery"]["recovered"])
        self.assertGreater(result["recovery"]["confirmation_latency_s"], 1.25)
        self.assertGreater(result["force_active_samples"], 0)
        for row in rows:
            if row["time_s"] > 12:
                row["payload_acc_world"] = [10, 0, 0]
        self.assertFalse(support.summarize(rows, 12, 1.25)["recovery"]["recovered"])
        self.assertIsNone(support.summarize(rows[:100], 12, 1.25)["recovery"])


if __name__ == "__main__":
    unittest.main()
