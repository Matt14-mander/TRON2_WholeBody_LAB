import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


path = Path(__file__).parents[1] / "exts/bipedal_locomotion/bipedal_locomotion/controllers/ocs2_live.py"
spec = importlib.util.spec_from_file_location("ocs2_live_under_test", path)
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


class LiveSafetyTests(unittest.TestCase):
    def test_hot_update_retry_and_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.json"
            source = live.LiveTargetFile(path)
            self.assertIsNone(source.poll())
            path.write_text('{"id":', encoding="utf-8")
            with self.assertRaises(ValueError):
                source.poll()
            target = {"id": "one", "position_world": [0.3, 0, 0.9], "quaternion_wxyz": [2, 0, 0, 0]}
            path.write_text(json.dumps(target), encoding="utf-8")
            target_id, (_, quaternion, _) = source.poll()
            np.testing.assert_array_equal(quaternion, [1, 0, 0, 0])
            # Rejected by the workspace guard: not yet acknowledged.
            self.assertIsNotNone(source.poll())
            source.accept(target_id)
            self.assertIsNone(source.poll())
            target["id"] = "two"
            path.write_text(json.dumps(target), encoding="utf-8")
            self.assertEqual(source.poll()[0], "two")

    def test_rate_limits_through_reversal(self):
        smoother = live.ArmTargetSmoother(np.zeros(6), -np.ones(6), np.ones(6))
        old_position, old_velocity = smoother.position.copy(), smoother.velocity.copy()
        for sign in (1, -1, 1):
            for _ in range(100):
                position, velocity = smoother.step(np.full(6, sign * 0.8), 0.02)
                self.assertLessEqual(np.abs(position - old_position).max(), 0.4 * 0.02 + 1e-9)
                self.assertLessEqual(np.abs(velocity - old_velocity).max(), 1.0 * 0.02 + 1e-9)
                self.assertTrue(np.all(np.abs(position) <= 1))
                old_position, old_velocity = position, velocity

    def test_unsafe_reference_never_mutates_smoother(self):
        smoother = live.ArmTargetSmoother(np.zeros(6), -np.ones(6), np.ones(6))
        for desired in (np.full(6, 2), np.full(6, np.nan)):
            with self.assertRaises(ValueError):
                smoother.step(desired, 0.02)
            np.testing.assert_array_equal(smoother.position, np.zeros(6))

    def test_arrival_and_zero_quaternion_rejected(self):
        for quaternion, arrival in (([0, 0, 0, 0], 1), ([1, 0, 0, 0], 1.1)):
            with self.assertRaises(ValueError):
                live.target_values([0, 0, 0], quaternion, arrival)


if __name__ == "__main__":
    unittest.main()
