import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "ocs2" / "generate_trajectory_dataset.py"
_SPEC = importlib.util.spec_from_file_location("ocs2_dataset_generator_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class Ocs2DatasetGeneratorTest(unittest.TestCase):
    def test_scenarios_are_reproducible_and_in_bounds(self):
        arguments = (5, 42, (0.15, 0.30), (-0.15, 0.15), (0.75, 0.95), (0.5, 0.75, 1.0))
        first = _MODULE.generate_scenarios(*arguments)
        second = _MODULE.generate_scenarios(*arguments)
        self.assertEqual(first, second)
        self.assertEqual([item.arrival_time for item in first], [0.5, 0.75, 1.0, 0.5, 0.75])
        for scenario in first:
            self.assertGreaterEqual(scenario.target_x, 0.15)
            self.assertLessEqual(scenario.target_x, 0.30)
            self.assertGreaterEqual(scenario.target_y, -0.15)
            self.assertLessEqual(scenario.target_y, 0.15)
            self.assertGreaterEqual(scenario.target_z, 0.75)
            self.assertLessEqual(scenario.target_z, 0.95)

    def test_trajectory_metrics(self):
        header = [f"column_{index}" for index in range(52)]
        rows = [[0.0] * 52, [0.0] * 52]
        rows[0][0] = 0.0
        rows[1][0] = 1.5
        rows[1][7] = -2.0
        rows[1][13] = 3.0
        rows[1][19] = -0.4
        rows[1][22] = 20.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(rows)
            metrics = _MODULE.trajectory_metrics(path)
        self.assertEqual(metrics["samples"], 2)
        self.assertEqual(metrics["duration"], 1.5)
        self.assertEqual(metrics["max_arm_velocity"], 2.0)
        self.assertEqual(metrics["max_arm_effort"], 3.0)
        self.assertEqual(metrics["max_base_command"], 0.4)
        self.assertEqual(metrics["max_wrench"], 20.0)


if __name__ == "__main__":
    unittest.main()
