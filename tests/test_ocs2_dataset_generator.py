import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "ocs2" / "generate_trajectory_dataset.py"
_SPEC = importlib.util.spec_from_file_location("ocs2_dataset_generator_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class Ocs2DatasetGeneratorTest(unittest.TestCase):
    @staticmethod
    def _arm_limits():
        return ((-1.0, 1.0, 5.0, 100.0),) * 6

    @staticmethod
    def _write_rows(path, rows, header=None):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header or _MODULE.TRAJECTORY_COLUMNS)
            writer.writerows(rows)

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
        rows = [[0.0] * 52, [0.0] * 52]
        rows[0][0] = 0.0
        rows[1][0] = 1.5
        rows[1][7] = -2.0
        rows[1][13] = 3.0
        rows[1][19] = -0.4
        rows[1][22] = 20.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            self._write_rows(path, rows)
            metrics = _MODULE.trajectory_metrics(path, self._arm_limits())
        self.assertEqual(metrics["samples"], 2)
        self.assertEqual(metrics["duration"], 1.5)
        self.assertEqual(metrics["max_arm_velocity"], 2.0)
        self.assertEqual(metrics["max_arm_effort"], 3.0)
        self.assertEqual(metrics["max_base_command"], 0.4)
        self.assertEqual(metrics["max_wrench"], 20.0)
        self.assertEqual(metrics["max_wrench_force"], 20.0)
        self.assertEqual(metrics["max_wrench_torque"], 0.0)
        self.assertEqual(metrics["min_arm_position_margin"], 1.0)

    def test_rejects_unsafe_joint_and_base_command(self):
        rows = [[0.0] * 52, [0.0] * 52]
        rows[1][0] = 0.02
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            rows[1][4] = -1.1  # arm4 is beyond its URDF lower limit.
            self._write_rows(path, rows)
            with self.assertRaisesRegex(ValueError, "Unsafe arm4 position"):
                _MODULE.trajectory_metrics(path, self._arm_limits())
            rows[1][4] = 0.0
            rows[1][19] = 1.1
            self._write_rows(path, rows)
            with self.assertRaisesRegex(ValueError, "walk-policy range"):
                _MODULE.trajectory_metrics(path, self._arm_limits())

    def test_rejects_wrong_header_and_nonmonotonic_time(self):
        rows = [[0.0] * 52, [0.0] * 52]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            self._write_rows(path, rows, [f"column_{index}" for index in range(52)])
            with self.assertRaisesRegex(ValueError, "52-column"):
                _MODULE.trajectory_metrics(path, self._arm_limits())
            self._write_rows(path, rows)
            with self.assertRaisesRegex(ValueError, "increase strictly"):
                _MODULE.trajectory_metrics(path, self._arm_limits())

    def test_contract_matches_existing_walk_actor_dimensions(self):
        contract = _MODULE.trajectory_contract(self._arm_limits())
        self.assertEqual(len(contract["columns"]), 52)
        self.assertEqual(contract["future_wrench"]["shape"], (5, 6))
        self.assertEqual(contract["walk_actor"]["input_widths"], (3, 42, 30, 3))
        self.assertEqual(contract["walk_actor"]["input_width"], 78)
        self.assertEqual(contract["walk_actor"]["leg_action_width"], 10)

    def test_reads_arm_limits_from_export_urdf(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.urdf"
            joints = "".join(
                f'<joint name="arm{i}_Joint" type="revolute">'
                '<limit lower="-1" upper="1" velocity="5" effort="100"/>'
                '</joint>' for i in range(1, 7)
            )
            path.write_text(f"<robot name=\"test\">{joints}</robot>", encoding="utf-8")
            self.assertEqual(_MODULE.load_arm_limits(path), self._arm_limits())

    def test_batch_records_only_safe_exports_with_walk_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urdf = root / "robot.urdf"
            joints = "".join(
                f'<joint name="arm{i}_Joint" type="revolute">'
                '<limit lower="-1" upper="1" velocity="5" effort="100"/>'
                '</joint>' for i in range(1, 7)
            )
            urdf.write_text(f'<robot name="test">{joints}</robot>', encoding="utf-8")
            task_info = root / "task.info"
            task_info.write_text("model {}\n", encoding="utf-8")
            output_dir = root / "dataset"
            args = argparse.Namespace(
                task_info=task_info, robot_urdf=urdf, library_dir=root / "codegen",
                output_dir=output_dir, count=2, seed=42, target_mode="relative",
                x_range=None, y_range=None, z_range=None, arrival_times=[1.0],
                sample_period=0.02, resume=False,
            )

            def fake_export(command, **_kwargs):
                self.assertIn("--relative-target", command)
                output = Path(command[7])
                rows = [[0.0] * 52, [0.0] * 52]
                rows[1][0] = 0.02
                if output.stem.endswith("000001"):
                    rows[1][4] = -1.1
                self._write_rows(output, rows)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(_MODULE, "parse_args", return_value=args), mock.patch.object(
                _MODULE.subprocess, "run", side_effect=fake_export
            ):
                self.assertEqual(_MODULE.main(), 0)

            with (output_dir / "manifest.csv").open(newline="") as stream:
                successes = list(csv.DictReader(stream))
            with (output_dir / "failures.csv").open(newline="") as stream:
                failures = list(csv.DictReader(stream))
            self.assertEqual(len(successes), 1)
            self.assertEqual(len(failures), 1)
            self.assertEqual(successes[0]["target_mode"], "relative")
            self.assertIn("Unsafe arm4 position", failures[0]["error"])
            self.assertFalse((output_dir / "trajectories" / "trajectory_000001.csv").exists())
            contract = json.loads((output_dir / "contract.json").read_text(encoding="utf-8"))
            self.assertEqual(len(contract["columns"]), 52)
            self.assertEqual(contract["walk_actor"]["input_width"], 78)


if __name__ == "__main__":
    unittest.main()
