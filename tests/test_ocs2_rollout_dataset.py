import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


_MODULE_PATH = (
    Path(__file__).parents[1]
    / "exts"
    / "bipedal_locomotion"
    / "bipedal_locomotion"
    / "controllers"
    / "ocs2_rollout_dataset.py"
)
_SPEC = importlib.util.spec_from_file_location("ocs2_rollout_dataset_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class Ocs2RolloutDatasetTest(unittest.TestCase):
    def test_external_wrench_assignments_are_balanced(self):
        ids = [f"trajectory_{index:06d}" for index in range(21)]
        assignments = _MODULE.make_external_wrench_assignments(ids, seed=7)
        conditions = [(item.axis, item.sign) for item in assignments.values()]
        for condition in (
            ("zero", 0), ("fx", 1), ("fx", -1), ("fy", 1),
            ("fy", -1), ("mz", 1), ("mz", -1),
        ):
            self.assertEqual(conditions.count(condition), 3)
        first_block = [assignments[trajectory_id] for trajectory_id in ids[:7]]
        self.assertEqual(len({(item.axis, item.sign) for item in first_block}), 7)

    def test_external_wrench_profiles_and_window(self):
        excitation = _MODULE.ExternalWrenchExcitation("fy", -1, "ramp", 8.0)
        np.testing.assert_allclose(
            _MODULE.evaluate_external_wrench(excitation, 0.9, 1.0, 0.8), np.zeros(6)
        )
        wrench = _MODULE.evaluate_external_wrench(excitation, 1.4, 1.0, 0.8)
        self.assertAlmostEqual(wrench[1], -4.0)
        np.testing.assert_allclose(
            _MODULE.evaluate_external_wrench(excitation, 1.8, 1.0, 0.8), np.zeros(6)
        )

    def test_paired_protocol_has_every_condition_per_profile(self):
        jobs = _MODULE.make_paired_external_wrench_excitations(
            force_amplitude=5.0, torque_amplitude=1.0
        )
        self.assertEqual(len(jobs), 21)
        self.assertEqual(len({suffix for suffix, _ in jobs}), 21)
        for profile in ("step", "ramp", "sine"):
            conditions = {
                (item.axis, item.sign) for _, item in jobs if item.profile == profile
            }
            self.assertEqual(len(conditions), 7)

    def test_wrench_transform_shifts_torque_reference_point(self):
        wrench = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 2.0])
        transformed = _MODULE.transform_wrench_to_base_origin(
            wrench,
            payload_position_world=np.asarray([0.0, 1.0, 0.0]),
            payload_quaternion_world=np.asarray([1.0, 0.0, 0.0, 0.0]),
            base_position_world=np.zeros(3),
            base_quaternion_world=np.asarray([1.0, 0.0, 0.0, 0.0]),
        )
        # r x F = [0, 1, 0] x [1, 0, 0] = [0, 0, -1].
        np.testing.assert_allclose(transformed, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def test_manifest_loading_and_split_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectories = root / "trajectories"
            trajectories.mkdir()
            fieldnames = [
                "trajectory_id", "trajectory_file", "target_x", "target_y", "target_z",
                "target_qw", "target_qx", "target_qy", "target_qz", "arrival_time",
            ]
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for index in range(20):
                    path = trajectories / f"trajectory_{index:06d}.csv"
                    path.touch()
                    writer.writerow({
                        "trajectory_id": f"trajectory_{index:06d}",
                        "trajectory_file": str(path.relative_to(root)),
                        "target_x": 0.2,
                        "target_y": 0.0,
                        "target_z": 0.85,
                        "target_qw": 1.0,
                        "target_qx": 0.0,
                        "target_qy": 0.0,
                        "target_qz": 0.0,
                        "arrival_time": 0.75,
                    })
            specs = _MODULE.load_trajectory_manifest(manifest)
            assignments = _MODULE.make_split_assignments(
                [spec.trajectory_id for spec in specs], seed=42
            )
        self.assertEqual(len(specs), 20)
        self.assertEqual(list(assignments.values()).count("train"), 16)
        self.assertEqual(list(assignments.values()).count("validation"), 2)
        self.assertEqual(list(assignments.values()).count("test"), 2)

    def test_episode_npz_round_trip(self):
        frames = [
            {"time": np.asarray(0.0), "state": np.asarray([1.0, 2.0])},
            {"time": np.asarray(0.02), "state": np.asarray([3.0, 4.0])},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "episode.npz"
            _MODULE.save_episode_npz(
                output, frames, {"trajectory_id": "trajectory_000000", "accepted": True}
            )
            with np.load(output) as data:
                np.testing.assert_allclose(data["time"], [0.0, 0.02])
                np.testing.assert_allclose(data["state"], [[1.0, 2.0], [3.0, 4.0]])
                self.assertEqual(str(data["meta_trajectory_id"]), "trajectory_000000")
                self.assertTrue(bool(data["meta_accepted"]))

    def test_split_file_rejects_changed_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "split.json"
            _MODULE.write_split_file(output, {"trajectory_000000": "train"}, seed=42)
            _MODULE.write_split_file(output, {"trajectory_000000": "train"}, seed=42)
            with self.assertRaisesRegex(ValueError, "does not match"):
                _MODULE.write_split_file(output, {"trajectory_000000": "test"}, seed=42)

    def test_json_contract_rejects_changed_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "wrench_contract.json"
            _MODULE.write_json_contract(output, {"amplitude": 5.0, "axis": ["fx"]})
            _MODULE.write_json_contract(output, {"axis": ["fx"], "amplitude": 5.0})
            with self.assertRaisesRegex(ValueError, "does not match"):
                _MODULE.write_json_contract(output, {"amplitude": 8.0, "axis": ["fx"]})


if __name__ == "__main__":
    unittest.main()
