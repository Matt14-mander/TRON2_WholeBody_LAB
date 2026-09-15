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


if __name__ == "__main__":
    unittest.main()
