import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np


_CONTROLLERS = (
    Path(__file__).parents[1]
    / "exts"
    / "bipedal_locomotion"
    / "bipedal_locomotion"
    / "controllers"
)
_PACKAGE = types.ModuleType("bipedal_locomotion")
_PACKAGE.__path__ = [str(_CONTROLLERS.parent)]
_CONTROLLERS_PACKAGE = types.ModuleType("bipedal_locomotion.controllers")
_CONTROLLERS_PACKAGE.__path__ = [str(_CONTROLLERS)]
sys.modules["bipedal_locomotion"] = _PACKAGE
sys.modules["bipedal_locomotion.controllers"] = _CONTROLLERS_PACKAGE


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _CONTROLLERS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("bipedal_locomotion.controllers.ocs2_interface", "ocs2_interface.py")
_TRAJECTORY_MODULE = _load("bipedal_locomotion.controllers.ocs2_trajectory", "ocs2_trajectory.py")
Ocs2Trajectory = _TRAJECTORY_MODULE.Ocs2Trajectory
TRAJECTORY_COLUMNS = _TRAJECTORY_MODULE.TRAJECTORY_COLUMNS


class Ocs2TrajectoryTest(unittest.TestCase):
    def _write_trajectory(self, path: Path) -> np.ndarray:
        data = np.vstack(
            (
                np.arange(len(TRAJECTORY_COLUMNS), dtype=np.float64),
                np.arange(len(TRAJECTORY_COLUMNS), dtype=np.float64) + 10.0,
            )
        )
        data[:, 0] = (0.0, 1.0)
        np.savetxt(path, data, delimiter=",", header=",".join(TRAJECTORY_COLUMNS), comments="")
        return data

    def test_interpolation_and_terminal_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            data = self._write_trajectory(path)
            trajectory = Ocs2Trajectory.from_csv(path)

            middle = trajectory.sample(0.5)
            np.testing.assert_allclose(middle.arm_position, 0.5 * (data[0, 1:7] + data[1, 1:7]))
            np.testing.assert_allclose(
                middle.base_wrench_prediction.reshape(-1),
                0.5 * (data[0, 22:52] + data[1, 22:52]),
            )

            terminal = trajectory.sample(2.0)
            np.testing.assert_array_equal(terminal.arm_position, data[-1, 1:7])
            np.testing.assert_array_equal(terminal.arm_velocity, np.zeros(6))
            np.testing.assert_array_equal(terminal.base_velocity_command, np.zeros(3))
            np.testing.assert_array_equal(terminal.arm_feedforward_effort, data[-1, 13:19])

    def test_rejects_non_monotonic_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            data = self._write_trajectory(path)
            data[1, 0] = 0.0
            np.savetxt(path, data, delimiter=",", header=",".join(TRAJECTORY_COLUMNS), comments="")
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                Ocs2Trajectory.from_csv(path)


if __name__ == "__main__":
    unittest.main()
