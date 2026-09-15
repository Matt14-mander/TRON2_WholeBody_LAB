"""Validated offline OCS2 trajectory format and interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ocs2_interface import Ocs2MpcSolution, validate_mpc_solution


def _column_names() -> tuple[str, ...]:
    names = ["time"]
    names.extend(f"arm_q{index}" for index in range(1, 7))
    names.extend(f"arm_dq{index}" for index in range(1, 7))
    names.extend(f"arm_tau{index}" for index in range(1, 7))
    names.extend(("base_vx", "base_vy", "base_wz"))
    for sample in range(5):
        names.extend(f"w{sample}_{component}" for component in ("fx", "fy", "fz", "tx", "ty", "tz"))
    return tuple(names)


TRAJECTORY_COLUMNS = _column_names()


@dataclass(frozen=True)
class Ocs2Trajectory:
    """A 52-column trajectory exported by ``tron2_ocs2_trajectory_export``."""

    data: np.ndarray

    @classmethod
    def from_csv(cls, path: str | Path) -> "Ocs2Trajectory":
        trajectory_path = Path(path).expanduser()
        if not trajectory_path.is_file():
            raise FileNotFoundError(f"OCS2 trajectory does not exist: {trajectory_path}")
        with trajectory_path.open("r", encoding="utf-8") as stream:
            header = tuple(part.strip() for part in stream.readline().strip().split(","))
        if header != TRAJECTORY_COLUMNS:
            raise ValueError(
                f"Unexpected OCS2 trajectory header ({len(header)} columns); "
                f"expected the {len(TRAJECTORY_COLUMNS)}-column v1 format."
            )
        data = np.loadtxt(trajectory_path, delimiter=",", skiprows=1, ndmin=2, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != len(TRAJECTORY_COLUMNS):
            raise ValueError(
                f"OCS2 trajectory must have shape (N, {len(TRAJECTORY_COLUMNS)}), got {data.shape}."
            )
        if data.shape[0] < 2:
            raise ValueError("OCS2 trajectory must contain at least two samples.")
        if not np.all(np.isfinite(data)):
            raise ValueError("OCS2 trajectory contains NaN or Inf.")
        if abs(data[0, 0]) > 1e-9:
            raise ValueError(f"OCS2 trajectory must start at time zero, got {data[0, 0]}.")
        if np.any(np.diff(data[:, 0]) <= 0.0):
            raise ValueError("OCS2 trajectory timestamps must be strictly increasing.")
        return cls(data=data)

    @property
    def duration(self) -> float:
        return float(self.data[-1, 0])

    def sample(self, time_s: float) -> Ocs2MpcSolution:
        if not np.isfinite(time_s) or time_s < 0.0:
            raise ValueError("Trajectory playback time must be finite and non-negative.")
        timestamps = self.data[:, 0]
        if time_s >= timestamps[-1]:
            row = self.data[-1].copy()
            # Terminal hold: retain position/gravity compensation and wrench,
            # but never keep commanding terminal velocity or base motion.
            row[7:13] = 0.0
            row[19:22] = 0.0
        else:
            upper = int(np.searchsorted(timestamps, time_s, side="right"))
            lower = max(0, upper - 1)
            if lower == upper:
                row = self.data[lower].copy()
            else:
                alpha = (time_s - timestamps[lower]) / (timestamps[upper] - timestamps[lower])
                row = (1.0 - alpha) * self.data[lower] + alpha * self.data[upper]
        solution = Ocs2MpcSolution(
            time=time_s,
            arm_position=row[1:7].copy(),
            arm_velocity=row[7:13].copy(),
            arm_feedforward_effort=row[13:19].copy(),
            base_velocity_command=row[19:22].copy(),
            base_wrench_prediction=row[22:52].reshape(5, 6).copy(),
        )
        validate_mpc_solution(solution)
        return solution
