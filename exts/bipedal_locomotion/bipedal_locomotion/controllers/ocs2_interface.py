"""Transport-neutral data contract between Isaac Lab and the OCS2 arm MPC.

The C++ OCS2 node may use ROS 2, shared memory, or another transport. Keeping
the numerical contract here prevents transport choices from leaking into the
policy and environment code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


WRENCH_PREDICTION_TIMES = np.asarray((0.0, 0.2, 0.4, 0.6, 0.8), dtype=np.float64)


@dataclass(frozen=True)
class Ocs2MpcObservation:
    """State and reference sent to OCS2.

    Quaternions use ``[w, x, y, z]`` and all floating-base quantities use the
    world frame unless the field name explicitly contains ``body``.
    """

    time: float
    base_position_world: np.ndarray
    base_quaternion_world: np.ndarray
    base_twist_body: np.ndarray
    arm_position: np.ndarray
    arm_velocity: np.ndarray
    end_effector_target_position_world: np.ndarray
    end_effector_target_quaternion_world: np.ndarray


@dataclass(frozen=True)
class Ocs2MpcSolution:
    """First control sample and horizon information returned by OCS2."""

    time: float
    arm_position: np.ndarray
    arm_velocity: np.ndarray
    arm_feedforward_effort: np.ndarray
    base_velocity_command: np.ndarray
    # Shape (5, 6), samples at WRENCH_PREDICTION_TIMES, expressed in the
    # current base frame as [Fx, Fy, Fz, Tx, Ty, Tz].
    base_wrench_prediction: np.ndarray


def validate_mpc_solution(solution: Ocs2MpcSolution, arm_dof: int = 6) -> None:
    """Fail fast on malformed or non-finite solver output."""
    expected_vectors = {
        "arm_position": (arm_dof,),
        "arm_velocity": (arm_dof,),
        "arm_feedforward_effort": (arm_dof,),
        "base_velocity_command": (3,),
        "base_wrench_prediction": (len(WRENCH_PREDICTION_TIMES), 6),
    }
    for name, expected_shape in expected_vectors.items():
        value = np.asarray(getattr(solution, name))
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}.")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains NaN or Inf.")
