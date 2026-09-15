"""Transport-neutral data contract between Isaac Lab and the OCS2 arm MPC.

The C++ OCS2 node may use ROS 2, shared memory, or another transport. Keeping
the numerical contract here prevents transport choices from leaking into the
policy and environment code.
"""

from __future__ import annotations

from dataclasses import dataclass
import socket
from typing import Iterable

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


class Ocs2BridgeError(RuntimeError):
    """Communication, protocol, or solver failure reported by the OCS2 bridge."""


def _finite_vector(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf.")
    return array


def validate_mpc_observation(observation: Ocs2MpcObservation, arm_dof: int = 6) -> None:
    """Fail fast before malformed state is sent to the native solver."""
    if not np.isfinite(observation.time):
        raise ValueError("observation time must be finite.")
    expected = {
        "base_position_world": (3,),
        "base_quaternion_world": (4,),
        "base_twist_body": (6,),
        "arm_position": (arm_dof,),
        "arm_velocity": (arm_dof,),
        "end_effector_target_position_world": (3,),
        "end_effector_target_quaternion_world": (4,),
    }
    for name, shape in expected.items():
        _finite_vector(name, getattr(observation, name), shape)
    if np.linalg.norm(observation.base_quaternion_world) < 1e-9:
        raise ValueError("base quaternion has zero norm.")
    if np.linalg.norm(observation.end_effector_target_quaternion_world) < 1e-9:
        raise ValueError("end-effector target quaternion has zero norm.")


def validate_mpc_solution(solution: Ocs2MpcSolution, arm_dof: int = 6) -> None:
    """Fail fast on malformed or non-finite solver output."""
    if not np.isfinite(solution.time):
        raise ValueError("solution time must be finite.")
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


class Ocs2TcpClient:
    """Persistent localhost client for the line-oriented ``tron2_ocs2_bridge`` protocol."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_s: float = 0.1):
        if not 0 < port < 65536:
            raise ValueError(f"Invalid TCP port: {port}.")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive.")
        self._address = (host, port)
        self._timeout_s = timeout_s
        self._socket: socket.socket | None = None
        self._receive_buffer = bytearray()
        self._request_id = 0

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._receive_buffer.clear()

    def _connect(self) -> socket.socket:
        if self._socket is None:
            self._socket = socket.create_connection(self._address, timeout=self._timeout_s)
            self._socket.settimeout(self._timeout_s)
        return self._socket

    def _readline(self) -> str:
        connection = self._connect()
        while b"\n" not in self._receive_buffer:
            chunk = connection.recv(4096)
            if not chunk:
                raise Ocs2BridgeError("OCS2 bridge closed the TCP connection.")
            self._receive_buffer.extend(chunk)
            if len(self._receive_buffer) > 65536:
                raise Ocs2BridgeError("OCS2 bridge response exceeded 64 KiB.")
        line, _, remainder = self._receive_buffer.partition(b"\n")
        self._receive_buffer = bytearray(remainder)
        return line.decode("ascii", errors="strict")

    @staticmethod
    def _flatten(values: Iterable[np.ndarray | float]) -> list[float]:
        flattened: list[float] = []
        for value in values:
            array = np.asarray(value, dtype=np.float64)
            flattened.extend(array.reshape(-1).tolist())
        return flattened

    def solve(self, observation: Ocs2MpcObservation) -> Ocs2MpcSolution:
        validate_mpc_observation(observation)
        self._request_id += 1
        request_id = self._request_id
        values = self._flatten(
            (
                observation.time,
                observation.base_position_world,
                observation.base_quaternion_world,
                observation.base_twist_body,
                observation.arm_position,
                observation.arm_velocity,
                observation.end_effector_target_position_world,
                observation.end_effector_target_quaternion_world,
            )
        )
        request = "SOLVE " + str(request_id) + " " + " ".join(f"{value:.17g}" for value in values) + "\n"
        try:
            self._connect().sendall(request.encode("ascii"))
            response = self._readline().split()
        except (OSError, UnicodeError) as error:
            self.close()
            raise Ocs2BridgeError(f"OCS2 bridge communication failed: {error}") from error

        if len(response) >= 2 and response[0] == "ERR":
            message = " ".join(response[2:]) if len(response) > 2 else "unknown solver error"
            raise Ocs2BridgeError(message)
        if not response or response[0] != "OK":
            raise Ocs2BridgeError(f"Malformed OCS2 bridge response: {' '.join(response)}")
        if len(response) != 54:
            raise Ocs2BridgeError(f"OCS2 bridge returned {len(response)} fields; expected 54.")
        if int(response[1]) != request_id:
            raise Ocs2BridgeError(f"OCS2 bridge response id {response[1]} does not match {request_id}.")
        data = np.asarray([float(value) for value in response[2:]], dtype=np.float64)
        solution = Ocs2MpcSolution(
            time=float(data[0]),
            arm_position=data[1:7],
            arm_velocity=data[7:13],
            arm_feedforward_effort=data[13:19],
            base_velocity_command=data[19:22],
            base_wrench_prediction=data[22:52].reshape(5, 6),
        )
        validate_mpc_solution(solution)
        return solution

    def reset(self) -> None:
        self._request_id += 1
        request_id = self._request_id
        try:
            self._connect().sendall(f"RESET {request_id}\n".encode("ascii"))
            response = self._readline().split()
        except (OSError, UnicodeError) as error:
            self.close()
            raise Ocs2BridgeError(f"OCS2 bridge reset failed: {error}") from error
        if response != ["OK_RESET", str(request_id)]:
            raise Ocs2BridgeError(f"Malformed OCS2 reset response: {' '.join(response)}")
