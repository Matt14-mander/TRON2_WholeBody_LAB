import importlib.util
from dataclasses import replace
from pathlib import Path
import sys
import threading
import time
import unittest

import numpy as np


_MODULE_PATH = (
    Path(__file__).parents[1]
    / "exts"
    / "bipedal_locomotion"
    / "bipedal_locomotion"
    / "controllers"
    / "ocs2_interface.py"
)
_SPEC = importlib.util.spec_from_file_location("ocs2_interface_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
Ocs2MpcObservation = _MODULE.Ocs2MpcObservation
Ocs2MpcSolution = _MODULE.Ocs2MpcSolution
Ocs2AsyncClient = _MODULE.Ocs2AsyncClient
Ocs2TcpClient = _MODULE.Ocs2TcpClient
validate_mpc_observation = _MODULE.validate_mpc_observation


class _FakeSocket:
    def __init__(self, response: bytes):
        self.response = response
        self.sent = bytearray()

    def sendall(self, data: bytes) -> None:
        self.sent.extend(data)

    def recv(self, _: int) -> bytes:
        response, self.response = self.response, b""
        return response

    def close(self) -> None:
        pass


class Ocs2TcpClientTest(unittest.TestCase):
    def test_solve_protocol_round_trip(self):
        payload = np.arange(52, dtype=np.float64)
        fake_socket = _FakeSocket(("OK 1 " + " ".join(map(str, payload)) + "\n").encode("ascii"))
        client = Ocs2TcpClient()
        client._socket = fake_socket
        observation = Ocs2MpcObservation(
            time=0.25,
            base_position_world=np.zeros(3),
            base_quaternion_world=np.array([1.0, 0.0, 0.0, 0.0]),
            base_twist_body=np.zeros(6),
            arm_position=np.zeros(6),
            arm_velocity=np.zeros(6),
            end_effector_target_position_world=np.array([0.35, 0.0, 0.85]),
            end_effector_target_quaternion_world=np.array([1.0, 0.0, 0.0, 0.0]),
            end_effector_arrival_time=1.0,
        )

        solution = client.solve(observation)

        sent_fields = fake_socket.sent.decode("ascii").split()
        self.assertEqual(len(sent_fields), 36)
        self.assertEqual(float(sent_fields[-1]), 1.0)
        self.assertEqual(solution.time, 0.0)
        np.testing.assert_array_equal(solution.arm_position, np.arange(1, 7))
        np.testing.assert_array_equal(solution.base_velocity_command, np.arange(19, 22))
        np.testing.assert_array_equal(solution.base_wrench_prediction, np.arange(22, 52).reshape(5, 6))

    def test_rejects_invalid_arrival_time(self):
        observation = Ocs2AsyncClientTest._observation(0.0)
        for invalid in (-0.1, float("nan")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_mpc_observation(
                        replace(observation, end_effector_arrival_time=invalid)
                    )


class _ControlledClient:
    def __init__(self):
        self.release = threading.Event()
        self.started = threading.Event()
        self.closed = False

    def solve(self, observation):
        self.started.set()
        self.release.wait(timeout=1.0)
        return Ocs2MpcSolution(
            time=observation.time,
            arm_position=np.zeros(6),
            arm_velocity=np.zeros(6),
            arm_feedforward_effort=np.zeros(6),
            base_velocity_command=np.zeros(3),
            base_wrench_prediction=np.zeros((5, 6)),
        )

    def reset(self):
        pass

    def close(self):
        self.closed = True
        self.release.set()


class Ocs2AsyncClientTest(unittest.TestCase):
    @staticmethod
    def _observation(simulation_time):
        return Ocs2MpcObservation(
            time=simulation_time,
            base_position_world=np.zeros(3),
            base_quaternion_world=np.array([1.0, 0.0, 0.0, 0.0]),
            base_twist_body=np.zeros(6),
            arm_position=np.zeros(6),
            arm_velocity=np.zeros(6),
            end_effector_target_position_world=np.array([0.35, 0.0, 0.85]),
            end_effector_target_quaternion_world=np.array([1.0, 0.0, 0.0, 0.0]),
        )

    def test_submit_does_not_block_and_publishes_result(self):
        transport = _ControlledClient()
        client = Ocs2AsyncClient(transport)
        started = time.monotonic()
        client.submit(self._observation(0.25))
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertTrue(transport.started.wait(timeout=1.0))
        self.assertIsNone(client.status().solution)

        transport.release.set()
        deadline = time.monotonic() + 1.0
        while client.status().solution is None and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertEqual(client.status().solution.time, 0.25)
        client.close()
        self.assertTrue(transport.closed)


if __name__ == "__main__":
    unittest.main()
