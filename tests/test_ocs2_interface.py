import importlib.util
from pathlib import Path
import sys
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
Ocs2TcpClient = _MODULE.Ocs2TcpClient


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
        )

        solution = client.solve(observation)

        self.assertEqual(len(fake_socket.sent.decode("ascii").split()), 35)
        self.assertEqual(solution.time, 0.0)
        np.testing.assert_array_equal(solution.arm_position, np.arange(1, 7))
        np.testing.assert_array_equal(solution.base_velocity_command, np.arange(19, 22))
        np.testing.assert_array_equal(solution.base_wrench_prediction, np.arange(22, 52).reshape(5, 6))


if __name__ == "__main__":
    unittest.main()
