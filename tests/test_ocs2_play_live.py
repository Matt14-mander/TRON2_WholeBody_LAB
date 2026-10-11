"""Exercise live update safety decisions without requiring an Isaac installation."""
import importlib.util
from pathlib import Path
import sys
import time
import types
import unittest

import numpy as np


folder = Path(__file__).parents[1] / "exts/bipedal_locomotion/bipedal_locomotion/controllers"


def load_bridge():
    # Load as an isolated package and restore simulator import stubs immediately.
    package = types.ModuleType("live_bridge_test")
    package.__path__ = [str(folder)]
    sys.modules[package.__name__] = package
    names = ("torch", "isaaclab", "isaaclab.assets")
    previous = {name: sys.modules.get(name) for name in names}
    sys.modules["torch"] = types.ModuleType("torch")
    sys.modules["isaaclab"] = types.ModuleType("isaaclab")
    assets = types.ModuleType("isaaclab.assets")
    assets.Articulation = object
    sys.modules["isaaclab.assets"] = assets
    try:
        spec = importlib.util.spec_from_file_location("live_bridge_test.ocs2_play_bridge", folder / "ocs2_play_bridge.py")
        bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge)
        return bridge
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


module = load_bridge()
interface = sys.modules["live_bridge_test.ocs2_interface"]


class BridgeSafetyTest(unittest.TestCase):
    def bridge(self, solution=None, duration=0.02, error=None):
        bridge = module.Ocs2PlayBridge.__new__(module.Ocs2PlayBridge)
        bridge._env = types.SimpleNamespace(step_dt=0.02)
        bridge._robot = types.SimpleNamespace(data=types.SimpleNamespace(body_pos_w=np.zeros((1, 1, 3))))
        bridge._ee_body_id = 0
        bridge._numpy = lambda value: value[0].copy()
        bridge._target_position_world = np.zeros(3)
        bridge._target_quaternion_world = np.array([1, 0, 0, 0])
        bridge._target_deadline_s = 1.0
        bridge.target_id, bridge.target_revision = "test", 1
        bridge._last_submission_time = 1.0
        bridge._update_period_s, bridge._max_solution_age_s = 0.1, 0.5
        bridge._next_retry_wall_time = -np.inf
        bridge._reported_error_serial = 0
        bridge._failure_count = 0
        bridge._pending_message_printed = False
        bridge._reported_solution_time = None
        bridge._warn = lambda message: None
        bridge._observation = lambda: types.SimpleNamespace(time=1.0)
        status = interface.Ocs2AsyncStatus(solution, duration, error, int(error is not None), False,
            completed_wall_time=time.monotonic(), solution_serial=1)
        bridge._async_client = types.SimpleNamespace(status=lambda: status, submit=lambda obs: None)
        bridge._smoother = sys.modules["live_bridge_test.ocs2_live"].ArmTargetSmoother(np.zeros(6), -np.ones(6), np.ones(6))
        bridge.applied = []
        bridge._set_arm_targets = lambda q, dq, tau: bridge.applied.append((q.copy(), dq.copy(), tau.copy()))
        bridge._write_policy_inputs = lambda *args: None
        bridge._fixed_command = lambda commands: None
        bridge.fallback = False
        def fallback(*args):
            bridge.fallback = True
        bridge._fallback = fallback
        return bridge

    @staticmethod
    def solution(timestamp=1.0, command=(-0.29, -0.08, 0.0)):
        return interface.Ocs2MpcSolution(timestamp, np.full(6, 0.2), np.zeros(6),
            np.zeros(6), np.asarray(command), np.zeros((5, 6)))

    def test_pending_falls_back(self):
        bridge = self.bridge()
        self.assertFalse(bridge.update(None, None))
        self.assertTrue(bridge.fallback)
        self.assertEqual(bridge.diagnostics["failure_reason"], "pending")

    def test_wall_clock_expiry_even_when_simulation_did_not_advance(self):
        bridge = self.bridge(self.solution(), duration=15.0)
        self.assertFalse(bridge.update(None, None))
        self.assertEqual(bridge.diagnostics["failure_reason"], "stale_solution")
        self.assertEqual(bridge.applied, [])

    def test_simulation_expiry(self):
        bridge = self.bridge(self.solution(timestamp=0.0))
        self.assertFalse(bridge.update(None, None))
        self.assertTrue(bridge.fallback)

    def test_free_base_solver_rejected(self):
        bridge = self.bridge(self.solution(command=(0, 0, 0)))
        self.assertFalse(bridge.update(None, None))
        self.assertTrue(bridge.fallback)
        self.assertEqual(bridge.applied, [])

    def test_fresh_arm_solution_is_rate_limited(self):
        bridge = self.bridge(self.solution())
        self.assertTrue(bridge.update(None, None))
        self.assertFalse(bridge.fallback)
        q, dq, effort = bridge.applied[0]
        self.assertLessEqual(np.max(q), 0.02 * 0.02 + 1e-9)
        np.testing.assert_array_equal(effort, np.zeros(6))

    def test_error_rejects_previously_fresh_solution(self):
        bridge = self.bridge(self.solution(), error="failed")
        self.assertFalse(bridge.update(None, None))
        self.assertEqual(bridge.diagnostics["failure_reason"], "solver_error")


if __name__ == "__main__":
    unittest.main()
