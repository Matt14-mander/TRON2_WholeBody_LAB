import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/hardware"))
from monitor_sfyg_arm import Monitor
from sfyg_arm_guard import ArmGuard
from shadow_sfyg_locomotion import Coordinates, LEG_NAMES
from sfyg_lifecycle import sdk_packet


class MonitorTests(unittest.TestCase):
    def test_settle_and_hold_loss(self):
        for fault in ("enable", "mode", "q", "dq", "nan"):
            m = Monitor(np.zeros(6))
            self.assertFalse(m.tick(np.zeros(6), np.zeros(6), True, True, 0))
            self.assertTrue(m.tick(np.zeros(6), np.zeros(6), True, True, 2))
            q, dq = np.zeros(6), np.zeros(6)
            if fault == "q": q[2] = .041
            if fault == "dq": dq[0] = .051
            if fault == "nan": q[0] = np.nan
            with self.assertRaises(RuntimeError) as raised:
                m.tick(q, dq, fault != "enable", fault != "mode", 3)
            self.assertIn("error_rad=", str(raised.exception))
            self.assertIn("dq_rad_s=", str(raised.exception))
            if fault == "enable": self.assertIn("enabled=False", str(raised.exception))
            if fault == "mode": self.assertIn("can_j=False", str(raised.exception))
            self.assertFalse(m.ready)

    def test_guard_stale_restart_and_not_ready(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"status.json"
            data = dict(schema="sfyg_arm_monitor_v1", session="a", monotonic_s=10,
                        state="HOLD", ready=True, motor_output=False,
                        enabled=True, can_j=True, policy_sha256="sha")
            p.write_text(json.dumps(data))
            guard = ArmGuard(p, "sha")
            guard.check(10.1)
            with self.assertRaises(RuntimeError): guard.check(10.3)
            for change in ({"session": "b"}, {"ready": False}, {"enabled": False},
                           {"can_j": False}, {"state": "FAULT"}, {"motor_output": True}):
                p.write_text(json.dumps(data | change))
                with self.assertRaises(RuntimeError): guard.check(10.1)

    def test_xyzw_and_inverse_joint_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"map.json"
            data = dict(sdk_names=LEG_NAMES[::-1], zero_rad=[.1]*10,
                        sign=[-1]*10, imu_to_base=np.eye(3).tolist(), quaternion_order="xyzw")
            p.write_text(json.dumps(data))
            c = Coordinates(p)
            q = np.linspace(-1, 1, 10)
            # +90 degree roll: world-down projected in body is -Y.
            converted = c.convert((0,0,q,np.zeros(10)),
                                  (0,0,[2**-.5,0,0,2**-.5],[1,2,3]))
            np.testing.assert_allclose(converted[3], [0,-1,0], atol=1e-7)
            packet = sdk_packet(dict(q=converted[0], kp=np.ones(10), kd=np.ones(10)), c)
            np.testing.assert_allclose(packet["q"], q)
            # Pitch distinguishes quaternion conventions even if yaw/level does not.
            g = c.convert((0,0,q,np.zeros(10)),
                          (0,0,[0,2**-.5,0,2**-.5],[0]*3))[3]
            np.testing.assert_allclose(g, [1,0,0], atol=1e-7)
