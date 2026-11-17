import json
import io
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/hardware"))
import hold_sfyg_arm as arm
from sfyg_arm_guard import ArmGuard
import run_sfyg_lifecycle as legs


def profile():
    data = json.loads((ROOT / "scripts/hardware/sfyg_arm_hold.example.json").read_text())
    return data


class ArmTests(unittest.TestCase):
    def test_readonly_never_constructs_sender_even_on_exit(self):
        class Bus:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def recv(self, timeout): return None
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(status_file=Path(d)/"status.json", execute=False, can="fake")
            p = arm.Profile(profile(), [0, 1.57, -1.48, 0, 0, 0])
            with patch.object(sys, "stdin", io.StringIO("quit supported\n")), \
                 patch.object(arm, "Sender", side_effect=AssertionError("sender in read-only")):
                arm.run_locked(args, p, SimpleNamespace(Bus=lambda **kw: Bus()))
            result = json.loads(args.status_file.read_text())
            self.assertFalse(result["execute"])
            self.assertFalse(result["ready"])
            self.assertEqual(result["state"], "STOPPED")

    def test_leg_execute_requires_live_arm_file_before_sdk(self):
        args = SimpleNamespace(execute=True, bundle_dir=Path("fake"), preflight=Path("fake"))
        with patch.object(legs, "load_bundle", return_value=({"policy_sha256": arm.POLICY_SHA}, None)), \
             patch.object(legs, "validate_preflight", return_value={}), \
             patch.object(legs, "subscribe", side_effect=AssertionError("SDK before arm gate")):
            with self.assertRaisesRegex(ValueError, "arm_status_file"):
                legs.run(args)

    def test_preflight_mapping_and_limits(self):
        data = profile()
        with self.assertRaisesRegex(ValueError, "preflight"):
            arm.Profile(data, [0, 1.57, -1.48, 0, 0, 0], True)
        data.update(operator="test", **{k: True for k in arm.CHECKS})
        data["sign"][0] = -1
        data["zero_rad"][0] = .2
        p = arm.Profile(data, [.1, 1.57, -1.48, 0, 0, 0], True)
        self.assertAlmostEqual(p.target[0], .1)
        with self.assertRaisesRegex(ValueError, "limits"):
            arm.Profile(data, [0, -1, 0, 0, 0, 0])
        data["zero_rad"][1] = float("nan")
        with self.assertRaises(ValueError):
            arm.Profile(data, [0]*6)

    def test_smooth_prepare_hold_and_faults(self):
        p = arm.Profile(profile(), [0, 1.57, -1.48, 0, 0, 0])
        ctl = arm.Hold(p)
        q = p.target.copy(); q[1] -= .2
        self.assertIsNone(ctl.tick(q, np.zeros(6), True, True, 0))
        ctl.start(q, np.zeros(6), 0)
        np.testing.assert_allclose(ctl.tick(q, np.zeros(6), True, True, 0), q)
        for now in np.arange(.02, 12, .02):
            q = ctl.tick(q, np.zeros(6), True, True, now)
        self.assertTrue(ctl.ready)
        self.assertEqual(ctl.state, "HOLD")
        with self.assertRaisesRegex(RuntimeError, "acknowledgement"):
            ctl.tick(q, np.zeros(6), False, True, 12)
        q[1] += .06
        with self.assertRaisesRegex(RuntimeError, "hold lost"):
            ctl.tick(q, np.zeros(6), True, True, 12)

    def test_folded_start_refused(self):
        p = arm.Profile(profile(), [0, 1.57, -1.48, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "folded"):
            arm.Hold(p).start(np.zeros(6), np.zeros(6), 0)

    def test_all_feedback_required_and_fault_bits(self):
        fb = arm.Feedback()
        for i in fb.IDS:
            data = bytearray(8)
            if 0x261 <= i <= 0x266:
                data[5] = 0x40
            if i == 0x2A1:
                data[0] = data[2] = 1
            if i == 0x2A5:
                data[:] = struct.pack(">ii", 90000, 0)
            msg = SimpleNamespace(arbitration_id=i, data=data, timestamp=10.,
                                  is_extended_id=False, is_remote_frame=False, is_error_frame=False)
            fb.update(msg, wall=10., mono=20.)
        q, dq, enabled, mode = fb.snapshot(20.1)
        self.assertAlmostEqual(q[0], np.pi/2)
        self.assertTrue(enabled and mode)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            fb.snapshot(20.2)
        fb.frames[0x261] = (20., bytes([0,0,0,0,0,0x41,0,0]))
        with self.assertRaisesRegex(RuntimeError, "fault"):
            fb.snapshot(20.1)

    def test_guard_rejects_stale_readonly_restart_and_fault(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"arm.json"
            status = dict(schema="sfyg_arm_hold_v1", session="one", monotonic_s=10.,
                          state="HOLD", ready=True, execute=True, policy_sha256=arm.POLICY_SHA)
            arm.publish_status(path, status)
            guard = ArmGuard(path, arm.POLICY_SHA)
            guard.check(10.1)
            with self.assertRaises(RuntimeError):
                guard.check(10.3)
            for key, value in [("execute", False), ("state", "FAULT"), ("session", "two")]:
                arm.publish_status(path, dict(status, **{key: value}))
                with self.assertRaises(RuntimeError):
                    guard.check(10.1)

    def test_arm_loss_damps_legs_in_publisher(self):
        sent = []
        coords = SimpleNamespace(order=np.arange(10), sign=np.ones(10), zero=np.zeros(10))
        guard = SimpleNamespace(check=lambda: (_ for _ in ()).throw(RuntimeError("arm lost")))
        lease = legs.MotorLease(sent.append, SimpleNamespace(snapshot=lambda *a: True),
                                coords, [5]*10, clock=lambda: 10., arm_guard=guard)
        lease.joy_at = 10.
        lease.renew(dict(q=[0]*10, kp=[100]*10, kd=[5]*10), 10.)
        lease.pump()
        self.assertIn("arm lost", str(lease.fault))
        self.assertEqual(sent[-1]["kp"], [0]*10)

    def test_sender_primes_before_enable_no_gripper_or_reset(self):
        calls = []
        class SDK:
            def __init__(self, channel): pass
            def __getattr__(self, name):
                return lambda *a, **k: calls.append((name, a, k))
        with patch.dict(sys.modules, piper_sdk=SimpleNamespace(C_PiperInterface_V2=SDK)):
            sender = arm.Sender("can0")
            sender.start([0, np.pi/2, -np.pi/2, 0, 0, 0])
            sender.stop(); sender.close()
        self.assertEqual(calls[0], ("ConnectPort", (), {"piper_init": False}))
        self.assertEqual(calls[1][0], "JointCtrl")
        self.assertEqual(calls[1][1][1:3], (90000, -90000))
        self.assertEqual([c[1][0] for c in calls if c[0] == "EnableArm"], list(range(1,7)))
        self.assertNotIn("ResetPiper", [c[0] for c in calls])


if __name__ == "__main__":
    unittest.main()
