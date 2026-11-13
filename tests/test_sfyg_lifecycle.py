import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/hardware"))
import sfyg_lifecycle as core
import run_sfyg_lifecycle as hardware


class ConstantPolicy(torch.nn.Module):
    def __init__(self, value=.4):
        super().__init__()
        self.value = value

    def forward(self, obs, history, command):
        return torch.full((1, 10), self.value)


def config():
    return dict(defaults=[0.]*10, actuators=[dict(kp=100., kd=5., effort=40.)]*10,
                gait=[.8, .5, .5, .2], physics_dt=.005, decimation=4,
                policy_sha256=core.POLICY_SHA, checkpoint_path="test",
                terms=[dict(name=n, width=w, scale=None, clip=None)
                       for n, w in zip(core.TERM_NAMES, core.WIDTHS)])


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.ctl = core.Lifecycle(config(), ConstantPolicy())
        self.now = -.02

    def tick(self, event=None, request=None, q=None):
        self.now += .02
        return self.ctl.tick(self.now, np.zeros(10) if q is None else q, np.zeros(10),
                             np.zeros(3), [0., 0., -1.], event, request)

    def ready(self):
        self.tick("prepare_supported")
        for _ in range(360):
            self.tick()
        self.assertTrue(self.ctl.ready)

    def policy(self):
        self.ready()
        self.tick("start_grounded")
        for _ in range(20):
            self.tick()
        self.assertEqual(self.ctl.state, "POLICY")

    def test_captured_pose_no_initial_target_jump(self):
        q = np.arange(10)*.03
        p = self.tick("prepare_supported", q=q)
        np.testing.assert_equal(p["q"], q)
        np.testing.assert_equal(p["kp"], np.zeros(10))

    def test_deep_fold_is_refused_before_arming(self):
        with self.assertRaisesRegex(ValueError, "deep-fold"):
            self.tick("prepare_supported", q=np.full(10, -2.))
        self.assertEqual(self.ctl.state, "WAIT")

    def test_start_gate_history_and_steady_policy(self):
        with self.assertRaises(ValueError):
            self.tick("start_grounded")
        self.ready()
        p = self.tick("start_grounded")
        np.testing.assert_equal(p["q"], np.zeros(10))
        np.testing.assert_equal(self.ctl.hist, [self.ctl.obs]*10)
        previous = self.ctl.previous.copy()
        self.tick()
        np.testing.assert_allclose(self.ctl.obs[26:36], previous)
        for _ in range(20):
            self.tick()
        np.testing.assert_allclose(self.ctl.target, .1)
        np.testing.assert_allclose(self.ctl.previous, .4)
        self.assertGreater(self.ctl.obs[36], 0.)  # clock continues at zero velocity

    def test_stop_blend_does_not_jump_to_full_action(self):
        self.ready()
        self.tick("start_grounded")
        self.tick("stop")
        self.assertEqual(self.ctl.state, "BLEND")
        self.assertLess(max(self.ctl.previous), .01)
        for _ in range(20):
            self.tick()
        self.assertEqual(self.ctl.state, "STOPPING")

    def test_start_rechecks_current_pose(self):
        self.ready()
        with self.assertRaises(ValueError):
            self.tick("start_grounded", q=np.ones(10)*.2)
        self.assertEqual(self.ctl.state, "READY")

    def test_ramp_stop_release(self):
        self.policy()
        self.tick(request=[.2, 0, 0])
        for _ in range(110):
            self.tick()
        self.assertAlmostEqual(float(self.ctl.command[0]), .2, places=6)
        with self.assertRaises(ValueError):
            self.tick("release_supported")
        self.tick("stop")
        for _ in range(220):
            p = self.tick()
        self.assertEqual(self.ctl.state, "ZERO")
        self.assertGreater(min(p["kp"]), 0.)  # stop preserves policy support
        p = self.tick("release_supported")
        np.testing.assert_allclose(p["q"], self.ctl.target)
        gains = []
        for _ in range(160):
            p = self.tick()
            gains.append(p["kp"][0])
        self.assertEqual(self.ctl.state, "DONE")
        self.assertTrue(np.all(np.diff(gains) <= 1e-10))
        np.testing.assert_equal(p["kp"], np.zeros(10))

    def test_deadline_fault_latches(self):
        self.policy()
        self.now += .1
        p = self.tick()
        self.assertEqual(self.ctl.state, "FAULT")
        np.testing.assert_equal(p["kp"], np.zeros(10))
        self.tick()
        self.assertEqual(self.ctl.state, "FAULT")

    def test_bad_command_can_retry_without_losing_balance_controller(self):
        self.policy()
        when = self.now+.02
        values = (when, np.zeros(10), np.zeros(10), np.zeros(3), [0,0,-1])
        with self.assertRaises(core.CommandRejected):
            self.ctl.tick(*values, requested=[.3,0,0])
        p = self.ctl.tick(*values)
        self.assertEqual(self.ctl.state, "POLICY")
        self.assertGreater(min(p["kp"]), 0)

    def test_deadline_cannot_be_overridden_by_start(self):
        self.ready()
        self.now += .1
        p = self.tick("start_grounded")
        self.assertEqual(self.ctl.state, "FAULT")
        np.testing.assert_equal(p["kp"], np.zeros(10))

    def test_bad_inference_and_torque_projection(self):
        self.policy()
        self.ctl.policy = ConstantPolicy(1000.)
        p = self.tick()
        np.testing.assert_allclose(p["q"], .4)
        np.testing.assert_allclose(self.ctl.previous, 1000.)  # source keeps pre-effort-limit action
        self.ctl.policy = ConstantPolicy(float("nan"))
        p = self.tick()
        self.assertEqual(self.ctl.state, "FAULT")
        np.testing.assert_equal(p["kp"], np.zeros(10))

    def test_inverse_coordinates_and_gain_order(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"coords.json"
            path.write_text(json.dumps(dict(sdk_names=core.LEG_NAMES[::-1],
                zero_rad=list(np.arange(10)*.1), sign=[-1]*10, imu_to_base=np.eye(3).tolist())))
            coords = core.Coordinates(path)
            q = np.arange(10)*.2
            packet = core.sdk_packet(dict(q=q, kp=np.arange(10)+1, kd=np.arange(10)+11), coords)
            restored, *_ = coords.convert((0, 0, packet["q"], [0]*10), (0, 0, [1,0,0,0], [0]*3))
            np.testing.assert_allclose(restored, q)
            np.testing.assert_equal(packet["kp"], np.arange(10, 0, -1))


class HardwareTest(unittest.TestCase):
    def lease(self):
        self.now, self.sent = 1., []
        samples = SimpleNamespace(snapshot=lambda *args: True)
        lease = hardware.MotorLease(self.sent.append, samples, core.Coordinates(None), [5.]*10, lambda: self.now)
        lease.joy_at = self.now
        return lease

    def test_unarmed_no_output_expiry_damps_and_cannot_rearm(self):
        lease = self.lease()
        lease.pump()
        self.assertEqual(self.sent, [])
        p = dict(q=[0.]*10, kp=[100.]*10, kd=[5.]*10)
        lease.renew(p, self.now)
        lease.pump()
        self.assertEqual(self.sent[-1]["kp"], [100.]*10)
        self.now += .061
        lease.pump()
        self.assertEqual(self.sent[-1]["kp"], [0.]*10)
        lease.renew(p, self.now)
        lease.pump()
        self.assertEqual(self.sent[-1]["kp"], [0.]*10)

    def test_sensor_remote_inference_watchdogs(self):
        p = dict(q=[0.]*10, kp=[100.]*10, kd=[5.]*10)
        lease = self.lease()
        lease.renew(p, self.now)
        lease.samples.snapshot = lambda *args: None
        lease.pump()
        self.assertIn("stale", lease.fault)
        self.assertEqual(self.sent[-1]["kp"], [0.]*10)
        lease = self.lease()
        lease.renew(p, self.now)
        lease.joystick(SimpleNamespace(buttons=[0,0,1,0,1,0]))
        lease.pump()
        self.assertIn("abort", lease.fault)
        self.assertEqual(self.sent[-1]["kp"], [0.]*10)
        lease = self.lease()
        lease.renew(p, self.now-.041)
        self.assertIn("inference", lease.fault)

    def test_no_execute_never_constructs_motor_sender(self):
        def connect(ip, samples):
            samples.joint(SimpleNamespace(stamp=1, q=[0]*10, dq=[0]*10))
            samples.imu(SimpleNamespace(stamp=1, quat=[1,0,0,0], gyro=[0]*3))
            return object()  # No publishRobotCmd or subscribeSensorJoy methods.
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(bundle_dir=Path(d), output_dir=Path(d)/"out",
                                   preflight=None, execute=False, robot_ip="test")
            with patch.object(hardware, "load_bundle", return_value=(config(), ConstantPolicy())), \
                 patch.object(hardware, "subscribe", connect), patch.object(sys, "stdin", io.StringIO("quit\n")), \
                 patch.object(hardware, "make_sender", side_effect=AssertionError("motor sender constructed")):
                hardware.run(args)
            result = json.loads((args.output_dir/"summary.json").read_text())
            self.assertFalse(result["motor_output"])
            self.assertEqual(result["publish_count"], 0)

    def test_execute_refuses_absent_arm_and_unverified_coordinates(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"preflight.json"
            profile = dict(policy_sha256=core.POLICY_SHA, operator="operator")
            profile.update({key: True for key in hardware.CHECKS})
            profile["arm_in_carry_pose_and_payload_0p5kg"] = False
            path.write_text(json.dumps(profile))
            args = SimpleNamespace(execute=True, preflight=path)
            with self.assertRaisesRegex(ValueError, "bare robot"):
                hardware.validate_preflight(args, config())


if __name__ == "__main__":
    unittest.main()
