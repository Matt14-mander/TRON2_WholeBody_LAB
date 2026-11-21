import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/hardware'))
from sfyg_supported_prepare import SupportedPrepare, RECORDED, CommandRejected
from run_sfyg_lifecycle import validate_preflight, SUSPENDED_CHECKS
from sfyg_lifecycle import Lifecycle
import prepare_sfyg_legs as hardware


def config():
    return dict(defaults=[0,0,-3.14159,0,0,0,0,3.14159,0,0],
                actuators=[dict(kp=100, kd=5, effort=40)]*10,
                policy_sha256='test')


class PrepareTests(unittest.TestCase):
    def test_path_hold_no_policy_and_existing_window_unchanged(self):
        cfg = config()
        old = Lifecycle(cfg, None)
        with self.assertRaises(CommandRejected):
            old.validate_command(RECORDED, np.zeros(10), 0, 'prepare_supported', None)
        c = SupportedPrepare(cfg)
        q = RECORDED + .03
        preview = c.preview(q, np.zeros(10), np.array([0, 0, -1]))
        self.assertTrue(preview['start_gate_passed'])
        targets = []
        for i in range(1201):
            p = c.tick(i*.02, q, np.zeros(10), [0,0,-1], 'prepare suspended' if i == 0 else None)
            q = np.array(p['q'])
            targets.append(q.copy())
        self.assertEqual(c.state, 'HOLD')
        np.testing.assert_allclose(q, cfg['defaults'])
        for waypoint in preview['move_waypoints']:
            frame = round((1. + waypoint['move_time_s'])/.02)
            np.testing.assert_allclose(targets[frame], waypoint['q_rad'], atol=1e-14)
        self.assertLess(np.max(np.abs(np.diff(targets, axis=0)))/.02, .1)
        self.assertLess(np.max(np.abs(np.diff(targets, n=2, axis=0)))/.02**2, .1)
        with self.assertRaises(CommandRejected): c.tick(24.02, q, np.zeros(10), [0,0,-1], 'start grounded')

    def test_reject_deep_pose_and_initial_velocity(self):
        for q, dq, gravity in (
            (RECORDED+.081, np.zeros(10), [0, 0, -1]),
            (RECORDED, np.ones(10)*.11, [0, 0, -1]),
            (RECORDED, np.zeros(10), [0, .5, -np.sqrt(.75)]),
            (RECORDED, np.zeros(10), [0, 0, -.5]),
        ):
            c = SupportedPrepare(config())
            preview = c.preview(q, dq, np.array(gravity))
            self.assertFalse(preview['start_gate_passed'])
            self.assertNotIn('move_waypoints', preview)
            with self.assertRaises(CommandRejected) as refusal:
                c.tick(0, q, dq, gravity, 'prepare suspended')
            self.assertEqual(preview['start_rejection'], str(refusal.exception))
            self.assertEqual(c.state, 'WAIT')

    def test_faults_and_stop_release(self):
        for kind in ('tilt', 'speed', 'track', 'deadline', 'abort'):
            c = SupportedPrepare(config())
            c.tick(0, RECORDED, np.zeros(10), [0,0,-1], 'prepare suspended')
            q, dq, g, t = RECORDED.copy(), np.zeros(10), [0,0,-1], .02
            if kind == 'tilt': g = [0,.5,-np.sqrt(.75)]
            if kind == 'speed': dq[0] = 1.01
            if kind == 'track': q[3] += .3
            if kind == 'deadline': t = .1
            p = c.tick(t,q,dq,g,'abort' if kind == 'abort' else None)
            self.assertEqual(c.state, 'FAULT')
            self.assertEqual(p['kp'], [0.]*10)
        c = SupportedPrepare(config())
        q = RECORDED.copy()
        for i in range(501):
            event = {0:'prepare suspended', 150:'stop', 300:'release supported'}.get(i)
            p = c.tick(i*.02,q,np.zeros(10),[0,0,-1],event)
            q = np.array(p['q'])
        self.assertEqual(c.state,'DONE')
        self.assertNotIn('HOLD', [e['state'] for e in c.events])

    def test_preflight_does_not_assume_enable(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'profile.json'
            profile = json.loads((ROOT/'scripts/hardware/sfyg_supported_prepare.example.json').read_text())
            args = SimpleNamespace(preflight=path, execute=True, suspended_prepare=True)
            path.write_text(json.dumps(profile))
            with self.assertRaises(ValueError):
                validate_preflight(args, config())
            profile.update({key: True for key in SUSPENDED_CHECKS})
            profile.update(operator='reviewed', policy_sha256='test', payload_mass_kg=0)
            path.write_text(json.dumps(profile))
            self.assertEqual(validate_preflight(args, config()), profile)
            for key in SUSPENDED_CHECKS + ('payload_mass_kg', 'quaternion_order'):
                path.write_text(json.dumps(profile | {key: False}))
                with self.assertRaises(ValueError):
                    validate_preflight(args, config())

    def test_preview_never_constructs_sender(self):
        class Clock:
            t = 0.
            def now(self): return self.t
            def sleep(self, dt): self.t += dt
        clock = Clock()
        class Samples:
            def snapshot(self, *args):
                stamp = int(round(clock.t*1000000))+1
                return (clock.t, (stamp,clock.t,RECORDED,np.zeros(10)),
                        (stamp,clock.t,[1,0,0,0],[0,0,0]))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            profile = root/'profile.json'
            profile.write_text((ROOT/'scripts/hardware/sfyg_supported_prepare.example.json').read_text())
            a = SimpleNamespace(bundle_dir=root, profile=profile, output_dir=root/'out', robot_ip='test')
            with patch.object(hardware,'configuration',return_value=config()), \
                 patch.object(hardware,'Samples',Samples), \
                 patch.object(hardware,'subscribe',return_value=object()), \
                 patch.object(hardware.time,'monotonic',clock.now), \
                 patch.object(hardware.time,'sleep',clock.sleep):
                hardware.run(a)
            report = json.loads((a.output_dir/'summary.json').read_text())
            self.assertFalse(report['motor_output'])
            self.assertEqual(report['policy_calls'],0)
            self.assertEqual(report['state'],'WAIT')
            preview = json.loads((a.output_dir/'preview.json').read_text())
            self.assertTrue(preview['start_gate_passed'])
            self.assertEqual(preview['move_waypoints'][0]['q_rad'], RECORDED.tolist())
            np.testing.assert_allclose(preview['move_waypoints'][-1]['q_rad'], config()['defaults'])
