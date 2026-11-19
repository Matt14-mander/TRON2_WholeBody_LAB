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
from prepare_sfyg_legs import validate, CHECKS
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
        q = RECORDED.copy()
        targets = []
        for i in range(1201):
            p = c.tick(i*.02, q, np.zeros(10), [0,0,-1], 'prepare suspended' if i == 0 else None)
            q = np.array(p['q'])
            targets.append(q.copy())
        self.assertEqual(c.state, 'HOLD')
        np.testing.assert_allclose(q, cfg['defaults'])
        self.assertLess(np.max(np.abs(np.diff(targets, axis=0)))/.02, .1)
        self.assertLess(np.max(np.abs(np.diff(targets, n=2, axis=0)))/.02**2, .1)
        with self.assertRaises(CommandRejected): c.tick(24.02, q, np.zeros(10), [0,0,-1], 'start grounded')

    def test_reject_deep_pose_and_initial_velocity(self):
        for q, dq in ((RECORDED+.081, np.zeros(10)), (RECORDED, np.ones(10)*.11)):
            c = SupportedPrepare(config())
            with self.assertRaises(CommandRejected): c.tick(0,q,dq,[0,0,-1],'prepare suspended')
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
        profile = dict(quaternion_order='wxyz')
        validate(profile, config(), False)
        with self.assertRaises(ValueError): validate(profile, config(), True)
        profile.update({k:True for k in CHECKS})
        profile.update(operator='reviewed', policy_sha256='test',payload_mass_kg=0)
        validate(profile,config(),True)
        for key in CHECKS:
            with self.assertRaises(ValueError): validate(profile|{key:False},config(),True)

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
            a = SimpleNamespace(bundle_dir=root,profile=profile,execute=False,
                output_dir=root/'out',arm_status_file=root/'missing.json',robot_ip='test')
            with patch.object(hardware,'configuration',return_value=config()), \
                 patch.object(hardware,'Samples',Samples), \
                 patch.object(hardware,'subscribe',return_value=object()), \
                 patch.object(hardware,'make_sender',side_effect=AssertionError('motor sender')), \
                 patch.object(hardware.time,'monotonic',clock.now), \
                 patch.object(hardware.time,'sleep',clock.sleep):
                hardware.run(a)
            report = json.loads((a.output_dir/'summary.json').read_text())
            self.assertFalse(report['motor_output'])
            self.assertEqual(report['policy_calls'],0)
            self.assertEqual(report['state'],'WAIT')
