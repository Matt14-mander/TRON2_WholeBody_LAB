import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/hardware"))
import prepare_sfyg_arm as prep
import hold_sfyg_arm as arm


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "scripts/hardware/sfyg_piper_x_prepare.example.json").read_text())
        self.profile = arm.Profile(self.data, [0, 1.5707963, -1.4835299, 0, 0, 0])

    def controller(self, directory, execute=False, start=None):
        start = np.zeros(6) if start is None else start
        path = Path(directory)/"plan.json"
        plan = prep.build_plan(self.profile, self.data, start)
        path.write_text(json.dumps(plan))
        self.data.update(staged_path_reviewed=True, mapping_evidence="fixture verified", limits_evidence="device checked")
        args = SimpleNamespace(execute=execute, plan_file=path,
                               plan_sha256=prep.hashlib.sha256(path.read_bytes()).hexdigest())
        return prep.Segmented(self.profile, args, self.data)

    def test_preview_preserves_real_out_of_range_start_and_no_sender(self):
        q = np.radians([-.025, -5.652, 2.562, 33.993, .349, 0])
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(execute=False, plan_file=Path(d)/"plan.json")
            ctl = prep.Segmented(self.profile, args, self.data)
            with patch.object(arm, "Sender", side_effect=AssertionError("preview sends")):
                self.assertIsNone(ctl.tick(q, np.zeros(6), False, False, 0))
                self.assertFalse(ctl.handle("next 1 supported", q, np.zeros(6), 0))
            plan = json.loads(args.plan_file.read_text())
            self.assertEqual({v["joint"] for v in plan["violations"]}, {2, 3})
            with self.assertRaisesRegex(ValueError, "limits"):
                prep.load_plan(args.plan_file, prep.hashlib.sha256(args.plan_file.read_bytes()).hexdigest(), self.profile, self.data)

    def test_review_binding_hash_limits_and_endpoint(self):
        with tempfile.TemporaryDirectory() as d:
            ctl = self.controller(d, True)
            args = ctl.args
            with self.assertRaisesRegex(ValueError, "SHA256"):
                prep.load_plan(args.plan_file, "bad", self.profile, self.data)
            self.data["zero_rad"][0] = .01
            with self.assertRaisesRegex(ValueError, "mapping"):
                prep.load_plan(args.plan_file, args.plan_sha256, self.profile, self.data)

    def test_segment_confirmation_continuity_pause_and_final_guard(self):
        with tempfile.TemporaryDirectory() as d:
            ctl = self.controller(d, True)
            q, dq, now = np.zeros(6), np.zeros(6), 0.
            self.assertFalse(ctl.handle("prepare supported", q, dq, now))
            self.assertTrue(ctl.handle("next 1 supported", q, dq, now))
            np.testing.assert_array_equal(ctl.tick(q, dq, False, False, now), q)
            self.assertFalse(ctl.handle("next 2 supported", q, dq, now))
            for segment in range(len(ctl.plan["segments_sdk_rad"])):
                if segment:
                    self.assertFalse(ctl.handle(f"next {segment+1} supported", q, dq, now))
                until = now+ctl.ack_delay+ctl.duration+2
                while now < until:
                    now += .02
                    prior = q.copy()
                    q = ctl.tick(q, dq, True, True, now)
                    self.assertLessEqual(max(abs(q-prior)), .05*.02+1e-6)
                self.assertEqual(ctl.index, segment+1)
                self.assertEqual(ctl.ready, segment+1 == len(ctl.plan["segments_sdk_rad"]))
                if not ctl.ready:
                    self.assertEqual(ctl.state, "PAUSED")
            self.assertEqual(ctl.state, "HOLD")
            with self.assertRaisesRegex(RuntimeError, "acknowledgement"):
                ctl.tick(q, dq, False, True, now+1)

    def test_changed_start_and_missing_review_refused(self):
        with tempfile.TemporaryDirectory() as d:
            ctl = self.controller(d, True)
            with self.assertRaisesRegex(RuntimeError, "start changed"):
                ctl.handle("next 1 supported", np.array([.03, 0, 0, 0, 0, 0]), np.zeros(6), 0)
            self.data["staged_path_reviewed"] = False
            with self.assertRaisesRegex(ValueError, "reviewed"):
                prep.Segmented(self.profile, ctl.args, self.data)

    def test_motion_faults_and_settle_timeout(self):
        for failure in ("speed", "tracking", "timeout"):
            with tempfile.TemporaryDirectory() as d:
                ctl = self.controller(d, True)
                q, dq = np.zeros(6), np.zeros(6)
                ctl.handle("next 1 supported", q, dq, 0)
                if failure == "speed":
                    dq[0] = .21
                    with self.assertRaisesRegex(RuntimeError, "overspeed"):
                        ctl.tick(q, dq, True, True, 1)
                elif failure == "tracking":
                    with self.assertRaisesRegex(RuntimeError, "tracking"):
                        ctl.tick(q, dq, True, True, ctl.ack_delay+ctl.duration)
                else:
                    q = ctl.target.copy(); q[0] = .03
                    with self.assertRaisesRegex(RuntimeError, "timeout"):
                        ctl.tick(q, dq, True, True, ctl.ack_delay+ctl.duration+11)

    def test_waypoints_and_bad_plan_rejected(self):
        self.data["sdk_waypoints_rad"] = [[0, .2, -.1, .1, 0, 0]]
        plan = prep.build_plan(self.profile, self.data, np.zeros(6))
        np.testing.assert_allclose(plan["segments_sdk_rad"][0], self.data["sdk_waypoints_rad"][0])
        with tempfile.TemporaryDirectory() as d:
            ctl = self.controller(d, True)
            for points in ([[0]*6], [[0, 1.57, -1.48, 0, 0, 0]], [[0]*6]*201):
                bad = dict(ctl.plan, segments_sdk_rad=points)
                ctl.args.plan_file.write_text(json.dumps(bad))
                sha = prep.hashlib.sha256(ctl.args.plan_file.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    prep.load_plan(ctl.args.plan_file, sha, self.profile, self.data)

    def test_runner_preview_never_constructs_sender(self):
        class Bus:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def recv(self, timeout): return None
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(status_file=Path(d)/"status.json", execute=False, can="fake",
                                   plan_file=Path(d)/"plan.json")
            args.controller_factory = lambda p: prep.Segmented(p, args, self.data)
            with patch.object(sys, "stdin", io.StringIO("quit supported\n")), \
                 patch.object(arm, "Sender", side_effect=AssertionError("sender in preview")):
                arm.run_locked(args, self.profile, SimpleNamespace(Bus=lambda **kw: Bus()))
            status = json.loads(args.status_file.read_text())
            self.assertFalse(status["ready"])
            self.assertFalse(status["execute"])


if __name__ == "__main__":
    unittest.main()
