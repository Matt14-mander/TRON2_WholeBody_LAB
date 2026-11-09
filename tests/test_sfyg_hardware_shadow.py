"""Exercise live read-only subscription and replay without robot hardware."""
import contextlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/hardware"), str(ROOT / "rsl_rl")]
import shadow_sfyg_student as shadow
from rsl_rl.modules import ActorCritic
from sfyg_policy import sha256
from sfyg_student_core import StudentPolicy
from sfyg_teacher_core import Settings, execution_contract, runner_config, process_action


class HardwareShadowTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".test-temp").mkdir(exist_ok=True)
        tmp = tempfile.TemporaryDirectory(dir=ROOT / ".test-temp")
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name)
        self.checkpoint = self.path / "fixture.pt"
        self.names = shadow.HOLD_NAMES[:6]
        self.nominal = np.array([0., 1.57, -1.48, 0., 0., 0.])
        interface = dict(names=self.names, defaults=self.nominal.tolist())
        self.teacher = execution_contract("low", interface, Settings())
        torch.manual_seed(12)
        with contextlib.redirect_stdout(io.StringIO()):
            template = ActorCritic(147, 177, 9, **runner_config()["policy"])
        self.model = StudentPolicy(template.actor).eval()
        torch.save(dict(iteration=500, policy=self.model.state_dict(), student_contract=dict(
            schema="sfyg_student_v1", actor_input=144, raw_output=9, teacher_contract=self.teacher)),
            self.checkpoint)
        self.cal = dict(student_sha256=sha256(self.checkpoint), low_checkpoint_sha256="low",
                        q_bias_rad=[.001, -.002, .003, 0., .001, .002], zero_n=-.013,
                        gain=1.5444, cap_n=8.)
        self.args = SimpleNamespace(can="test", replay=None, seconds=3.2,
                                    checkpoint=self.checkpoint, arm_names=self.names,
                                    calibration=self.path / "cal.json", pose_tolerance=.1,
                                    output_dir=self.path / "live")
        self.args.calibration.write_text(json.dumps(self.cal))

    def test_passive_can_and_replay_match_reference_on_every_frame(self):
        # Only recv exists; any attempted CAN transmission fails the test.
        origin = 10.
        nominal_mdeg = np.rint(np.degrees(self.nominal)*1000).astype(int)
        speed_units = [-50, 4, 15, -10, 0, 32]
        def feedback():
            for i in range(320):
                q = nominal_mdeg + i
                frames = [(0x2A5+j, struct.pack(">ii", *q[2*j:2*j+2])) for j in range(3)]
                frames += [(0x251+j, struct.pack(">hhi", v, 700, -123)) for j,v in enumerate(speed_units)]
                for ident, data in frames:
                    yield SimpleNamespace(arbitration_id=ident, data=data, timestamp=origin+i*.01,
                                          is_extended_id=False, is_remote_frame=False, is_error_frame=False)
        stream = feedback()
        class Bus:
            def __init__(self, **kwargs):
                assert kwargs["interface"] == "socketcan" and len(kwargs["can_filters"]) == 9

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def recv(self, timeout):
                return next(stream)

        can_module = ModuleType("can")
        can_module.Bus = Bus
        clock = [round((origin+i*.01)*1e9) for i in range(320)]
        with patch.dict(sys.modules, {"can": can_module}), patch.object(shadow.time, "time_ns", side_effect=clock):
            shadow.run(self.args)
        steps = self.args.output_dir / "steps.jsonl"
        self.args.replay, self.args.can = steps, None
        self.args.output_dir = self.path / "replay"
        shadow.run(self.args)
        self.assertEqual(steps.read_bytes(), (self.args.output_dir / "steps.jsonl").read_bytes())

        rows = [json.loads(line) for line in steps.read_text().splitlines()]
        self.assertEqual(len(rows), 160)
        with torch.inference_mode():
            for name in ("native", "mujoco_cal"):
                frames = np.zeros((160, 18), dtype=np.float32)
                bias = np.asarray(self.cal["q_bias_rad"]) if name == "mujoco_cal" else 0.
                frames[1:, :6] = np.radians((nominal_mdeg + 2*np.arange(1, 160)[:, None])/1000) - self.nominal + bias
                frames[1:, 6:12] = np.asarray(speed_units)/1000
                padded = np.pad(frames, ((7, 0), (0, 0)))
                h = torch.from_numpy(np.stack([padded[i:i+8].flatten() for i in range(160)]))
                np.testing.assert_allclose([r[name]["history144"] for r in rows], h, atol=1e-7)
                w = self.model.estimate(h)
                if name == "mujoco_cal":
                    w[:, 0] = ((w[:, 0]*40 + .013)*1.5444).clamp(max=8)/40
                raw = self.model.actor(torch.cat((h, w), dim=1))
                cmd = process_action(raw, torch.arange(160)*.02, Settings())
                for field, reference in (("raw_vx", raw[:, 0]), ("cmd_vx", cmd[:, 0]),
                                         ("estimated_fx_n", 40*w[:, 0])):
                    np.testing.assert_allclose([r[name][field] for r in rows], reference, atol=1e-5)

    def test_bad_state_or_binding_stops(self):
        with self.assertRaisesRegex(ValueError, "Expected named"):
            shadow.ordered_arm(["arm"], [0], [0], self.names)
        with self.assertRaisesRegex(ValueError, "Missing"):
            list(shadow.timed_samples([{"stamp_ns": 1}, {"stamp_ns": 50_000_001}], .02))
        with self.assertRaisesRegex(ValueError, "does not belong"):
            shadow.Shadow(self.checkpoint, dict(self.cal, student_sha256="other"))
        self.args.replay = self.path / "input.jsonl"
        self.args.can = None
        self.args.replay.write_text(json.dumps(dict(stamp_ns=1, arm_names=self.names,
            q_rad=[0]*6, dq_rad_s=[0]*6)))
        with self.assertRaisesRegex(ValueError, "Outside carry pose"):
            shadow.run(self.args)
        report = json.loads((self.args.output_dir / "summary.json").read_text())
        self.assertFalse(report["completed"])
        self.assertEqual(report["frames"], 0)
