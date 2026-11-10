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
import shadow_sfyg_locomotion as shadow


class ProbeActor(torch.nn.Module):
    def forward(self, obs, history, command):
        return obs[:, 6:16] + history[:, 0, 6:16] + obs[:, 26:36] + command[:, :1]


class ShadowTest(unittest.TestCase):
    def test_coordinates(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "coordinates.json"
            path.write_text(json.dumps(dict(sdk_names=shadow.LEG_NAMES[::-1], zero_rad=[1]*10,
                sign=[-1]*10, imu_to_base=[[0,-1,0],[1,0,0],[0,0,1]])))
            coords = shadow.Coordinates(path)
            q, dq, angular, gravity = coords.convert((0,0,range(10),range(10)),
                (0,0,[2**-.5,2**-.5,0,0],[1,0,0]))
            np.testing.assert_equal(q, 1-np.arange(10)[::-1])
            np.testing.assert_equal(dq, -np.arange(10)[::-1])
            np.testing.assert_allclose(angular, [0,1,0])
            np.testing.assert_allclose(gravity, [1,0,0], atol=1e-7)

    def test_receiver(self):
        samples = shadow.Samples()
        joint = SimpleNamespace(stamp=1, q=[1]*10, dq=[0]*10)
        imu = SimpleNamespace(stamp=9, quat=[1,0,0,0], gyro=[0]*3)
        with patch.object(shadow.time, "monotonic", return_value=0):
            samples.joint(joint)
            samples.imu(imu)
            joint.q[0] = 9
            self.assertEqual(samples.snapshot(.04,.015)[1][2][0], 1)
        with patch.object(shadow.time, "monotonic", return_value=.03):
            samples.joint(joint)  # duplicated stamp must not refresh reception time
            imu.stamp += 1
            samples.imu(imu)
            self.assertIsNone(samples.snapshot(.04,.015))
        with patch.object(shadow.time, "monotonic", return_value=.08):
            self.assertIsNone(samples.snapshot(.04,.1))
        joint.stamp = 0
        samples.joint(joint)
        with self.assertRaisesRegex(ValueError, "backwards"):
            samples.snapshot(.04,.015)

    def test_run_and_replay(self):
        class SDK:
            def init(self, ip):
                return ip == "10.192.1.2"
            def subscribeRobotState(self, cb):
                self.joint = cb
            def subscribeImuData(self, cb):
                self.imu = cb

        sdk = SDK()  # Deliberately has no command or mode-changing methods.
        now = [0.]
        def feed(delay):
            now[0] += delay
            stamp = int(now[0]*1e9)
            sdk.joint(SimpleNamespace(stamp=stamp, q=[now[0]]*10, dq=[0]*10))
            sdk.imu(SimpleNamespace(stamp=stamp+2000000000, quat=[1,0,0,0], gyro=[0]*3))

        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            actor = torch.jit.script(ProbeActor())
            actor.save(str(path / "policy.pt"))
            cfg = dict(schema="sfyg_mujoco_v1", source_commit="f441d67", phase="robust",
                policy_sha256=shadow.sha256(path/"policy.pt"), checkpoint_sha256="test",
                checkpoint_path="test", leg_names=shadow.LEG_NAMES, physics_dt=.005, decimation=4,
                gait=[1.5,.5,.5,.1], defaults=[0]*10, command_limits=[1,.5,1.5],
                terms=[dict(name=n,width=w,scale=None,clip=None)
                       for n,w in zip(shadow.TERM_NAMES, shadow.WIDTHS)])
            (path/"deployment.json").write_text(json.dumps(cfg))
            args = SimpleNamespace(bundle_dir=path, output_dir=path/"out", coordinates=None,
                command=[.2,0,0], duration=.06, robot_ip="10.192.1.2", startup_timeout=1,
                max_age_ms=40, max_skew_ms=15)
            from types import ModuleType
            pkg = ModuleType("limxsdk")
            sub = ModuleType("limxsdk.robot")
            pkg.robot = sub
            sub.Robot = lambda kind: sdk
            sub.RobotType = SimpleNamespace(Tron2="TRON2")
            modules = {"limxsdk":pkg, "limxsdk.robot":sub, "limxsdk.robot.Robot":sub.Robot,
                       "limxsdk.robot.RobotType":sub.RobotType}
            with patch.dict(sys.modules, modules), patch.object(shadow.time,"monotonic",lambda: now[0]), \
                    patch.object(shadow.time,"sleep",feed):
                shadow.run(args)
            rows = [json.loads(s) for s in (path/"out/steps.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            np.testing.assert_equal(rows[0]["history"], [rows[0]["observation"]]*10)
            for i, r in enumerate(rows):
                inputs = [torch.tensor(r[k])[None] for k in ("observation","history","command")]
                np.testing.assert_allclose(actor(*inputs)[0], r["action"])
                np.testing.assert_equal(r["observation"][26:36], rows[i-1]["action"] if i else [0]*10)
                np.testing.assert_equal(r["history"][-1], r["observation"])
            self.assertTrue(json.loads((path/"out/summary.json").read_text())["completed"])
            (path/"policy.pt").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError,"export"):
                shadow.run(args)


if __name__ == "__main__":
    unittest.main()
