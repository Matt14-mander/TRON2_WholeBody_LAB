"""Task-space frames, COM motion, time alignment and data validity."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

spec = importlib.util.spec_from_file_location("task_space_telemetry",
    Path(__file__).resolve().parents[1] / "scripts/mujoco/task_space_telemetry.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(payload=True):
    chain = ''.join(f'<body pos=".1 0 0"><joint name="arm{i}_Joint" axis="0 1 0"/>'
                    '<geom type="sphere" size=".02" mass=".1"/>' for i in range(1, 7))
    load = ('<body name="paint_payload_Link" pos=".15 0 0"><geom type="box" '
            'pos=".02 0 0" size=".05 .02 .01" mass=".5"/></body>') if payload else ''
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="base_Link">'
        '<freejoint/><geom type="sphere" size=".03" mass="1"/>'+chain+
        '<body name="gripper_base_Link" pos=".1 0 0">'+load+'</body>'+'</body>'*6+
        '</body></worldbody></mujoco>')
    channel = SimpleNamespace(spec=SimpleNamespace(joints=[f"arm{i}_Joint" for i in range(1, 7)]),
                              map=[(7+i, 6+i, i) for i in range(6)], has_command=True,
                              cmd=dict(q=[0.]*6))
    data = mujoco.MjData(model)
    core = SimpleNamespace(model=model, channels=[channel], data=data, paused=False, manual_mode=False)
    return core


class TaskTests(unittest.TestCase):
    def test_actual_base_world_error_frames_and_payload_com(self):
        with tempfile.TemporaryDirectory() as tmp:
            core = fixture()
            recorder = module.TaskSpaceRecorder(Path(tmp)/"sim")
            recorder.attach(core)
            core.data.qpos[:3] = [2., 3., .8]
            core.data.qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
            core.data.qpos[7] = .2
            before = core.data.qpos.copy()
            recorder.capture(core.data, core.channels[0].cmd, 100.)
            row = recorder.rows[-1]
            rb = np.array(row["base_rotation_world"]).reshape(3, 3)
            for key in ("ee", "payload"):
                point = row[key]
                np.testing.assert_allclose(point["position_error_world_m"],
                    rb@point["position_error_base_m"], atol=1e-12)
                self.assertGreater(point["orientation_error_rad"], .19)
            np.testing.assert_array_equal(core.data.qpos, before)
            bid = recorder.ids["payload"]
            np.testing.assert_allclose(row["payload"]["position_world_m"], recorder.actual.xipos[bid])
            self.assertAlmostEqual(np.linalg.norm(recorder.actual.xipos[bid]-recorder.actual.xpos[bid]), .02)
            # Move only the whole robot: base-frame arm error must be invariant.
            core.data.qpos[:3] += [10., -4., 2.]
            recorder.capture(core.data, core.channels[0].cmd, 100.01)
            np.testing.assert_allclose(row["ee"]["position_error_base_m"],
                recorder.rows[-1]["ee"]["position_error_base_m"], atol=1e-12)

    def test_translating_base_acceleration_is_net_and_reset_clears_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            core = fixture()
            recorder = module.TaskSpaceRecorder(Path(tmp)/"sim")
            recorder.attach(core)
            for t in (.1, .11):
                core.data.time = t
                core.data.qpos[0] = .5*2*t*t
                core.data.qvel[0] = 2*t
                recorder.capture(core.data, core.channels[0].cmd, 100+t)
            for key in ("ee", "payload"):
                np.testing.assert_allclose(recorder.rows[-1][key]["acceleration_world_m_s2"], [2, 0, 0], atol=1e-10)
                np.testing.assert_allclose(recorder.rows[-1][key]["position_error_base_m"], 0., atol=1e-12)
            recorder.on_reset(core)
            core.data.time = 0
            recorder.capture(core.data, core.channels[0].cmd, 101.)
            self.assertIsNone(recorder.rows[-1]["payload"]["acceleration_world_m_s2"])

    def test_alignment_phase_summary_and_missing_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core = fixture(payload=False)
            recorder = module.TaskSpaceRecorder(root/"sim")
            recorder.attach(core)
            for i in range(3):
                core.data.time = i*.01
                recorder.capture(core.data, core.channels[0].cmd, 100.005+i*.01)
            recorder.save()
            arms = root/"arms"
            arms.mkdir()
            (arms/"contract.json").write_text(json.dumps(dict(clock=recorder.contract["clock"])))
            (arms/"arm_report.json").write_text(json.dumps(dict(outcome=dict(completed=True))))
            steps = [dict(monotonic_s=100+i*.01, phase="hold" if i < 2 else "complete") for i in range(4)]
            (arms/"arm_steps.jsonl").write_text('\n'.join(map(json.dumps, steps)))
            report = module.analyze(root/"sim", arms, root/"analysis")
            self.assertEqual(report["all"]["samples"], 3)
            self.assertEqual(report["phases"]["hold"]["samples"], 2)
            self.assertIsNone(report["all"]["payload"]["position_error_base_m"])
            self.assertEqual(report["all"]["ee"]["position_error_base_m"]["norm_rms"], 0.)
            (arms/"arm_steps.jsonl").write_text(json.dumps(dict(phase="hold")))
            with self.assertRaisesRegex(ValueError, "timestamps"):
                module.analyze(root/"sim", arms, root/"invalid")


if __name__ == "__main__":
    unittest.main()
