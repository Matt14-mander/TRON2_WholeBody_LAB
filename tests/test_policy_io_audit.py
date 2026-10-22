"""Verify passive capture and detect planted deployment contract faults."""
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("policy_io_audit", ROOT/"scripts/mujoco/policy_io_audit.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    contract = dict(default_q=[0.]*18, action_scale=[.25]*10,
        ang_vel_scale=.25, dof_pos_scale=1., dof_vel_scale=.05,
        observation_clip=100., gait=[.8, .5, .5, .2], requested_command=[.15, 0, 0])
    contract["default_q"][2], contract["default_q"][7] = -3.14159, 3.14159
    rows, previous, history = [], np.zeros(10), None
    for i in range(4):
        phase = 2*np.pi*((i+1)*.8/50)
        q, dq = np.array(contract["default_q"]), np.zeros(18)
        q[0], dq[0] = .1*i, .2*i
        proprio = np.concatenate(([0, 0, .01], [0, 0, -1], q[:10]-np.array(contract["default_q"][:10]), dq[:10]*.05,
            previous, [np.sin(phase), np.cos(phase)], [.8, .5, .5, .2])).astype(np.float32)
        history = np.tile(proprio, 10) if history is None else np.r_[history[42:], proprio]
        encoder, wrench = np.ones(3)*.01, np.tile([0, 0, -71., 0, 9., 0], (5, 1))
        action = np.arange(10)*.1+i*.01
        row = dict(source=dict(stamp=100+i*.02, q=q.tolist(), dq=dq.tolist(), gyro=[0, 0, .04],
                    quaternion=[1, 0, 0, 0], last_action=previous.tolist(), command=[.15, 0, 0]),
            proprio=proprio.tolist(), wrench_prediction=wrench.tolist(), command=[.15, 0, 0],
            encoder=dict(input=history.tolist(), output=encoder.tolist()),
            policy=dict(input=np.r_[encoder, proprio, (wrench*[.01,.01,.01,.05,.05,.05]).reshape(-1), [.15,0,0]].tolist(),
                        output=action.tolist()),
            published_targets=(np.array(contract["default_q"])+np.r_[action*.25, np.zeros(8)]).tolist(), publish_stamp=100+i*.02)
        rows.append(row)
        previous = action
    return rows, contract


class AuditTests(unittest.TestCase):
    def test_session_tap_runs_once_and_preserves_output_without_aliasing(self):
        class Session:
            calls = 0
            def run(self, names, inputs):
                self.calls += 1
                return [inputs["x"]*2]
        recorder = module.PolicyIORecorder("unused", {})
        recorder.begin({}, np.zeros(42), SimpleNamespace(wrench_prediction=np.zeros((5, 6)),
            base_command=np.zeros(3), arm_position=np.zeros(6)))
        session = Session()
        tap = module.SessionTap(session, recorder, "policy")
        vector = np.array([1., 2.])
        result = tap.run(None, {"x": vector})
        vector[:] = 9
        np.testing.assert_array_equal(result[0], [2, 4])
        np.testing.assert_array_equal(recorder.pending["policy"]["input"], [1, 2])
        recorder.publish(np.ones(18))
        self.assertIsNone(recorder.pending)
        self.assertEqual(session.calls, 1)

    def test_detects_zero_yaw_violation_history_reversal_and_mapping_error(self):
        rows, contract = fixture()
        self.assertTrue(module.audit_rows(rows, contract)[0]["contract_pass"])
        bad = copy.deepcopy(rows)
        bad[1]["policy"]["input"][77] = -.03
        self.assertFalse(module.audit_rows(bad, contract)[0]["checks"]["command"]["pass_check"])
        bad = copy.deepcopy(rows)
        bad[2]["encoder"]["input"] = np.asarray(bad[2]["encoder"]["input"]).reshape(10, 42)[::-1].reshape(-1).tolist()
        self.assertFalse(module.audit_rows(bad, contract)[0]["checks"]["history"]["pass_check"])
        bad = copy.deepcopy(rows)
        bad[3]["published_targets"][0:2] = bad[3]["published_targets"][1::-1]
        self.assertFalse(module.audit_rows(bad, contract)[0]["checks"]["published_leg_targets"]["pass_check"])

    def test_native_joint_name_reorder_and_quaternion_sign(self):
        rows, contract = fixture()
        names = [f"joint{i}" for i in range(18)]
        clock = dict(host="test", wall_minus_monotonic_s=123.)
        contract.update(clock=clock, wire_joint_names=names)
        native = [dict(monotonic_s=r["source"]["stamp"], segment=0,
            native_io=dict(q=r["source"]["q"][::-1], dq=r["source"]["dq"][::-1],
                imu_quaternion_wxyz=[-1, 0, 0, 0], imu_gyro=r["source"]["gyro"],
                base_quaternion_wxyz=[1, 0, 0, 0], base_gyro_body=r["source"]["gyro"],
                received_q=r["published_targets"][::-1])) for r in rows]
        report = module.native_comparison(rows, native, contract, dict(clock=clock, wire_joint_names=names[::-1]))
        for value in report["differences"].values():
            self.assertEqual(value["max_abs"], 0.)

    def test_pytorch_replay_uses_recorded_input_and_detects_modified_output(self):
        import torch
        spec = importlib.util.spec_from_file_location("audit_core_test",
            ROOT/"exts/bipedal_locomotion/bipedal_locomotion/controllers/paint_teacher_core.py")
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
        actor = core.mlp([78, 512, 512, 256, 128, 10])
        actor.add_module("9", core.CheckpointScaledTanh(5.))
        encoder = core.mlp([420, 256, 128, 3])
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/"model.pt"
            torch.save(dict(model_state_dict={"actor."+k:v for k,v in actor.state_dict().items()},
                            encoder_state_dict={"encoder."+k:v for k,v in encoder.state_dict().items()}), checkpoint)
            row = {}
            with torch.no_grad():
                for name, net, width in (("encoder", encoder, 420), ("policy", actor, 78)):
                    x = torch.randn(width)
                    row[name] = dict(input=x.tolist(), output=net(x).tolist())
            self.assertTrue(all(x["pass_check"] for x in module.replay([row], checkpoint).values()))
            row["policy"]["output"][0] += .01
            self.assertFalse(module.replay([row], checkpoint)["policy"]["pass_check"])


if __name__ == "__main__":
    unittest.main()
