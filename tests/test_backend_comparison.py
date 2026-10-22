"""Cross-backend command reference, heading statistics and C0 condition checks."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load_module("backend_compare", ROOT/"scripts/mujoco/compare_support_backends.py")
wrapper = load_module("backend_wrapper", ROOT/"scripts/mujoco/run_wholebody_v3.py")
support = load_module("backend_support", ROOT/"scripts/rsl_rl/validate_paint_support.py")


def write(path, value, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(map(json.dumps, value)) if lines else json.dumps(value), encoding="utf-8")


class BackendTests(unittest.TestCase):
    def test_continuous_command_reference_matches_isaac(self):
        for t in np.linspace(0, 8, 801):
            expected = support.targets(float(t), "forward", [.15, 0, 0], [0]*6)[0]
            np.testing.assert_allclose(np.array([.15, 0, 0])*wrapper.paint_command_scale(t), expected)

    def test_unwrap_heading_and_initial_heading_displacement(self):
        times = np.linspace(5, 29, 241)
        angle = np.radians(179+2*(times-5))
        matrices = np.array([[[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]] for a in angle])
        positions = np.zeros((len(times), 3))
        result = compare.pose_summary(times, matrices, positions)
        self.assertAlmostEqual(result["yaw_slope_deg_s"], 2.)
        self.assertAlmostEqual(result["yaw_net_deg"], 48.)
        with self.assertRaises(ValueError):
            compare.pose_summary([0, 0, 1], matrices[:3], positions[:3])

    def test_end_to_end_and_reject_one_sided_gravity_compensation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            isaac, native, model = root/"isaac", root/"native", root/"model"
            args = dict(arm_pose="nominal", payload_mode="attached", payload_mass=.5,
                        payload_dimensions=[.2, .12, .08], payload_offset=[.15, 0, 0], low_wrench="gravity")
            write(isaac/"contract.json", dict(args=args, nominal_arm=[0]*6, checkpoint_sha256="same"))
            ma = dict(args, arm_pose="configured", arm_gravity_compensation=False,
                      command_profile="paint", command=[.15, 0, 0], model_root=str(model))
            clock = dict(host="test", wall_minus_monotonic_s=0.)
            write(native/"control/contract.json", dict(arguments=ma, clock=clock))
            write(native/"sim/task_contract.json", dict(attached=True, error=None,
                arguments=dict(external_wrench=None), clock=clock))
            write(model/"SFYG_TRON2A/v3_source.json", dict(checkpoint_sha256="same"))
            ir, mr, ar = [], [], []
            for t in np.linspace(0, 30, 301):
                angle = np.radians(-2*t)
                matrix = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
                ir.append(dict(time_s=t, base_quaternion_world_wxyz=[np.cos(angle/2), 0, 0, np.sin(angle/2)],
                    base_position_world=[.15*t, 0, 0], base_velocity=[.15, 0, 0], base_command=[.15, 0, 0],
                    arm_q=[0]*6, arm_target=[0]*6, force_payload_local=[0]*3, low_wrench_physical=[0]*30))
                mr.append(dict(monotonic_s=100+t, sim_time_s=t, segment=0,
                    base_rotation_world=np.array(matrix).reshape(-1).tolist(), base_position_world_m=[.15*t, 0, 0]))
                ar.append(dict(monotonic_s=100+t, target=[0]*6, position_error=[0]*6))
            write(isaac/"forward_trial0.jsonl", ir, True)
            write(isaac/"episodes.jsonl", [dict(case="forward", trial=0, completed=True)], True)
            write(native/"sim/task_steps.jsonl", mr, True)
            write(native/"control/arm_steps.jsonl", ar, True)
            write(native/"control/arm_report.json", dict(outcome=dict(completed=True)))
            result = compare.compare(isaac, native, "forward", 0, root/"out")
            self.assertAlmostEqual(result["isaac"]["yaw_slope_deg_s"], -2.)
            self.assertAlmostEqual(result["mujoco"]["yaw_slope_deg_s"], -2.)
            ma["arm_gravity_compensation"] = True
            write(native/"control/contract.json", dict(arguments=ma, clock=clock))
            with self.assertRaisesRegex(ValueError, "conditions differ"):
                compare.compare(isaac, native, "forward", 0, root/"invalid")


if __name__ == "__main__":
    unittest.main()
