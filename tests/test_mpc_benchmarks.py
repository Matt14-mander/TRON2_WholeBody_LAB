import importlib.util
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts/ocs2" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


prepare = module("prepare_mpc_benchmarks")
report = module("summarize_mpc_benchmarks")


class MpcBenchmarksTest(unittest.TestCase):
    def test_variants_preserve_constraints_and_horizon(self):
        source = (REPO / "ocs2_ws/src/tron2_ocs2/config/task_arm_live.info").read_text()
        variants = prepare.configurations(source)
        for name, value in variants.items():
            self.assertIn("activate        true", value)  # self collision
            self.assertIn("timeHorizon          1.0", value)
            self.assertIn("maxNumIterations              2", value)
            self.assertIn("checkNumericalStability       true", value)
            self.assertIn("vx -0.29", value)
        self.assertIn("algorithm                     ILQR", variants["analytic_ilqr"])
        self.assertIn("integratorType           RK4", variants["analytic_ilqr_rk4"])

    def test_ambiguous_setting_refused(self):
        with self.assertRaises(ValueError):
            prepare.replace_setting("nThreads 1\nnThreads 3\n", "nThreads", 2)

    def test_extracts_body_twist_and_relative_arrival(self):
        frame = {"control_sim_time": 3.0, "target_deadline_sim_time": 3.4,
                 "target_position_world": [0, 1, 2], "target_quaternion_wxyz": [1, 0, 0, 0],
                 "state": {"base_position_world": [0, 0, 0.74], "base_quaternion_world": [1, 0, 0, 0],
                           "base_linear_velocity_body": [-0.2, -0.1, 0], "base_angular_velocity_body": [0, 0, 0.1],
                           "arm_position_actual": [0] * 6, "arm_velocity_actual": [0] * 6}}
        observation = prepare.observation_from_frame(frame)
        self.assertEqual(observation["base_twist_body"], [-0.2, -0.1, 0, 0, 0, 0.1])
        self.assertAlmostEqual(observation["end_effector_arrival_time"], 0.4)

    def test_cold_slow_request_is_separate(self):
        rows = [{"cold": True, "solve_duration_s": 15}, {"cold": False, "solve_duration_s": 0.07}]
        result = report.summarize(rows, 2)
        self.assertEqual(result["warm_p95_ms"], 70)
        self.assertEqual(result["cold_durations_ms"], [15000])
        self.assertTrue(result["latency_gate_passed"])

    def test_failures_missing_and_resets_cannot_pass_gate(self):
        for rows in ([{"cold": True, "solve_duration_s": 0.01}],
                     [{"cold": True, "solve_duration_s": 0.01}, {"error": "failed"}],
                     [{"cold": True, "solve_duration_s": 0.01}] * 2,
                     [{"cold": True, "solve_duration_s": 0.01}, {"cold": False, "solve_duration_s": -0.01}],
                     [{"cold": True, "solve_duration_s": 0.01}, {"cold": False, "solve_duration_s": 0.101}]):
            self.assertFalse(report.summarize(rows, 2)["latency_gate_passed"])


if __name__ == "__main__":
    unittest.main()
