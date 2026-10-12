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
    def test_collision_gate_requires_all_checks_and_initial_clearance(self):
        rows = [{"cold": True, "solve_duration_s": 0.03, "collision_checked": True,
                 "initial_collision_margin": -0.01, "minimum_collision_margin": -0.01},
                {"cold": False, "solve_duration_s": 0.03, "collision_checked": True,
                 "initial_collision_margin": 0.02, "minimum_collision_margin": 0.01}]
        result = report.summarize(rows, 2)
        self.assertTrue(result["latency_gate_passed"])
        self.assertFalse(result["collision_gate_passed"])
        self.assertTrue(result["worst_collision"]["cold"])
        rows[0]["minimum_collision_margin"] = 0.0
        rows[0]["initial_collision_margin"] = 0.0
        self.assertTrue(report.summarize(rows, 2)["collision_gate_passed"])
        rows[0]["collision_checked"] = False
        self.assertIsNone(report.summarize(rows, 2)["collision_gate_passed"])
        self.assertIsNone(report.summarize(rows[1:], 2)["collision_gate_passed"])

    def test_variants_preserve_constraints_and_horizon(self):
        source = (REPO / "ocs2_ws/src/tron2_ocs2/config/task_arm_live.info").read_text()
        variants = prepare.configurations(source)
        for name, value in variants.items():
            self.assertIn("activate        true", value)  # self collision
            self.assertIn("timeHorizon          1.0", value)
            iterations = 4 if name == "analytic_ilqr_4iter" else 8 if name.startswith("analytic_ilqr_8iter") else 2
            self.assertIn(f"maxNumIterations              {iterations}", value)
            self.assertIn("checkNumericalStability       true", value)
            self.assertIn("vx -0.29", value)
        self.assertIn("algorithm                     ILQR", variants["analytic_ilqr"])
        for count in (4, 8):
            self.assertEqual(
                prepare.replace_setting(variants[f"analytic_ilqr_{count}iter"], "maxNumIterations", 2),
                variants["analytic_ilqr"])
        self.assertIn("integratorType           RK4", variants["analytic_ilqr_rk4"])

    def test_penalty_comparison_changes_only_strength(self):
        source = (REPO / "ocs2_ws/src/tron2_ocs2/config/task_arm_live.info").read_text()
        variants = prepare.configurations(source)
        for suffix, strength in (("mu10", "1e-2"), ("mu100", "1e-1")):
            candidate = variants[f"analytic_ilqr_8iter_{suffix}"]
            self.assertIn(f"boxConstraintMu {strength}", candidate)
            self.assertEqual(prepare.replace_setting(candidate, "boxConstraintMu", "1e-3"),
                             variants["analytic_ilqr_8iter"])
        self.assertEqual(prepare.replace_setting(variants["analytic_ilqr_8iter_mu100_acc19"],
                                                "armAcceleration", "20.0"),
                         variants["analytic_ilqr_8iter_mu100"])

    def test_selects_actual_in_bound_state_without_clipping(self):
        lower, upper = prepare.arm_position_bounds()
        outside = {"event": "control_response", "step": 100,
                   "state": {"arm_position_actual": [0.0, 0.04, 0.000022, 0.0, 0.0, 0.0]}}
        inside = {"event": "control_response", "step": 101,
                  "state": {"arm_position_actual": [0.0, 0.04, -0.001, 0.0, 0.0, 0.0]}}
        selected, margin, skipped = prepare.select_frame([outside, inside], 100, lower, upper, 1e-6)
        self.assertIs(selected, inside)
        self.assertAlmostEqual(margin, 0.001)
        self.assertEqual(skipped, 1)
        self.assertEqual(outside["state"]["arm_position_actual"][2], 0.000022)
        with self.assertRaises(ValueError):
            prepare.select_frame([outside], 100, lower, upper, 1e-6)
        selected, margin, skipped = prepare.select_frame([outside, inside], 100, lower, upper)
        self.assertIs(selected, outside)  # Default retains the original comparison.

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
        self.assertIsNone(result["joint_input_gate_passed"])

    def test_worst_position_keeps_cold_and_warm_provenance(self):
        rows = [{"sample": 0, "cold": True, "solve_duration_s": 0.02,
                 "minimum_arm_position_margin": -0.27, "worst_arm_position_index": 3,
                 "initial_arm_position_margin": -0.00002, "worst_arm_position_time": 2.8},
                {"sample": 1, "cold": False, "solve_duration_s": 0.02,
                 "minimum_arm_position_margin": 0.02},
                {"sample": 2, "cold": False, "solve_duration_s": 0.02,
                 "minimum_arm_position_margin": -0.1, "worst_arm_position_index": 2}]
        result = report.summarize(rows, 3)
        self.assertEqual(result["cold_worst_position"]["sample"], 0)
        self.assertEqual(result["warm_worst_position"]["sample"], 2)
        self.assertEqual(result["warm_position_violation_records"], 1)
        self.assertEqual(result["cold_worst_position"]["worst_arm_position_time"], 2.8)
        self.assertAlmostEqual(result["warm_minimum_arm_position_margin"], -0.1)

    def test_current_ee_reference_preserves_measured_state(self):
        frame = {"control_sim_time": 2.0, "target_deadline_sim_time": 0.0,
                 "target_position_world": [9, 8, 7], "target_quaternion_wxyz": [0, 1, 0, 0],
                 "state": {"base_position_world": [0, 0, 0.74], "base_quaternion_world": [1, 0, 0, 0],
                           "base_linear_velocity_body": [-0.2, 0, 0], "base_angular_velocity_body": [0, 0, 0],
                           "arm_position_actual": [0.1] * 6, "arm_velocity_actual": [0.2] * 6,
                           "end_effector_position_world": [0.2, 0.3, 1.2],
                           "end_effector_quaternion_world": [1, 0, 0, 0]}}
        recorded = prepare.observation_from_frame(frame)
        current = prepare.observation_from_frame(frame, "current_ee")
        self.assertEqual(current["end_effector_target_position_world"], [0.2, 0.3, 1.2])
        self.assertEqual(current["end_effector_target_quaternion_world"], [1, 0, 0, 0])
        self.assertEqual(current["end_effector_arrival_time"], 1.0)
        self.assertEqual(recorded["end_effector_arrival_time"], 0.0)
        for key in ("arm_position", "arm_velocity", "base_twist_body", "time"):
            self.assertEqual(current[key], recorded[key])
        del frame["state"]["end_effector_quaternion_world"]
        with self.assertRaises(KeyError):
            prepare.observation_from_frame(frame, "current_ee")

    def test_fast_unsafe_plan_fails_separate_gate_including_cold(self):
        safe = {"solve_duration_s": 0.027, "minimum_arm_position_margin": 0.1,
                "maximum_arm_speed": 4.0, "maximum_arm_acceleration": 18.0,
                "base_velocity_command": [-0.29, -0.08, 0.0]}
        rows = [dict(safe, cold=True), dict(safe, cold=False)]
        self.assertTrue(report.summarize(rows, 2)["joint_input_gate_passed"])
        for key, value in (("minimum_arm_position_margin", -0.69),
                           ("maximum_arm_speed", 5.44), ("maximum_arm_acceleration", 31.03),
                           ("maximum_arm_acceleration", float("nan"))):
            rows[0] = dict(safe, cold=True, **{key: value})
            result = report.summarize(rows, 2)
            self.assertTrue(result["latency_gate_passed"])
            self.assertFalse(result["joint_input_gate_passed"])
            self.assertIn(key, result["joint_input_gate_failures"])

    def test_failures_missing_and_resets_cannot_pass_gate(self):
        for rows in ([{"cold": True, "solve_duration_s": 0.01}],
                     [{"cold": True, "solve_duration_s": 0.01}, {"error": "failed"}],
                     [{"cold": True, "solve_duration_s": 0.01}] * 2,
                     [{"cold": True, "solve_duration_s": 0.01}, {"cold": False, "solve_duration_s": -0.01}],
                     [{"cold": True, "solve_duration_s": 0.01}, {"cold": False, "solve_duration_s": 0.101}]):
            self.assertFalse(report.summarize(rows, 2)["latency_gate_passed"])


if __name__ == "__main__":
    unittest.main()
