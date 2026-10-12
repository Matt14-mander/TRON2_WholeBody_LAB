"""Prepare controlled solver comparisons and extract one measured observation.

This never starts Isaac or actuates a robot. Generated configs are benchmark
candidates; do not adopt them for live control on latency evidence alone.
"""
import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


REPO = Path(__file__).resolve().parents[2]


def replace_setting(text, key, value):
    pattern = rf"(?m)^(\s*{re.escape(key)}\s+)([^\s;]+)"
    result, count = re.subn(pattern, lambda match: match[1] + str(value), text)
    if count != 1:
        raise ValueError(f"Expected exactly one {key} setting; found {count}.")
    return result


def configurations(source):
    baseline = replace_setting(source, "analyticDynamics", "false")
    baseline = replace_setting(baseline, "consistentInitializer", "false")
    analytic = replace_setting(source, "analyticDynamics", "true")
    analytic_default = replace_setting(analytic, "consistentInitializer", "false")
    ilqr = replace_setting(analytic, "algorithm", "ILQR")
    eight_iterations = replace_setting(ilqr, "maxNumIterations", 8)
    # Compare the algorithm alone first. RK4 rollout then limits the number
    # of trajectory knots to a fixed grid, at a separately measurable change.
    rk4 = replace_setting(ilqr, "integratorType", "RK4")
    return {
        "ad_slq": baseline,
        "analytic_slq_default_init": analytic_default,
        "analytic_slq": analytic,
        "analytic_ilqr": ilqr,
        "analytic_ilqr_4iter": replace_setting(ilqr, "maxNumIterations", 4),
        "analytic_ilqr_8iter": eight_iterations,
        "analytic_ilqr_8iter_mu10": replace_setting(eight_iterations, "boxConstraintMu", "1e-2"),
        "analytic_ilqr_8iter_mu100": replace_setting(eight_iterations, "boxConstraintMu", "1e-1"),
        "analytic_ilqr_8iter_mu100_acc19": replace_setting(
            replace_setting(eight_iterations, "boxConstraintMu", "1e-1"), "armAcceleration", "19.0"),
        "analytic_ilqr_rk4": rk4,
        "analytic_ilqr_rk4_1thread": replace_setting(rk4, "nThreads", 1),
    }


def observation_from_frame(frame, target_mode="recorded"):
    state = frame["state"]
    if target_mode not in ("recorded", "current_ee"):
        raise ValueError(f"Unknown target mode: {target_mode}")
    simulation_time = float(frame["control_sim_time"])
    deadline = frame.get("target_deadline_sim_time")
    observation = {
        "time": simulation_time,
        "base_position_world": state["base_position_world"],
        "base_quaternion_world": state["base_quaternion_world"],
        "base_twist_body": state["base_linear_velocity_body"] + state["base_angular_velocity_body"],
        "arm_position": state["arm_position_actual"],
        "arm_velocity": state["arm_velocity_actual"],
        "end_effector_target_position_world": frame["target_position_world"],
        "end_effector_target_quaternion_world": frame["target_quaternion_wxyz"],
        "end_effector_arrival_time": 0.0 if deadline is None else max(0.0, float(deadline) - simulation_time),
    }
    if target_mode == "current_ee":
        # Keep measured q/dq unchanged; change only the reference for this
        # comparison. This remains a frozen-state computational benchmark.
        observation["end_effector_target_position_world"] = state["end_effector_position_world"]
        observation["end_effector_target_quaternion_world"] = state["end_effector_quaternion_world"]
        observation["end_effector_arrival_time"] = 1.0
    for key, value in observation.items():
        if not np.all(np.isfinite(np.asarray(value, dtype=float))):
            raise ValueError(f"Non-finite observation field: {key}")
    if observation["end_effector_arrival_time"] > 1.0 + 1e-9:
        raise ValueError("Captured arrival exceeds the benchmark's 1 s horizon.")
    return observation


def arm_position_bounds():
    root = ET.parse(REPO / "robot_description/tron2/SFYG_TRON2A/urdf/robot.urdf").getroot()
    limits = [root.find(f"joint[@name='arm{i}_Joint']/limit") for i in range(1, 7)]
    return (np.asarray([float(limit.attrib["lower"]) for limit in limits]),
            np.asarray([float(limit.attrib["upper"]) for limit in limits]))


def select_frame(frames, step, lower, upper, minimum_margin=None):
    skipped = 0
    for frame in frames:
        if frame.get("event") != "control_response" or frame.get("step", -1) < step:
            continue
        q = np.asarray(frame["state"]["arm_position_actual"], dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("Invalid measured arm position in diagnostic frame.")
        margin = float(min(np.min(q - lower), np.min(upper - q)))
        if minimum_margin is not None and margin < minimum_margin:
            skipped += 1
            continue
        return frame, margin, skipped
    raise ValueError(f"No qualifying control_response at/after step {step}; "
                     f"minimum position margin={minimum_margin}, skipped={skipped}. "
                     "Capture another diagnostic episode; measured states are not clipped.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="logs/ocs2_benchmark")
    parser.add_argument("--task", type=Path, default=REPO / "ocs2_ws/src/tron2_ocs2/config/task_arm_live.info")
    parser.add_argument("--diagnostic", type=Path, help="Existing live JSONL; optional for nominal-state benchmark.")
    parser.add_argument("--step", type=int, default=100, help="First control_response with step >= this value.")
    parser.add_argument("--target-mode", choices=("recorded", "current_ee"), default="recorded",
                        help="Recorded reference, or measured EE pose with a new 1 s arrival.")
    parser.add_argument("--minimum-arm-position-margin", type=float,
                        help="Select a measured frame inside URDF limits by this many radians; never clip q.")
    args = parser.parse_args()
    if args.target_mode == "current_ee" and not args.diagnostic:
        parser.error("current_ee requires --diagnostic with a measured EE pose.")
    if args.minimum_arm_position_margin is not None and (
            not args.diagnostic or not np.isfinite(args.minimum_arm_position_margin) or
            args.minimum_arm_position_margin < 0):
        parser.error("Minimum arm position margin requires --diagnostic and a finite non-negative value.")
    variants = configurations(args.task.read_text(encoding="utf-8"))
    observation = None
    if args.diagnostic:
        with args.diagnostic.open(encoding="utf-8") as stream:
            try:
                frame, initial_margin, skipped = select_frame(
                    (json.loads(line) for line in stream if line.strip()), args.step,
                    *arm_position_bounds(), args.minimum_arm_position_margin)
            except ValueError as error:
                parser.error(str(error))
            observation = observation_from_frame(frame, args.target_mode)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, config in variants.items():
        (output / f"{name}.info").write_text(config, encoding="utf-8")
    if observation is not None:
        (output / "observation.json").write_text(json.dumps(observation, indent=2) + "\n", encoding="utf-8")
        metadata = {"target_mode": args.target_mode, "diagnostic": str(args.diagnostic),
                    "episode": frame.get("episode"), "step": frame.get("step"),
                    "simulation_time": observation["time"],
                    "initial_arm_position_margin": initial_margin,
                    "required_arm_position_margin": args.minimum_arm_position_margin,
                    "skipped_position_frames": skipped,
                    "arrival_s": observation["end_effector_arrival_time"]}
        measured = frame["state"].get("end_effector_position_world")
        if measured is not None:
            metadata["recorded_target_distance_m"] = float(np.linalg.norm(
                np.asarray(frame["target_position_world"]) - np.asarray(measured)))
            metadata["benchmark_target_distance_m"] = float(np.linalg.norm(
                np.asarray(observation["end_effector_target_position_world"]) - np.asarray(measured)))
        (output / "observation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(variants)} configurations in {output.resolve()}")
    print("Use identical observation.json, sample count and period for every variant.")


if __name__ == "__main__":
    main()
