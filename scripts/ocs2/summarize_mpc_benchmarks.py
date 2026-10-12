"""Summarize native/TCP benchmark JSONL, preserving failures and cold starts."""
import argparse
import json
from pathlib import Path

import numpy as np


def joint_input_gate(rows, expected_samples):
    """Check every solve, including cold; no collision/tracking claim is made.

    Limits match task_arm_live.info. Missing metrics mean unknown, not passed.
    """
    failures = set()
    missing = False
    if len(rows) != expected_samples:
        failures.add("record_count")
    for row in rows:
        if "error" in row:
            failures.add("solve_error")
        if not np.isfinite(row.get("solve_duration_s", float("nan"))) or row.get("solve_duration_s", -1) < 0:
            failures.add("invalid_solve_duration")
        for key, limit, lower in (("minimum_arm_position_margin", 0.0, True),
                                  ("maximum_arm_speed", 5.0, False),
                                  ("maximum_arm_acceleration", 20.0, False)):
            if key not in row:
                missing = True
                continue
            value = row[key]
            if not np.isfinite(value) or (value < limit - 1e-6 if lower else value > limit + 1e-6):
                failures.add(key)
        command = row.get("base_velocity_command")
        if command is None:
            missing = True
        elif (np.asarray(command).shape != (3,) or not np.all(np.isfinite(command)) or
              np.max(np.abs(np.asarray(command) - [-0.29, -0.08, 0.0])) > 1e-4):
            failures.add("fixed_base_command")
    return {"joint_input_gate_passed": False if failures else (None if missing else True),
            "joint_input_gate_failures": sorted(failures),
            "joint_input_metrics_missing": missing,
            "joint_input_limits": {"arm_speed": 5.0, "arm_acceleration": 20.0,
                                   "position_tolerance_rad": 1e-6}}


def summarize(rows, expected_samples=20, period=0.1):
    successful = [r for r in rows if "error" not in r and
                  np.isfinite(r.get("solve_duration_s", float("nan"))) and
                  r["solve_duration_s"] >= 0.0]
    warm = [r for r in successful if r.get("cold") is False]
    durations = [r["solve_duration_s"] for r in warm]
    result = {"records": len(rows), "successful": len(successful),
              "errors": len(rows) - len(successful),
              "missing_records": max(0, expected_samples - len(rows)),
              "cold_records": sum(bool(r.get("cold")) for r in successful),
              "cold_durations_ms": [r["solve_duration_s"] * 1000 for r in successful if r.get("cold")],
              "warm_records": len(warm), "period_s": period}
    result.update(joint_input_gate(rows, expected_samples))
    collision_rows = [r for r in successful if r.get("collision_checked") is True and
                      np.isfinite(r.get("minimum_collision_margin", float("nan")))]
    # Exact knot distances for configured pairs/geometry, not a whole-robot or
    # continuous-trajectory certificate. Missing/disabled checks never pass.
    result["collision_gate_passed"] = (None if len(collision_rows) != expected_samples or
                                      len(rows) != expected_samples else
                                      all(r["minimum_collision_margin"] >= 0.0 for r in collision_rows))
    if collision_rows:
        worst = min(collision_rows, key=lambda r: r["minimum_collision_margin"])
        result["minimum_collision_margin"] = worst["minimum_collision_margin"]
        result["collision_violation_records"] = sum(r["minimum_collision_margin"] < 0 for r in collision_rows)
        result["worst_collision"] = {key: worst[key] for key in (
            "sample", "cold", "observation_sim_time", "initial_collision_margin",
            "minimum_collision_margin", "worst_collision_time", "worst_collision_pair_index") if key in worst}
        result["collision_validation_ms_max"] = max(r.get("collision_validation_ms", 0.0) for r in collision_rows)
    if not durations:
        result["latency_gate_passed"] = False
        return result
    result.update(warm_p50_ms=float(np.percentile(durations, 50) * 1000),
                  warm_p95_ms=float(np.percentile(durations, 95) * 1000),
                  warm_max_ms=float(max(durations) * 1000),
                  warm_deadline_miss_fraction=float(np.mean(np.asarray(durations) > period)))
    # A P95 alone can hide repeated deadline misses; require all measured warm
    # requests below deadline, and all expected records, for this initial gate.
    result["latency_gate_passed"] = (len(rows) == expected_samples and
        len(successful) == expected_samples and len(warm) == expected_samples - 1 and
        max(durations) <= period)
    for key in ("cost", "equality_sse", "iterations", "knots", "optimization_ms",
                "maximum_arm_speed", "maximum_arm_acceleration"):
        values = [r[key] for r in warm if key in r]
        if values:
            result[key + "_median"] = float(np.median(values))
            result[key + "_max"] = float(max(values))
    margins = [r["minimum_arm_position_margin"] for r in successful if "minimum_arm_position_margin" in r]
    if margins:
        result["minimum_arm_position_margin"] = float(min(margins))
    for name, group in (("cold", [r for r in successful if r.get("cold") is True]), ("warm", warm)):
        available = [r for r in group if "minimum_arm_position_margin" in r]
        if available:
            worst = min(available, key=lambda r: r["minimum_arm_position_margin"])
            result[name + "_minimum_arm_position_margin"] = worst["minimum_arm_position_margin"]
            result[name + "_position_violation_records"] = sum(
                r["minimum_arm_position_margin"] < -1e-6 for r in available)
            result[name + "_worst_position"] = {key: worst[key] for key in (
                "sample", "observation_sim_time", "minimum_arm_position_margin", "initial_arm_position_margin",
                "worst_arm_position_index", "worst_arm_position_time", "worst_arm_position_value",
                "arm_lower_position_limits", "arm_upper_position_limits") if key in worst}
    commands = [r["base_velocity_command"] for r in successful if "base_velocity_command" in r]
    if commands:
        result["max_fixed_base_command_error"] = float(np.max(np.abs(
            np.asarray(commands) - np.asarray([-0.29, -0.08, 0.0]))))
    return result


def read_records(path):
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Expected a JSON object")
            records.append(record)
        except (ValueError, json.JSONDecodeError):
            # ros2 or a failed executable may write non-JSON to stdout. Count it
            # as an error instead of hiding it or losing the other comparisons.
            records.append({"error": f"Invalid JSON benchmark record at line {number}"})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--expected-samples", type=int, default=20)
    parser.add_argument("--period", type=float, default=0.1)
    args = parser.parse_args()
    if args.expected_samples < 2 or not np.isfinite(args.period) or args.period <= 0:
        parser.error("Require at least two samples and a finite positive period.")
    for path in args.paths:
        rows = read_records(path)
        print(json.dumps({"file": str(path), **summarize(rows, args.expected_samples, args.period)}, ensure_ascii=False))
    print("Latency, joint/input and collision gates are separate. Joint/input limits match task_arm_live.info; "
          "collision checks cover configured pairs at trajectory knots only. Timed live tracking/stability remain unverified.")


if __name__ == "__main__":
    main()
