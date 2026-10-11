"""Summarize native/TCP benchmark JSONL, preserving failures and cold starts."""
import argparse
import json
from pathlib import Path

import numpy as np


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
    print("Latency gate is computational only. Review cost/constraints, then verify timed live tracking and stability.")


if __name__ == "__main__":
    main()
