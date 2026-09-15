#!/usr/bin/env python3
"""Generate a reproducible batch of offline TRON2 OCS2 expert trajectories."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random
import subprocess
import time


MANIFEST_FIELDS = (
    "trajectory_id",
    "seed",
    "target_x",
    "target_y",
    "target_z",
    "target_qw",
    "target_qx",
    "target_qy",
    "target_qz",
    "arrival_time",
    "sample_period",
    "samples",
    "duration",
    "max_arm_velocity",
    "max_arm_effort",
    "max_base_command",
    "max_wrench",
    "solve_wall_time",
    "trajectory_file",
)

FAILURE_FIELDS = (
    "trajectory_id",
    "seed",
    "target_x",
    "target_y",
    "target_z",
    "arrival_time",
    "error",
)


@dataclass(frozen=True)
class Scenario:
    index: int
    target_x: float
    target_y: float
    target_z: float
    arrival_time: float

    @property
    def trajectory_id(self) -> str:
        return f"trajectory_{self.index:06d}"


def finite_pair(values: list[float], name: str) -> tuple[float, float]:
    if len(values) != 2 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain two finite values.")
    lower, upper = values
    if lower > upper:
        raise ValueError(f"{name} lower bound must not exceed its upper bound.")
    return lower, upper


def generate_scenarios(
    count: int,
    seed: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    arrival_times: tuple[float, ...],
) -> list[Scenario]:
    if count <= 0:
        raise ValueError("count must be positive.")
    if not arrival_times or not all(math.isfinite(value) and value > 0.0 for value in arrival_times):
        raise ValueError("arrival times must be finite and positive.")
    rng = random.Random(seed)
    return [
        Scenario(
            index=index,
            target_x=rng.uniform(*x_range),
            target_y=rng.uniform(*y_range),
            target_z=rng.uniform(*z_range),
            arrival_time=arrival_times[index % len(arrival_times)],
        )
        for index in range(count)
    ]


def trajectory_metrics(path: Path) -> dict[str, float | int]:
    samples = 0
    duration = 0.0
    maxima = {"max_arm_velocity": 0.0, "max_arm_effort": 0.0,
              "max_base_command": 0.0, "max_wrench": 0.0}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) != 52:
            raise ValueError(f"Expected 52 trajectory columns, got {len(header)}.")
        for row in reader:
            if len(row) != 52:
                raise ValueError(f"Expected 52 values, got {len(row)}.")
            values = [float(value) for value in row]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Trajectory contains NaN or Inf.")
            samples += 1
            duration = values[0]
            maxima["max_arm_velocity"] = max(maxima["max_arm_velocity"], *(abs(v) for v in values[7:13]))
            maxima["max_arm_effort"] = max(maxima["max_arm_effort"], *(abs(v) for v in values[13:19]))
            maxima["max_base_command"] = max(maxima["max_base_command"], *(abs(v) for v in values[19:22]))
            maxima["max_wrench"] = max(maxima["max_wrench"], *(abs(v) for v in values[22:52]))
    if samples < 2:
        raise ValueError("Trajectory contains fewer than two samples.")
    return {"samples": samples, "duration": duration, **maxima}


def append_row(path: Path, fields: tuple[str, ...], row: dict[str, object]) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def existing_ids(manifest: Path) -> set[str]:
    if not manifest.is_file():
        return set()
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        return {row["trajectory_id"] for row in csv.DictReader(stream)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-info", required=True, type=Path)
    parser.add_argument("--robot-urdf", required=True, type=Path)
    parser.add_argument("--library-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--x-range", type=float, nargs=2, default=(0.15, 0.30))
    parser.add_argument("--y-range", type=float, nargs=2, default=(-0.15, 0.15))
    parser.add_argument("--z-range", type=float, nargs=2, default=(0.75, 0.95))
    parser.add_argument("--arrival-times", type=float, nargs="+", default=(0.5, 0.75, 1.0))
    parser.add_argument("--sample-period", type=float, default=0.02)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path, label in ((args.task_info, "task info"), (args.robot_urdf, "robot URDF")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not math.isfinite(args.sample_period) or args.sample_period <= 0.0:
        raise ValueError("sample period must be finite and positive.")
    x_range = finite_pair(args.x_range, "x range")
    y_range = finite_pair(args.y_range, "y range")
    z_range = finite_pair(args.z_range, "z range")
    arrival_times = tuple(args.arrival_times)
    scenarios = generate_scenarios(
        args.count, args.seed, x_range, y_range, z_range, arrival_times
    )

    trajectories_dir = args.output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    args.library_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.csv"
    failures = args.output_dir / "failures.csv"
    completed = existing_ids(manifest) if args.resume else set()
    if not args.resume and (manifest.exists() or failures.exists()):
        raise FileExistsError(
            f"Dataset metadata already exists in {args.output_dir}; use --resume or a new output directory."
        )

    success_count = len(completed)
    failure_count = 0
    for scenario in scenarios:
        if scenario.trajectory_id in completed:
            continue
        output = trajectories_dir / f"{scenario.trajectory_id}.csv"
        command = [
            "ros2", "run", "tron2_ocs2", "tron2_ocs2_trajectory_export",
            str(args.task_info), str(args.robot_urdf), str(args.library_dir), str(output),
            str(scenario.target_x), str(scenario.target_y), str(scenario.target_z),
            "1.0", "0.0", "0.0", "0.0", str(args.sample_period),
            str(scenario.arrival_time),
        ]
        print(
            f"[{scenario.index + 1}/{len(scenarios)}] {scenario.trajectory_id}: "
            f"target=({scenario.target_x:.3f}, {scenario.target_y:.3f}, {scenario.target_z:.3f}), "
            f"arrival={scenario.arrival_time:.3f}s",
            flush=True,
        )
        started = time.monotonic()
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        wall_time = time.monotonic() - started
        if result.returncode == 0:
            try:
                metrics = trajectory_metrics(output)
                append_row(manifest, MANIFEST_FIELDS, {
                    "trajectory_id": scenario.trajectory_id,
                    "seed": args.seed,
                    "target_x": scenario.target_x,
                    "target_y": scenario.target_y,
                    "target_z": scenario.target_z,
                    "target_qw": 1.0,
                    "target_qx": 0.0,
                    "target_qy": 0.0,
                    "target_qz": 0.0,
                    "arrival_time": scenario.arrival_time,
                    "sample_period": args.sample_period,
                    **metrics,
                    "solve_wall_time": wall_time,
                    "trajectory_file": str(output.relative_to(args.output_dir)),
                })
                success_count += 1
                continue
            except (OSError, ValueError) as error:
                failure_message = str(error)
        else:
            failure_message = (result.stderr or result.stdout or "unknown exporter error").strip()
        output.unlink(missing_ok=True)
        append_row(failures, FAILURE_FIELDS, {
            "trajectory_id": scenario.trajectory_id,
            "seed": args.seed,
            "target_x": scenario.target_x,
            "target_y": scenario.target_y,
            "target_z": scenario.target_z,
            "arrival_time": scenario.arrival_time,
            "error": failure_message,
        })
        failure_count += 1
        print(f"[WARN] {scenario.trajectory_id} failed: {failure_message}", flush=True)

    print(f"Dataset complete: successes={success_count}, failures={failure_count}, root={args.output_dir}")
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
