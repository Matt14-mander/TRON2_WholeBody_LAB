"""Analyze matched external-wrench rollouts against zero-wrench controls."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


SIGNALS = (
    "base_linear_velocity_body",
    "base_angular_velocity_body",
    "projected_gravity_body",
)


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def rms_difference(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    difference = np.asarray(left)[mask] - np.asarray(right)[mask]
    return float(np.sqrt(np.mean(np.square(difference))))


def aligned_episode_pair(
    root: Path, treatment_row: dict[str, str], control_row: dict[str, str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    treatment_path = root / treatment_row["episode_file"]
    control_path = root / control_row["episode_file"]
    with np.load(treatment_path, allow_pickle=False) as data:
        treatment = {key: data[key].copy() for key in (*SIGNALS, "episode_time", "external_wrench_active")}
    with np.load(control_path, allow_pickle=False) as data:
        control = {key: data[key].copy() for key in (*SIGNALS, "episode_time")}
    count = min(len(treatment["episode_time"]), len(control["episode_time"]))
    return treatment, control, count


def compare_pair(
    root: Path, treatment_row: dict[str, str], control_row: dict[str, str]
) -> dict[str, float | int | str | bool]:
    treatment, control, count = aligned_episode_pair(root, treatment_row, control_row)
    active = treatment["external_wrench_active"][:count].astype(bool)
    times = treatment["episode_time"][:count]
    if np.any(active):
        active_end = float(times[np.flatnonzero(active)[-1]])
        recovery = (times > active_end) & (times <= active_end + 1.0)
    else:
        recovery = np.zeros(count, dtype=bool)
    result: dict[str, float | int | str | bool] = {
        "trajectory_id": treatment_row["trajectory_id"],
        "source_trajectory_id": treatment_row["source_trajectory_id"],
        "axis": treatment_row["external_wrench_axis"],
        "sign": int(treatment_row["external_wrench_sign"]),
        "profile": treatment_row["external_wrench_profile"],
        "amplitude": float(treatment_row["external_wrench_amplitude"]),
        "treatment_accepted": as_bool(treatment_row["accepted"]),
        "control_accepted": as_bool(control_row["accepted"]),
        "incremental_fall": (
            as_bool(control_row["accepted"]) and not as_bool(treatment_row["accepted"])
        ),
        "max_tilt_delta_deg": (
            float(treatment_row["max_base_tilt_deg"])
            - float(control_row["max_base_tilt_deg"])
        ),
    }
    for signal in SIGNALS:
        result[f"active_{signal}_rmse"] = rms_difference(
            treatment[signal][:count], control[signal][:count], active
        )
        result[f"recovery_{signal}_rmse"] = rms_difference(
            treatment[signal][:count], control[signal][:count], recovery
        )
    return result


def zero_repeat_noise(root: Path, rows: list[dict[str, str]]) -> dict[str, float]:
    """Estimate rollout variability using zero-wrench step/ramp/sine repeats."""
    by_source = defaultdict(list)
    for row in rows:
        if row["external_wrench_axis"] == "zero" and as_bool(row["accepted"]):
            by_source[row["source_trajectory_id"]].append(row)
    values = defaultdict(list)
    for source_rows in by_source.values():
        source_rows.sort(key=lambda row: row["external_wrench_profile"])
        for index, left_row in enumerate(source_rows):
            for right_row in source_rows[index + 1:]:
                left, right, count = aligned_episode_pair(root, left_row, right_row)
                times = left["episode_time"][:count]
                active_window = (times >= 1.0) & (times < 1.8)
                recovery_window = (times >= 1.8) & (times < 2.8)
                for signal in SIGNALS:
                    values[f"active_{signal}_rmse"].append(rms_difference(
                        left[signal][:count], right[signal][:count], active_window
                    ))
                    values[f"recovery_{signal}_rmse"].append(rms_difference(
                        left[signal][:count], right[signal][:count], recovery_window
                    ))
    return {
        key: float(np.nanmedian(samples)) for key, samples in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir")
    parser.add_argument(
        "--output_csv",
        help="Optional path for per-pair metrics; defaults to paired_analysis.csv in the dataset.",
    )
    args = parser.parse_args()

    root = Path(args.dataset_dir).expanduser().resolve()
    manifest_path = root / "rollout_manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "trajectory_id", "source_trajectory_id", "external_wrench_axis",
        "external_wrench_sign", "external_wrench_profile", "external_wrench_amplitude", "accepted",
        "max_base_tilt_deg", "episode_file",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Not a paired schema-2 rollout manifest: {manifest_path}")

    controls = {
        (row["source_trajectory_id"], row["external_wrench_profile"]): row
        for row in rows if row["external_wrench_axis"] == "zero"
    }
    treatments = [row for row in rows if row["external_wrench_axis"] != "zero"]
    metrics = []
    for row in treatments:
        key = (row["source_trajectory_id"], row["external_wrench_profile"])
        if key not in controls:
            raise ValueError(f"Missing matched zero-wrench control for {key}")
        metrics.append(compare_pair(root, row, controls[key]))

    output = Path(args.output_csv).expanduser().resolve() if args.output_csv else root / "paired_analysis.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    noise = zero_repeat_noise(root, rows)
    control_failures = sum(not as_bool(row["accepted"]) for row in controls.values())
    print(
        f"episodes={len(rows)} controls={len(controls)} control_failures={control_failures} "
        f"treatment_pairs={len(metrics)}"
    )
    print(f"incremental_falls={sum(bool(item['incremental_fall']) for item in metrics)}")
    grouped = defaultdict(list)
    for item in metrics:
        grouped[(
            str(item["axis"]), int(item["sign"]), str(item["profile"]),
            float(item["amplitude"]),
        )].append(item)
    for condition, samples in sorted(grouped.items()):
        stable_samples = [
            item for item in samples
            if bool(item["control_accepted"]) and bool(item["treatment_accepted"])
        ]
        tilt = np.asarray([
            float(item["max_tilt_delta_deg"]) for item in stable_samples
        ])
        active_linear = np.asarray([
            float(item["active_base_linear_velocity_body_rmse"]) for item in stable_samples
        ])
        falls = sum(bool(item["incremental_fall"]) for item in samples)
        if stable_samples:
            detail = (
                f"tilt_delta_mean={np.nanmean(tilt):.3f}deg "
                f"active_linear_velocity_rmse={np.nanmean(active_linear):.4f}m/s"
            )
        else:
            detail = "metrics=unavailable(control_or_treatment_failed)"
        print(
            f"condition={condition} n={len(samples)} stable_pairs={len(stable_samples)} "
            f"incremental_falls={falls} {detail}"
        )
    stable_metrics = [
        item for item in metrics
        if bool(item["control_accepted"]) and bool(item["treatment_accepted"])
    ]
    if not noise:
        print("zero_repeat_noise_median=unavailable(fewer_than_two_accepted_controls_per_source)")
    else:
        print("zero_repeat_noise_median:")
    for key, value in sorted(noise.items()):
        treatment_values = np.asarray([float(item[key]) for item in stable_metrics])
        treatment_median = float(np.nanmedian(treatment_values))
        ratio = treatment_median / value if value > 1e-12 else float("inf")
        print(
            f"  {key}: noise={value:.6f} treatment={treatment_median:.6f} "
            f"signal_to_repeat_ratio={ratio:.2f}"
        )
    print(f"per_pair_csv={output}")


if __name__ == "__main__":
    main()
