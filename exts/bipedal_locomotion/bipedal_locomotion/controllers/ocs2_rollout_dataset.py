"""Manifest, split, and storage utilities for OCS2 Isaac Lab rollouts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Ocs2TrajectorySpec:
    trajectory_id: str
    path: Path
    target_position: np.ndarray
    target_quaternion: np.ndarray
    arrival_time: float


def load_trajectory_manifest(path: str | Path) -> list[Ocs2TrajectorySpec]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"OCS2 trajectory manifest does not exist: {manifest_path}")
    required = {
        "trajectory_id", "trajectory_file", "target_x", "target_y", "target_z",
        "target_qw", "target_qx", "target_qy", "target_qz", "arrival_time",
    }
    specs = []
    seen = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Trajectory manifest is missing columns: {sorted(missing)}")
        for row in reader:
            trajectory_id = row["trajectory_id"].strip()
            if not trajectory_id or trajectory_id in seen:
                raise ValueError(f"Invalid or duplicate trajectory_id: {trajectory_id!r}")
            trajectory_path = Path(row["trajectory_file"]).expanduser()
            if not trajectory_path.is_absolute():
                trajectory_path = manifest_path.parent / trajectory_path
            trajectory_path = trajectory_path.resolve()
            if not trajectory_path.is_file():
                raise FileNotFoundError(f"Trajectory file does not exist: {trajectory_path}")
            target_position = np.asarray(
                [row["target_x"], row["target_y"], row["target_z"]], dtype=np.float64
            )
            target_quaternion = np.asarray(
                [row["target_qw"], row["target_qx"], row["target_qy"], row["target_qz"]],
                dtype=np.float64,
            )
            arrival_time = float(row["arrival_time"])
            if (
                not np.all(np.isfinite(target_position))
                or not np.all(np.isfinite(target_quaternion))
                or np.linalg.norm(target_quaternion) < 1e-9
                or not np.isfinite(arrival_time)
                or arrival_time <= 0.0
            ):
                raise ValueError(f"Invalid target metadata for {trajectory_id}.")
            specs.append(
                Ocs2TrajectorySpec(
                    trajectory_id=trajectory_id,
                    path=trajectory_path,
                    target_position=target_position,
                    target_quaternion=target_quaternion / np.linalg.norm(target_quaternion),
                    arrival_time=arrival_time,
                )
            )
            seen.add(trajectory_id)
    if not specs:
        raise ValueError("Trajectory manifest contains no trajectories.")
    return specs


def make_split_assignments(
    trajectory_ids: Sequence[str], seed: int = 42
) -> dict[str, str]:
    ids = list(trajectory_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Trajectory ids must be unique when creating splits.")
    random.Random(seed).shuffle(ids)
    train_end = int(0.8 * len(ids))
    validation_end = train_end + int(0.1 * len(ids))
    return {
        trajectory_id: (
            "train" if index < train_end else "validation" if index < validation_end else "test"
        )
        for index, trajectory_id in enumerate(ids)
    }


def write_split_file(path: str | Path, assignments: Mapping[str, str], seed: int) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, "assignments": dict(sorted(assignments.items()))}
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"Existing split file does not match split_seed={seed}: {output}"
            )
        return
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def save_episode_npz(
    path: str | Path, frames: Sequence[Mapping[str, np.ndarray | float | int | bool]],
    metadata: Mapping[str, np.ndarray | float | int | str | bool],
) -> None:
    if not frames:
        raise ValueError("Cannot save an empty rollout episode.")
    keys = tuple(frames[0].keys())
    if any(tuple(frame.keys()) != keys for frame in frames):
        raise ValueError("Every rollout frame must contain identical ordered keys.")
    arrays = {}
    for key in keys:
        arrays[key] = np.stack([np.asarray(frame[key]) for frame in frames])
        if arrays[key].dtype.kind in "fc" and not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"Rollout field {key} contains NaN or Inf.")
    for key, value in metadata.items():
        arrays[f"meta_{key}"] = np.asarray(value)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, output)


def append_manifest_row(path: str | Path, fieldnames: Sequence[str], row: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def completed_trajectory_ids(*paths: str | Path) -> set[str]:
    completed = set()
    for path in paths:
        candidate = Path(path)
        if not candidate.is_file():
            continue
        with candidate.open("r", encoding="utf-8", newline="") as stream:
            completed.update(row["trajectory_id"] for row in csv.DictReader(stream))
    return completed
