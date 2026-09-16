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


@dataclass(frozen=True)
class ExternalWrenchExcitation:
    """One deterministic external-wrench validation condition.

    The wrench is applied to the selected payload-proxy body at its center of
    mass and is expressed in that body's local frame.  ``axis == "zero"`` is
    the no-disturbance control condition.
    """

    axis: str
    sign: int
    profile: str
    amplitude: float

    @property
    def channel(self) -> int | None:
        return {"fx": 0, "fy": 1, "mz": 5}.get(self.axis)


def make_external_wrench_assignments(
    trajectory_ids: Sequence[str],
    seed: int = 42,
    force_amplitude: float = 5.0,
    torque_amplitude: float = 1.0,
    profiles: Sequence[str] = ("step", "ramp", "sine"),
) -> dict[str, ExternalWrenchExcitation]:
    """Create balanced, deterministic zero/+/- Fx/Fy/Mz assignments."""
    ids = list(trajectory_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Trajectory ids must be unique when assigning external wrenches.")
    if force_amplitude <= 0.0 or torque_amplitude <= 0.0:
        raise ValueError("External-wrench amplitudes must be positive.")
    profiles = tuple(profiles)
    allowed_profiles = {"step", "ramp", "sine"}
    if not profiles or any(profile not in allowed_profiles for profile in profiles):
        raise ValueError(f"Profiles must be selected from {sorted(allowed_profiles)}.")

    conditions = (
        ("zero", 0), ("fx", 1), ("fx", -1), ("fy", 1),
        ("fy", -1), ("mz", 1), ("mz", -1),
    )
    generator = random.Random(seed)
    assignments = {}
    for block_start in range(0, len(ids), len(conditions)):
        block_conditions = list(conditions)
        generator.shuffle(block_conditions)
        profile = profiles[(block_start // len(conditions)) % len(profiles)]
        for trajectory_id, (axis, sign) in zip(
            ids[block_start:block_start + len(conditions)], block_conditions
        ):
            amplitude = 0.0 if axis == "zero" else (
                torque_amplitude if axis == "mz" else force_amplitude
            )
            assignments[trajectory_id] = ExternalWrenchExcitation(
                axis=axis, sign=sign, profile=profile, amplitude=amplitude
            )
    return assignments


def make_paired_external_wrench_excitations(
    force_amplitude: float = 5.0,
    torque_amplitude: float = 1.0,
    profiles: Sequence[str] = ("step", "ramp", "sine"),
) -> list[tuple[str, ExternalWrenchExcitation]]:
    """Return the full matched zero/+/- Fx/Fy/Mz x profile protocol."""
    # Reuse assignment validation so paired and unpaired protocols accept the
    # same amplitude/profile domain.
    make_external_wrench_assignments(
        ["validation"], 0, force_amplitude, torque_amplitude, profiles
    )
    conditions = (
        ("zero", 0), ("fx", 1), ("fx", -1), ("fy", 1),
        ("fy", -1), ("mz", 1), ("mz", -1),
    )
    jobs = []
    for profile in profiles:
        for axis, sign in conditions:
            amplitude = 0.0 if axis == "zero" else (
                torque_amplitude if axis == "mz" else force_amplitude
            )
            sign_name = "zero" if sign == 0 else "pos" if sign > 0 else "neg"
            episode_suffix = f"{profile}_{axis}_{sign_name}"
            jobs.append((episode_suffix, ExternalWrenchExcitation(
                axis=axis, sign=sign, profile=profile, amplitude=amplitude
            )))
    return jobs


def evaluate_external_wrench(
    excitation: ExternalWrenchExcitation,
    time_s: float,
    start_time_s: float,
    duration_s: float,
) -> np.ndarray:
    """Evaluate a local-frame wrench ordered [Fx, Fy, Fz, Mx, My, Mz]."""
    if duration_s <= 0.0:
        raise ValueError("External-wrench duration must be positive.")
    wrench = np.zeros(6, dtype=np.float64)
    channel = excitation.channel
    if channel is None or time_s < start_time_s or time_s >= start_time_s + duration_s:
        return wrench
    phase = np.clip((time_s - start_time_s) / duration_s, 0.0, 1.0)
    if excitation.profile == "step":
        envelope = 1.0
    elif excitation.profile == "ramp":
        envelope = phase
    elif excitation.profile == "sine":
        envelope = np.sin(np.pi * phase)
    else:
        raise ValueError(f"Unsupported external-wrench profile: {excitation.profile}")
    wrench[channel] = excitation.sign * excitation.amplitude * envelope
    return wrench


def quaternion_to_rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("Quaternion must be a finite [w, x, y, z] vector.")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero.")
    w, x, y, z = quaternion / norm
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def transform_wrench_to_base_origin(
    wrench_payload: np.ndarray,
    payload_position_world: np.ndarray,
    payload_quaternion_world: np.ndarray,
    base_position_world: np.ndarray,
    base_quaternion_world: np.ndarray,
) -> np.ndarray:
    """Transform a payload-frame wrench at payload COM to the base origin."""
    wrench_payload = np.asarray(wrench_payload, dtype=np.float64)
    if wrench_payload.shape != (6,) or not np.all(np.isfinite(wrench_payload)):
        raise ValueError("Payload wrench must be a finite six-vector.")
    rotation_world_payload = quaternion_to_rotation_matrix(payload_quaternion_world)
    rotation_world_base = quaternion_to_rotation_matrix(base_quaternion_world)
    rotation_base_world = rotation_world_base.T
    force_base = rotation_base_world @ (rotation_world_payload @ wrench_payload[:3])
    torque_base_at_payload = rotation_base_world @ (
        rotation_world_payload @ wrench_payload[3:]
    )
    lever_base = rotation_base_world @ (
        np.asarray(payload_position_world, dtype=np.float64)
        - np.asarray(base_position_world, dtype=np.float64)
    )
    torque_base_at_base = torque_base_at_payload + np.cross(lever_base, force_base)
    return np.concatenate((force_base, torque_base_at_base))


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


def write_json_contract(path: str | Path, payload: Mapping[str, object]) -> None:
    """Write an immutable dataset contract or verify an existing one."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = json.loads(json.dumps(payload, sort_keys=True))
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != normalized:
            raise ValueError(f"Existing dataset contract does not match this run: {output}")
        return
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
