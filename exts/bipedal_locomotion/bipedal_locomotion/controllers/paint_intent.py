"""Simulator-independent PD interaction protocol and causal intent samples.

Labels are external-on-payload [Fx, Fy, Mz] at its COM in its local frame.
They are NOT arm-on-base reaction forces or a desired end-effector pose.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

ARM_JOINT_NAMES = tuple(f"arm{i}_Joint" for i in range(1, 7))
CONDITIONS = ("zero", "fx_pos", "fx_neg", "fy_pos", "fy_neg", "mz_pos", "mz_neg")
SCHEMA = "paint_pd_response_v1"


def collection_conditions(payload_mode, selected=None):
    conditions = tuple(CONDITIONS if selected is None else selected)
    if payload_mode not in ("attached", "none") or not conditions or len(set(conditions)) != len(conditions) or any(c not in CONDITIONS for c in conditions):
        raise ValueError("Choose attached/none and unique valid collection conditions.")
    if payload_mode == "none" and conditions != ("zero",):
        raise ValueError("No-payload mode is a zero-force support diagnostic; use --conditions zero.")
    return conditions


@dataclass(frozen=True)
class InteractionProtocol:
    warmup: float = 2.0
    duration: float = 2.0
    recovery: float = 2.0
    force: float = 5.0
    torque: float = 0.5
    profile: str = "trapezoid"

    def __post_init__(self):
        values = np.array([self.warmup, self.duration, self.recovery, self.force, self.torque])
        if not np.isfinite(values).all() or self.warmup < 0 or self.recovery < 0 or self.duration <= 0 or self.force <= 0 or self.torque <= 0:
            raise ValueError("Protocol needs finite nonnegative settling times and positive duration/amplitudes.")
        if self.profile not in ("trapezoid", "sine"):
            raise ValueError("Unsupported interaction profile.")

    @property
    def total_time(self):
        return self.warmup + self.duration + self.recovery

    def wrench(self, condition: str, time: float) -> np.ndarray:
        if condition not in CONDITIONS or not np.isfinite(time):
            raise ValueError("Unknown condition or nonfinite time.")
        result = np.zeros(6, dtype=np.float64)
        phase = (time - self.warmup) / self.duration
        if condition == "zero" or not 0 <= phase < 1:
            return result
        envelope = min(phase / 0.1, 1.0, (1.0 - phase) / 0.1) if self.profile == "trapezoid" else np.sin(np.pi * phase)
        axis, direction = condition.split("_")
        channel = {"fx": 0, "fy": 1, "mz": 5}[axis]
        result[channel] = (1 if direction == "pos" else -1) * (self.torque if axis == "mz" else self.force) * envelope
        return result


def make_history_samples(q, dq, applied_position, wrench, valid, history=4):
    """Rows are post-step states paired with the command/wrench of that interval.

    Invalid rows break history; reset responses must never cross into a sample.
    Feature order per row is q(6), dq(6), executed position command(6), oldest first.
    """
    arrays = [np.asarray(value, dtype=np.float32) for value in (q, dq, applied_position, wrench)]
    if not isinstance(history, int) or history < 1:
        raise ValueError("history must be a positive integer.")
    count = len(arrays[0])
    if any(a.shape != (count, width) for a, width in zip(arrays, (6, 6, 6, 3))) or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Invalid/nonfinite intent arrays.")
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != (count,):
        raise ValueError("Invalid validity mask.")
    features = np.concatenate(arrays[:3], axis=1)
    windows, labels, indices = [], [], []
    for end in range(history - 1, count):
        start = end + 1 - history
        if valid[start:end + 1].all():
            windows.append(features[start:end + 1].reshape(-1))
            labels.append(arrays[3][end])
            indices.append(end)
    return (np.asarray(windows, dtype=np.float32).reshape(-1, history * 18),
            np.asarray(labels, dtype=np.float32).reshape(-1, 3), np.asarray(indices, dtype=np.int64))


class IntentHistory:
    """Online adapter for the exact exported model input; clear on every reset."""
    def __init__(self, history=4):
        if not isinstance(history, int) or history < 1:
            raise ValueError("history must be positive.")
        self.history = history
        self.rows = []

    def reset(self):
        self.rows.clear()

    def update(self, q, dq, previous_applied_position):
        row = np.concatenate([np.asarray(v, dtype=np.float32) for v in (q, dq, previous_applied_position)])
        if row.shape != (18,) or not np.isfinite(row).all():
            self.reset()
            raise ValueError("Expected three finite six-vectors in radians/radians per second.")
        self.rows.append(row.copy())
        self.rows = self.rows[-self.history:]
        return None if len(self.rows) < self.history else np.concatenate(self.rows)
