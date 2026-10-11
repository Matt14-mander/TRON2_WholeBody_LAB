"""Simulator-independent safety and hot-target input for live arm diagnostics."""

import json
from pathlib import Path

import numpy as np


FIXED_WALK_COMMAND = np.asarray((-0.29, -0.08, 0.0))


def target_values(position, quaternion, arrival_s):
    position = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("Target position must contain three finite world coordinates.")
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("Target quaternion must contain four finite WXYZ values.")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-9:
        raise ValueError("Target quaternion has zero norm.")
    if not np.isfinite(arrival_s) or not 0.0 <= arrival_s <= 1.0:
        raise ValueError("Arrival time must be within the 1 s MPC horizon.")
    return position.copy(), quaternion / norm, float(arrival_s)


class LiveTargetFile:
    """Poll a small JSON command; each new id is accepted exactly once.

    Writers should atomically replace the file. Malformed/partially written
    input raises ValueError and leaves the last accepted id unchanged.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.last_id = None

    def poll(self):
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            target_id = data["id"]
            if not isinstance(target_id, str) or not target_id:
                raise ValueError("Target id must be a non-empty string.")
            if target_id == self.last_id:
                return None
            values = target_values(data["position_world"], data["quaternion_wxyz"], data.get("arrival_s", 1.0))
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid live target JSON: {error}") from error
        return target_id, values

    def accept(self, target_id):
        self.last_id = target_id


class ArmTargetSmoother:
    """Bound position increments and velocity changes at the policy step rate."""

    def __init__(self, position, lower, upper, speed=0.4, acceleration=1.0):
        if not np.isfinite(speed) or speed <= 0 or not np.isfinite(acceleration) or acceleration <= 0:
            raise ValueError("Arm speed and acceleration must be finite and positive.")
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        if self.lower.shape != (6,) or self.upper.shape != (6,) or not np.all(np.isfinite([self.lower, self.upper])) or np.any(self.lower >= self.upper):
            raise ValueError("Arm limits must be finite ordered six-vectors.")
        self.position = np.clip(np.asarray(position).copy(), self.lower, self.upper)
        self.velocity = np.zeros(6)
        self.speed, self.acceleration = speed, acceleration

    def step(self, desired, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Control timestep must be finite and positive.")
        desired = np.asarray(desired)
        if desired.shape != (6,) or not np.all(np.isfinite(desired)):
            raise ValueError("Arm position must contain six finite values.")
        if np.any(desired < self.lower) or np.any(desired > self.upper):
            raise ValueError("Arm solution exceeds joint position limits.")
        error = desired - self.position
        # Braking speed prevents a large overshoot as the reference stops.
        requested = np.sign(error) * np.minimum(self.speed, np.sqrt(2 * self.acceleration * np.abs(error)))
        velocity = self.velocity + np.clip(requested - self.velocity, -self.acceleration * dt, self.acceleration * dt)
        position = np.clip(self.position + velocity * dt, self.lower, self.upper)
        self.velocity = (position - self.position) / dt
        self.position = position
        return self.position.copy(), self.velocity.copy()


class JsonlDiagnostics:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("w", encoding="utf-8")

    def write(self, record):
        def convert(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value).__name__)
        self.stream.write(json.dumps(record, default=convert, allow_nan=False) + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()
