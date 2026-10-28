"""Keyboard payload forces and per-robot FT admittance, with no scene access."""
from threading import Lock

import numpy as np


class PayloadKeyboard:
    """W/S: Fx, A/D: Fy, Q/E: yaw torque, X: release, R: initial, P: pause."""
    def __init__(self, initial=(0, 0, 0), force_step=.5, torque_step=.05, max_force=4., max_torque=.2):
        self.initial = np.asarray(initial, float).copy()
        self.force_step, self.torque_step = force_step, torque_step
        self.max_force, self.max_torque = max_force, max_torque
        if (self.initial.shape != (3,) or not np.isfinite(self.initial).all()
                or not np.isfinite([force_step, torque_step, max_force, max_torque]).all()
                or min(force_step, torque_step, max_force) <= 0 or max_torque < 0
                or np.linalg.norm(self.initial[:2]) > max_force or abs(self.initial[2]) > max_torque):
            raise ValueError("Invalid finite payload force/torque or keyboard limits")
        self.requested, self.paused = self.initial.copy(), False
        self.lock = Lock()

    def on_key(self, key):
        with self.lock:
            if key == ord("P"):
                self.paused = not self.paused
            elif key == ord("X"):
                self.requested[:] = 0
            elif key == ord("R"):
                self.requested[:] = self.initial
            elif key in (ord("W"), ord("S"), ord("A"), ord("D"), ord("Q"), ord("E")):
                index = 0 if key in (ord("W"), ord("S")) else 1 if key in (ord("A"), ord("D")) else 2
                delta = self.torque_step if index == 2 else self.force_step
                self.requested[index] += delta if key in (ord("W"), ord("A"), ord("Q")) else -delta
                norm = np.linalg.norm(self.requested[:2])
                if norm > self.max_force:
                    self.requested[:2] *= self.max_force / norm
                self.requested[2] = np.clip(self.requested[2], -self.max_torque, self.max_torque)
            else:
                return
            print(f"[KEY] payload WORLD Fx/Fy/Tz={self.requested.round(3)}; paused={self.paused}", flush=True)

    def snapshot(self):
        with self.lock:
            return self.requested.copy(), self.paused


def slew(current, target, dt, limits):
    result = np.asarray(current, float).copy()
    delta = np.asarray(target)[:2] - result[:2]
    norm = np.linalg.norm(delta)
    if norm:
        result[:2] += delta * min(1., limits[0] * dt / norm)
    result[2] += np.clip(target[2] - result[2], -limits[1] * dt, limits[1] * dt)
    return result


class LocalAdmittance:
    """M*dv/dt + D*v = filtered local grasp wrench, without formation feedback.

    Receives only this robot's measured world wrench, base axes and timing.
    Raw measurements are never replaced by a share of the applied object force.
    """
    def __init__(self, mass=4., damping=8., yaw_inertia=.2, yaw_damping=1.,
                 filter_tau=.3, force_deadband=.15, torque_deadband=.03,
                 max_speed=.15, max_yaw=.2, acceleration=.10, yaw_acceleration=.3):
        values = [mass, damping, yaw_inertia, yaw_damping, filter_tau, max_speed, max_yaw, acceleration, yaw_acceleration]
        if not np.isfinite([*values, force_deadband, torque_deadband]).all() or min(values) <= 0 or min(force_deadband,torque_deadband) < 0:
            raise ValueError("Admittance parameters must be finite and positive; deadbands nonnegative")
        self.mass = np.array([mass,mass,yaw_inertia],float)
        self.damping = np.array([damping,damping,yaw_damping],float)
        self.filter_tau = filter_tau
        self.deadband = np.array([force_deadband,force_deadband,torque_deadband])
        self.max_speed, self.max_yaw = max_speed, max_yaw
        self.acceleration, self.yaw_acceleration = acceleration, yaw_acceleration
        self.reset()

    def reset(self):
        self.bias = np.zeros(6)
        self.bias_samples = 0
        self.filtered_world = np.zeros(6)
        self.command_world = np.zeros(3)
        self.command = np.zeros(3)
        self.corrected_local = np.zeros(3)

    def update(self, measured_world, base_rotation, dt, settling=False, calibrating=False):
        measured_world, r = np.asarray(measured_world,float), np.asarray(base_rotation,float)
        if measured_world.shape != (6,) or r.shape != (3,3) or not np.isfinite([*measured_world,*r.ravel(),dt]).all() or dt <= 0:
            raise ValueError("Finite own wrench, rotation and positive dt required")
        if settling:
            if calibrating:
                self.bias_samples += 1
                self.bias += (measured_world - self.bias) / self.bias_samples
            self.filtered_world[:] = 0
            self.command_world[:] = 0
            self.command[:] = 0
            self.corrected_local[:] = 0
            return self.command.copy()
        # Keep filter/virtual velocity in WORLD axes to avoid mixing body axes
        # when the robot turns; rotate own wrench and output at the current step.
        alpha = -np.expm1(-dt / self.filter_tau)
        self.filtered_world += alpha * (measured_world - self.bias - self.filtered_world)
        local = np.r_[r.T @ self.filtered_world[:3], r.T @ self.filtered_world[3:]]
        sensed = local[[0,1,5]]
        self.corrected_local = np.sign(sensed) * np.maximum(np.abs(sensed) - self.deadband, 0.)
        current_local = np.r_[(r.T @ np.r_[self.command_world[:2],0.])[:2], self.command_world[2]]
        decay = np.exp(-self.damping * dt / self.mass)
        target = decay * current_local + (1 - decay) * self.corrected_local / self.damping
        norm = np.linalg.norm(target[:2])
        if norm > self.max_speed:
            target[:2] *= self.max_speed / norm
        target[2] = np.clip(target[2],-self.max_yaw,self.max_yaw)
        self.command = slew(current_local,target,dt,[self.acceleration,self.yaw_acceleration])
        self.command_world = np.r_[(r @ np.r_[self.command[:2],0.])[:2],self.command[2]]
        return self.command.copy()
