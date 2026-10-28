"""Role interface: followers receive own signals only; no simulator/leader handle."""
from dataclasses import dataclass
from threading import Lock
from typing import Protocol
import numpy as np

from sfyg_cooperative_control import LocalAdmittance, slew


@dataclass(frozen=True)
class FollowerInput:
    time_s: float
    dt: float
    settling: bool
    calibrating: bool
    grasp_world: np.ndarray  # object -> own robot, at own grasp point
    base_rotation: np.ndarray
    arm_position_error: np.ndarray
    arm_velocity: np.ndarray
    previous_arm_target_offset: np.ndarray


@dataclass(frozen=True)
class HighAction:
    velocity: np.ndarray  # own base axes [vx, vy, wz], physical SI units
    arm_offset: np.ndarray  # six radians; currently only zero is supported


class FollowerPolicy(Protocol):
    def reset(self): ...
    def update(self, observation: FollowerInput) -> HighAction: ...
    def diagnostics(self) -> dict: ...


class AdmittanceFollower:
    def __init__(self, axes='forward', hold=False, **kwargs):
        if axes not in ('forward','planar'): raise ValueError('Unknown follower axes')
        self.axes, self.hold = axes, hold
        self.controller = LocalAdmittance(**kwargs)
        self.suggestion = np.zeros(3)

    def reset(self):
        self.controller.reset()
        self.suggestion[:] = 0

    def update(self, observation):
        o = observation
        command = self.controller.update(o.grasp_world,o.base_rotation,o.dt,
            settling=o.settling,calibrating=o.calibrating)
        if self.axes=='forward':
            command[1:] = 0
            self.controller.command[:] = command
            self.controller.command_world[:] = np.r_[
                (o.base_rotation @ np.r_[command[:2],0.])[:2],command[2]]
        self.suggestion = command.copy()
        return HighAction(np.zeros(3) if self.hold else command,np.zeros(6))

    def diagnostics(self):
        c = self.controller
        return dict(kind='hold' if self.hold else 'ft_admittance',axes=self.axes,
            suggestion=self.suggestion.tolist(),bias_world=c.bias.tolist(),
            bias_samples=c.bias_samples,filtered_local_fx_fy_tz=c.corrected_local.tolist())


def make_follower(kind, **kwargs):
    if kind in ('teacher','student'):
        raise NotImplementedError(f'{kind} adapter is reserved, not deployed: validate observation/force '
            'semantics and checkpoint manifest first. External applied force is NOT local grasp reaction.')
    if kind not in ('admittance','hold'): raise ValueError('Unknown follower policy')
    return AdmittanceFollower(hold=kind=='hold',**kwargs)


def checked_action(action, max_speed, max_yaw):
    command,arm = np.asarray(action.velocity,float),np.asarray(action.arm_offset,float)
    if command.shape!=(3,) or arm.shape!=(6,) or not np.isfinite(np.r_[command,arm]).all():
        raise ValueError('Finite velocity3/arm6 high action required')
    if np.any(arm != 0): raise ValueError('This demo holds the original arm pose; active arm adapter required')
    if np.linalg.norm(command[:2])>max_speed+1e-8 or abs(command[2])>max_yaw+1e-8:
        raise ValueError('High action exceeds declared command bounds')
    return command.copy()


class LeaderKeyboard:
    """Forward-only keyboard velocity for the leader, never the shared object."""
    def __init__(self, initial=.1, maximum=.15, step=.025):
        if not np.isfinite([initial,maximum,step]).all() or not 0<=initial<=maximum or min(maximum,step)<=0:
            raise ValueError('Valid forward leader velocity limits required')
        self.initial,self.maximum,self.step = initial,maximum,step
        self.requested,self.paused = initial,False
        self.lock = Lock()

    def on_key(self,key):
        with self.lock:
            if key==ord('W'): self.requested=min(self.maximum,self.requested+self.step)
            elif key==ord('S'): self.requested=max(0.,self.requested-self.step)
            elif key==ord('X'): self.requested=0.
            elif key==ord('R'): self.requested=self.initial
            elif key==ord('P'): self.paused=not self.paused
            else: return
            print(f'[KEY] LEADER ONLY vx={self.requested:.3f} m/s paused={self.paused}',flush=True)

    def snapshot(self):
        with self.lock: return self.requested,self.paused


def leader_command(previous, request, dt, acceleration, inactive):
    target = np.array([0. if inactive else request,0.,0.])
    return slew(previous,target,dt,[acceleration,acceleration])
