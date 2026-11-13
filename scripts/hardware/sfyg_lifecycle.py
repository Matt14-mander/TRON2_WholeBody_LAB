"""Deployment-only lifecycle shared by the SDK adapter and the MuJoCo test.

No SDK imports, training changes or policy weight updates. All angles are in
policy joint order. PREPARE/RELEASE require external physical support.
"""
import json
from pathlib import Path

import numpy as np
import torch

from shadow_sfyg_locomotion import (Coordinates, LEG_NAMES, TERM_NAMES, WIDTHS,
                                    ObservationHistory, make_observation, sha256, vec)


POLICY_SHA = "7e349aa6b3fe2cc62a77e874a26bb4c5742bd774d669d2120bab162d9d17809b"
CHECKPOINT_SHA = "09874b232d6f0b5834724129b24b4a307b84d9395aa408b6440b702ac9266114"
DEPLOYMENT_SHA = "3695b77527f248933aa7578154326bb394e1f601b937e2156660a81a93bd5c7e"


class CommandRejected(ValueError):
    """Operator command refused without abandoning the currently active controller."""


def load_bundle(path):
    path = Path(path)
    if sha256(path / "deployment.json") != DEPLOYMENT_SHA:
        raise ValueError("Formal deployment.json changed; do not change gains/defaults to pass startup")
    cfg = json.loads((path / "deployment.json").read_text(encoding="utf-8"))
    if (cfg.get("schema") != "sfyg_mujoco_v1" or cfg.get("phase") != "robust"
            or cfg.get("source_commit") != "f441d67" or cfg["leg_names"] != LEG_NAMES
            or cfg["checkpoint_sha256"] != CHECKPOINT_SHA
            or cfg["policy_sha256"] != POLICY_SHA or sha256(path / "policy.pt") != POLICY_SHA
            or [(t["name"], t["width"]) for t in cfg["terms"]] != list(zip(TERM_NAMES, WIDTHS))
            or not np.isclose(cfg["physics_dt"] * cfg["decimation"], .02)):
        raise ValueError("Expected the original formal robust model_5000 export")
    torch.set_num_threads(1)
    with (path / "policy.pt").open("rb") as stream:
        policy = torch.jit.load(stream, map_location="cpu").eval()
    with torch.inference_mode():
        for _ in range(3):
            value = policy(torch.zeros(1, 42), torch.zeros(1, 10, 42), torch.zeros(1, 3))
    if value.shape != (1, 10) or not torch.isfinite(value).all():
        raise ValueError("Invalid policy output")
    return cfg, policy


def smooth(x):
    x = float(np.clip(x, 0., 1.))
    return x*x*x*(10 + x*(-15 + 6*x))


def sdk_packet(packet, coords):
    """Inverse of Coordinates.convert; gains permute but never change sign."""
    result = {}
    for name in ("q", "kp", "kd"):
        values = vec(packet[name], 10)
        ordered = np.empty(10)
        ordered[coords.order] = values * coords.sign + coords.zero if name == "q" else values
        result[name] = ordered.tolist()
    return result


class Lifecycle:
    POLICY_STATES = {"BLEND", "POLICY", "STOPPING", "ZERO"}

    def __init__(self, cfg, policy):
        self.cfg, self.policy = cfg, policy
        self.default = np.asarray(cfg["defaults"][:10], float)
        self.kp = np.array([a["kp"] for a in cfg["actuators"][:10]])
        self.kd = np.array([a["kd"] for a in cfg["actuators"][:10]])
        self.effort = np.array([a["effort"] for a in cfg["actuators"][:10]])
        self.state, self.reason = "WAIT", ""
        self.history = ObservationHistory()
        self.previous = np.zeros(10, np.float32)
        self.command = np.zeros(3, np.float32)
        self.requested = np.zeros(3, np.float32)
        self.raw = np.zeros(10, np.float32)
        self.step = 0
        self.last_time = None
        self.since = 0.
        self.stable_since = None
        self.ready = False
        self.target = self.default.copy()
        self.events = []
        self.obs = self.hist = None
        self.pending_stop = False
        self.tracking_bad_since = None

    def transition(self, state, now):
        self.state, self.since = state, now
        self.events.append(dict(time_s=now, state=state))

    def fault(self, reason, now):
        if self.state != "FAULT":
            self.reason = str(reason)
            self.transition("FAULT", now)

    def validate_command(self, q, dq, tilt, event, requested):
        # Run before advancing time/history. The caller can retry this SAME tick
        # with no event after a typo/early start, retaining balance control.
        if event == "prepare_supported":
            if self.state != "WAIT":
                raise CommandRejected("Prepare requires WAIT; restart after release/fault")
            if tilt > 20 or max(abs(dq)) > .5 or max(abs(q-self.default)) > .6:
                raise CommandRejected("Prepare requires supported upright near-standing pose (each joint within 0.6 rad); deep-fold get-up is unvalidated")
        if event == "start_grounded":
            if (self.state != "READY" or not self.ready or max(abs(q-self.default)) >= .12
                    or max(abs(dq)) >= .25 or tilt >= 12):
                raise CommandRejected("Start requires settled preparation pose, upright base and grounded feet confirmation")
        if requested is not None:
            try:
                request = vec(requested, 3)
            except ValueError as exc:
                raise CommandRejected(str(exc)) from exc
            if request[0] < 0 or request[0] > .2 or np.any(request[1:] != 0):
                raise CommandRejected("Commissioning envelope is vx 0..0.20, vy=wz=0")
            if self.state != "POLICY":
                raise CommandRejected("Velocity command requires completed zero-command blend")
        if event == "release_supported" and self.state not in {"READY", "ZERO"}:
            raise CommandRejected("Release requires READY or stopped ZERO, then external support")

    def tick(self, now, q, dq, angular, gravity, event=None, requested=None):
        """50 Hz tick. Failures latch FAULT; no automatic restart or history reset."""
        q, dq, angular, gravity = vec(q, 10), vec(dq, 10), vec(angular, 3), vec(gravity, 3)
        if not np.isfinite(now):
            raise ValueError("Invalid clock")
        tilt = float(np.rad2deg(np.arccos(np.clip(-gravity[2], -1, 1))))
        self.validate_command(q, dq, tilt, event, requested)
        if not .99 <= np.linalg.norm(gravity) <= 1.01:
            self.fault("invalid gravity vector", now)
        dt = .02 if self.last_time is None else now - self.last_time
        self.last_time = now
        if not .005 <= dt <= .05:
            self.fault("control deadline/clock failure", now)
        if self.state in self.POLICY_STATES and (tilt > 35 or np.max(abs(dq)) > 16):
            self.fault("tilt or joint velocity limit", now)
        if event == "abort":
            self.fault("operator abort", now)
        if self.state == "FAULT":
            return self.damping(q)
        if event == "prepare_supported":
            self.origin = q.copy()
            self.prepare_duration = max(6., 1.875 * float(max(abs(q-self.default))) / .35)
            self.transition("PREPARE", now)
        if event == "start_grounded":
            self.history.reset()
            self.previous.fill(0)
            self.step = 0
            self.command.fill(0)
            self.requested.fill(0)
            self.transition("BLEND", now)
        if requested is not None:
            request = vec(requested, 3)
            self.requested = request.astype(np.float32)
        if event == "stop" and self.state in self.POLICY_STATES:
            self.requested.fill(0)
            if self.state == "BLEND":
                self.pending_stop = True
            else:
                self.transition("STOPPING", now)
        if event == "release_supported":
            self.origin = self.target.copy()
            self.transition("RELEASE", now)

        kp, kd = self.kp.copy(), self.kd.copy()
        if self.state == "WAIT":
            target, kp = q.copy(), np.zeros(10)
        elif self.state in {"PREPARE", "READY"}:
            elapsed = now-self.since
            if self.state == "PREPARE":
                # Gain capture at measured position before beginning the pose interpolation.
                gain = smooth(elapsed / .25)
                alpha = smooth((elapsed-1.) / self.prepare_duration)
                target = self.origin + alpha*(self.default-self.origin)
                kp *= gain
                if elapsed >= self.prepare_duration+1.:
                    self.transition("READY", now)
            else:
                target = self.default.copy()
            stable = max(abs(q-self.default)) < .12 and max(abs(dq)) < .25 and tilt < 12
            self.stable_since = (now if self.stable_since is None else self.stable_since) if stable else None
            self.ready = self.state == "READY" and self.stable_since is not None and now-self.stable_since >= 1.
        elif self.state in self.POLICY_STATES:
            if self.state == "BLEND" and now-self.since >= .3:
                self.transition("STOPPING" if self.pending_stop else "POLICY", now)
            self.command += np.clip(self.requested-self.command, -.1*dt, .1*dt)
            if self.state in {"BLEND", "ZERO"}:
                self.command.fill(0)
            self.obs = make_observation(angular, gravity, q, dq, self.previous, self.step, self.cfg)
            self.hist = self.history.push(self.obs)
            with torch.inference_mode():
                raw = self.policy(torch.from_numpy(self.obs[None]), torch.from_numpy(self.hist[None]),
                                  torch.from_numpy(self.command[None]))
            if raw.shape != (1, 10) or not torch.isfinite(raw).all():
                self.fault("non-finite/invalid inference", now)
                return self.damping(q)
            self.raw = raw[0].cpu().numpy().copy()
            alpha = smooth((now-self.since)/.3) if self.state == "BLEND" else 1.
            applied = alpha*self.raw
            target = self.default + .25*applied
            # Same torque feasibility projection as official deployment, using OUR gains.
            target = np.clip(target, q+(kd*dq-self.effort)/kp, q+(kd*dq+self.effort)/kp)
            # Training stores the action BEFORE actuator effort saturation. Keep
            # that contract: only startup blend changes the requested action.
            self.previous = applied.astype(np.float32)
            self.step += 1
            if self.state == "STOPPING" and now-self.since >= 4. and max(abs(self.command)) < 1e-6:
                self.transition("ZERO", now)
        elif self.state == "RELEASE":
            target = self.origin.copy()
            kp *= 1-smooth((now-self.since)/2.)
            if now-self.since >= 2.:
                self.transition("DAMP", now)
        else:
            target, kp = q.copy(), np.zeros(10)
            if self.state == "DAMP" and now-self.since >= 1.:
                self.transition("DONE", now)
        if self.state in {"PREPARE", "READY", "RELEASE"}:
            if max(abs(dq)) > 3. or tilt > 20:
                self.fault("supported preparation/release motion limit", now)
                return self.damping(q)
            error = float(max(abs(target-q)))
            self.tracking_bad_since = (now if self.tracking_bad_since is None else self.tracking_bad_since) if error > .5 else None
            if self.tracking_bad_since is not None and now-self.tracking_bad_since > .25:
                self.fault("preparation/release joint tracking error", now)
                return self.damping(q)
            # With ramped Kp near zero, solving for a torque-limited target can
            # produce enormous angles. Scale gains instead; keep the pose path.
            tau = kp*(target-q)-kd*dq
            scale = np.minimum(1., self.effort/np.maximum(abs(tau), 1e-9))
            kp *= scale
            kd *= scale
        self.target = target.copy()
        return dict(q=target.tolist(), kp=kp.tolist(), kd=kd.tolist(), state=self.state)

    def damping(self, q):
        return dict(q=vec(q, 10).tolist(), kp=[0.]*10, kd=self.kd.tolist(), state=self.state)
