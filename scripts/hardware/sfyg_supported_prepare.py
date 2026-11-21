"""Suspended, feet-clear preparation only. No policy evaluation or auto handoff."""
import json
import numpy as np
from sfyg_lifecycle import POLICY_SHA, DEPLOYMENT_SHA, smooth, CommandRejected
from shadow_sfyg_locomotion import sha256, vec

# Recorded 2026-11-19 pose. This is a narrow commissioning envelope, not get-up.
RECORDED = np.array([-.3032, .0429, -2.983, -.741586, .4645,
                     -.285685, .1141, 3.3212, -.6927, .4043])


def configuration(bundle):
    if sha256(bundle/'deployment.json') != DEPLOYMENT_SHA or sha256(bundle/'policy.pt') != POLICY_SHA:
        raise ValueError('Expected unchanged formal robust5000 bundle')
    # Only verify the weights hash. Never load/evaluate the actor.
    return json.loads((bundle/'deployment.json').read_text())


def tilt_degrees(gravity):
    return float(np.degrees(np.arccos(np.clip(-gravity[2], -1, 1))))


def start_rejection(q, dq, gravity, tilt):
    if not .99 <= np.linalg.norm(gravity) <= 1.01:
        return 'Invalid gravity vector'
    if max(abs(q-RECORDED)) > .08:
        return 'Outside tested suspended start: recorded pose +/-0.08rad'
    if max(abs(dq)) > .1:
        return 'Suspended start is not still: joint speed exceeds 0.1rad/s'
    if tilt >= 12:
        return 'Suspended start tilt must be below 12deg'
    return None


class SupportedPrepare:
    def __init__(self, cfg):
        self.default = vec(cfg['defaults'][:10], 10)
        self.kp = np.array([a['kp'] for a in cfg['actuators'][:10]], float)
        self.kd = np.array([a['kd'] for a in cfg['actuators'][:10]], float)
        self.effort = np.array([a['effort'] for a in cfg['actuators'][:10]], float)
        self.state, self.reason = 'WAIT', ''
        self.target = self.default.copy()
        self.origin = None
        self.last = self.stable = self.bad_since = None
        self.since = 0.
        self.events = []
        self.paused_gain = 1.

    def transition(self, state, now):
        self.state, self.since = state, now
        self.events.append(dict(state=state, time_s=now))

    def fault(self, reason, now):
        if self.state != 'FAULT':
            self.reason = str(reason)
            self.transition('FAULT', now)

    def move_duration(self, origin):
        # Quintic path: max slope 1.875, max curvature <5.774.
        delta = max(abs(self.default-origin))
        return max(16., 1.875*delta/.1, np.sqrt(5.774*delta/.1))

    def move_target(self, origin, fraction):
        return origin + smooth(fraction)*(self.default-origin)

    def preview(self, q, dq, gravity):
        tilt = tilt_degrees(gravity)
        error = q-RECORDED
        rejection = start_rejection(q, dq, gravity, tilt)
        report = dict(measured_start_rad=q.tolist(), start_error_rad=error.tolist(),
                      start_within_envelope=bool(max(abs(error)) <= .08),
                      max_joint_speed_rad_s=float(max(abs(dq))), tilt_deg=tilt,
                      start_gate_passed=rejection is None, start_rejection=rejection,
                      prepare_target=self.default.tolist(), policy_calls=0, motor_output=False)
        if rejection is None:
            duration = self.move_duration(q)
            report['move_duration_s'] = duration
            report['move_waypoints'] = [
                dict(move_time_s=fraction*duration, q_rad=self.move_target(q, fraction).tolist())
                for fraction in (0., .25, .5, .75, 1.)
            ]
        return report

    def settled(self, q, dq):
        return max(abs(q-self.target)) < .08 and max(abs(dq)) < .1

    def tick(self, now, q, dq, gravity, event=None):
        q, dq, gravity = vec(q, 10), vec(dq, 10), vec(gravity, 3)
        tilt = tilt_degrees(gravity)
        if event == 'prepare suspended':
            if self.state != 'WAIT':
                raise CommandRejected('Prepare only from WAIT')
            rejection = start_rejection(q, dq, gravity, tilt)
            if rejection is not None:
                raise CommandRejected(rejection)
        elif event == 'release supported' and self.state not in ('HOLD', 'PAUSED'):
            raise CommandRejected('Release only from settled HOLD or PAUSED')
        elif event not in (None, 'prepare suspended', 'release supported', 'stop', 'abort'):
            raise CommandRejected('No policy/start command exists in this controller')
        dt = .02 if self.last is None else now-self.last
        self.last = now
        if not np.isfinite(now) or not .005 <= dt <= .05 or not .99 <= np.linalg.norm(gravity) <= 1.01:
            self.fault('Clock/deadline/gravity invalid', now)
        if event == 'abort': self.fault('operator abort', now)
        if self.state == 'FAULT': return self.packet(q, np.zeros(10), self.kd)
        if event == 'prepare suspended':
            self.origin = q.copy()
            self.target = q.copy()
            self.duration = self.move_duration(self.origin)
            self.transition('CAPTURE', now)
        if event == 'stop' and self.state in ('CAPTURE', 'MOVE', 'SETTLE'):
            # Freeze the current reference, never jump to default or switch policy.
            self.paused_gain = smooth((now-self.since)/.25) if self.state == 'CAPTURE' else 1.
            self.transition('PAUSING', now)
            self.stable = None
        if event == 'release supported': self.transition('RELEASE', now)
        kp, kd = self.kp.copy(), self.kd.copy()
        elapsed = now-self.since
        if self.state == 'WAIT':
            return self.packet(q, np.zeros(10), kd)
        if self.state == 'CAPTURE':
            kp *= smooth(elapsed/.25)
            if elapsed >= 1.:
                self.transition('MOVE', now)
        elif self.state == 'MOVE':
            self.target = self.move_target(self.origin, elapsed/self.duration)
            if elapsed >= self.duration:
                self.transition('SETTLE', now)
        elif self.state == 'RELEASE':
            kp *= self.paused_gain*(1-smooth(elapsed/2.))
            if elapsed >= 2.: self.transition('DAMP', now)
        elif self.state == 'DAMP':
            kp *= 0
            if elapsed >= 1.: self.transition('DONE', now)
        elif self.state == 'DONE': kp *= 0
        elif self.state in ('PAUSING', 'PAUSED'): kp *= self.paused_gain
        if self.state in ('SETTLE', 'PAUSING'):
            self.stable = (now if self.stable is None else self.stable) if self.settled(q, dq) else None
            if self.stable is not None and now-self.stable >= 2.:
                self.transition('HOLD' if self.state == 'SETTLE' else 'PAUSED', now)
            elif now-self.since > 12.: self.fault('settling timeout', now)
        if tilt >= 12 or max(abs(dq)) > 1.:
            self.fault('supported motion tilt/velocity limit', now)
        # A narrow swept corridor around this tested path, including gravity sag.
        low = np.minimum(self.origin, self.default)-.15
        high = np.maximum(self.origin, self.default)+.15
        releasing = self.state in ('RELEASE', 'DAMP', 'DONE')
        if not releasing and (np.any(q < low) or np.any(q > high)):
            self.fault('outside supported pose corridor', now)
        err = max(abs(q-self.target))
        self.bad_since = (now if self.bad_since is None else self.bad_since) if err > .12 else None
        if not releasing and (err > .25 or (self.bad_since is not None and now-self.bad_since >= .25)):
            self.fault('preparation tracking error', now)
        if self.state == 'FAULT': return self.packet(q, np.zeros(10), self.kd)
        tau = kp*(self.target-q)-kd*dq
        scale = np.minimum(1., self.effort/np.maximum(abs(tau), 1e-9))
        return self.packet(self.target, kp*scale, kd*scale)

    def packet(self, target, kp, kd):
        return dict(q=target.tolist(), kp=kp.tolist(), kd=kd.tolist(), state=self.state)
