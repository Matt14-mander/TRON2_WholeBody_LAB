"""Simulator-independent stand/walk fine-tuning contracts and warm-start loading."""
from __future__ import annotations

import math

import torch

SCHEMA = "wholebody_standwalk_v1"
EVAL_CASES = {
    "stand": (0, (0., 0., 0.)),
    "walk": (1, (-.25, 0., 0.)),
    "start_stop": (2, (-.25, 0., 0.)),
    "stop_start": (3, (-.25, 0., 0.)),
    "slow": (1, (-.05, 0., 0.)),
    "forward": (1, (.15, 0., 0.)),
    "left": (1, (0., .1, 0.)),
    "right": (1, (0., -.1, 0.)),
    "yaw_left": (1, (0., 0., .2)),
    "yaw_right": (1, (0., 0., -.2)),
}


def standing_mask(command, previous, enter=(.02, .03), leave=(.04, .06)):
    """Hysteresis based on actual body-frame vx/vy/wz; no sampled standing flag."""
    if not (0 <= enter[0] < leave[0] and 0 <= enter[1] < leave[1]):
        raise ValueError("Standing enter thresholds must be smaller than leave thresholds.")
    linear = command[:, :2].norm(dim=-1)
    angular = command[:, 2].abs()
    return torch.where(previous,
                       (linear < leave[0]) & (angular < leave[1]),
                       (linear <= enter[0]) & (angular <= enter[1]))


def scheduled_velocity(time_s, cases, target, transition_s=1., first_hold_s=5., second_hold_s=6.):
    """0: stand, 1: walk, 2: stand-walk-stand, 3: walk-stand-walk.

    Episode time is the only clock; no hidden schedule state crosses a reset.
    A cubic smoothstep gives zero endpoint slopes, including in the stop command.
    """
    if any(not math.isfinite(x) or x <= 0 for x in (transition_s, first_hold_s, second_hold_s)):
        raise ValueError("Hold and transition times must be positive and finite.")
    def blend(start):
        u = ((time_s - start) / transition_s).clamp(0, 1)
        return u.square() * (3 - 2 * u)
    walking = blend(first_hold_s) - blend(first_hold_s + transition_s + second_hold_s)
    fraction = torch.where(cases == 2, walking,
                           torch.where(cases == 3, 1 - walking, (cases == 1).to(target.dtype)))
    return target * fraction[:, None]


def sample_cases(count, stage, device, generator=None):
    if stage not in ("stand", "transition", "expanded"):
        raise ValueError("Stage must be stand, transition or expanded.")
    u = torch.rand(count, device=device, generator=generator)
    if stage == "stand":
        return (u >= .6).long()  # 60% stand, 40% known walk; no transitions yet.
    # Retain static stand/walk tests while explicitly practicing both transitions.
    return torch.where(u < .2, 0, torch.where(u < .4, 1, torch.where(u < .7, 2, 3)))


def _check_state(expected, actual, label):
    if set(expected) != set(actual):
        raise ValueError(f"{label} keys do not match; missing={sorted(set(expected)-set(actual))}, "
                         f"unexpected={sorted(set(actual)-set(expected))}")
    for key, value in actual.items():
        if expected[key].shape != value.shape:
            raise ValueError(f"{label} shape mismatch for {key}: {tuple(value.shape)} != {tuple(expected[key].shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{label} contains NaN/Inf: {key}")


def load_warm_start(actor_critic, encoder, checkpoint, allow_critic_input_reset=False):
    """Validate everything BEFORE copying any weights. Never silently skip actor weights.

    Only a critic first-layer input-width change may be reset (e.g. added body
    mass observation). The rest of the critic is strict. Optimizers/iteration
    counters are intentionally not loaded: this is a new fine-tuning run.
    """
    if "model_state_dict" not in checkpoint or "encoder_state_dict" not in checkpoint:
        raise ValueError("Warm start requires model_state_dict and encoder_state_dict.")
    state = checkpoint["model_state_dict"]
    expected = actor_critic.state_dict()
    if set(state) != set(expected):
        raise ValueError("Actor/critic checkpoint keys must match the complete repository model.")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Source actor/critic contains NaN/Inf.")
    actor = {k: v for k, v in state.items() if k.startswith("actor.") or k == "logstd"}
    _check_state({k: v for k, v in expected.items() if k in actor}, actor, "Actor")
    _check_state(encoder.state_dict(), checkpoint["encoder_state_dict"], "History encoder")
    critic = {k: v for k, v in state.items() if k.startswith("critic.")}
    reinitialized = []
    mismatch = [k for k in critic if critic[k].shape != expected[k].shape]
    if mismatch:
        key = "critic.0.weight"
        if (not allow_critic_input_reset or mismatch != [key]
                or critic[key].ndim != 2 or critic[key].shape[0] != expected[key].shape[0]
                or abs(critic[key].shape[1] - expected[key].shape[1]) != 1):
            raise ValueError(f"Unsupported critic architecture mismatch: {mismatch}")
        # The body-mass width changes the placement of later features. Reusing
        # arbitrary input columns would be incorrect; initialize this layer anew.
        critic[key] = expected[key].clone()
        critic["critic.0.bias"] = torch.zeros_like(expected["critic.0.bias"])
        reinitialized = [key, "critic.0.bias"]
    _check_state({k: v for k, v in expected.items() if k.startswith("critic.")}, critic, "Critic")
    actor_critic.load_state_dict({**actor, **critic}, strict=True)
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    return {"source_iteration": checkpoint.get("iter"), "critic_reinitialized": reinitialized}


def freeze_history_encoder(algorithm):
    algorithm.encoder.requires_grad_(False)
    algorithm.encoder.eval()
    # Repository PPO owns a separate supervised encoder optimizer, not actor PPO.
    algorithm.extra_optimizer = None


class EpisodeMetrics:
    """Evaluate the actual pre-reset state, excluding initial/stop settling transients."""
    def __init__(self):
        self.steps = 0
        self.peak_tilt_deg = 0.
        self.tracking_sum = 0.
        self.yaw_tracking_sum = 0.
        self.stand_linear_sum = self.stand_yaw_sum = 0.
        self.stand_count = 0
        self.quiet_since = None
        self.stand_anchor = None
        self.max_stand_drift_m = 0.
        self.stop_latency_s = None
        self.was_moving = False

    def update(self, time_s, command, linear_velocity, yaw_rate, position_xy, tilt_deg, dt):
        self.steps += 1
        self.peak_tilt_deg = max(self.peak_tilt_deg, float(tilt_deg))
        self.tracking_sum += float((linear_velocity[:2] - command[:2]).square().sum())
        self.yaw_tracking_sum += (float(yaw_rate) - float(command[2])) ** 2
        quiet = float(command[:2].norm()) <= .02 and abs(float(command[2])) <= .03
        if not quiet:
            self.was_moving = True
            self.quiet_since = None
            self.stand_anchor = None
            return
        if self.quiet_since is None:
            self.quiet_since = time_s - dt
        speed = float(linear_velocity[:2].norm())
        if self.was_moving and self.stop_latency_s is None and speed < .05 and abs(float(yaw_rate)) < .05:
            self.stop_latency_s = time_s - self.quiet_since
        if time_s - self.quiet_since < 1.:
            return
        self.stand_count += 1
        self.stand_linear_sum += speed ** 2
        self.stand_yaw_sum += float(yaw_rate) ** 2
        if self.stand_anchor is None:
            self.stand_anchor = position_xy.clone()
        self.max_stand_drift_m = max(self.max_stand_drift_m, float((position_xy - self.stand_anchor).norm()))

    def result(self, duration_s, completed, require_standing=False, require_stop=False):
        lin_rms = math.sqrt(self.stand_linear_sum / self.stand_count) if self.stand_count else None
        yaw_rms = math.sqrt(self.stand_yaw_sum / self.stand_count) if self.stand_count else None
        tracking = math.sqrt(self.tracking_sum / max(1, self.steps))
        yaw_tracking = math.sqrt(self.yaw_tracking_sum / max(1, self.steps))
        passed = bool(completed and self.peak_tilt_deg < 15. and tracking < .1 and yaw_tracking < .1)
        if require_standing:
            passed &= bool(lin_rms is not None and lin_rms < .05 and yaw_rms < .05)
        if require_stop:
            passed &= self.stop_latency_s is not None and self.stop_latency_s < 2.
        return {"duration_s": duration_s, "completed": bool(completed), "passed": bool(passed),
                "max_tilt_deg": self.peak_tilt_deg, "linear_tracking_rmse": tracking,
                "yaw_tracking_rmse": yaw_tracking,
                "standing_linear_rms": lin_rms, "standing_yaw_rms": yaw_rms,
                "standing_samples": self.stand_count, "max_standing_drift_m": self.max_stand_drift_m,
                "stop_latency_s": self.stop_latency_s}
