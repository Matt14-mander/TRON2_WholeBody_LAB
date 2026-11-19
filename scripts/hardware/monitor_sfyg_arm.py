"""Receive-only monitor of an already enabled controller holding carry in CAN/J.

No SDK construction, enable, target, reset, disable or stop commands. Quitting
invalidates the leg heartbeat; it does NOT stop the arm. Stop legs first.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
import uuid

import numpy as np
from hold_sfyg_arm import Feedback, Profile, publish_status, POLICY_SHA, DEPLOYMENT_SHA


def bridge_stopped():
    service = subprocess.run(["systemctl", "is-active", "arm-mros.service"],
                             capture_output=True, text=True, timeout=1)
    if service.stdout.strip() != "inactive":
        raise RuntimeError("arm-mros.service must be inactive")
    process = subprocess.run(["pgrep", "-af", r"/home/guest/arm/(arm_bridge.py|config_arm_can)"],
                             capture_output=True, text=True, timeout=1)
    if process.returncode != 1:
        raise RuntimeError("Arm bridge process present or process check failed")


class Monitor:
    def __init__(self, target):
        self.target = np.asarray(target)
        self.stable = None
        self.ready = False

    def tick(self, q, dq, enabled, mode, now):
        if (not enabled or not mode or not np.isfinite(q).all()
                or not np.isfinite(dq).all()
                or max(abs(q-self.target)) >= .04 or max(abs(dq)) >= .05):
            self.ready = False
            error = np.asarray(q)-self.target
            raise RuntimeError(
                "Carry hold lost: "
                f"enabled={bool(enabled)} can_j={bool(mode)}; "
                f"q_finite={bool(np.isfinite(q).all())} "
                f"dq_finite={bool(np.isfinite(dq).all())}; "
                f"error_rad={error.tolist()} (abs must be <0.04); "
                f"dq_rad_s={np.asarray(dq).tolist()} (abs must be <0.05)")
        self.stable = now if self.stable is None else self.stable
        self.ready = now-self.stable >= 2
        return self.ready


def run(args):
    import can
    import fcntl
    cfg_bytes = (args.bundle_dir / "deployment.json").read_bytes()
    if hashlib.sha256(cfg_bytes).hexdigest() != DEPLOYMENT_SHA:
        raise ValueError("Expected formal robust5000 deployment")
    if hashlib.sha256((args.bundle_dir / "policy.pt").read_bytes()).hexdigest() != POLICY_SHA:
        raise ValueError("Policy hash mismatch")
    data = json.loads(args.profile.read_text())
    if not data.get("operator", "").strip() or data.get("mapping_checked") is not True:
        raise ValueError("Review arm SDK/simulation mapping and operator in profile first")
    cfg = json.loads(cfg_bytes)
    profile = Profile(data, cfg["defaults"][10:16], execute=False)
    monitor = Monitor(profile.target)
    args.status_file.parent.mkdir(parents=True, exist_ok=True)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    session = uuid.uuid4().hex
    state, reason, q, dq = "WAIT", "", None, None
    feedback = Feedback()
    def report():
        publish_status(args.status_file, dict(
            schema="sfyg_arm_monitor_v1", session=session, monotonic_s=time.monotonic(),
            state=state, ready=state == "HOLD", motor_output=False,
            enabled=state == "HOLD", can_j=state == "HOLD", policy_sha256=POLICY_SHA,
            reason=reason, q_sdk_rad=None if q is None else q.tolist(),
            dq_sdk_rad=None if dq is None else dq.tolist(), target_sdk_rad=profile.target.tolist(),
            raw_can={hex(i): {"age_s": time.monotonic()-stamp, "hex": raw.hex(" ")}
                     for i, (stamp, raw) in feedback.frames.items()}))
    with args.status_file.with_suffix(".monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            report()
            bridge_stopped()
            filters = [dict(can_id=i, can_mask=0x7FF, extended=False) for i in sorted(Feedback.IDS)]
            with can.Bus(interface="socketcan", channel=args.can, can_filters=filters,
                         ignore_config=True) as bus:
                started = due = next_bridge = time.monotonic()
                seen = False
                while not stopped:
                    msg = bus.recv(.002)
                    if msg is not None:
                        feedback.update(msg)
                    now = time.monotonic()
                    if now < due:
                        continue
                    if seen and now-due > .1:
                        raise RuntimeError("Monitor scheduling deadline missed")
                    due = now+.02
                    if now >= next_bridge:
                        bridge_stopped()
                        next_bridge = time.monotonic()+1
                    try:
                        q, dq, enabled, mode = feedback.snapshot(time.monotonic())
                    except RuntimeError:
                        if not seen and now-started < 3:
                            continue
                        raise
                    seen = True
                    profile.check_limits(q)
                    ready = monitor.tick(q, dq, enabled, mode, time.monotonic())
                    new_state = "HOLD" if ready else "SETTLING"
                    if new_state != state:
                        print(f"[ARM MONITOR] {new_state}; no motor output", flush=True)
                    state = new_state
                    report()
            state, reason = "STOPPED", "operator stopped monitor; arm controller unchanged"
        except BaseException as exc:
            state, reason = "FAULT", str(exc)
            raise
        finally:
            report()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle_dir", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--status_file", type=Path, required=True)
    p.add_argument("--can", default="can0")
    run(p.parse_args())
