"""Piper V2 carry hold on Linux SocketCAN. Default: passive feedback only.

No automatic homing, reset, gripper commands or disabling on exit. Hardware
mapping and firmware compatibility must be verified in the profile first.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import struct
import tempfile
import threading
import time
import uuid

import numpy as np

POLICY_SHA = "7e349aa6b3fe2cc62a77e874a26bb4c5742bd774d669d2120bab162d9d17809b"
DEPLOYMENT_SHA = "3695b77527f248933aa7578154326bb394e1f601b937e2156660a81a93bd5c7e"
CHECKS = ("mapping_checked", "piper_v2_firmware_checked", "exclusive_arm_control",
          "low_level_dev_mode", "gantry_and_arm_support_ready", "path_clear",
          "payload_and_gripper_secured")


def vector(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("Expected six finite joint values")
    return value


class Profile:
    def __init__(self, data, target, execute=False):
        self.sign, self.zero = vector(data["sign"]), vector(data["zero_rad"])
        self.order = data["sdk_to_sim"]
        if sorted(self.order) != list(range(6)) or not np.isin(self.sign, [-1, 1]).all():
            raise ValueError("Invalid joint permutation or signs")
        self.low, self.high = vector(data["sdk_min_rad"]), vector(data["sdk_max_rad"])
        if np.any(self.low >= self.high):
            raise ValueError("Invalid physical joint limits")
        # Each SDK joint indexes a simulation joint. q_sdk = sign*q_sim + zero.
        self.target = vector(target)[self.order] * self.sign + self.zero
        self.check_limits(self.target)
        if execute and (not str(data.get("operator", "")).strip()
                        or any(data.get(k) is not True for k in CHECKS)):
            raise ValueError("Arm preflight incomplete; verify each check on hardware")

    def check_limits(self, q):
        q = vector(q)
        if np.any(q < self.low) or np.any(q > self.high):
            raise ValueError("Arm position outside verified SDK limits")
        return q


class Feedback:
    """Require every position, velocity, driver and controller frame to be fresh."""
    IDS = set(range(0x2A5, 0x2A8)) | set(range(0x251, 0x257)) | set(range(0x261, 0x267)) | {0x2A1}

    def __init__(self):
        self.frames = {}

    def update(self, msg, wall=None, mono=None):
        if msg.arbitration_id not in self.IDS or msg.is_extended_id or msg.is_remote_frame or msg.is_error_frame:
            return
        wall = time.time() if wall is None else wall
        mono = time.monotonic() if mono is None else mono
        if len(msg.data) != 8 or not math.isfinite(msg.timestamp) or abs(wall-msg.timestamp) > .15:
            raise RuntimeError("Malformed/stale CAN feedback")
        self.frames[msg.arbitration_id] = (mono, bytes(msg.data))

    def snapshot(self, now):
        if self.frames.keys() != self.IDS or any(now-t > .15 or now < t for t, _ in self.frames.values()):
            raise RuntimeError("Missing/stale arm feedback")
        q = []
        for i in range(0x2A5, 0x2A8):
            q.extend(np.radians(np.asarray(struct.unpack(">ii", self.frames[i][1])) / 1000))
        dq = [struct.unpack_from(">h", self.frames[i][1])[0]/1000 for i in range(0x251, 0x257)]
        flags = [self.frames[i][1][5] for i in range(0x261, 0x267)]
        status = self.frames[0x2A1][1]
        if any(f & 0xBF for f in flags) or status[1] or status[6] or status[7]:
            raise RuntimeError("Arm controller/driver fault (no automatic reset)")
        return vector(q), vector(dq), all(f & 0x40 for f in flags), status[0] == 1 and status[2] == 1


class Hold:
    def __init__(self, profile):
        self.profile = profile
        self.state, self.ready, self.stable = "WAIT", False, None

    def start(self, q, dq, now):
        self.profile.check_limits(q)
        if self.state != "WAIT" or max(abs(q-self.profile.target)) > .6 or max(abs(dq)) > .1:
            raise ValueError("Require WAIT, stationary supported arm within 0.6 rad of carry; no folded-pose planner")
        self.origin, self.started = q.copy(), now
        self.duration = max(8., 1.875*max(abs(q-self.profile.target))/.08)
        self.state = "PREPARE"

    def tick(self, q, dq, enabled, mode, now):
        self.profile.check_limits(q)
        if self.state == "WAIT":
            return None
        elapsed = now-self.started
        if elapsed > 2 and (not enabled or not mode):
            raise RuntimeError("Arm enable/CAN MOVE J acknowledgement lost")
        if max(abs(dq)) > .35:
            raise RuntimeError("Arm velocity exceeded commissioning limit")
        x = float(np.clip((elapsed-2)/self.duration, 0, 1))
        alpha = x*x*x*(10+x*(-15+6*x))
        target = self.origin + alpha*(self.profile.target-self.origin)
        if max(abs(q-target)) > .15:
            raise RuntimeError("Arm tracking error")
        settled = x == 1 and enabled and mode and max(abs(q-self.profile.target)) < .04 and max(abs(dq)) < .05
        self.stable = (now if self.stable is None else self.stable) if settled else None
        self.ready = self.stable is not None and now-self.stable >= 1
        if self.ready:
            self.state = "HOLD"
        if self.state == "HOLD" and not settled:
            raise RuntimeError("Carry hold lost")
        return self.profile.check_limits(target)


class Sender:
    def __init__(self, channel):
        from piper_sdk import C_PiperInterface_V2
        self.sdk = C_PiperInterface_V2(channel)
        self.sdk.ConnectPort(piper_init=False)

    def target(self, q):
        self.sdk.JointCtrl(*[int(round(math.degrees(x)*1000)) for x in q])

    def start(self, q):
        self.target(q)  # Prime measured position before requesting control/enable.
        self.sdk.MotionCtrl_2(0x01, 0x01, 10, 0x00)
        for joint in range(1, 7):
            self.sdk.EnableArm(joint)  # Never enable/open the gripper implicitly.

    def stop(self):
        self.sdk.EmergencyStop(0x01)  # Never reset/resume/disable automatically.

    def close(self):
        self.sdk.DisconnectPort()


def publish_status(path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def run(args):
    import can
    deployment = (args.bundle_dir / "deployment.json").read_bytes()
    if hashlib.sha256(deployment).hexdigest() != DEPLOYMENT_SHA:
        raise ValueError("Expected unchanged formal robust_5000 deployment.json")
    cfg = json.loads(deployment)
    if cfg["policy_sha256"] != POLICY_SHA:
        raise ValueError("Wrong policy bundle")
    profile = Profile(json.loads(args.profile.read_text(encoding="utf-8")), cfg["defaults"][10:16], args.execute)
    args.status_file.parent.mkdir(parents=True, exist_ok=True)
    # Linux advisory lock is released after crashes; no stale lock-file deletion needed.
    import fcntl
    channel_key = hashlib.sha256(args.can.encode()).hexdigest()[:16]
    with (Path(tempfile.gettempdir()) / ("sfyg_arm_"+channel_key+".lock")).open("a") as channel_lock, (args.status_file.with_suffix(".lock")).open("a") as lock:
        fcntl.flock(channel_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(args, profile, can)


def run_locked(args, profile, can):
    feedback, hold, commands = Feedback(), Hold(profile), queue.Queue()
    session = uuid.uuid4().hex
    sender = None
    active = False
    reason = ""
    def inputs():
        import sys
        for line in sys.stdin:
            commands.put(line.strip())
        commands.put("abort")
    threading.Thread(target=inputs, daemon=True).start()
    old = {s: signal.signal(s, lambda *_: commands.put("abort")) for s in (signal.SIGINT, signal.SIGTERM)}
    def report(q=None):
        publish_status(args.status_file, dict(schema="sfyg_arm_hold_v1", session=session,
            monotonic_s=time.monotonic(), state=hold.state, ready=hold.ready,
            execute=args.execute, policy_sha256=POLICY_SHA, reason=reason,
            q_sdk_rad=None if q is None else q.tolist(), target_sdk_rad=profile.target.tolist()))
    try:
        report()
        print("Passive WAIT. Commands: prepare supported | quit supported | abort", flush=True)
        print("Mapped carry target (SDK radians):", profile.target.tolist(), flush=True)
        filters = [dict(can_id=i, can_mask=0x7FF, extended=False) for i in sorted(Feedback.IDS)]
        with can.Bus(interface="socketcan", channel=args.can, can_filters=filters, ignore_config=True) as bus:
            started = due = time.monotonic()
            last_state = None
            while True:
                msg = bus.recv(.005)
                if msg is not None:
                    feedback.update(msg)
                now = time.monotonic()
                if now < due:
                    continue
                if active and now-due > .1:
                    raise RuntimeError("Arm control scheduling deadline")
                due = now+.02
                command = commands.get_nowait() if not commands.empty() else ""
                if command in {"abort", "quit supported"}:
                    hold.state, hold.ready = "STOPPED", False
                    break
                try:
                    q, dq, enabled, mode = feedback.snapshot(now)
                except RuntimeError:
                    if not active and now-started < 3:
                        continue
                    raise
                if command == "prepare supported":
                    if not args.execute:
                        print("Read-only: restart with --execute after preflight", flush=True)
                    else:
                        try:
                            hold.start(q, dq, now)
                        except ValueError as exc:
                            print("[REJECTED]", exc, flush=True)
                        else:
                            sender = Sender(args.can)
                            active = True
                            sender.start(q)
                elif command:
                    print("Unknown command", flush=True)
                target = hold.tick(q, dq, enabled, mode, now)
                if target is not None:
                    sender.target(target)
                report(q)
                if last_state != (hold.state, hold.ready):
                    print(f"[ARM] {hold.state} ready={hold.ready} q={q.tolist()}", flush=True)
                    last_state = (hold.state, hold.ready)
    except BaseException as exc:
        hold.state, hold.ready, reason = "FAULT", False, str(exc)
        raise
    finally:
        try:
            report()
        finally:
            try:
                if sender is not None and active:
                    sender.stop()
            finally:
                if sender is not None:
                    sender.close()
                for s, handler in old.items():
                    signal.signal(s, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle_dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--status_file", type=Path, required=True)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args())
