"""Read-only SFYG S1 observer. An independent controller holds the carry pose."""
import argparse
from contextlib import closing
from datetime import datetime
import json
import math
from pathlib import Path
import struct
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/mujoco"))
from sfyg_policy import HOLD_NAMES
from sfyg_student_policy import S1_CHECKPOINT, StudentCheckpointPolicy
from sfyg_teacher_core import History, process_action


def ordered_arm(names, q, dq, arm_names):
    names = list(names)
    if (len(set(names)) != len(names) or len(q) != len(names) or len(dq) != len(names)
            or any(name not in names for name in arm_names)):
        raise ValueError(f"Expected named arm q/dq in radians: {arm_names}; received {names}")
    indices = [names.index(name) for name in arm_names]
    arm = np.asarray(q, dtype=float)[indices]
    velocity = np.asarray(dq, dtype=float)[indices]
    if not (np.isfinite(arm).all() and np.isfinite(velocity).all()):
        raise ValueError("Nonfinite arm feedback")
    return arm.tolist(), velocity.tolist()


def piper_samples(channel):
    """Listen to broadcast feedback on an already configured Linux SocketCAN link."""
    import can

    positions = range(0x2A5, 0x2A8)
    speeds = range(0x251, 0x257)
    ids = set(positions) | set(speeds)
    filters = [dict(can_id=i, can_mask=0x7ff, extended=False) for i in sorted(ids)]
    q, dq = np.zeros(6), np.zeros(6)
    stamps = {}
    deadline = time.monotonic() + 3
    with can.Bus(interface="socketcan", channel=channel, can_filters=filters,
                 ignore_config=True) as bus:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = sorted(ids - stamps.keys())
                raise TimeoutError(f"No coherent Piper X state; missing IDs: {[hex(i) for i in missing]}")
            msg = bus.recv(remaining)
            if msg is None:
                continue
            i = msg.arbitration_id
            if i not in ids or msg.is_extended_id or msg.is_remote_frame or msg.is_error_frame:
                continue
            if len(msg.data) != 8:
                raise ValueError(f"Invalid Piper feedback length on {i:#x}")
            stamp = round(msg.timestamp * 1e9)
            stamps[i] = stamp
            if i in positions:
                j = 2 * (i - positions.start)
                q[j:j+2] = np.radians(np.array(struct.unpack(">ii", msg.data)) / 1000)
            else:
                dq[i - speeds.start] = struct.unpack_from(">h", msg.data)[0] / 1000
            if stamps.keys() != ids:
                continue
            newest = max(stamps.values())
            span = newest - min(stamps.values())
            if span > 20_000_000:
                continue
            received = time.time_ns()
            if abs(received - newest) > 250_000_000:
                raise ValueError("Stale SocketCAN feedback or receiver processing backlog >250 ms")
            yield dict(stamp_ns=newest, arm_names=HOLD_NAMES[:6], q_rad=q.tolist(),
                       dq_rad_s=dq.tolist(), receive_unix_ns=received, can_span_ms=span/1e6)
            stamps.clear()
            deadline = time.monotonic() + 3


def replay_samples(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def timed_samples(samples, dt):
    """Pick the first observed packet at/after each source-clock control tick."""
    period = round(dt * 1e9)
    due = previous = None
    for packet in samples:
        stamp = packet["stamp_ns"]
        if not isinstance(stamp, int) or stamp <= 0 or (previous is not None and stamp <= previous):
            raise ValueError(f"Invalid or non-increasing source nanosecond stamp {stamp}")
        previous = stamp
        if due is None:
            due = stamp
        if stamp < due:
            continue
        if stamp - due >= period:
            raise ValueError(f"Missing 50 Hz state frame: packet late by {(stamp-due)/1e6:.2f} ms")
        yield packet, (stamp - due) / 1e6
        due += period


class Shadow:
    def __init__(self, checkpoint, calibration):
        self.policy = StudentCheckpointPolicy(checkpoint)
        teacher = self.policy.manifest["teacher_contract"]
        if (calibration["student_sha256"] != self.policy.sha or
                calibration["low_checkpoint_sha256"] != teacher["low_sha256"]):
            raise ValueError("Calibration does not belong to this student/locomotion checkpoint")
        interface = teacher["low_interface"]
        indices = [interface["names"].index(name) for name in HOLD_NAMES[:6]]
        self.nominal = np.asarray(interface["defaults"])[indices]
        self.bias = np.asarray(calibration["q_bias_rad"], dtype=float)
        self.zero = float(calibration["zero_n"])
        self.gain = float(calibration["gain"])
        self.cap = float(calibration["cap_n"])
        if (self.bias.shape != (6,) or not np.isfinite(self.bias).all() or
                not math.isfinite(self.zero) or self.gain != 1.5444 or self.cap != 8):
            raise ValueError("Expected the six q offsets and fixed gain/cap from the 6 N handoff report")
        self.history = History(2, 8, 18, "cpu")

    @torch.inference_mode()
    def infer(self, tick, q, dq):
        p = self.policy
        error = np.asarray(q) - self.nominal
        # Same sensor frame for both branches. Target offsets are zero because
        # the external controller is holding the checkpoint's nominal target.
        frame = np.zeros((2, 18), dtype=np.float32)
        frame[0, :6], frame[1, :6] = error, error + self.bias
        frame[:, 6:12] = dq
        history = self.history.append(torch.from_numpy(frame), tick,
                                      valid=torch.full((2,), tick > 0)).flatten(1)
        wrench = p.actor.estimate(history)
        unscaled = wrench[:, 0].clone() * 40
        wrench[1, 0] = ((unscaled[1] - self.zero) * self.gain).clamp(max=self.cap) / 40
        raw = p.actor.actor(torch.cat((history, wrench), dim=1))
        commands = process_action(raw, p.time(tick), p.settings)
        return {name: dict(history144=history[i].tolist(), fx_before_gain_n=unscaled[i].item(),
                           estimated_fx_n=(40*wrench[i, 0]).item(), raw_vx=raw[i, 0].item(),
                           cmd_vx=commands[i, 0].item())
                for i, name in enumerate(("native", "mujoco_cal"))}


def run(args):
    if not math.isfinite(args.pose_tolerance) or args.pose_tolerance <= 0:
        raise ValueError("Pose tolerance must be finite and positive")
    torch.set_num_threads(1)
    cal = json.loads(args.calibration.read_text(encoding="utf-8"))
    observer = Shadow(args.checkpoint, cal)
    dt = observer.policy.settings.dt
    seconds = 30. if args.seconds is None and args.can else args.seconds
    if seconds is not None and (not math.isfinite(seconds) or seconds < dt):
        raise ValueError("Duration must contain at least one control frame")
    count = None if seconds is None else round(seconds / dt)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    print(f"[SHADOW] carry target rad={observer.nominal.tolist()}  Piper J1..J6={HOLD_NAMES[:6]}", flush=True)
    report = dict(completed=False, frames=0, checkpoint_sha256=observer.policy.sha, dt=dt,
                  arm_names=HOLD_NAMES[:6], carry_target_rad=observer.nominal.tolist(),
                  pose_tolerance_rad=args.pose_tolerance, q_bias_rad=observer.bias.tolist(),
                  zero_n=observer.zero, gain=observer.gain, cap_n=observer.cap,
                  source=args.can or str(args.replay), output=str(out))
    source = piper_samples(args.can) if args.can else replay_samples(args.replay)
    try:
        with closing(source), (out / "steps.jsonl").open("x", encoding="utf-8", buffering=1) as log:
            for tick, (packet, late) in enumerate(timed_samples(source, dt)):
                q, dq = ordered_arm(packet["arm_names"], packet["q_rad"], packet["dq_rad_s"], HOLD_NAMES[:6])
                deviation = float(np.max(np.abs(q - observer.nominal)))
                if deviation > args.pose_tolerance:
                    raise ValueError(f"Outside carry pose: max error {deviation:.4f} rad, q={q}")
                pair = observer.infer(tick, q, dq)
                row = dict(tick=tick, time_s=tick*dt, stamp_ns=packet["stamp_ns"],
                           receive_unix_ns=packet.get("receive_unix_ns"), sample_late_ms=late,
                           arm_names=HOLD_NAMES[:6], q_rad=q, dq_rad_s=dq,
                           can_span_ms=packet.get("can_span_ms"), **pair)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                report["frames"] += 1
                if tick % 50 == 0:
                    print(f"[{tick*dt:5.1f}s] Fx N / raw / cmd: " + "   ".join(
                        f"{k}={v['estimated_fx_n']:+.2f}/{v['raw_vx']:+.4f}/{v['cmd_vx']:.4f}"
                        for k, v in pair.items()), flush=True)
                if report["frames"] == count:
                    break
        if not report["frames"] or (count is not None and report["frames"] != count):
            raise ValueError("Input ended before the requested observations were recorded")
        report["completed"] = True
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = str(exc) or type(exc).__name__
        raise
    finally:
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[REPORT] {out}  frames={report['frames']} completed={report['completed']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--can", help="Piper X SocketCAN channel, e.g. can0")
    source.add_argument("--replay", type=Path, help="Replay recorded steps.jsonl without an SDK")
    parser.add_argument("--checkpoint", type=Path, default=S1_CHECKPOINT)
    parser.add_argument("--calibration", type=Path, required=True, help="6 N handoff summary.json")

    parser.add_argument("--seconds", type=float, help="Live default 30 s; replay default entire file")
    parser.add_argument("--pose-tolerance", type=float, default=.10, help="Max observed joint error, radians")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs/hardware" /
                        ("student500_shadow_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    run(parser.parse_args())
