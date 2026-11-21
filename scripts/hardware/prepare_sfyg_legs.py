"""Read-only preview of the suspended leg preparation path."""
import argparse
import json
from pathlib import Path
import time

from shadow_sfyg_locomotion import Samples, Coordinates, subscribe
from sfyg_supported_prepare import SupportedPrepare, RECORDED, configuration


def run(args):
    cfg = configuration(args.bundle_dir)
    coords = Coordinates(args.profile)
    if coords.config.get("quaternion_order") != "wxyz":
        raise ValueError("Suspended preparation requires raw SDK wxyz")
    ctl = SupportedPrepare(cfg)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    (out/"interface.json").write_text(json.dumps(dict(
        profile=coords.config, policy_inference=False, motor_output=False,
        recorded_start=RECORDED.tolist(), default=ctl.default.tolist(),
        hardware_validated=False), indent=2), encoding="utf-8")
    samples = Samples()
    start = due = None
    last_stamp = None
    frames = 0
    reason = "completed"
    try:
        robot = subscribe(args.robot_ip, samples)  # Keep the SDK subscription owner alive.
        waiting = time.monotonic()
        with (out/"steps.jsonl").open("w", encoding="utf-8", buffering=1) as stream:
            while True:
                now = time.monotonic()
                sample = samples.snapshot(.04, .015)
                if sample is None:
                    if start is not None or now-waiting > 12:
                        raise RuntimeError("Missing fresh joint/IMU")
                    time.sleep(.002)
                    continue
                if start is None:
                    start = due = now
                if now-due > .03:
                    raise RuntimeError("Control deadline missed")
                _, joint, imu = sample
                stamp = joint[0], imu[0]
                if last_stamp and (stamp[0] == last_stamp[0] or stamp[1] == last_stamp[1]):
                    raise RuntimeError("Duplicate joint/IMU sample")
                last_stamp = stamp
                q, dq, _, gravity = coords.convert(joint, imu)
                packet = ctl.tick(now-start, q, dq, gravity)
                stream.write(json.dumps(dict(time_s=now-start, state=ctl.state, q=q.tolist(), dq=dq.tolist(),
                    gravity=gravity.tolist(), packet=packet, motor_output=False), allow_nan=False)+"\n")
                frames += 1
                if ctl.state == "FAULT":
                    raise RuntimeError(ctl.reason)
                if now-start >= 10:
                    preview = ctl.preview(q, dq, gravity)
                    (out/"preview.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")
                    print("[PREVIEW]", json.dumps(preview), flush=True)
                    break
                due += .02
                time.sleep(max(0., due-time.monotonic()))
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (out/"summary.json").write_text(json.dumps(dict(
            reason=reason, state=ctl.state, frames=frames, policy_calls=0,
            motor_output=False, events=ctl.events, hardware_validated=False), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle_dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--arm_status_file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--robot_ip", default="10.192.1.2")
    run(parser.parse_args())
