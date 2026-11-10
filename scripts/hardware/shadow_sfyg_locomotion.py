"""Read TRON2 state and record frozen SFYG inference. SDK access is subscription-only."""
import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mujoco"))
from sfyg_policy import LEG_NAMES, TERM_NAMES, WIDTHS, ObservationHistory, make_observation, sha256


def vec(values, size):
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {size} finite values")
    return result


class Samples:
    """Copy SDK-owned arrays inside callbacks. Timestamp the two arrivals independently."""
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {}
        self.error = None

    def store(self, name, msg, fields):
        received = time.monotonic()
        with self.lock:
            try:
                stamp = int(msg.stamp)
                previous = self.data.get(name)
                if previous and stamp < previous[0]:
                    raise ValueError(f"{name} SDK clock moved backwards")
                if previous and stamp == previous[0]:
                    return
                self.data[name] = (stamp, received, *(tuple(getattr(msg, f)) for f in fields))
            except Exception as exc:
                self.error = str(exc)

    def joint(self, msg):
        self.store("joint", msg, ("q", "dq"))

    def imu(self, msg):
        self.store("imu", msg, ("quat", "gyro"))

    def snapshot(self, max_age, max_skew):
        with self.lock:
            if self.error is not None:
                raise ValueError(self.error)
            if len(self.data) < 2:
                return None
            joint, imu = self.data["joint"], self.data["imu"]
            now = time.monotonic()
        if now - min(joint[1], imu[1]) > max_age or abs(joint[1] - imu[1]) > max_skew:
            return None
        return now, joint, imu


def subscribe(robot_ip, samples):
    import limxsdk.robot.Robot as Robot
    import limxsdk.robot.RobotType as RobotType

    robot = Robot(RobotType.Tron2)
    if not robot.init(robot_ip):
        raise RuntimeError(f"SDK init failed for {robot_ip}")
    if robot.subscribeRobotState(samples.joint) is False or robot.subscribeImuData(samples.imu) is False:
        raise RuntimeError("SDK state subscription failed")
    return robot


class Coordinates:
    def __init__(self, path):
        self.config = json.loads(path.read_text()) if path else dict(
            sdk_names=LEG_NAMES, zero_rad=[0.] * 10, sign=[1.] * 10,
            imu_to_base=np.eye(3).tolist())
        names = self.config["sdk_names"]
        if len(names) != 10 or set(names) != set(LEG_NAMES):
            raise ValueError("sdk_names must list each of the ten policy joint names exactly once")
        self.order = [names.index(name) for name in LEG_NAMES]
        self.zero = vec(self.config["zero_rad"], 10)
        self.sign = vec(self.config["sign"], 10)
        self.rotation = np.asarray(self.config["imu_to_base"], dtype=float)
        if not np.isin(self.sign, [-1., 1.]).all():
            raise ValueError("Expected joint directions +1 or -1")
        if (self.rotation.shape != (3, 3)
                or not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-6)
                or not np.allclose(np.cross(self.rotation[0], self.rotation[1]), self.rotation[2], atol=1e-6)):
            raise ValueError("imu_to_base must be a proper rotation matrix")

    def convert(self, joint, imu):
        q = (vec(joint[2], 10)[self.order] - self.zero) * self.sign
        dq = vec(joint[3], 10)[self.order] * self.sign
        quat = vec(imu[2], 4)
        norm = np.linalg.norm(quat)
        if not .9 <= norm <= 1.1:
            raise ValueError("Invalid SDK wxyz quaternion norm")
        w, x, y, z = quat / norm
        gravity = [2*(w*y-x*z), -2*(y*z+w*x), 2*(x*x+y*y)-1]
        return q, dq, self.rotation @ vec(imu[3], 3), self.rotation @ gravity


def run(args, connect=subscribe):
    bundle, out = args.bundle_dir, args.output_dir
    cfg = json.loads((bundle / "deployment.json").read_text())
    if (cfg.get("schema") != "sfyg_mujoco_v1" or cfg.get("source_commit") != "f441d67"
            or cfg.get("phase") != "robust" or cfg["leg_names"] != LEG_NAMES
            or [(t["name"], t["width"]) for t in cfg["terms"]] != list(zip(TERM_NAMES, WIDTHS))
            or sha256(bundle / "policy.pt") != cfg["policy_sha256"]):
        raise ValueError("Expected matching frozen robust SFYG export and leg observation contract")
    dt = cfg["physics_dt"] * cfg["decimation"]
    timing = [dt, args.duration, args.startup_timeout, args.max_age_ms, args.max_skew_ms]
    if not all(math.isfinite(x) and x > 0 for x in timing):
        raise ValueError("Durations and timing limits must be positive and finite")
    cmd = vec(args.command, 3).astype(np.float32)
    if np.any(abs(cmd) > cfg["command_limits"]):
        raise ValueError("Shadow command exceeds exported limits")
    coords = Coordinates(args.coordinates)
    torch.set_num_threads(1)
    with (bundle / "policy.pt").open("rb") as src:
        model = torch.jit.load(src, map_location="cpu").eval()
    command_tensor = torch.from_numpy(cmd[None])
    with torch.no_grad():
        for _ in range(3):
            warm = model(torch.zeros(1, 42), torch.zeros(1, 10, 42), command_tensor)
        if warm.shape != (1, 10) or not torch.isfinite(warm).all():
            raise ValueError("Invalid SFYG actor output")
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(policy_sha256=cfg["policy_sha256"], checkpoint_sha256=cfg["checkpoint_sha256"],
                    checkpoint=cfg["checkpoint_path"], policy_period_s=dt, robot_ip=args.robot_ip,
                    coordinates=coords.config, coordinate_source=str(args.coordinates or "identity_unverified"),
                    last_action_source="previous shadow prediction, initial zero",
                    max_age_ms=args.max_age_ms, max_skew_ms=args.max_skew_ms, motor_output=False)
    (out / "interface.json").write_text(json.dumps(metadata, indent=2))
    samples = Samples()
    history = ObservationHistory()
    previous = np.zeros(10, dtype=np.float32)
    count = 0
    gaps = 0
    segment_step = 0
    reason = "completed"
    print(f"[SHADOW] subscription only; coordinates={metadata['coordinate_source']}; output={out}", flush=True)
    try:
        robot = connect(args.robot_ip, samples)  # Keep SDK subscription owner alive throughout the run.
        start_wait = time.monotonic()
        start = None
        with (out / "steps.jsonl").open("x", buffering=1) as stream, torch.no_grad():
            while start is None or due < start + args.duration - 1e-9:
                sample = samples.snapshot(args.max_age_ms / 1000, args.max_skew_ms / 1000)
                if sample is None or (count and (sample[1][0] == last_joint or sample[2][0] == last_imu)):
                    deadline = start_wait + args.startup_timeout if start is None else due + dt
                    if time.monotonic() >= deadline:
                        raise RuntimeError("No fresh pair before deadline (missing, duplicate, stale or skewed)")
                    time.sleep(.001)
                    continue
                now, joint, imu = sample
                if start is None:
                    start = now
                    due = now
                elif now >= start + args.duration:
                    break
                if now - due >= dt:
                    gaps += 1
                    history.reset()
                    previous.fill(0)
                    segment_step = 0
                    due = now
                last_joint, last_imu = joint[0], imu[0]
                q, dq, angular, gravity = coords.convert(joint, imu)
                obs = make_observation(angular, gravity, q, dq, previous, segment_step, cfg)
                hist = history.push(obs)
                action = model(torch.from_numpy(obs[None]), torch.from_numpy(hist[None]), command_tensor)
                if action.shape != (1, 10) or not torch.isfinite(action).all():
                    raise ValueError("Expected ten finite shadow actions")
                previous = action[0].cpu().numpy().copy()
                row = dict(step=count, segment=gaps, segment_step=segment_step, time_s=now-start, monotonic_s=now,
                           joint_stamp=joint[0], imu_stamp=imu[0], joint_rx_s=joint[1], imu_rx_s=imu[1],
                           sdk_q=list(map(float, joint[2])), sdk_dq=list(map(float, joint[3])),
                           sdk_quat_wxyz=list(map(float, imu[2])), sdk_gyro=list(map(float, imu[3])),
                           q=q.tolist(), dq=dq.tolist(), gravity=gravity.tolist(), angular=angular.tolist(),
                           observation=obs.tolist(), history=hist.tolist(), command=cmd.tolist(),
                           action=previous.tolist())
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                count += 1
                segment_step += 1
                due += dt
                time.sleep(max(0., due - time.monotonic()))
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary = dict(completed=reason == "completed", frames=count, timing_gaps=gaps,
                       reason=reason, motor_output=False)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print("[SHADOW]", summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True, help="New directory")
    parser.add_argument("--robot_ip", default="10.192.1.2")
    parser.add_argument("--coordinates", type=Path, help="Optional joint and IMU coordinate mapping JSON")
    parser.add_argument("--command", nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument("--duration", type=float, default=10.)
    parser.add_argument("--startup_timeout", type=float, default=12.)
    parser.add_argument("--max_age_ms", type=float, default=40.)
    parser.add_argument("--max_skew_ms", type=float, default=15.)
    run(parser.parse_args())
