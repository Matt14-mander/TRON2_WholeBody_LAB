"""Preview a measured-start Piper X joint-space path; explicitly confirm each segment.

No collision planner: an operator must review the entire physical swept path.
Default is passive CAN reception, including when the measured start is out of limits.
Final HOLD uses the existing leg interlock; do not restart another arm sender.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import hold_sfyg_arm as arm

MAPPING_KEYS = ("sdk_to_sim", "sign", "zero_rad", "sdk_min_rad", "sdk_max_rad")


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def binding(data, target):
    return digest(dict(mapping={k: data[k] for k in MAPPING_KEYS}, target=target.tolist(),
                       waypoints=data.get("sdk_waypoints_rad", [])))


def build_plan(profile, data, q):
    """Subdivide reviewed joint waypoints; straight interpolation is not obstacle avoidance."""
    q = arm.vector(q)
    points = [arm.vector(v) for v in data.get("sdk_waypoints_rad", [])]
    points.append(profile.target.copy())
    segments, origin = [], q.copy()
    for point in points:
        count = max(1, int(np.ceil(np.max(np.abs(point-origin))/.2)))
        start = origin.copy()
        for i in range(1, count+1):
            segments.append((start+(point-start)*(i/count)).tolist())
        origin = point
    violations = []
    for index, point in enumerate([q.tolist()] + segments):
        for j, value in enumerate(point):
            if not profile.low[j] <= value <= profile.high[j]:
                violations.append(dict(point=index, joint=j+1, rad=value,
                                       low=float(profile.low[j]), high=float(profile.high[j])))
    return dict(schema="sfyg_arm_prepare_v1", policy_sha256=arm.POLICY_SHA,
                binding=binding(data, profile.target), start_sdk_rad=q.tolist(),
                segments_sdk_rad=segments, violations=violations,
                collision_checked=False, max_segment_rad=.2, max_speed_rad_s=.05)


def load_plan(path, sha, profile, data):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("Reviewed plan SHA256 mismatch")
    plan = json.loads(raw)
    if (plan.get("schema") != "sfyg_arm_prepare_v1"
            or plan.get("policy_sha256") != arm.POLICY_SHA
            or plan.get("binding") != binding(data, profile.target)):
        raise ValueError("Plan does not match current policy/mapping/limits")
    points = [arm.vector(plan["start_sdk_rad"])]
    points += [arm.vector(v) for v in plan["segments_sdk_rad"]]
    if len(points) < 2 or len(points) > 200:
        raise ValueError("Invalid plan length")
    for point in points:
        profile.check_limits(point)
    if any(np.max(np.abs(b-a)) > .200001 for a, b in zip(points, points[1:])):
        raise ValueError("Plan segment exceeds 0.2 rad")
    if not np.allclose(points[-1], profile.target, atol=1e-8, rtol=0):
        raise ValueError("Plan must end at mapped carry target")
    return plan


class Segmented:
    instructions = ("Preview: no motor output. Execute: next N supported confirms exactly one segment. "
                    "Wait for PAUSED before the next confirmation. quit supported exits; final HOLD stays alive.")

    def __init__(self, profile, args, data):
        self.profile, self.args, self.data = profile, args, data
        self.state, self.ready = "WAIT", False
        self.index, self.stable, self.plan = 0, None, None
        self.current = None
        if args.execute:
            if (data.get("staged_path_reviewed") is not True
                    or not str(data.get("mapping_evidence", "")).strip()
                    or not str(data.get("limits_evidence", "")).strip()):
                raise ValueError("Require staged_path_reviewed and mapping/limits evidence in profile")
            if not args.plan_sha256:
                raise ValueError("Execute requires explicit --plan_sha256 from reviewed preview")
            self.plan = load_plan(args.plan_file, args.plan_sha256, profile, data)

    def preview(self, q):
        plan = build_plan(self.profile, self.data, q)
        self.args.plan_file.parent.mkdir(parents=True, exist_ok=True)
        # Never silently overwrite a reviewed preview.
        with self.args.plan_file.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, indent=2, allow_nan=False)
        print("[PREVIEW] measured SDK degrees:", np.degrees(q).round(3).tolist(), flush=True)
        sim = np.empty(6)
        sim[self.profile.order] = (q-self.profile.zero)/self.profile.sign
        print("[PREVIEW] inferred simulation degrees (UNVERIFIED mapping):",
              np.degrees(sim).round(3).tolist(), flush=True)
        for i, point in enumerate(plan["segments_sdk_rad"], 1):
            print(f"[SEGMENT {i}] SDK deg={np.degrees(point).round(3).tolist()}", flush=True)
        print("[LIMIT CONFLICTS]", plan["violations"], flush=True)
        print("[PLAN]", self.args.plan_file, "SHA256="+
              hashlib.sha256(self.args.plan_file.read_bytes()).hexdigest(), flush=True)
        print("No collision guarantee. Out-of-limit start cannot execute; verify calibration first.", flush=True)
        self.plan = plan

    def handle(self, command, q, dq, now):
        expected = f"next {self.index+1} supported"
        if not self.args.execute:
            print("[REJECTED] Passive preview; no motor sender", flush=True)
            return False
        if command != expected or self.state not in {"WAIT", "PAUSED"}:
            print(f"[REJECTED] state={self.state}; expected '{expected}' at WAIT/PAUSED", flush=True)
            return False
        self.profile.check_limits(q)
        if max(abs(dq)) > .05:
            raise RuntimeError("Arm must be stationary before segment confirmation")
        prior = (self.plan["start_sdk_rad"] if self.index == 0
                 else self.plan["segments_sdk_rad"][self.index-1])
        if max(abs(q-arm.vector(prior))) > .02:
            raise RuntimeError("Measured start changed >0.02 rad; generate/review a new plan")
        self.origin = q.copy()
        self.target = arm.vector(self.plan["segments_sdk_rad"][self.index])
        self.started, self.stable = now, None
        self.duration = max(4., 1.875*max(abs(self.target-self.origin))/.05)
        prime = self.index == 0
        self.ack_delay = 2. if prime else 0.
        self.state = "MOVING"
        print(f"[MOVE] segment={self.index+1} duration={self.duration:.2f}s", flush=True)
        return prime

    def tick(self, q, dq, enabled, mode, now):
        if not self.args.execute:
            if self.plan is None:
                self.preview(q)
            return None
        if self.state == "WAIT":
            return None
        self.profile.check_limits(q)
        elapsed = now-self.started
        if elapsed > self.ack_delay and not (enabled and mode):
            raise RuntimeError("Arm enable/CAN MOVE J acknowledgement lost")
        if max(abs(dq)) > .20:
            raise RuntimeError("Arm overspeed")
        if self.state in {"PAUSED", "HOLD"}:
            if max(abs(q-self.current)) > .04:
                raise RuntimeError("Segment hold lost")
            return self.current.copy()
        x = float(np.clip((elapsed-self.ack_delay)/self.duration, 0, 1))
        alpha = x*x*x*(10+x*(-15+6*x))
        target = self.origin+alpha*(self.target-self.origin)
        self.profile.check_limits(target)
        if max(abs(q-target)) > .12:
            raise RuntimeError("Segment tracking error")
        settled = (x == 1 and enabled and mode and max(abs(q-self.target)) < .02
                   and max(abs(dq)) < .03)
        self.stable = (now if self.stable is None else self.stable) if settled else None
        if self.stable is not None and now-self.stable >= 1:
            self.index += 1
            self.current = self.target.copy()
            self.ready = self.index == len(self.plan["segments_sdk_rad"])
            self.state = "HOLD" if self.ready else "PAUSED"
            print(f"[SEGMENT DONE] {self.index}/{len(self.plan['segments_sdk_rad'])} {self.state}", flush=True)
        if elapsed > self.ack_delay+self.duration+10:
            raise RuntimeError("Segment settle timeout")
        return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle_dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--plan_file", type=Path, required=True,
                        help="New output in preview; existing reviewed plan in execute")
    parser.add_argument("--plan_sha256")
    parser.add_argument("--status_file", type=Path, required=True)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if hashlib.sha256((args.bundle_dir / "policy.pt").read_bytes()).hexdigest() != arm.POLICY_SHA:
        raise ValueError("Expected unchanged formal robust_5000 policy.pt")
    data = json.loads(args.profile.read_text(encoding="utf-8"))
    args.controller_factory = lambda profile: Segmented(profile, args, data)
    arm.run(args)


if __name__ == "__main__":
    main()
