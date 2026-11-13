"""Formal robust_5000 lifecycle test. Fixture ONLY during prepare and release."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hardware"))
from sfyg_lifecycle import Lifecycle, load_bundle
from sfyg_bridge import SFYGBridge


def run(args):
    cfg, policy = load_bundle(args.bundle_dir)
    env = SFYGBridge(cfg, delay_steps=args.delay_steps)
    ctl = Lifecycle(cfg, policy)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    # A test fixture holds the base while legs move from a bent preparation pose.
    # This is deliberately NOT a claim of autonomous rising from ground crouch.
    bent = np.array([-.15, .02, -3.14159, -.30, .17, -.11, -.03, 3.14159, -.26, .16])
    if args.initial_pose == "folded_supported":
        bent = np.array([-3.0014, .084, -3.0211, -2.62219, 1.5945,
                         -2.99739, .0836, 3.1288, -2.6189, 1.5955])
        try:
            ctl.tick(0., bent, np.zeros(10), np.zeros(3), [0,0,-1], "prepare_supported")
        except ValueError as exc:
            result = dict(passed=True, kind="expected_prepare_rejection", reason=str(exc),
                          simulated_motion=False, motor_output=False, hardware_validated=False)
            (out/"summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(result)
            return
        raise AssertionError("Deep folded preparation must be refused")
    env.data.qpos[env.qpos[:10]] = bent
    mujoco.mj_forward(env.model, env.data)
    root_pose = env.data.qpos[:7].copy()
    queue = deque(maxlen=args.delay_steps+1)
    rows, packet = [], None
    walk_start = stop_time = release_time = None
    last_state = None
    feet = [env.model.body(n).id for n in ("ankle_pitch_L_Link", "ankle_pitch_R_Link")]
    releases, last_contact = np.zeros(2, int), None
    with (out / "steps.jsonl").open("w", encoding="utf-8") as stream:
        for tick in range(int(55/cfg["physics_dt"])):
            now = tick*cfg["physics_dt"]
            if tick % cfg["decimation"] == 0:
                linear, angular, gravity = env.base_state()
                q = env.data.qpos[env.qpos[:10]].copy()
                dq = env.data.qvel[env.qvel[:10]].copy()
                event = "prepare_supported" if tick == 0 else None
                command = None
                if ctl.state == "READY" and ctl.ready:
                    event = "start_grounded"
                if ctl.state == "POLICY" and walk_start is None:
                    walk_start = now
                    command = [args.vx, 0., 0.]
                if walk_start is not None and now >= walk_start+12 and stop_time is None:
                    event, stop_time = "stop", now
                if ctl.state == "ZERO" and stop_time is not None and now >= stop_time+7:
                    event, release_time = "release_supported", now
                    root_pose = env.data.qpos[:7].copy()
                packet = ctl.tick(now, q, dq, angular, gravity, event, command)
                contacting = np.zeros(2, bool)
                for contact in env.data.contact:
                    b1, b2 = env.model.geom_bodyid[[contact.geom1, contact.geom2]]
                    for i, body in enumerate(feet):
                        contacting[i] |= (b1 == body and b2 == 0) or (b2 == body and b1 == 0)
                if last_contact is not None and ctl.state in ctl.POLICY_STATES:
                    releases += last_contact & ~contacting
                last_contact = contacting
                row = dict(time_s=now, state=ctl.state, q=q.tolist(), dq=dq.tolist(),
                           target=packet["q"], kp=packet["kp"], command=ctl.command.tolist(),
                           raw_action=ctl.raw.tolist(), applied_action=ctl.previous.tolist(),
                           velocity=linear.tolist(), angular=angular.tolist(),
                           tilt_deg=float(np.degrees(np.arccos(np.clip(-gravity[2], -1, 1)))),
                           height=float(env.data.xpos[env.base, 2]),
                           fixture=ctl.state not in ctl.POLICY_STATES)
                stream.write(json.dumps(row)+"\n")
                rows.append(row)
                if ctl.state != last_state:
                    print(f"[LIFECYCLE] t={now:.2f} state={ctl.state} ready={ctl.ready}", flush=True)
                    last_state = ctl.state
                if ctl.state in {"FAULT", "DONE"}:
                    break
            queue.append(packet)
            p = queue[0]
            ids = env.motors[:10]
            env.model.actuator_gainprm[ids, 0] = p["kp"]
            env.model.actuator_biasprm[ids, 1] = -np.array(p["kp"])
            env.model.actuator_biasprm[ids, 2] = -np.array(p["kd"])
            env.data.ctrl[ids] = p["q"]
            env.data.ctrl[env.motors[10:]] = env.defaults[10:]
            mujoco.mj_step(env.model, env.data)
            if ctl.state not in ctl.POLICY_STATES:
                env.data.qpos[:7] = root_pose
                env.data.qvel[:6] = 0
            mujoco.mj_forward(env.model, env.data)
    free = [r for r in rows if not r["fixture"]]
    plateau = [r for r in free if walk_start is not None and walk_start+5 <= r["time_s"] < walk_start+12]
    tail = [r for r in free if stop_time is not None and stop_time+5 <= r["time_s"] < stop_time+7]
    tail_rms = np.sqrt(np.mean(np.square([r["velocity"] for r in tail]), axis=0)).tolist() if tail else None
    mean_vx = float(np.mean([r["velocity"][0] for r in plateau])) if plateau else None
    max_tilt = max((r["tilt_deg"] for r in free), default=180.)
    boundaries = [dict(time_s=b["time_s"], transition=a["state"]+"->"+b["state"],
                       target_jump_rad=float(np.max(np.abs(np.array(b["target"])-a["target"]))))
                  for a, b in zip(rows, rows[1:]) if a["state"] != b["state"]]
    passed = (ctl.state == "DONE" and bool(plateau) and bool(tail) and max_tilt < 20
              and (mean_vx > .1 if args.vx else abs(mean_vx) < .05) and tail_rms[0] < .05)
    summary = dict(passed=passed, final_state=ctl.state, fault=ctl.reason, events=ctl.events,
                   policy_sha256=cfg["policy_sha256"], delay_steps=args.delay_steps,
                   initial_pose=args.initial_pose,
                   requested_vx=args.vx, mean_vx=mean_vx, tail_rms_xyz=tail_rms,
                   max_free_tilt_deg=max_tilt, foot_releases=releases.tolist(),
                   boundary_target_changes=boundaries,
                   free_policy_seconds=len(free)*.02, arm_and_payload_present=True,
                   hardware_validated=False, fixture="base pose constrained ONLY in prepare/release; free during policy/stop",
                   audit=env.audit())
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "audit"}, indent=2))
    if not passed:
        raise RuntimeError(f"Lifecycle gate failed: {out / 'summary.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--delay_steps", type=int, choices=range(5), default=0)
    p.add_argument("--vx", type=float, choices=[0., .2], default=.2)
    p.add_argument("--initial_pose", choices=["bent", "folded_supported"], default="bent")
    run(p.parse_args())
