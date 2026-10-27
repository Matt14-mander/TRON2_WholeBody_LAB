"""Single SFYG sim-to-sim: frozen actor+velocity estimator, held arm and real load."""
import argparse
from contextlib import nullcontext
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import torch
import mujoco
import numpy as np

from sfyg_bridge import ROOT, SFYGBridge
from sfyg_policy import sha256


def command_at(time_s, requested, profile, duration):
    if profile == "constant":
        return requested.copy()
    # 3 s stand, 3 s acceleration, 3 s deceleration, 3 s final hold.
    factor = np.clip(min((time_s - 3.) / 3., (duration - 3. - time_s) / 3.), 0., 1.)
    return requested * factor


def run(args):
    cfg = json.loads((args.bundle_dir / "deployment.json").read_text(encoding="utf-8"))
    if cfg.get("schema") != "sfyg_mujoco_v1":
        raise ValueError("Use the bundle exported by export_sfyg.py")
    policy_path = args.bundle_dir / "policy.pt"
    if sha256(policy_path) != cfg["policy_sha256"]:
        raise ValueError("Policy SHA256 differs from deployment.json")
    with policy_path.open("rb") as stream:
        policy = torch.jit.load(stream, map_location="cpu").eval()
    torch.set_num_threads(1)
    requested = np.array(args.command, np.float32)
    if not np.isfinite(requested).all() or np.any(abs(requested) > cfg["command_limits"]):
        raise ValueError("Command exceeds source vx/vy/wz limits +/-1/.5/1.5")
    if not np.isfinite(args.duration) or args.duration <= 0 or (args.profile == "start-stop" and args.duration < 15):
        raise ValueError("Positive duration required; start-stop needs at least 15 s")
    if args.fps < 1 or not np.isfinite(args.force_world).all() or np.linalg.norm(args.force_world) > 2:
        raise ValueError("Positive fps and horizontal payload force norm <=2 N required")
    bridge = SFYGBridge(cfg, args.payload_mass, args.friction, args.delay_steps, args.seed)
    if bridge.payload is None and any(args.force_world):
        raise ValueError("Source phase has no load on which to apply force")
    output = args.output_dir or ROOT / "logs/mujoco" / ("sfyg_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    audit = bridge.audit()
    if max(audit["authored_mass_error_kg"], audit["authored_com_error_m"], audit["authored_inertia_error"]) > 1e-6:
        raise ValueError("Authored mass/COM/inertia audit failed")
    with torch.inference_mode():
        sample = policy(torch.zeros(1, 42), torch.zeros(1, 10, 42), torch.zeros(1, 3))
    if sample.shape != (1, 10) or not torch.isfinite(sample).all():
        raise ValueError("Export must produce finite ten leg actions")
    audit.update(policy_sha256=cfg["policy_sha256"], checkpoint_sha256=cfg["checkpoint_sha256"],
                 phase=cfg["phase"], physics_dt=cfg["physics_dt"], decimation=cfg["decimation"],
                 mujoco_version=mujoco.__version__, force_world_xy=args.force_world)
    (output / "interface.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"[MuJoCo] phase={cfg['phase']} mass={audit['total_mass_kg']:.4f} kg; "
          f"42+10x42+3 -> 10; control={1/(cfg['physics_dt']*cfg['decimation']):.0f}Hz; "
          f"delays={bridge.delays}; output={output}", flush=True)
    if args.check_only:
        print("[CHECK] model, policy and authored mass/inertia checks passed; no rollout performed")
        return output
    context = nullcontext(None)
    if not args.headless:
        from mujoco import viewer as mj_viewer
        context = mj_viewer.launch_passive(bridge.model, bridge.data)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 3., -18., 135.
    renderer = writer = None
    rows, trace = [], []
    reason, error = "completed", None
    initial_pos = bridge.data.xpos[bridge.base].copy()
    contacts_previous = None
    releases = np.zeros(2, int)
    feet = [bridge.model.body(name).id for name in ("ankle_pitch_L_Link", "ankle_pitch_R_Link")]
    next_frame = 0.
    try:
        if args.video:
            import imageio.v2 as imageio
            renderer = mujoco.Renderer(bridge.model, height=720, width=960)
            writer = imageio.get_writer(str(output / "preview.mp4"), fps=args.fps, codec="libx264")
        with (output / "steps.csv").open("w", newline="", encoding="utf-8") as stream, context as viewer:
            columns = ["time_s", "command_vx", "command_vy", "command_wz", "vx", "vy", "wz", "vz",
                       "x", "y", "height_m", "tilt_deg", "arm_error_rad", "force_x", "force_y",
                       "left_contact", "right_contact", "torque_saturation_fraction", "estimated_vx", "estimated_vy", "estimated_vz"]
            csv_writer = csv.DictWriter(stream, fieldnames=columns)
            csv_writer.writeheader()
            while bridge.data.time < args.duration - cfg["physics_dt"] / 2:
                started = time.monotonic()
                if viewer is not None and not viewer.is_running():
                    reason = "viewer_closed"
                    break
                command = command_at(bridge.data.time, requested, args.profile, args.duration)
                obs, history = bridge.observation()
                with torch.inference_mode():
                    tensors = (torch.from_numpy(obs[None]), torch.from_numpy(history[None]), torch.from_numpy(command[None]))
                    action = policy(*tensors)[0].cpu().numpy().copy()
                    estimate = policy.estimate_velocity(tensors[1])[0].cpu().numpy().copy()
                bridge.set_action(action)
                force = np.r_[args.force_world, 0.] * np.clip((bridge.data.time - .5) / .2, 0., 1.)
                for _ in range(cfg["decimation"]):
                    bridge.physics_step(force if bridge.payload is not None else None)
                bridge.control_steps += 1
                # Refresh link frames/velocities to the post-integration qpos.
                mujoco.mj_forward(bridge.model, bridge.data)
                linear, angular, gravity = bridge.base_state()
                pos = bridge.data.xpos[bridge.base]
                tilt = float(np.rad2deg(np.arccos(np.clip(-gravity[2], -1, 1))))
                held = bridge.data.qpos[bridge.qpos[10:16]] - bridge.defaults[10:16]
                contacting = np.zeros(2, bool)
                for contact in bridge.data.contact:
                    b1, b2 = bridge.model.geom_bodyid[[contact.geom1, contact.geom2]]
                    for i, body in enumerate(feet):
                        contacting[i] |= (b1 == body and b2 == 0) or (b2 == body and b1 == 0)
                if contacts_previous is not None:
                    releases += contacts_previous & ~contacting
                contacts_previous = contacting
                effort = np.array([a["effort"] for a in cfg["actuators"]])
                torque = bridge.data.actuator_force[bridge.motors]
                values = [float(bridge.data.time), *command, linear[0], linear[1], angular[2], linear[2],
                          pos[0], pos[1], pos[2], tilt, max(abs(held)), *force[:2], *contacting.astype(int),
                          np.mean(abs(torque) >= effort * .999), *estimate]
                row = dict(zip(columns, map(float, values)))
                csv_writer.writerow(row)
                rows.append(row)
                if args.trace:
                    trace.append(dict(observation=obs, history=history, commands=command, action=action,
                                      target=bridge.target.copy(), joint_pos=bridge.data.qpos[bridge.qpos].copy()))
                if len(rows) % 50 == 0 or len(rows) == 1:
                    print(f"[MJ] t={bridge.data.time:.2f} cmd={command.round(3)} "
                          f"actual(vx,vy,wz)={np.r_[linear[:2],angular[2]].round(3)} tilt={tilt:.2f}", flush=True)
                camera.lookat[:] = pos + [0, 0, .1]
                if writer is not None and bridge.data.time >= next_frame:
                    renderer.update_scene(bridge.data, camera)
                    writer.append_data(renderer.render())
                    next_frame += 1. / args.fps
                if viewer is not None:
                    viewer.cam.lookat[:] = camera.lookat
                    if len(rows) == 1:
                        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = camera.distance, camera.elevation, camera.azimuth
                    viewer.sync()
                    time.sleep(max(0., cfg["physics_dt"] * cfg["decimation"] - (time.monotonic() - started)))
                if not np.isfinite(bridge.data.qpos).all() or not np.isfinite(bridge.data.qvel).all():
                    reason = "nonfinite_state"
                    break
                if tilt > 45 or pos[2] < .45:
                    reason = "fall"
                    break
                # MuJoCo can reset unstable states internally. Never count this
                # as a successful rollout or keep looping on the reset time.
                if bridge.data.time < bridge.control_steps * cfg["physics_dt"] * cfg["decimation"] - 1e-6:
                    reason = "simulator_reset"
                    break
    except Exception as exc:
        reason, error = "error", repr(exc)
        raise
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        kept = [r for r in rows if r["time_s"] >= 1.]
        report = dict(kind="sfyg_mujoco_diagnostic", is_acceptance=False, reason=reason, error=error,
            completed=reason == "completed" and bridge.data.time >= args.duration - cfg["physics_dt"],
            duration_s=float(bridge.data.time), profile=args.profile, requested_command=requested.tolist(),
            displacement_world_m=(bridge.data.xpos[bridge.base] - initial_pos).tolist(),
            mean_vx_vy_wz_after_1s=[float(np.mean([r[k] for r in kept])) for k in ("vx", "vy", "wz")] if kept else None,
            rmse_vx_vy_wz_after_1s=[float(np.sqrt(np.mean([(r[k]-r[c])**2 for r in kept])))
                for k,c in zip(("vx","vy","wz"),("command_vx","command_vy","command_wz"))] if kept else None,
            max_tilt_deg=max((r["tilt_deg"] for r in rows), default=0),
            peak_arm_error_rad=max((r["arm_error_rad"] for r in rows), default=0),
            foot_contact_releases=releases.tolist(), interface=audit)
        (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if trace:
            np.savez_compressed(output / "trace.npz", **{key:np.array([r[key] for r in trace]) for key in trace[0]})
        print(f"[REPORT] reason={reason}, sim_time={bridge.data.time:.2f}s; {output / 'evaluation.json'}", flush=True)
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle_dir", type=Path, required=True)
    p.add_argument("--command", type=float, nargs=3, default=[.25, 0, 0])
    p.add_argument("--duration", type=float, default=20.)
    p.add_argument("--profile", choices=("constant", "start-stop"), default="constant")
    p.add_argument("--payload_mass", type=float)
    p.add_argument("--friction", type=float)
    p.add_argument("--delay_steps", type=int, choices=range(5))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_world", type=float, nargs=2, default=[0., 0.], metavar=("FX", "FY"))
    p.add_argument("--output_dir", type=Path)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--video", action="store_true")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--check_only", action="store_true")
    run(p.parse_args())
