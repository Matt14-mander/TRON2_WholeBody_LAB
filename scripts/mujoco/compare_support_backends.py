"""Compare steady fixed-arm C0 task conditions, retaining native physics differences."""
import argparse
import json
from pathlib import Path

import numpy as np


def rotation(q):
    w, x, y, z = np.asarray(q, dtype=float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def load(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def pose_summary(times, matrices, positions):
    times, matrices, positions = np.asarray(times), np.asarray(matrices), np.asarray(positions)
    if len(times) < 3 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Need at least three finite, strictly increasing pose timestamps")
    yaw = np.unwrap(np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0]))
    roll = np.degrees(np.arctan2(matrices[:, 2, 1], matrices[:, 2, 2]))
    pitch = np.degrees(np.arcsin(np.clip(-matrices[:, 2, 0], -1, 1)))
    rate = np.degrees(np.gradient(yaw, times))
    c, s = np.cos(yaw[0]), np.sin(yaw[0])
    delta = positions[-1]-positions[0]
    return dict(samples=len(times), duration_sim_s=float(times[-1]-times[0]),
        yaw_net_deg=float(np.degrees(yaw[-1]-yaw[0])),
        yaw_slope_deg_s=float(np.polyfit(times-times[0], np.degrees(yaw-yaw[0]), 1)[0]),
        yaw_rate_mean_deg_s=float(rate.mean()), yaw_rate_std_deg_s=float(rate.std()),
        roll_mean_deg=float(roll.mean()), roll_std_deg=float(roll.std()),
        pitch_mean_deg=float(pitch.mean()), pitch_std_deg=float(pitch.std()),
        displacement_initial_heading_m=[float(c*delta[0]+s*delta[1]), float(-s*delta[0]+c*delta[1]), float(delta[2])])


def compare(isaac_dir, mujoco_case, case, trial, output, start_s=5., end_s=29.):
    isaac_dir, mujoco_case, output = map(Path, (isaac_dir, mujoco_case, output))
    ic = json.loads((isaac_dir/"contract.json").read_text(encoding="utf-8"))
    mc = json.loads((mujoco_case/"control/contract.json").read_text(encoding="utf-8"))
    sc = json.loads((mujoco_case/"sim/task_contract.json").read_text(encoding="utf-8"))
    ia, ma = ic["args"], mc["arguments"]
    manifest = json.loads((Path(ma["model_root"])/"SFYG_TRON2A/v3_source.json").read_text(encoding="utf-8"))
    expected = [0., 0., 0.] if case == "hold" else [.15, 0., 0.]
    checks = dict(checkpoint_hash=ic["checkpoint_sha256"] == manifest["checkpoint_sha256"],
        fixed_arm_nominal=ia["arm_pose"] == "nominal" and ma["arm_pose"] == "configured",
        payload_attached=ia["payload_mode"] == "attached",
        payload_mass=bool(np.isclose(ia["payload_mass"], ma["payload_mass"])),
        payload_dimensions=bool(np.allclose(ia["payload_dimensions"], ma["payload_dimensions"])),
        payload_offset=bool(np.allclose(ia["payload_offset"], ma["payload_offset"])),
        gravity_preview=ia["low_wrench"] == ma["low_wrench"] == "gravity",
        no_mujoco_arm_ff=not ma["arm_gravity_compensation"],
        command_profile=ma["command_profile"] == "paint",
        fixed_command=bool(np.allclose(ma["command"], expected)),
        no_mujoco_external_force=sc["arguments"].get("external_wrench") is None)
    if not all(checks.values()):
        raise ValueError(f"C0 task conditions differ: {checks}")
    source = load(isaac_dir/f"{case}_trial{trial}.jsonl")
    ir = [r for r in source if start_s <= r["time_s"] <= end_s]
    if not ir or "base_quaternion_world_wxyz" not in ir[0]:
        raise ValueError("Need a new Isaac rollout with root quaternion logging")
    if any(np.linalg.norm(r["force_payload_local"]) > 1e-8 for r in ir):
        raise ValueError("C0 requires zero physical external interaction force")
    if any(not np.allclose(r["base_command"], expected, atol=1e-5) for r in ir):
        raise ValueError("Isaac command not settled within comparison window")
    arms = load(mujoco_case/"control/arm_steps.jsonl")
    if not arms or arms[0].get("monotonic_s") is None:
        raise ValueError("Need control monotonic timestamps")
    origin = arms[0]["monotonic_s"]
    native = load(mujoco_case/"sim/task_steps.jsonl")
    if not sc.get("attached") or sc.get("error"):
        raise ValueError("Invalid simulator recorder")
    if len({r["segment"] for r in native}) != 1:
        raise ValueError("Do not reset the simulator during a comparison trial")
    clock_a, clock_b = mc["clock"], sc["clock"]
    if clock_a["host"] != clock_b["host"] or abs(clock_a["wall_minus_monotonic_s"]-clock_b["wall_minus_monotonic_s"]) > 1:
        raise ValueError("MuJoCo controller and simulator clock sessions differ")
    mr = [r for r in native if start_s <= r["monotonic_s"]-origin <= end_s]
    ar = [r for r in arms if start_s <= r["monotonic_s"]-origin <= end_s]
    if not ar or not np.allclose(ar[0]["target"], ic["nominal_arm"], atol=1e-5):
        raise ValueError("Nominal arm targets differ")
    im = pose_summary([r["time_s"] for r in ir],
        [rotation(r["base_quaternion_world_wxyz"]) for r in ir], [r["base_position_world"] for r in ir])
    mm = pose_summary([r["sim_time_s"] for r in mr],
        [np.array(r["base_rotation_world"]).reshape(3, 3) for r in mr], [r["base_position_world_m"] for r in mr])
    im["arm_error_rms_rad"] = np.sqrt(np.mean((np.array([r["arm_q"] for r in ir])-
        np.array([r["arm_target"] for r in ir]))**2, axis=0)).tolist()
    mm["arm_error_rms_rad"] = np.sqrt(np.mean(np.array([r["position_error"] for r in ar])**2, axis=0)).tolist()
    im["base_velocity_body_mean"] = np.mean([r["base_velocity"] for r in ir], axis=0).tolist()
    # Same temporal window, but native sensor/actuator physics are deliberately retained.
    ie = load(isaac_dir/"episodes.jsonl")
    native_outcome = json.loads((mujoco_case/"control/arm_report.json").read_text())["outcome"]
    report = dict(stage="C0 task-condition diagnostic, not identical dynamics", case=case, trial=trial,
        window_control_s=[start_s, end_s], task_checks=checks,
        isaac=im, mujoco=mm, isaac_outcome=[r for r in ie if r["case"] == case and r["trial"] == trial],
        mujoco_outcome=native_outcome,
        isaac_first_wrench=source[0]["low_wrench_physical"][:6],
        limitations=["Isaac implicit PD at 200Hz versus native MuJoCo MIT PD at 1000Hz; no physics retuning in C0.",
            "Arm gravity feedforward OFF in both. Source .pt and exported ONNX already independently checked.",
            "Same continuous 3-4s quintic command reference; Isaac teacher samples targets at 25Hz, MuJoCo at 50Hz. Compare 5-29s steady window.",
            "Each backend computes gravity wrench from its own actual arm/payload asset. Native inertia/contact/actuator parameters may differ.",
            "MuJoCo time selection uses same-host controller monotonic origin; yaw slope uses native simulator time."])
    for backend in (im, mm):
        if backend["duration_sim_s"] < (end_s-start_s)*.9:
            raise ValueError("Insufficient steady-window coverage; inspect termination/outcome first")
    output.mkdir(parents=True, exist_ok=False)
    (output/"backend_comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-dir", required=True)
    parser.add_argument("--mujoco-case", required=True)
    parser.add_argument("--case", choices=("hold", "forward"), required=True)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-s", type=float, default=5.)
    parser.add_argument("--end-s", type=float, default=29.)
    args = parser.parse_args()
    compare(args.isaac_dir, args.mujoco_case, args.case, args.trial, args.output_dir, args.start_s, args.end_s)
