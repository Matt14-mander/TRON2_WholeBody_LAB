"""Separate heading drift, payload inclination and arm-normal tracking offline."""
import argparse
import json
from pathlib import Path

import numpy as np


def rotation(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        raise ValueError("Need a finite nonzero wxyz quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def angles(vectors, reference):
    return np.degrees(np.arccos(np.clip(np.sum(vectors * reference, axis=-1), -1., 1.)))


def remove_heading(vectors, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack((c*vectors[:, 0]+s*vectors[:, 1],
                            -s*vectors[:, 0]+c*vectors[:, 1], vectors[:, 2]))


def stats(values):
    x = np.asarray(values)
    return dict(mean=float(x.mean()), std=float(x.std()), rms=float(np.sqrt(np.mean(x*x))),
                min=float(x.min()), max=float(x.max()), peak_abs=float(np.max(np.abs(x))),
                peak_to_peak=float(np.ptp(x)), p95_abs=float(np.percentile(np.abs(x), 95)))


def series(rows, axis=(0., 0., 1.), point="payload", settle_s=2.):
    if not np.isfinite(settle_s) or settle_s < 0:
        raise ValueError("settle_s must be finite and nonnegative")
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        raise ValueError("Need a finite nonzero payload normal axis")
    axis = axis/np.linalg.norm(axis)
    original = rows
    rows = [r for r in rows if r["time_s"] >= settle_s and r.get(point) is not None]
    if len(rows) < 3:
        raise ValueError("Need at least three settled samples for the selected point")
    if len({r["segment"] for r in rows}) != 1:
        raise ValueError("Cannot mix simulator reset segments")
    t = np.array([r["sim_time_s"] for r in rows])
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Simulator timestamps must increase strictly")
    rb = np.array([r["base_rotation_world"] for r in rows]).reshape(-1, 3, 3)
    if (not np.isfinite(rb).all() or not np.allclose(rb.transpose(0, 2, 1)@rb, np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(rb), 1., atol=1e-5)):
        raise ValueError("Invalid base rotation matrix")
    if np.any(np.hypot(rb[:, 0, 0], rb[:, 1, 0]) < 1e-6):
        raise ValueError("Base forward axis vertical: heading undefined")
    yaw = np.unwrap(np.arctan2(rb[:, 1, 0], rb[:, 0, 0]))
    roll = np.arctan2(rb[:, 2, 1], rb[:, 2, 2])
    pitch = np.arcsin(np.clip(-rb[:, 2, 0], -1., 1.))
    up = np.array([r[point]["up_world"] for r in rows])
    if not np.isfinite(up).all() or not np.allclose(np.linalg.norm(up, axis=1), 1., atol=1e-5):
        raise ValueError("Invalid recorded normal")
    heading_free = remove_heading(up, yaw)
    base_relative = np.einsum("nji,nj->ni", rb, up)
    target = np.array([rotation(r[point]["target_quaternion_world_wxyz"])@axis for r in rows])
    tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1., 1.)))
    target_tilt = np.degrees(np.arccos(np.clip(target[:, 2], -1., 1.)))
    # Keep the old first-control reference AND a common settled reference.
    first = next(r[point]["up_world"] for r in original if r.get(point) is not None)
    output = dict(time_s=np.array([r["time_s"] for r in rows]), sim_time_s=t,
        phase=np.array([r["phase"] for r in rows]),
        heading_delta_deg=np.degrees(yaw-yaw[0]),
        heading_rate_deg_s=np.degrees(np.gradient(yaw, t)),
        base_roll_deg=np.degrees(roll), base_pitch_deg=np.degrees(pitch),
        tilt_world_deg=tilt, target_tilt_world_deg=target_tilt,
        tilt_error_to_command_deg=tilt-target_tilt,
        normal_from_first_control_deg=angles(up, first),
        normal_from_settled_deg=angles(up, up[0]),
        heading_removed_normal_from_settled_deg=angles(heading_free, heading_free[0]),
        base_relative_normal_from_settled_deg=angles(base_relative, base_relative[0]),
        normal_error_to_command_deg=angles(up, target))
    positions = np.array([r["base_position_world_m"] for r in rows])
    output["base_displacement_initial_heading_m"] = remove_heading(positions-positions[0], np.full(len(yaw), yaw[0]))
    output["acceleration_norm_m_s2"] = np.array([np.linalg.norm(r[point]["acceleration_world_m_s2"])
        if r[point].get("acceleration_world_m_s2") is not None else np.nan for r in rows])
    return output


def summarize(data, mask):
    n = int(np.sum(mask))
    if not n:
        return None
    t = data["sim_time_s"][mask]
    heading = data["heading_delta_deg"][mask]
    elapsed = t[-1]-t[0]
    result = dict(samples=n, duration_sim_s=float(elapsed),
        heading_net_change_deg=float(heading[-1]-heading[0]),
        heading_linear_slope_deg_s=float(np.polyfit(t-t[0], heading, 1)[0]) if n > 1 else None,
        base_displacement_initial_heading_m=(data["base_displacement_initial_heading_m"][mask][-1]
            - data["base_displacement_initial_heading_m"][mask][0]).tolist())
    for key, values in data.items():
        if key in ("time_s", "sim_time_s", "phase", "base_displacement_initial_heading_m"):
            continue
        values = values[mask]
        result[key] = stats(values[np.isfinite(values)]) if np.isfinite(values).any() else None
    return result


def plot_series(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
    groups = [(["heading_delta_deg"], "Base heading change (deg)"),
              (["base_roll_deg", "base_pitch_deg"], "Base attitude (deg)"),
              (["tilt_world_deg", "target_tilt_world_deg"], "Normal inclination (deg)"),
              (["normal_from_settled_deg", "heading_removed_normal_from_settled_deg",
                "base_relative_normal_from_settled_deg"], "Normal change, settled ref (deg)"),
              (["normal_error_to_command_deg", "tilt_error_to_command_deg"], "Normal tracking error (deg)"),
              (["acceleration_norm_m_s2"], "Net linear acceleration (m/s2)")]
    for ax, (keys, label) in zip(axes, groups):
        for key in keys:
            ax.plot(data["time_s"], data[key], label=key)
        boundaries = np.flatnonzero(data["phase"][1:] != data["phase"][:-1])+1
        for i in boundaries:
            ax.axvline(data["time_s"][i], color="grey", alpha=.35, linestyle="--")
        ax.set_ylabel(label)
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Control elapsed time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze(steps, output, settle_s=2., point="payload", axis=None, plot=False):
    steps, output = Path(steps), Path(output)
    rows = [json.loads(line) for line in steps.read_text(encoding="utf-8").splitlines() if line.strip()]
    source = steps.parent / "task_report.json"
    source_report = json.loads(source.read_text(encoding="utf-8")) if source.is_file() else None
    if axis is None:
        if source_report is None:
            raise ValueError("Need sibling task_report.json for the normal axis, or provide --up-axis")
        axis = source_report["definitions"]["payload_up_axis"]
    data = series(rows, axis, point, settle_s)
    report = dict(source=str(steps.resolve()), point=point, up_axis=axis, settle_s=settle_s,
        arm_outcome=source_report.get("arm_outcome") if source_report else None,
        alignment_coverage=source_report.get("alignment_coverage") if source_report else None,
        definitions=dict(heading="Unwrapped ZYX yaw of base forward axis; rates use simulator time, not body angular velocity.",
            heading_removed="Rz(-base_yaw) times the recorded world normal; retains base roll/pitch and intended arm motion.",
            base_relative="Full measured base rotation removed; retains arm motion, including commanded motion.",
            tracking="Normal angle to received arm-target FK at measured base pose. Not a fixed-world PAINT target.",
            tilt="Absolute angle of chosen normal to world +Z. Std/peak-to-peak describe variation; RMS near 90deg is not wobble amplitude.",
            reference="Normal-change references use the first post-settle sample, except explicitly named first_control. Angles are not additive attribution percentages.",
            drift="Heading net change and slope are descriptive. For these straight-walk tests commanded yaw rate is zero."),
        settled=summarize(data, np.ones(len(data["time_s"]), dtype=bool)),
        phases={phase: summarize(data, data["phase"] == phase) for phase in sorted(set(data["phase"]))})
    output.mkdir(parents=True, exist_ok=False)
    (output/"heading_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    with (output/"heading_steps.jsonl").open("w", encoding="utf-8") as stream:
        for i in range(len(data["time_s"])):
            row = {k: v[i].tolist() for k, v in data.items()}
            if not np.isfinite(data["acceleration_norm_m_s2"][i]):
                row["acceleration_norm_m_s2"] = None
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    if plot:
        plot_series(data, output/"heading_payload.png")
    print(f"\n=== {steps} ===")
    for label, block in {"settled": report["settled"], **report["phases"]}.items():
        print(f"\n{label}: {block['samples']} samples")
        for key in ("heading_net_change_deg", "heading_linear_slope_deg_s", "base_displacement_initial_heading_m"):
            print(key, block[key])
        for key in ("heading_rate_deg_s", "base_roll_deg", "base_pitch_deg", "tilt_world_deg",
                    "target_tilt_world_deg", "normal_from_first_control_deg", "normal_from_settled_deg",
                    "heading_removed_normal_from_settled_deg", "base_relative_normal_from_settled_deg",
                    "normal_error_to_command_deg", "tilt_error_to_command_deg"):
            print(key, {k: round(block[key][k], 5) for k in ("mean", "std", "peak_abs", "peak_to_peak")})
    print(f"Saved: {output}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", required=True, help="Existing analysis/task_aligned.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--settle-s", type=float, default=2.)
    parser.add_argument("--point", choices=("payload", "ee"), default="payload")
    parser.add_argument("--up-axis", nargs=3, type=float)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    analyze(args.steps, args.output_dir, args.settle_s, args.point, args.up_axis, args.plot)
