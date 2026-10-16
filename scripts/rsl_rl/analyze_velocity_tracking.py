"""Separate mean tracking bias from periodic velocity fluctuations (no simulator).

Period averaging supplements instantaneous RMSE; it does not change acceptance
criteria. Use fixed-command cases only. The period is supplied, not inferred.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

FIXED_CASES = ("stand", "walk", "slow", "forward", "left", "right", "yaw_left", "yaw_right")


def analyze(rows, settle_s, frequency):
    rows = sorted((r for r in rows if r["time_s"] >= settle_s), key=lambda r: r["time_s"])
    if len(rows) < 4:
        raise ValueError("Need at least four samples after settling")
    t = np.array([r["time_s"] for r in rows])
    command = np.array([r["command"] for r in rows])
    velocity = np.array([r["velocity"][:2] for r in rows])
    yaw = np.array([r["yaw_rate"] for r in rows])
    if not all(np.isfinite(v).all() for v in (t, command, velocity, yaw)):
        raise ValueError("Non-finite time/command/velocity")
    if np.any(np.diff(t) <= 0) or command.shape != (len(t), 3):
        raise ValueError("Invalid time ordering or command dimensions")
    if np.max(np.abs(command - command[0])) > 1e-5:
        raise ValueError("This analysis requires a fixed command, not an entire transition episode")
    dt = float(np.median(np.diff(t)))
    error = velocity - command[:, :2]
    bias = error.mean(axis=0)
    fluctuation = error.std(axis=0)

    # Complete, non-overlapping time windows of one known gait period. Windows
    # can contain alternating 62/63 samples at 50Hz for the 0.8Hz gait.
    period = 1. / frequency
    first = math.ceil((t[0] - 1e-8) / period)
    last = math.floor((t[-1] + dt - 1e-8) / period)
    means = []
    for index in range(first, last):
        lo, hi = index * period, (index + 1) * period
        mask = (t >= lo - 1e-8) & (t < hi - 1e-8)
        sample_t = t[mask]
        if (len(sample_t) < 4 or sample_t[0] > lo + 1.1 * dt
                or sample_t[-1] < hi - 1.1 * dt or len(sample_t) < .9 * period / dt):
            continue
        means.append(error[mask].mean(axis=0))

    # Interpolate onto a uniform grid for FFT if sample spacing is irregular.
    uniform_t = np.arange(t[0], t[-1] + dt * .25, dt)
    vy = np.interp(uniform_t, t, velocity[:, 1])
    centered = vy - vy.mean()
    frequencies = np.fft.rfftfreq(len(centered), dt)
    power = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
    band = (frequencies >= .05) & (frequencies <= min(10., .5 / dt))
    peak = None
    if centered.std() > 1e-3 and band.any():
        peak = float(frequencies[np.flatnonzero(band)[np.argmax(power[band])]])
    summary = dict(samples=len(rows), command=command[0].tolist(),
        mean_velocity_xy=velocity.mean(axis=0).tolist(), mean_error_xy=bias.tolist(),
        error_std_xy=fluctuation.tolist(),
        instantaneous_xy_rmse=float(np.sqrt((error ** 2).sum(axis=1).mean())),
        mean_bias_norm=float(np.linalg.norm(bias)), fluctuation_norm=float(np.linalg.norm(fluctuation)),
        vy_variance_share_of_xy_mse=float(fluctuation[1] ** 2 / max((error ** 2).sum(axis=1).mean(), 1e-15)),
        period_s=period, complete_period_windows=len(means),
        period_mean_xy_rmse=float(np.sqrt(np.square(means).sum(axis=1).mean())) if means else None,
        yaw_mean=float(yaw.mean()), yaw_mean_error=float((yaw - command[:, 2]).mean()),
        yaw_rmse=float(np.sqrt(np.square(yaw - command[:, 2]).mean())),
        vy_dominant_frequency_hz=peak)
    return summary, (rows, t, command, velocity, yaw, frequencies, power)


def plot_series(path, data, frequency):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, t, command, velocity, yaw, frequencies, power = data
    contacts = all("foot_normal_force" in r for r in rows)
    fig, axes = plt.subplots(5 if contacts else 4, 1, figsize=(11, 12 if contacts else 10))
    for i, label in enumerate(("vx (m/s)", "vy (m/s)")):
        axes[i].plot(t, velocity[:, i], label="measured", linewidth=.8)
        axes[i].plot(t, command[:, i], "--", label="command")
        axes[i].set_ylabel(label)
        axes[i].set_xlabel("time (s)")
    axes[2].plot(t, yaw, label="measured", linewidth=.8)
    axes[2].plot(t, command[:, 2], "--", label="command")
    axes[2].set_ylabel("yaw rate (rad/s)")
    axes[2].set_xlabel("time (s)")
    axes[3].plot(frequencies, power, label="detrended vy spectrum")
    for multiple in (1, 2):
        axes[3].axvline(multiple * frequency, linestyle="--", color="gray",
                        label=f"{multiple} x configured gait frequency")
    axes[3].set_xlim(0, 5)
    axes[3].set_xlabel("frequency (Hz)")
    axes[3].set_ylabel("FFT power (relative)")
    if contacts:
        force = np.array([r["foot_normal_force"] for r in rows])
        axes[4].plot(t, force[:, 0], label="left foot Fz", linewidth=.8)
        axes[4].plot(t, force[:, 1], label="right foot Fz", linewidth=.8)
        axes[4].set_xlabel("time (s)")
        axes[4].set_ylabel("normal force (N)")
    for axis in axes:
        axis.legend(loc="upper right", fontsize=8)
        axis.grid(alpha=.25)
    fig.suptitle(path.stem)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--settle_s", type=float, default=2.)
    parser.add_argument("--gait_frequency", type=float, default=.8)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.settle_s) or args.settle_s < 0 or not math.isfinite(args.gait_frequency) or args.gait_frequency <= 0:
        parser.error("settle_s must be nonnegative; gait_frequency must be positive (finite)")
    source = Path(args.steps).expanduser().resolve()
    groups = defaultdict(list)
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["case"] in FIXED_CASES:
                groups[(row["case"], row["trial"])].append(row)
    if not groups:
        raise ValueError("No fixed-command evaluation cases found")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for (case, trial), rows in groups.items():
        result, data = analyze(rows, args.settle_s, args.gait_frequency)
        result.update(case=case, trial=trial)
        results.append(result)
        block = result["period_mean_xy_rmse"]
        block_text = f"{block:.4f}" if block is not None else "unavailable"
        print(f"{case:12s} trial={trial} raw={result['instantaneous_xy_rmse']:.4f} "
              f"bias={result['mean_bias_norm']:.4f} period_mean={block_text} "
              f"vy_std={result['error_std_xy'][1]:.4f} vy_peak_hz={result['vy_dominant_frequency_hz']}")
        if args.plot and trial == 0:
            plot_series(output / f"{case}_trial0.png", data, args.gait_frequency)
    (output / "analysis.json").write_text(json.dumps(dict(
        source=str(source), settle_s=args.settle_s, configured_gait_frequency=args.gait_frequency,
        note="Period-window averages supplement raw RMSE; original pass criteria remain unchanged.",
        results=results), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Analysis saved: {output}")


if __name__ == "__main__":
    main()
