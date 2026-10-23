"""Plot native transport telemetry: force/reference, actions/response and rewards."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path, help="Run directory containing rollout.jsonl")
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rows = [json.loads(line) for line in (args.run / "rollout.jsonl").read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("No transport telemetry was recorded.")
    t = np.arange(len(rows)) * .02
    reference = np.array([row.get("tracking_reference", row["reference"]) for row in rows])
    command = np.array([row["base_command"] for row in rows])
    actual = np.array([row["base_velocity"] for row in rows])
    force = np.array([row["force_world"][0] for row in rows])
    arm = np.array([row["arm_position"] for row in rows])
    target = np.array([row["arm_target"] for row in rows])
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True, constrained_layout=True)
    for j, name in enumerate(("Fx", "Fy", "Fz")):
        axes[0].plot(t, force[:, j], label=name)
    axes[0].set_ylabel("Payload force [N]")
    for j, name in enumerate(("vx", "vy", "wz")):
        axes[1].plot(t, reference[:, j], "--", label=f"reference {name}")
        axes[1].plot(t, actual[:, j], label=f"actual {name}")
        axes[2].plot(t, command[:, j], label=name)
    axes[1].set_ylabel("m/s or rad/s")
    axes[2].set_ylabel("Base command")
    for j in range(6):
        axes[3].plot(t, arm[:, j], label=f"q{j+1}")
        axes[3].plot(t, target[:, j], "--", alpha=.6)
    axes[3].set_ylabel("Arm q / dashed target [rad]")
    for name in rows[0]["rewards"]:
        axes[4].plot(t, [row["rewards"][name] for row in rows], label=name)
    axes[4].set_ylabel("Weighted reward / step")
    axes[4].set_xlabel("Concatenated control time [s]; red lines mark reset/terminal steps")
    for axis in axes:
        for i, row in enumerate(rows):
            if row["done_env0"]:
                axis.axvline(t[i], color="red", alpha=.2, linewidth=.7)
        axis.grid(alpha=.2)
        axis.legend(loc="upper right", ncol=3, fontsize=8)
    fig.suptitle("PAINT transport v4 — env 0 telemetry")
    path = args.run.resolve() / "transport_diagnostics.png"
    fig.savefig(path, dpi=150)
    print(path)


if __name__ == "__main__": main()
