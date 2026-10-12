"""Check complete PD force pairs and quantify arm response vs matched zero."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    args = parser.parse_args()
    root = Path(args.dataset).resolve()
    records = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    report = {"episodes": len(records), "accepted": sum(r["accepted"] for r in records), "pairs": [], "incomplete_groups": []}
    groups = {}
    for r in records:
        groups.setdefault(r["group_id"], {})[r["condition"]] = r
    for group_id, group in groups.items():
        if len(group) != 7 or not all(r["accepted"] for r in group.values()):
            report["incomplete_groups"].append(group_id)
            continue
        with np.load(root / group["zero"]["file"], allow_pickle=False) as baseline:
            for name, r in group.items():
                if name == "zero":
                    continue
                with np.load(root / r["file"], allow_pickle=False) as data:
                    if not np.allclose(data["time"], baseline["time"]) or not np.array_equal(data["applied_position"], baseline["applied_position"]):
                        raise ValueError("Pair timestamps/PD commands differ.")
                    active = np.any(np.abs(data["interaction_wrench"]) > 1e-6, axis=1)
                    if not active.any():
                        raise ValueError("Excited condition contains no applied wrench; check episode timing.")
                    delta_q = data["arm_position"] - baseline["arm_position"]
                    delta_dq = data["arm_velocity"] - baseline["arm_velocity"]
                    result = {"group": group_id, "condition": name,
                        "active_arm_position_rms_rad": np.sqrt((delta_q[active] ** 2).mean(0)).tolist(),
                        "active_arm_velocity_rms_rad_s": np.sqrt((delta_dq[active] ** 2).mean(0)).tolist(),
                        "pre_force_position_rms_rad": float(np.sqrt((delta_q[data["time"] <= data["interval_start"][np.flatnonzero(active)[0]]] ** 2).mean())),
                        "max_tilt_deg": float(data["tilt_deg"].max()), "min_joint_margin_rad": float(data["joint_margin"].min())}
                    if "intent_estimate_valid" in data:
                        mask = data["intent_estimate_valid"].astype(bool)
                        if mask.any():
                            result["intent_mae_N_N_Nm"] = np.abs(data["intent_estimate"][mask]-data["interaction_wrench"][mask]).mean(0).tolist()
                    report["pairs"].append(result)
    (root / "pair_analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pairs"]:
        raise SystemExit("No complete accepted seven-condition group; fix collection before training.")


if __name__ == "__main__":
    main()
