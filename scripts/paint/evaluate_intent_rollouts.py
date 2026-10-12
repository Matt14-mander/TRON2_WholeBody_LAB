"""Summarize online estimator errors, including zero-only and rejected runs."""
import argparse
import json
from pathlib import Path
import numpy as np


def error_metrics(prediction, truth):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if prediction.shape != truth.shape or truth.ndim != 2 or truth.shape[1] != 3 or not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("Expected finite, aligned N x 3 wrench arrays.")
    if not len(truth):
        return {"samples": 0}
    error = prediction - truth
    active = np.any(np.abs(truth) > 1e-6, axis=1)
    result = {"samples": len(truth), "mae_N_N_Nm": np.abs(error).mean(0).tolist(),
              "signed_bias_N_N_Nm": error.mean(0).tolist(), "rmse_N_N_Nm": np.sqrt((error**2).mean(0)).tolist(),
              "zero_predictor_mae_N_N_Nm": np.abs(truth).mean(0).tolist(),
              "active_samples": int(active.sum()), "zero_samples": int((~active).sum())}
    for name, mask in (("active", active), ("zero", ~active)):
        result[name+"_mae_N_N_Nm"] = np.abs(error[mask]).mean(0).tolist() if mask.any() else None
    result["per_excited_axis_mae"] = {name: float(np.abs(error[np.abs(truth[:, axis]) > 1e-6, axis]).mean())
        if np.any(np.abs(truth[:, axis]) > 1e-6) else None for axis, name in enumerate(("fx", "fy", "mz"))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    args = parser.parse_args()
    root = Path(args.dataset).resolve()
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((root/"contract.json").read_text(encoding="utf-8"))
    episodes = []
    collected = {"accepted": [], "rejected": []}
    for record in manifest:
        with np.load(root/record["file"], allow_pickle=False) as data:
            mask = data["intent_estimate_valid"].astype(bool)
            if "response_valid" in data:
                mask &= data["response_valid"].astype(bool)
            predicted = data["intent_estimate"][mask]
            truth = data["interaction_wrench"][mask]
            item = {"episode_id": record["episode_id"], "accepted": record["accepted"], "reason": record["reason"],
                    **error_metrics(predicted, truth)}
            if "interval_start" in data:
                before = mask & (data["interval_start"] < contract["protocol"]["warmup"])
                item["before_force"] = error_metrics(data["intent_estimate"][before], data["interaction_wrench"][before])
        episodes.append(item)
        if len(truth):
            collected["accepted" if record["accepted"] else "rejected"].append((predicted, truth))
    summary = {"episodes": len(manifest), "accepted_episodes": sum(r["accepted"] for r in manifest),
               "backend": contract.get("backend", "unspecified"), "base_mode": contract.get("base_mode", "policy"),
               "note": "Errors are observation-only metrics; zero-only runs do not validate force estimation."}
    for name, values in collected.items():
        summary[name] = error_metrics(np.concatenate([v[0] for v in values]), np.concatenate([v[1] for v in values])) if values else {"samples": 0}
    report = {"summary": summary, "per_episode": episodes}
    (root/"intent_evaluation.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not any(collected.values()):
        raise SystemExit("No valid estimator outputs were recorded; check --model_dir/--intent_model_dir and history length.")


if __name__ == "__main__":
    main()
