"""Train/export a PAINT-style estimator from MPC-free PD response datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers"))
from paint_intent import ARM_JOINT_NAMES, CONDITIONS, SCHEMA, make_history_samples


class IntentEstimator(nn.Module):
    """Exported forward accepts flattened SI history and returns [N, N, Nm]."""
    def __init__(self, mean, std, label_scale):
        super().__init__()
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std)
        self.register_buffer("label_scale", label_scale)
        self.history = len(mean) // 18
        self.network = nn.Sequential(nn.Linear(len(mean), 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(),
                                     nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 3))

    def forward(self, x):
        rows = x.reshape(-1, self.history, 18)
        # Tracking error carries the force response; retain the absolute command
        # as well for posture-dependent gravity/Jacobian effects.
        features = torch.cat((rows[:, :, :6] - rows[:, :, 12:18], rows[:, :, 6:18]), dim=2).reshape(-1, self.history * 18)
        return self.network((features - self.input_mean) / self.input_std) * self.label_scale


def load_groups(directories, history):
    groups = {}
    dt = None
    contracts = []
    for directory in directories:
        root = Path(directory).expanduser().resolve()
        contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
        if contract["schema"] != SCHEMA or contract["joint_names"] != list(ARM_JOINT_NAMES) or contract["label_frame"] != "payload_local" or contract["label_point"] != "payload_COM":
            raise ValueError(f"Unsupported dataset contract: {root}")
        if dt is not None and not np.isclose(dt, contract["step_dt"], rtol=0, atol=1e-8):
            raise ValueError("Do not mix control rates in one estimator.")
        dt = contract["step_dt"]
        contracts.append(contract)
        rows = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        by_group = {}
        for record in rows:
            by_group.setdefault(record["group_id"], []).append(record)
        for group_id, records in by_group.items():
            # Preserve all seven matched conditions; do not learn from survivor-only pairs.
            if len(records) != 7 or {r["condition"] for r in records} != set(CONDITIONS) or not all(r["accepted"] for r in records):
                continue
            for record in records:
                with np.load(root / record["file"], allow_pickle=False) as data:
                    if str(data["meta_schema"]) != SCHEMA or not bool(data["meta_accepted"]):
                        raise ValueError("Manifest and episode metadata disagree.")
                    x, y, _ = make_history_samples(data["arm_position"], data["arm_velocity"], data["applied_position"],
                                                   data["interaction_wrench"], data["response_valid"], history)
                if len(x):
                    # Same seed/hold posture stays in the same split even across mass/speed datasets.
                    groups.setdefault(group_id, []).append((x, y))
    if len(groups) < 5:
        raise ValueError("Need at least 5 complete independent pair groups; collect --repeats 30 for an initial model.")
    return groups, dt, contracts


def metrics(model, x, y, device):
    with torch.inference_mode():
        predictions = np.concatenate([model(torch.from_numpy(batch).to(device)).cpu().numpy() for batch in np.array_split(x, max(1, int(np.ceil(len(x) / 4096))))])
    error = predictions - y
    active = np.any(np.abs(y) > 1e-6, axis=1)
    result = {"samples": len(x), "mae_N_N_Nm": np.abs(error).mean(0).tolist(), "rmse_N_N_Nm": np.sqrt((error ** 2).mean(0)).tolist(),
              "zero_predictor_mae_N_N_Nm": np.abs(y).mean(0).tolist()}
    for name, mask in (("active", active), ("zero", ~active)):
        result[name + "_samples"] = int(mask.sum())
        result[name + "_mae_N_N_Nm"] = np.abs(error[mask]).mean(0).tolist() if mask.any() else None
    result["per_excited_axis_mae"] = {
        name: float(np.abs(error[np.abs(y[:, index]) > 1e-6, index]).mean()) if np.any(np.abs(y[:, index]) > 1e-6) else None
        for index, name in enumerate(("fx", "fy", "mz"))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.history < 1 or args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Invalid training configuration.")
    groups, dt, contracts = load_groups(args.datasets, args.history)
    keys = sorted(groups)
    generator = np.random.default_rng(args.seed)
    generator.shuffle(keys)
    n_val = n_test = max(1, int(len(keys) * 0.1))
    splits = {"train": keys[:len(keys)-n_val-n_test], "validation": keys[len(keys)-n_val-n_test:len(keys)-n_test], "test": keys[len(keys)-n_test:]}
    arrays = {}
    for name, ids in splits.items():
        pairs = [pair for group_id in ids for pair in groups[group_id]]
        arrays[name] = (np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs]))
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty model output directory.")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device(args.device)
    train_x, train_y = arrays["train"]
    label_scale = np.maximum(np.abs(train_y).max(0), 1e-3)
    rows = train_x.reshape(-1, args.history, 18)
    features = np.concatenate((rows[:, :, :6] - rows[:, :, 12:18], rows[:, :, 6:18]), axis=2).reshape(-1, args.history * 18)
    scale_floor = np.tile(np.array([1e-3] * 12 + [.1] * 6, dtype=np.float32), args.history)
    model = IntentEstimator(torch.from_numpy(features.mean(0)), torch.from_numpy(np.maximum(features.std(0), scale_floor)),
                            torch.from_numpy(label_scale)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    x = torch.from_numpy(train_x)
    y = torch.from_numpy(train_y)
    validation_x, validation_y = (torch.from_numpy(a).to(device) for a in arrays["validation"])
    best_loss = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(x))
        for ids in order.split(args.batch_size):
            optimizer.zero_grad()
            loss = (((model(x[ids].to(device)) - y[ids].to(device)) / model.label_scale) ** 2).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_loss = float((((model(validation_x) - validation_y) / model.label_scale) ** 2).mean())
        if not np.isfinite(val_loss):
            raise RuntimeError("Training diverged.")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"epoch={epoch+1} validation_normalized_mse={val_loss:.6f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    report = {name: metrics(model, *data, device) for name, data in arrays.items()}
    report["offline_baseline_gate_passed"] = bool(np.all(np.array(report["test"]["mae_N_N_Nm"]) < np.array(report["test"]["zero_predictor_mae_N_N_Nm"])))
    report["gate_meaning"] = "Held-out overall MAE beats a constant-zero predictor in all three channels; not a deployment approval."
    model = model.cpu()
    exported = torch.jit.script(model)
    with (output / "intent_estimator.ts").open("wb") as stream:
        torch.jit.save(exported, stream)
    # Verify the serialized artifact, not just the in-memory module.
    with (output / "intent_estimator.ts").open("rb") as stream:
        reloaded = torch.jit.load(stream)
    example = torch.from_numpy(arrays["test"][0][:8])
    torch.testing.assert_close(reloaded(example), model(example))
    with (output / "weights.pt").open("wb") as stream:
        torch.save(best_state, stream)
    metadata = {"schema": SCHEMA, "history": args.history, "input_size": args.history * 18, "step_dt": dt,
                "joint_names": list(ARM_JOINT_NAMES), "feature_order": "oldest to newest; each row q(6),dq(6),previous_applied_position(6)",
                "input_units": "rad, rad/s, rad", "output_order": ["Fx_N", "Fy_N", "Mz_Nm"], "output_frame": "payload_local",
                "output_point": "payload_COM", "output_semantics": "external_on_payload during preceding control interval",
                "deployment": "observation only; no robot command output", "normalization": "tracking error preprocessing and normalization embedded in TorchScript", "training_args": vars(args)}
    for filename, payload in (("metadata.json", metadata), ("metrics.json", report), ("splits.json", splits), ("source_contracts.json", contracts)):
        (output / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["test"], indent=2))
    print(f"Exported observation-only estimator: {output / 'intent_estimator.ts'}")


if __name__ == "__main__":
    main()
