"""Validate semantic fields and excitation coverage in rollout NPZ files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = (
    "ocs2_arm_on_base_wrench_plan",
    "future_wrench",
    "external_wrench_payload_at_body_com",
    "external_wrench_base_at_base_origin",
    "external_wrench_active",
    "payload_proxy_com_position_world",
    "base_link_position_world",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir")
    args = parser.parse_args()

    root = Path(args.dataset_dir).expanduser().resolve()
    contract_path = root / "wrench_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"Missing wrench contract: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    files = sorted((root / "episodes").rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No episode NPZ files under {root / 'episodes'}")

    coverage = Counter()
    active_frames = 0
    total_frames = 0
    peak_payload = np.zeros(6, dtype=np.float64)
    peak_base = np.zeros(6, dtype=np.float64)
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            missing = [name for name in REQUIRED_FIELDS if name not in data]
            if missing:
                raise ValueError(f"{path} is missing fields: {missing}")
            if not np.array_equal(data["future_wrench"], data["ocs2_arm_on_base_wrench_plan"]):
                raise ValueError(f"Deprecated future_wrench alias differs in {path}")
            axis = str(data["meta_external_wrench_axis"])
            sign = int(data["meta_external_wrench_sign"])
            profile = str(data["meta_external_wrench_profile"])
            coverage[(axis, sign, profile)] += 1
            payload_wrench = data["external_wrench_payload_at_body_com"]
            base_wrench = data["external_wrench_base_at_base_origin"]
            active = data["external_wrench_active"].astype(bool)
            if payload_wrench.shape[1:] != (6,) or base_wrench.shape != payload_wrench.shape:
                raise ValueError(f"Unexpected wrench shape in {path}")
            if not np.array_equal(active, np.any(payload_wrench != 0.0, axis=1)):
                raise ValueError(f"external_wrench_active is inconsistent in {path}")
            active_frames += int(active.sum())
            total_frames += len(active)
            peak_payload = np.maximum(peak_payload, np.max(np.abs(payload_wrench), axis=0))
            peak_base = np.maximum(peak_base, np.max(np.abs(base_wrench), axis=0))

    print(f"contract_schema={contract['schema_version']} episodes={len(files)}")
    print(f"active_frames={active_frames}/{total_frames}")
    for condition, count in sorted(coverage.items()):
        print(f"condition={condition} episodes={count}")
    print("peak_payload_local=" + np.array2string(peak_payload, precision=4))
    print("peak_base_origin=" + np.array2string(peak_base, precision=4))


if __name__ == "__main__":
    main()
