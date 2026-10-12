"""Replay recorded policy inputs and compare initial Isaac/MuJoCo observations."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


FIELDS = {"latent": (0, 3), "angular_velocity_scaled": (3, 6), "gravity": (6, 9),
          "leg_position_relative": (9, 19), "leg_velocity_scaled": (19, 29),
          "previous_action": (29, 39), "gait_phase": (39, 41), "gait_command": (41, 45),
          "wrench_prediction": (45, 75), "base_command": (75, 78)}


def load_zero(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    episode = next(item for item in manifest if item["condition"] == "zero")
    with np.load(directory / episode["file"]) as archive:
        required = ("policy_input", "policy_history", "locomotion_action")
        if any(key not in archive for key in required):
            raise ValueError(f"{directory}: rerun collection with policy input telemetry enabled.")
        arrays = {key: archive[key].astype(np.float32) for key in required}
    return episode, arrays


def replay(arrays, actor, encoder):
    with torch.inference_mode():
        inputs = torch.from_numpy(arrays["policy_input"])
        actions = actor(inputs).numpy()
        latent = encoder(torch.from_numpy(arrays["policy_history"])).numpy()
    return {"actor_replay_max_abs_error": float(np.abs(actions-arrays["locomotion_action"]).max()),
            "encoder_replay_max_abs_error": float(np.abs(latent-arrays["policy_input"][:, :3]).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac_dir", required=True)
    parser.add_argument("--mujoco_dir", required=True)
    parser.add_argument("--locomotion_dir", required=True)
    args = parser.parse_args()
    isaac_episode, isaac = load_zero(args.isaac_dir)
    mujoco_episode, mujoco = load_zero(args.mujoco_dir)
    directory = Path(args.locomotion_dir)
    with (directory / "policy.pt").open("rb") as stream:
        actor = torch.jit.load(stream, map_location="cpu").eval()
    with (directory / "encoder.pt").open("rb") as stream:
        encoder = torch.jit.load(stream, map_location="cpu").eval()
    report = {"isaac_episode": isaac_episode, "mujoco_episode": mujoco_episode,
              "isaac_replay": replay(isaac, actor, encoder), "mujoco_replay": replay(mujoco, actor, encoder),
              "first_policy_inputs": {},
              "note": "Replay verifies network export independently of dynamics. First-frame differences can reflect reset/history; later closed-loop trajectories are not identical-state comparisons."}
    for name, (start, end) in FIELDS.items():
        left, right = isaac["policy_input"][0, start:end], mujoco["policy_input"][0, start:end]
        report["first_policy_inputs"][name] = {"isaac": left.tolist(), "mujoco": right.tolist(),
                                              "max_abs_difference": float(np.abs(left-right).max())}
    path = Path(args.mujoco_dir) / "locomotion_input_comparison.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
