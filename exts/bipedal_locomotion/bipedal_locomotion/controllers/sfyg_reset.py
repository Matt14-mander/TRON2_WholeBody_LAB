"""Source provenance and strict weights-only transfer, without simulator imports."""
import hashlib
from pathlib import Path
import subprocess
import torch

BASE = "f441d67"
SCHEMA = "sfyg_source_reset_v1"
CORE_FILES = (
    "rsl_rl/rsl_rl/algorithm/ppo.py",
    "rsl_rl/rsl_rl/runner/on_policy_runner.py",
    "rsl_rl/rsl_rl/modules/actor_critic.py",
    "rsl_rl/rsl_rl/modules/mlp_encoder.py",
    "rsl_rl/rsl_rl/storage/rollout_storage.py",
    "exts/bipedal_locomotion/bipedal_locomotion/utils/wrappers/rsl_rl/rl_mlp_cfg.py",
    "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py",
    "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF_TRON2A/limx_base_env_cfg.py",
    "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_solefoot_tron2a_env_cfg.py",
    "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_solefoot_yg_tron2a_env_cfg.py",
    "exts/bipedal_locomotion/bipedal_locomotion/assets/config/solefoot_yg_tron2a_cfg.py",
    "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/rewards.py",
    "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/observations.py",
    "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/events.py",
    "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/commands/gait_command.py",
    "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/commands/commands_cfg.py",
)


def verify_source(root):
    """Git blob identity tolerates Windows checkout CRLF; actual bytes are also logged."""
    root = Path(root)
    result = {}
    for name in CORE_FILES:
        expected = subprocess.check_output(["git", "rev-parse", f"{BASE}:{name}"], cwd=root).strip()
        actual = subprocess.check_output(["git", "hash-object", "--path=" + name, name], cwd=root).strip()
        if actual != expected:
            raise ValueError(f"Source core differs from {BASE}: {name}")
        result[name] = {"git_blob": expected.decode(),
                        "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
    return result


def load_weights(algorithm, saved, phase, evaluation=False):
    contract = saved.get("sfyg_reset_contract", {})
    if contract.get("schema") != SCHEMA or contract.get("source_commit") != BASE:
        raise ValueError("Use a checkpoint from train_sfyg_reset.py; old v2/v3/v4 are excluded")
    old = contract.get("phase")
    if evaluation and old != phase:
        raise ValueError("Evaluation phase must match the checkpoint")
    if not evaluation and (old, phase) not in (("source", "source"), ("source", "carry"), ("carry", "carry")):
        raise ValueError("Only source -> carry or same-phase weights-only transfer is supported")
    for key in ("model_state_dict", "encoder_state_dict"):
        if not all(torch.isfinite(v).all() for v in saved[key].values()):
            raise ValueError("Checkpoint contains nonfinite weights")
    target = algorithm.actor_critic.state_dict()
    model = saved["model_state_dict"]
    if not evaluation and old == "source" and phase == "carry":
        # Adding an articulated payload adds one mass entry to the original
        # privileged critic. Its input changes; reset the ENTIRE critic.
        # Actor, logstd and encoder must still match exactly; no partial actor.
        keys = [k for k in target if not k.startswith("critic.")]
        if set(model) != set(target):
            raise ValueError("Unexpected actor/critic checkpoint keys")
        if any(model[k].shape != target[k].shape for k in keys):
            raise ValueError("Source/carry actor shape mismatch")
        model = {k: model[k] if k in keys else v for k,v in target.items()}
        print("[RESET] source -> carry: actor+encoder transferred; entire critic freshly initialized", flush=True)
    algorithm.actor_critic.load_state_dict(model, strict=True)
    algorithm.encoder.load_state_dict(saved["encoder_state_dict"], strict=True)
