"""Dedicated transport v4 Student runner; no change to locomotion PPO/runner."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import torch
from ..algorithm.paint_ppo import PaintPPO
from ..modules import ActorCritic
from ..modules.intent_estimator import IntentEstimator, StudentInference

STUDENT_SCHEMA = "paint_transport_student_v1"
STUDENT_IO = {"history": 8, "row": "q-nominal6,dq6,previous_applied_target-nominal6",
              "order": "oldest_to_newest", "input": 144, "estimate": 3, "actor_input": 147,
              "critic_input": 177, "action": 9, "wrench_scale": [40., 40., 10.],
              "wrench_frame": "base", "label": "current_c_at_pre_action_observation",
              "policy_distribution": "tanh_normal", "estimate_detached": True}


def fingerprint(module):
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check_environment(saved, current, teacher=False):
    """Explicit comparison; actor policy width is the sole stage-specific width."""
    expected_mode = "teacher" if teacher else "student"
    if saved.get("actor_mode") != expected_mode:
        raise ValueError(f"Expected {expected_mode} checkpoint")
    for key in ("schema", "low_sha256", "control_hz", "physics_hz", "episode_s", "arm_pd",
                "payload_height_range", "wrench_frame"):
        # JSON round-trip accepts tuple/list equivalence, not value changes.
        if json.dumps(saved.get(key), sort_keys=True) != json.dumps(current.get(key), sort_keys=True):
            raise ValueError(f"PAINT checkpoint environment mismatch: {key}")
    expected_shapes = deepcopy(current["observation_shapes"])
    expected_shapes["policy"] = [147 if teacher else 144]
    if saved.get("observation_shapes") != expected_shapes:
        raise ValueError("PAINT checkpoint observation contract mismatch")
    for key in ("arm_scale", "base_limits", "payload_mass_range", "max_wrench", "zero_force",
                "disturbance_scale", "terrain", "domain"):
        if json.dumps(saved.get("arguments", {}).get(key)) != json.dumps(current["arguments"].get(key)):
            raise ValueError(f"PAINT checkpoint setting mismatch: {key}")


class PaintRunner:
    def __init__(self, env, train_cfg, manifest, teacher_path, log_dir, device="cpu"):
        self.env, self.cfg, self.manifest = env, deepcopy(train_cfg), manifest
        self.device, self.log_dir = device, Path(log_dir)
        self.writer = None
        self.current_learning_iteration = 0
        self.teacher_sha256 = hashlib.sha256(Path(teacher_path).read_bytes()).hexdigest()
        checkpoint = torch.load(teacher_path, map_location=device, weights_only=False)
        check_environment(checkpoint.get("transport_contract", {}), manifest, teacher=True)
        policy_cfg = deepcopy(train_cfg["policy"])
        policy_cfg.pop("class_name", None)
        saved_policy = checkpoint.get("transport_contract", {}).get("policy_config")
        if saved_policy is not None and saved_policy != train_cfg["policy"]:
            raise ValueError("Teacher architecture/config differs from Student config")
        self.teacher = ActorCritic(147, 177, 9, **policy_cfg).to(device)
        self.teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if not all(torch.isfinite(v).all() for v in self.teacher.state_dict().values()):
            raise ValueError("Nonfinite Teacher weights")
        self.teacher.requires_grad_(False).eval()
        self.teacher_fingerprint = fingerprint(self.teacher)
        self.actor_critic = ActorCritic(147, 177, 9, **policy_cfg).to(device)
        # Guide does not require warm-starting from Teacher; retain the v4 zero-mean start.
        last = [m for m in self.actor_critic.actor if isinstance(m, torch.nn.Linear)][-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        self.estimator = IntentEstimator().to(device)
        algorithm_cfg = deepcopy(train_cfg["algorithm"])
        algorithm_cfg.pop("class_name", None)
        self.alg = PaintPPO(self.actor_critic, self.teacher, self.estimator, env.num_envs,
                            train_cfg["num_steps_per_env"], device=device, **algorithm_cfg)
        self.contract = {"schema": STUDENT_SCHEMA, "io": STUDENT_IO,
                         "teacher_sha256": self.teacher_sha256, "transport_contract": manifest}
        (self.log_dir / "student_contract.json").write_text(json.dumps(self.contract, indent=2) + "\n", encoding="utf-8")

    def save(self, path):
        self.verify_frozen_teacher()
        checkpoint = {"student_contract": self.contract, "train_cfg": self.cfg,
                    "model_state_dict": self.actor_critic.state_dict(),
                    "estimator_state_dict": self.estimator.state_dict(),
                    "optimizer_state_dict": self.alg.optimizer.state_dict(),
                    "estimator_optimizer_state_dict": self.alg.estimator_optimizer.state_dict(),
                    "iteration": self.current_learning_iteration,
                    "learning_rate": self.alg.learning_rate}
        with Path(path).open("wb") as stream:
            torch.save(checkpoint, stream)

    def load(self, path, resume=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        saved = checkpoint.get("student_contract", {})
        if (saved.get("schema") != STUDENT_SCHEMA or saved.get("io") != STUDENT_IO
                or saved.get("teacher_sha256") != self.teacher_sha256):
            raise ValueError("Student schema/I/O/Teacher SHA mismatch")
        check_environment(saved.get("transport_contract", {}), self.manifest)
        saved_cfg = checkpoint.get("train_cfg", {})
        keys = ("policy", "algorithm", "num_steps_per_env") if resume else ("policy",)
        if any(saved_cfg.get(key) != self.cfg.get(key) for key in keys):
            raise ValueError("Student model/optimizer configuration mismatch")
        for key in ("model_state_dict", "estimator_state_dict"):
            if not all(torch.isfinite(v).all() for v in checkpoint[key].values()):
                raise ValueError("Nonfinite Student checkpoint")
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.estimator.load_state_dict(checkpoint["estimator_state_dict"], strict=True)
        if resume:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.alg.estimator_optimizer.load_state_dict(checkpoint["estimator_optimizer_state_dict"])
            self.alg.learning_rate = checkpoint["learning_rate"]
            self.current_learning_iteration = checkpoint["iteration"]

    def verify_frozen_teacher(self):
        if fingerprint(self.teacher) != self.teacher_fingerprint or any(p.grad is not None for p in self.teacher.parameters()):
            raise RuntimeError("Frozen Teacher changed")
        return True

    def get_inference_policy(self, device=None):
        return StudentInference(deepcopy(self.estimator), deepcopy(self.actor_critic.actor)).to(device or self.device).eval()

    def export(self, path):
        module = self.get_inference_policy("cpu")
        scripted = torch.jit.script(module)
        with Path(path).open("wb") as stream:
            torch.jit.save(scripted, stream)
        with Path(path).open("rb") as stream:
            loaded = torch.jit.load(stream)
        with torch.no_grad():
            history = torch.randn(16, 144)
            torch.testing.assert_close(loaded(history), module(history))

    def learn(self, iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            raise ValueError("Student history starts at a real reset")
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(str(self.log_dir), flush_secs=10)
        obs = self.env.get_observations()
        obs = {key: value.to(self.device) for key, value in obs.items()}
        self.actor_critic.train()
        self.estimator.train()
        with (self.log_dir / "student_training.jsonl").open("a", encoding="utf-8") as stream:
            for _ in range(iterations):
                reward_sum, failures, timeouts = 0., 0, 0
                for _ in range(self.cfg["num_steps_per_env"]):
                    # teacher comes from the SAME pre-action obs, not post-step extras.
                    action = self.alg.act(obs["student"], obs["teacher"], obs["critic"])
                    with torch.no_grad():
                        next_obs, reward, done, info = self.env.step(action.to(self.env.device))
                    next_obs = {key: value.to(self.device) for key, value in next_obs.items()}
                    info = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in info.items()}
                    self.alg.process_env_step(reward.to(self.device), done.to(self.device), info, next_obs["student"])
                    reward_sum += float(reward.mean())
                    failures += int(info.get("transport_terminated", torch.zeros_like(done)).sum())
                    timeouts += int((info["time_outs"].bool() & ~info["transport_terminated"].bool()).sum())
                    obs = next_obs
                self.alg.compute_returns(obs["critic"])
                stats = self.alg.update()
                self.current_learning_iteration += 1
                stats = dict(stats, mean_step_reward=reward_sum/self.cfg["num_steps_per_env"], failures=failures, timeouts=timeouts)
                for key, value in stats.items():
                    self.writer.add_scalar("Student/" + key, value, self.current_learning_iteration)
                stream.write(json.dumps(dict(iteration=self.current_learning_iteration, **stats)) + "\n")
                stream.flush()
                print(f"[STUDENT] iteration={self.current_learning_iteration} " + json.dumps(stats), flush=True)
                if self.current_learning_iteration % self.cfg["save_interval"] == 0:
                    self.save(self.log_dir / f"model_{self.current_learning_iteration}.pt")
            self.save(self.log_dir / f"model_{self.current_learning_iteration}.pt")
