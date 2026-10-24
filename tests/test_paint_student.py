"""CPU checks of actual Student PPO, transitions, contracts, resume and export."""
from copy import deepcopy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rsl_rl"))
from rsl_rl.modules import ActorCritic
from rsl_rl.modules.intent_estimator import IntentEstimator
from rsl_rl.algorithm.paint_ppo import PaintPPO, normal_kl
from rsl_rl.runner.paint_runner import PaintRunner, fingerprint, check_environment


def policy():
    with contextlib.redirect_stdout(io.StringIO()):
        return ActorCritic(147, 177, 9, actor_hidden_dims=[16], critic_hidden_dims=[16],
                           bounded_distribution=True, init_noise_std=.1)


def algorithm(**kwargs):
    return PaintPPO(policy(), policy(), IntentEstimator(), 3, 3,
                    num_learning_epochs=1, num_mini_batches=2, **kwargs)


def fill(alg):
    for step in range(3):
        obs, teacher, critic = torch.randn(3, 144), torch.randn(3, 147), torch.randn(3, 177)
        alg.act(obs, teacher, critic)
        alg.process_env_step(torch.randn(3), torch.tensor([False, True, False]),
                             {"time_outs": torch.tensor([False, True, False])}, torch.randn(3, 144))
    alg.compute_returns(torch.randn(3, 177))


class FakeEnv:
    num_envs = 3
    device = "cpu"

    def __init__(self):
        self.tick = 0
        self.obs = {"student": torch.zeros(3, 144), "teacher": torch.zeros(3, 147), "critic": torch.zeros(3, 177)}

    def get_observations(self):
        return self.obs

    def step(self, action):
        assert action.shape == (3, 9) and action.abs().max() <= 1
        self.tick += 1
        for value in self.obs.values():
            value.fill_(self.tick / 100)
        return self.obs, torch.ones(3), torch.tensor([False, True, False]), {
            "time_outs": torch.tensor([False, True, False]),
            "transport_terminated": torch.zeros(3, dtype=torch.bool)}


def manifest():
    return {"schema": "paint_transport_v4", "actor_mode": "student", "low_sha256": "low123",
            "control_hz": 50, "physics_hz": 200, "episode_s": 10, "arm_pd": [60., 6.],
            "payload_height_range": [.44, .64], "wrench_frame": "base -> world -> each body local, applied every physics step",
            "observation_shapes": {"policy": [144], "student": [144], "teacher": [147], "critic": [177]},
            "arguments": {"arm_scale": .1, "base_limits": [.35, .2, .4], "payload_mass_range": [.5, .5],
                          "max_wrench": [3., 3., .5], "zero_force": False, "disturbance_scale": 0.,
                          "terrain": "flat", "domain": "nominal"}}


def runner(folder, env=None):
    cfg = {"policy": {"actor_hidden_dims": [16], "critic_hidden_dims": [16], "activation": "elu",
                       "bounded_distribution": True, "init_noise_std": .1},
           "algorithm": {"num_learning_epochs": 1, "num_mini_batches": 2, "desired_kl": None},
           "num_steps_per_env": 3, "save_interval": 1}
    teacher_manifest = deepcopy(manifest())
    teacher_manifest["actor_mode"] = "teacher"
    teacher_manifest["observation_shapes"]["policy"] = [147]
    teacher_path = folder / "teacher.pt"
    if not teacher_path.exists():
        with teacher_path.open("wb") as stream:
            torch.save({"model_state_dict": policy().state_dict(), "transport_contract": teacher_manifest}, stream)
    with contextlib.redirect_stdout(io.StringIO()):
        return PaintRunner(env or FakeEnv(), cfg, manifest(), teacher_path, folder)


class StudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)

    def test_kl_direction_and_tanh_invariance(self):
        mu, std = torch.tensor([[.1, -.4]]), torch.tensor([[.15, .1]])
        tm, ts = torch.tensor([[-.2, .1]]), torch.tensor([[.2, .13]])
        actual = normal_kl(mu, std, tm, ts)
        expected = torch.distributions.kl_divergence(torch.distributions.Normal(mu, std), torch.distributions.Normal(tm, ts)).sum(-1)
        torch.testing.assert_close(actual, expected)
        self.assertNotAlmostEqual(float(actual), float(normal_kl(tm, ts, mu, std)))
        torch.testing.assert_close(normal_kl(mu, std, mu, std), torch.zeros(1))
        z = torch.distributions.Normal(mu, std).sample((100000,))
        a = z.tanh()
        jac = torch.log1p(-a.square())
        empirical = ((torch.distributions.Normal(mu, std).log_prob(z)-jac)
                     - (torch.distributions.Normal(tm, ts).log_prob(z)-jac)).sum(-1).mean()
        self.assertLess(abs(float(empirical-actual)), .03)

    def test_snapshot_pairs_before_in_place_step(self):
        alg = algorithm()
        obs, teacher, critic = torch.ones(3, 144), torch.full((3, 147), 2.), torch.full((3, 177), 3.)
        alg.act(obs, teacher, critic)
        obs.fill_(10); teacher.fill_(20); critic.fill_(30)
        alg.process_env_step(torch.ones(3), torch.zeros(3), {}, obs)
        self.assertTrue((alg.storage.observations[0] == 1).all())
        self.assertTrue((alg.storage.teacher_observations[0] == 2).all())
        self.assertTrue((alg.storage.critic_obs[0] == 3).all())

    def test_timeout_and_failure_are_distinct(self):
        alg = algorithm()
        alg.act(torch.zeros(3, 144), torch.zeros(3, 147), torch.zeros(3, 177))
        values = alg.transition.values.squeeze(-1).clone()
        alg.process_env_step(torch.ones(3), torch.ones(3), {
            "time_outs": torch.ones(3, dtype=torch.bool), "transport_terminated": torch.tensor([False, True, False])}, torch.zeros(3, 144))
        expected = torch.ones(3) + alg.gamma * values * torch.tensor([1., 0., 1.])
        torch.testing.assert_close(alg.storage.rewards[0, :, 0], expected)

    def test_real_ppo_and_regression_update_freeze_teacher(self):
        alg = algorithm(desired_kl=None)
        teacher, actor, estimator = fingerprint(alg.teacher), fingerprint(alg.actor_critic), fingerprint(alg.estimator)
        fill(alg)
        stats = alg.update()
        self.assertEqual(stats["optimizer_steps"], 2)
        self.assertEqual(stats["estimator_steps"], 2)
        self.assertNotEqual(actor, fingerprint(alg.actor_critic))
        self.assertNotEqual(estimator, fingerprint(alg.estimator))
        self.assertEqual(teacher, fingerprint(alg.teacher))
        self.assertTrue(all(p.grad is None for p in alg.teacher.parameters()))
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in stats.values()))

    def test_estimate_is_detached_from_policy_gradients(self):
        alg = algorithm()
        obs = torch.randn(3, 144)
        estimate = alg.estimator(obs)
        alg.actor_critic.update_distribution(torch.cat((obs, estimate.detach()), -1))
        alg.actor_critic.action_mean.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in alg.estimator.parameters()))
        estimate.square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in alg.estimator.parameters()))

    def test_teacher_kl_updates_actor_without_reward_gradient(self):
        alg = algorithm(desired_kl=None, distill_coef=1., value_loss_coef=0., entropy_coef=0.)
        fill(alg)
        alg.storage.advantages.zero_()
        before = fingerprint(alg.actor_critic.actor)
        stats = alg.update()
        self.assertGreater(stats["teacher_kl"], 0.)
        self.assertNotEqual(before, fingerprint(alg.actor_critic.actor))

    def test_same_minibatch_regression_uses_current_normalized_label(self):
        alg = algorithm(desired_kl=None)
        fill(alg)
        seen = []
        for b in alg.storage.batches(2, 1):
            for row, teacher in zip(b["observations"], b["teacher_observations"]):
                index = (alg.storage.observations.flatten(0, 1) == row).all(-1).nonzero()[0, 0]
                torch.testing.assert_close(teacher, alg.storage.teacher_observations.flatten(0, 1)[index])
                seen.append(int(index))
        self.assertEqual(sorted(seen), list(range(9)))

    def test_early_stop_counts_rejected_batch(self):
        alg = algorithm(desired_kl=.001)
        fill(alg)
        alg.storage.mu.add_(10.)
        before = fingerprint(alg.actor_critic)
        stats = alg.update()
        self.assertEqual(stats["kl_checks"], 1)
        self.assertEqual(stats["optimizer_steps"], 0)
        self.assertEqual(stats["early_stopped"], 1.)
        self.assertLess(stats["learning_rate"], 1e-4)
        self.assertEqual(before, fingerprint(alg.actor_critic))

    def test_reject_shape_nonfinite_and_bad_settings(self):
        alg = algorithm()
        with self.assertRaises(ValueError):
            alg.act(torch.zeros(3, 143), torch.zeros(3, 147), torch.zeros(3, 177))
        with self.assertRaises(FloatingPointError):
            alg.act(torch.full((3, 144), float("nan")), torch.zeros(3, 147), torch.zeros(3, 177))
        with self.assertRaises(ValueError):
            algorithm(distill_coef=float("nan"))

    def test_contract_rejects_changed_low_and_action_settings(self):
        current = manifest()
        for key, val in (("low_sha256", "wrong"), ("control_hz", 25), ("payload_height_range", [0., 1.])):
            bad = deepcopy(current)
            bad[key] = val
            with self.assertRaises(ValueError):
                check_environment(bad, current)
        bad = deepcopy(current)
        bad["arguments"]["arm_scale"] = .5
        with self.assertRaises(ValueError):
            check_environment(bad, current)

    def test_cli_requires_teacher_and_student_checkpoint_where_needed(self):
        spec = importlib.util.spec_from_file_location("student_transport_cli", ROOT / "scripts/rsl_rl/train_transport_v4.py")
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        base = ["--locomotion_path", "low.pt", "--actor_mode", "student"]
        # Physical range validation is independently covered by transport tests.
        fake_core = SimpleNamespace(validate_settings=lambda *a: None)
        with patch.dict(sys.modules, {"bipedal_locomotion.controllers.paint_transport_v4": fake_core}):
            for extra in (["--mode", "train"], ["--controller", "student", "--teacher_checkpoint", "teacher.pt"],
                          ["--resume", "--teacher_checkpoint", "teacher.pt"]):
                with self.assertRaises(ValueError):
                    script.validate_args(script.parser_for_task().parse_args(base + extra))
            script.validate_args(script.parser_for_task().parse_args(base + ["--mode", "train", "--teacher_checkpoint", "teacher.pt"]))
        seen = []
        args = SimpleNamespace(controller="student")
        obs = {"policy": torch.zeros(2, 144), "teacher": torch.ones(2, 147)}
        script.make_action(args, None, obs, lambda x: seen.append(x.shape[-1]))
        self.assertEqual(seen, [144])

    def test_runner_resume_export_and_no_privilege_at_inference(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            folder = Path(tmp)
            run = runner(folder)
            teacher_hash = fingerprint(run.teacher)
            run.learn(1)
            expected_actor, expected_est = fingerprint(run.actor_critic), fingerprint(run.estimator)
            restored = runner(folder)
            restored.load(folder / "model_1.pt", resume=True)
            self.assertEqual(restored.current_learning_iteration, 1)
            self.assertTrue(restored.alg.optimizer.state)
            self.assertTrue(restored.alg.estimator_optimizer.state)
            self.assertEqual(expected_actor, fingerprint(restored.actor_critic))
            self.assertEqual(expected_est, fingerprint(restored.estimator))
            restored.cfg["save_interval"] = 10
            restored.load(folder / "model_1.pt", resume=True)
            restored.learn(1)
            self.assertTrue((folder / "model_2.pt").exists())
            restored.export(folder / "student_actor.pt")
            with (folder / "student_actor.pt").open("rb") as stream:
                exported = torch.jit.load(stream)
            history = torch.randn(5, 144)
            with torch.no_grad():
                torch.testing.assert_close(exported(history), restored.get_inference_policy()(history))
                self.assertLessEqual(float(exported(history).abs().max()), 1.)
            self.assertEqual(teacher_hash, fingerprint(restored.teacher))
            checkpoint = torch.load(folder / "model_2.pt", weights_only=False)
            checkpoint["student_contract"]["teacher_sha256"] = "wrong"
            with (folder / "bad.pt").open("wb") as stream:
                torch.save(checkpoint, stream)
            with self.assertRaises(ValueError):
                restored.load(folder / "bad.pt")
            run.writer.close(); restored.writer.close()


if __name__ == "__main__":
    unittest.main()
