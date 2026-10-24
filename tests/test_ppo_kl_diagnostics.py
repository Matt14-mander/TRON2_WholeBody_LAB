"""Regression tests for KL rejection, learning-rate control and encoder drift."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rsl_rl"))
from rsl_rl.algorithm import PPO
from rsl_rl.modules import ActorCritic, MLP_Encoder


class PPOKLTests(unittest.TestCase):
    def make_rollout(self, epochs=1, batches=1):
        torch.manual_seed(4)
        with contextlib.redirect_stdout(io.StringIO()):
            actor = ActorCritic(6, 6, 2, actor_hidden_dims=[8], critic_hidden_dims=[8], init_noise_std=.6)
            encoder = MLP_Encoder(4, 3, hidden_dims=[8], output_detach=True)
        algo = PPO(4, encoder, actor, num_learning_epochs=epochs, num_mini_batches=batches,
                   learning_rate=1e-4, max_learning_rate=3e-4, schedule="adaptive",
                   early_stop=True, desired_kl=.01)
        algo.init_storage(4, 4, [2], [6], [4], [1], [2])
        for _ in range(4):
            obs, history, commands, critic = torch.randn(4, 2), torch.randn(4, 4), torch.randn(4, 1), torch.randn(4, 5)
            with torch.no_grad():
                algo.act(obs, history, commands, critic)
                algo.process_env_step(torch.ones(4), torch.zeros(4, dtype=torch.bool), {}, obs)
        algo.compute_returns(torch.cat((critic, commands), -1))
        return algo

    def test_unchanged_first_batch_does_not_raise_learning_rate(self):
        algo = self.make_rollout()
        algo.update()
        self.assertEqual(algo.learning_rate, 1e-4)
        self.assertEqual(algo.last_update_stats["optimizer_steps"], 1)

    def test_rejected_first_batch_is_logged_and_reduces_lr(self):
        algo = self.make_rollout()
        # KL = 2 * .08**2 / (2 * .6**2) = .01778: above
        # early-stop threshold .015, below old LR reduction threshold .02.
        with torch.no_grad():
            algo.actor_critic.actor[-1].bias.add_(.08)
        before = [p.detach().clone() for p in algo.actor_critic.parameters()]
        losses = algo.update()
        self.assertGreater(losses[3], .015)
        self.assertEqual(algo.last_update_stats["optimizer_steps"], 0)
        self.assertEqual(algo.last_update_stats["early_stopped"], 1)
        self.assertAlmostEqual(algo.learning_rate, 1e-4 / 1.5)
        for old, new in zip(before, algo.actor_critic.parameters()):
            self.assertTrue(torch.equal(old, new))

    def test_rejected_second_batch_is_not_hidden_by_first_zero_kl(self):
        algo = self.make_rollout(epochs=2)
        step = algo.optimizer.step
        def large_step(*args, **kwargs):
            result = step(*args, **kwargs)
            with torch.no_grad():
                algo.actor_critic.actor[-1].bias.add_(.5)
            return result
        algo.optimizer.step = large_step
        losses = algo.update()
        self.assertEqual(algo.last_update_stats["optimizer_steps"], 1)
        self.assertEqual(algo.last_update_stats["kl_checks"], 2)
        self.assertGreater(losses[3], .1)
        self.assertGreater(algo.last_update_stats["max_checked_kl"], .1)

    def test_encoder_change_is_visible_even_with_no_actor_updates(self):
        algo = self.make_rollout()
        with torch.no_grad():
            algo.actor_critic.actor[-1].bias.add_(.08)
        step = algo.extra_optimizer.step
        def shifted_encoder(*args, **kwargs):
            result = step(*args, **kwargs)
            with torch.no_grad():
                algo.encoder.encoder[-1].bias.add_(1.)
            return result
        algo.extra_optimizer.step = shifted_encoder
        algo.update()
        self.assertGreater(algo.last_update_stats["encoder_policy_probe_kl"], 1e-5)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in algo.last_update_stats.values()))


if __name__ == "__main__":
    unittest.main()
