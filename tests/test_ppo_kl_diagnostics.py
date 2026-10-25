"""Regression tests for KL rejection, learning-rate control and encoder drift."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from copy import deepcopy

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

    def test_guard_rolls_back_actor_weights_and_adam_state(self):
        algo = self.make_rollout()
        algo.update_kl_guard = True
        algo.extra_optimizer = None
        for param in algo.actor_critic.parameters():
            param.grad = torch.zeros_like(param)
        algo.optimizer.step()
        before = deepcopy(algo.actor_critic.state_dict())
        state = deepcopy(algo.optimizer.state_dict()["state"])
        step = algo.optimizer.step
        def excessive_step(*args, **kwargs):
            value = step(*args, **kwargs)
            with torch.no_grad():
                algo.actor_critic.actor[-1].bias.add_(1.)
            return value
        algo.optimizer.step = excessive_step
        algo.update()
        self.assertEqual(algo.last_update_stats["actor_rejected_updates"], 1)
        self.assertEqual(algo.last_update_stats["optimizer_steps"], 0)
        self.assertLess(algo.last_update_stats["post_update_probe_kl"], 1e-6)
        for key, value in before.items():
            torch.testing.assert_close(value, algo.actor_critic.state_dict()[key], rtol=0, atol=0)
        for key, values in state.items():
            for field, value in values.items():
                torch.testing.assert_close(value, algo.optimizer.state_dict()["state"][key][field], rtol=0, atol=0)

    def test_guard_rolls_back_encoder_weights_and_adam_state(self):
        algo = self.make_rollout()
        algo.update_kl_guard = True
        algo.schedule = "fixed"
        for group in algo.optimizer.param_groups:
            group["lr"] = 0.
        for param in algo.encoder.parameters():
            param.grad = torch.zeros_like(param)
        algo.extra_optimizer.step()
        before = deepcopy(algo.encoder.state_dict())
        state = deepcopy(algo.extra_optimizer.state_dict()["state"])
        step = algo.extra_optimizer.step
        def excessive_step(*args, **kwargs):
            value = step(*args, **kwargs)
            with torch.no_grad():
                algo.encoder.encoder[-1].bias.add_(20.)
            return value
        algo.extra_optimizer.step = excessive_step
        algo.update()
        self.assertEqual(algo.last_update_stats["encoder_rejected_updates"], 1)
        self.assertEqual(algo.last_update_stats["encoder_steps"], 0)
        self.assertLess(algo.last_update_stats["encoder_policy_probe_kl"], 1e-6)
        for key, value in before.items():
            torch.testing.assert_close(value, algo.encoder.state_dict()[key], rtol=0, atol=0)
        for key, values in state.items():
            for field, value in values.items():
                torch.testing.assert_close(value, algo.extra_optimizer.state_dict()["state"][key][field], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
