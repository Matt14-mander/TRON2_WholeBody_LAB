"""Density, PPO gradient, export and legacy compatibility for bounded teacher."""
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
import torch
from torch.distributions import Normal, TransformedDistribution, TanhTransform

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("teacher_actor_test", ROOT / "rsl_rl/rsl_rl/modules/actor_critic.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DistributionTest(unittest.TestCase):
    def make_actor(self, bounded=True):
        with contextlib.redirect_stdout(io.StringIO()):
            return module.ActorCritic(12, 14, 9, actor_hidden_dims=[16, 16],
                critic_hidden_dims=[16], init_noise_std=.08, bounded_distribution=bounded)

    def test_density_matches_transformed_normal_and_ppo_ratio(self):
        actor = self.make_actor()
        obs = torch.randn(64, 12)
        a = actor.act(obs)
        old = actor.get_actions_log_prob(a).detach()
        expected = TransformedDistribution(actor.distribution, [TanhTransform()]).log_prob(a).sum(-1)
        torch.testing.assert_close(old, expected, atol=1e-5, rtol=1e-5)
        self.assertTrue((a.abs() < 1).all())
        actor.act(obs)
        torch.testing.assert_close((actor.get_actions_log_prob(a)-old).exp(), torch.ones(64))
        self.assertTrue(torch.isfinite(actor.entropy).all())
        loss = -((actor.get_actions_log_prob(a)-old).exp() * (a[:, 0]-.2)).mean() - .001*actor.entropy.mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None))
        self.assertTrue(any(p.grad.abs().sum() > 0 for p in actor.actor.parameters() if p.grad is not None))

    def test_zero_start_export_and_std_limits(self):
        actor = self.make_actor()
        head = [m for m in actor.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
        with torch.no_grad():
            head.weight.zero_(); head.bias.zero_()
            actor.logstd.fill_(20)
        obs = torch.randn(128, 12)
        torch.testing.assert_close(actor.act_inference(obs), torch.zeros(128, 9))
        actor.act(obs)
        torch.testing.assert_close(actor.action_std, torch.full((128, 9), .2))
        with torch.no_grad():
            head.bias.fill_(100.)
        self.assertTrue((actor.act_inference(obs).abs() < 1).all())
        # Trace tests the same deterministic actor delivered to export consumers.
        exported = torch.jit.trace(actor.actor, obs)
        torch.testing.assert_close(exported(obs), actor.act_inference(obs))

    def test_legacy_remains_normal(self):
        actor = self.make_actor(False)
        obs = torch.randn(8, 12)
        a = actor.act(obs)
        torch.testing.assert_close(actor.action_mean, actor.actor(obs))
        torch.testing.assert_close(actor.get_actions_log_prob(a),
            Normal(actor.action_mean, actor.action_std).log_prob(a).sum(-1))


if __name__ == "__main__":
    unittest.main()
