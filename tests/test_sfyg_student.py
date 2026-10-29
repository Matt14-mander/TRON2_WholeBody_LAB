"""S0 only: actual regression, frozen teacher isolation and deployable I/O."""
import io
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from sfyg_student_core import StudentPolicy, estimator_step, estimator_metrics, latent_normal_kl
from sfyg_teacher_core import History, Settings, process_action


class StudentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(2)
        self.teacher = nn.Sequential(nn.Linear(147, 16), nn.ELU(), nn.Linear(16, 9))
        self.student = StudentPolicy(self.teacher)

    def test_actor_update_does_not_change_teacher_or_estimator(self):
        teacher = [p.detach().clone() for p in self.teacher.parameters()]
        estimator = [p.detach().clone() for p in self.student.estimator.parameters()]
        optimizer = torch.optim.Adam(self.student.actor.parameters(), lr=.001)
        optimizer.zero_grad()
        self.student(torch.randn(8, 144)).square().mean().backward()
        optimizer.step()
        for before, after in zip(teacher, self.teacher.parameters()):
            torch.testing.assert_close(before, after)
            self.assertIsNone(after.grad)
        for before, after in zip(estimator, self.student.estimator.parameters()):
            torch.testing.assert_close(before, after)
            self.assertIsNone(after.grad)

    def test_real_regression_learns_and_preserves_actor(self):
        history = torch.randn(32, 144)
        target = torch.stack((history[:, 0]*.03+.1, history[:, 1]*.01, history[:, 2]*.01), -1)
        actor = [p.detach().clone() for p in self.student.actor.parameters()]
        optimizer = torch.optim.Adam(self.student.estimator.parameters(), lr=.001)
        first = estimator_step(self.student.estimator, optimizer, history, target)
        for _ in range(60):
            last = estimator_step(self.student.estimator, optimizer, history, target)
        self.assertLess(last, first*.05)
        for before, after in zip(actor, self.student.actor.parameters()):
            torch.testing.assert_close(before, after)
        with self.assertRaises(ValueError):
            estimator_step(self.student.estimator, torch.optim.Adam(self.student.parameters()), history, target)

    def test_export_accepts_only_history_and_preserves_physical_action(self):
        with torch.no_grad():
            self.student.actor[-1].weight.zero_()
            self.student.actor[-1].bias.fill_(.4)
        history = torch.randn(2, 144)
        buffer = io.BytesIO()
        torch.jit.save(torch.jit.script(self.student.eval()), buffer)
        buffer.seek(0)
        loaded = torch.jit.load(buffer)
        torch.testing.assert_close(loaded(history), self.student(history))
        torch.testing.assert_close(loaded.estimate(history), self.student.estimate(history))
        # No tanh or execution processing hidden in the exported raw actor.
        torch.testing.assert_close(loaded(history), torch.full((2, 9), .4))
        command = process_action(loaded(history), torch.tensor([2., 4.]), Settings())
        torch.testing.assert_close(command, torch.tensor([[0., 0., 0.], [.3, 0., 0.]]))
        with self.assertRaises((ValueError, torch.jit.Error)):
            loaded(torch.randn(2, 147))

    def test_history_reset_is_per_robot_and_does_not_leak_future(self):
        h = History(2, 8, 18, 'cpu')
        h.append(torch.ones(2, 18), 1)
        snapshot = h.data.flatten(1).clone()
        h.append(torch.full((2, 18), 2.), 2)
        self.assertTrue(bool((snapshot[:, -18:] == 1).all()))
        h.reset(torch.tensor([0]))
        self.assertFalse(bool(h.data[0].any()))
        self.assertTrue(bool((h.data[1, -1] == 2).all()))

    def test_kl_matches_torch_and_teacher_has_no_gradient(self):
        mu = torch.randn(4, 9, requires_grad=True)
        std = torch.full((4, 9), .2, requires_grad=True)
        tm = torch.randn(4, 9, requires_grad=True)
        ts = torch.full((4, 9), .3, requires_grad=True)
        kl = latent_normal_kl(mu, std, tm, ts)
        expected = torch.distributions.kl_divergence(torch.distributions.Normal(mu, std),
                                                   torch.distributions.Normal(tm, ts)).sum(-1)
        torch.testing.assert_close(kl, expected)
        kl.mean().backward()
        self.assertIsNotNone(mu.grad)
        self.assertIsNone(tm.grad)
        self.assertIsNone(ts.grad)

    def test_metrics_report_physical_units_and_bad_inputs(self):
        metrics = estimator_metrics(torch.tensor([[.1, -.05, .02]]), torch.zeros(1, 3))
        torch.testing.assert_close(torch.tensor(metrics['physical_rmse']), torch.tensor([4., 2., .2]))
        self.assertFalse(metrics['is_acceptance'])
        for h in (torch.randn(2, 147), torch.full((2, 144), float('nan'))):
            with self.assertRaises(ValueError): self.student(h)


if __name__ == '__main__':
    unittest.main()
