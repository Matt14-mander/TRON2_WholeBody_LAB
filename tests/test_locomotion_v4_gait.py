"""Gait timing, contact incentives, strict I/O and rejection of motionless policies."""
import math
from pathlib import Path
import sys
import types
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_locomotion_v4 import core, native_module


class GaitTests(unittest.TestCase):
    def test_alternation_and_height_velocity_consistency(self):
        gait = dict(core.DEFAULT_GAIT)
        time = torch.tensor([.125, .375, .625, .875]) / gait["frequency"]
        contact, height, velocity = core.gait_targets(time, gait)
        self.assertGreater(float(contact[0, 0]), .99)
        self.assertLess(float(contact[0, 1]), .01)
        torch.testing.assert_close(contact[:2], contact[2:].flip(-1))
        torch.testing.assert_close(height[:2], height[2:].flip(-1))
        dt = 1e-4
        derivative = (core.gait_targets(time + dt, gait)[1] - core.gait_targets(time - dt, gait)[1]) / (2 * dt)
        torch.testing.assert_close(velocity, derivative, atol=5e-4, rtol=.005)
        features = core.gait_features(time, gait)
        self.assertEqual(tuple(features.shape), (4, 6))
        torch.testing.assert_close(features[:, 0], torch.sin(2 * math.pi * time * .8))

    def test_changed_duty_keeps_swing_inside_noncontact_phase(self):
        gait = dict(core.DEFAULT_GAIT, duration=.6)
        t = torch.tensor([.55, .7, .8, .9]) / .8
        contact, height, velocity = core.gait_targets(t, gait)
        self.assertEqual(float(height[0, 0]), 0.)
        self.assertEqual(float(velocity[0, 0]), 0.)
        self.assertAlmostEqual(float(height[2, 0]), .2, places=5)
        self.assertLess(float(contact[2, 0]), .01)
        self.assertGreater(float(velocity[1, 0]), 0.)
        self.assertLess(float(velocity[3, 0]), 0.)

    def test_correct_gait_beats_two_feet_planted_and_standing_is_masked(self):
        mdp = native_module()
        time = torch.tensor([.125, .125]) / .8
        targets = core.gait_targets(time, core.DEFAULT_GAIT)
        forces = torch.zeros(2, 2, 3)
        forces[:, 0, 2] = 200.
        velocities = torch.zeros_like(forces)
        velocities[..., 2] = targets[2]
        good = core.gait_score(forces.norm(dim=-1), velocities, targets[1], targets)
        bad = core.gait_score(torch.full((2, 2), 100.), torch.zeros_like(forces), torch.zeros(2, 2), targets)
        self.assertTrue(bool((good > bad + .3).all()))
        standing = torch.tensor([False, True])
        env = types.SimpleNamespace(num_envs=2, step_dt=.02, episode_length_buf=time / .02,
            cfg=types.SimpleNamespace(locomotion_gait=dict(core.DEFAULT_GAIT)),
            command_manager=types.SimpleNamespace(get_term=lambda name: types.SimpleNamespace(is_standing_env=standing)))
        class Scene(dict): pass
        env.scene = Scene(robot=types.SimpleNamespace(data=types.SimpleNamespace(
            body_pos_w=torch.cat((torch.zeros(2, 2, 2), targets[1][..., None] + 2.), -1),
            body_lin_vel_w=velocities)))
        env.scene.sensors = {"contact_forces": types.SimpleNamespace(data=types.SimpleNamespace(net_forces_w=forces)),
            "height_scanner": types.SimpleNamespace(data=types.SimpleNamespace(ray_hits_w=torch.tensor([0., 0., 2.]).repeat(2, 81, 1)))}
        cfg = types.SimpleNamespace(name="robot", body_ids=[0, 1])
        reward = mdp.restored_gait(env, types.SimpleNamespace(name="contact_forces", body_ids=[0, 1]), cfg)
        self.assertGreater(float(reward[0]), 1.)
        self.assertEqual(float(reward[1]), 0.)
        torch.testing.assert_close(mdp.gait_observation(env), core.gait_features(time, core.DEFAULT_GAIT))
        # Reset uses the same clock immediately; no stale phase buffer survives.
        env.episode_length_buf[0] = 0
        torch.testing.assert_close(mdp.gait_observation(env)[0, :2], torch.tensor([0., 1.]))

    def test_new_contract_rejects_legacy_and_keeps_height_width(self):
        old = core.contract(117, "nominal", (.35, .2, .4), core.STAGES["nominal"])
        new = core.contract(123, "nominal", (.35, .2, .4), core.STAGES["nominal"], core.DEFAULT_GAIT)
        self.assertEqual(new["height_width"], 81)
        self.assertEqual(new["history_width"], 360)
        core.check_checkpoint({"v4_contract": new}, new)
        with self.assertRaisesRegex(ValueError, "schema"):
            core.check_checkpoint({"v4_contract": old}, new)
        changed = dict(new, gait=dict(core.DEFAULT_GAIT, frequency=1.))
        with self.assertRaisesRegex(ValueError, "gait"):
            core.check_checkpoint({"v4_contract": changed}, new)

    def test_walk_first_quota_and_minimum_targets(self):
        mdp = native_module()
        env = types.SimpleNamespace(num_envs=100, device="cpu", step_dt=.02,
                                    episode_length_buf=torch.zeros(100))
        cfg = types.SimpleNamespace(limits=(.35, .2, .4), profile="walk_first", evaluation_case=-1)
        cmd = mdp.VelocityCommand(cfg, env)
        self.assertEqual(torch.bincount(cmd.roles).tolist(), [10, 80, 5, 5])
        cmd._resample_command(torch.arange(100))
        minima = torch.tensor([.15, .1, .15])
        self.assertTrue(bool(((cmd.target == 0) | (cmd.target.abs() >= minima)).all()))

    def test_noop_slow_and_yaw_fail_but_tracking_passes(self):
        for command in (torch.tensor([.05, 0., 0.]), torch.tensor([0., 0., -.2])):
            for tracking in (False, True):
                metrics = core.MotionProgress()
                for step in range(1000):
                    velocity = torch.tensor([float(command[0]), 0., 0.]) if tracking else torch.zeros(3)
                    metrics.update((step + 1) * .02, command, velocity,
                                   float(command[2]) if tracking else 0., .02)
                self.assertEqual(metrics.result()["motion_passed"], tracking)

    def test_stop_requires_observed_motion(self):
        for tracking in (False, True):
            metrics = core.MotionProgress()
            for step in range(1000):
                command = torch.tensor([.2 if step < 500 else 0., 0., 0.])
                velocity = command.clone() if tracking else torch.zeros(3)
                metrics.update((step + 1) * .02, command, velocity, 0., .02)
            self.assertEqual(metrics.result(require_stop=True)["motion_passed"], tracking)

    def test_insufficient_moving_window_is_not_reported_as_motion_success(self):
        metrics = core.MotionProgress()
        metrics.update(.02, torch.tensor([.05, 0., 0.]), torch.zeros(3), 0., .02)
        self.assertFalse(metrics.result()["motion_passed"])


if __name__ == "__main__":
    unittest.main()
