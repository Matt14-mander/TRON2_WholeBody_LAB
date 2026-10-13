"""Stand/walk schedules, strict loading, PPO freezing and terminal telemetry.

Native terms use manager doubles; physical stability still requires Isaac Sim.
"""
import ast
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
CONTROLLERS = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers"
sys.path.insert(0, str(CONTROLLERS))
sys.path.insert(0, str(ROOT / "rsl_rl"))
import standwalk_core as core


def networks(critic_width=90):
    from rsl_rl.modules import ActorCritic, MLP_Encoder
    with contextlib.redirect_stdout(io.StringIO()):
        actor = ActorCritic(78, critic_width, 10, actor_hidden_dims=[16], critic_hidden_dims=[16])
        encoder = MLP_Encoder(420, 3, hidden_dims=[16], output_detach=True)
    return actor, encoder


def weights(actor, encoder):
    return {"model_state_dict": {k: v.clone() for k, v in actor.state_dict().items()},
            "encoder_state_dict": {k: v.clone() for k, v in encoder.state_dict().items()}, "iter": 20000}


def native_module():
    class Command:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env
            self.num_envs, self.device = env.num_envs, env.device

        def reset(self, env_ids):
            return {}
    managers = types.ModuleType("isaaclab.managers")
    managers.CommandTerm = Command
    managers.CommandTermCfg = type("CommandCfg", (), {})
    managers.SceneEntityCfg = lambda name, **kwargs: types.SimpleNamespace(name=name, **kwargs)
    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    gait = types.ModuleType("bipedal_locomotion.tasks.locomotion.mdp.commands.gait_command")
    gait.GaitCommand = Command
    legacy = types.ModuleType("bipedal_locomotion.tasks.locomotion.mdp")
    legacy.joint_orientation_l1 = lambda env, asset_cfg: torch.ones(env.num_envs)
    class GaitReward:
        def __init__(self, env):
            self._env = env
            self.command_name = "gait_command"

        def compute_contact_targets(self, gait_params):
            return self._env.command_manager.get_term(self.command_name).gait_indices.clone()
    legacy.GaitReward = GaitReward
    packages = {name: types.ModuleType(name) for name in (
        "isaaclab", "bipedal_locomotion", "bipedal_locomotion.controllers",
        "bipedal_locomotion.tasks", "bipedal_locomotion.tasks.locomotion")}
    packages.update({"isaaclab.managers": managers, "isaaclab.utils": utils,
        "bipedal_locomotion.controllers.standwalk_core": core,
        "bipedal_locomotion.tasks.locomotion.mdp": legacy,
        "bipedal_locomotion.tasks.locomotion.mdp.commands.gait_command": gait})
    spec = importlib.util.spec_from_file_location("standwalk_native_test", ROOT /
        "exts/bipedal_locomotion/bipedal_locomotion/tasks/standwalk/mdp.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, packages):
        spec.loader.exec_module(module)
    return module


class StandWalkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.native = native_module()

    def test_schedule_has_exact_holds_and_smooth_transition(self):
        times = torch.tensor([0., 5., 5.5, 6., 12., 12.5, 13., 20.])
        target = torch.tensor([[-.25, .1, .2]]).repeat(8, 1)
        result = core.scheduled_velocity(times, torch.full((8,), 2), target)
        fraction = torch.tensor([0., 0., .5, 1., 1., .5, 0., 0.])
        torch.testing.assert_close(result, target * fraction[:, None])
        inverse = core.scheduled_velocity(times, torch.full((8,), 3), target)
        torch.testing.assert_close(inverse + result, target)
        torch.testing.assert_close(core.scheduled_velocity(times, torch.zeros(8, dtype=torch.long), target), torch.zeros_like(target))

    def test_standing_hysteresis_uses_planar_speed_and_yaw(self):
        command = torch.tensor([[.03, 0., 0.], [.03, 0., 0.], [0., 0., .1], [0., 0., 0.]])
        result = core.standing_mask(command, torch.tensor([True, False, True, False]))
        self.assertEqual(result.tolist(), [True, False, False, True])

    def test_stage_distribution_retains_known_walk(self):
        generator = torch.Generator().manual_seed(7)
        cases = core.sample_cases(10000, "stand", "cpu", generator)
        self.assertGreater(float((cases == 0).float().mean()), .57)
        self.assertLess(float((cases == 0).float().mean()), .63)
        self.assertEqual(set(cases.tolist()), {0, 1})
        cases = core.sample_cases(10000, "transition", "cpu", generator)
        self.assertEqual(set(cases.tolist()), {0, 1, 2, 3})

    def test_strict_loading_is_atomic_for_actor_and_encoder(self):
        source_actor, source_encoder = networks()
        actor, encoder = networks()
        report = core.load_warm_start(actor, encoder, weights(source_actor, source_encoder))
        self.assertEqual(report["source_iteration"], 20000)
        for key, value in actor.state_dict().items():
            torch.testing.assert_close(value, source_actor.state_dict()[key], rtol=0, atol=0)
        saved = weights(source_actor, source_encoder)
        saved["encoder_state_dict"].pop("encoder.0.bias")
        before = {k: v.clone() for k, v in actor.state_dict().items()}
        with self.assertRaises(ValueError):
            core.load_warm_start(actor, encoder, saved)
        for key, value in actor.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_only_one_critic_mass_input_may_be_reinitialized(self):
        source_actor, source_encoder = networks()
        actor, encoder = networks(91)
        saved = weights(source_actor, source_encoder)
        with self.assertRaises(ValueError):
            core.load_warm_start(actor, encoder, saved)
        initial = actor.critic[0].weight.clone()
        report = core.load_warm_start(actor, encoder, saved, allow_critic_input_reset=True)
        self.assertEqual(report["critic_reinitialized"], ["critic.0.weight", "critic.0.bias"])
        torch.testing.assert_close(actor.critic[0].weight, initial)
        torch.testing.assert_close(actor.actor[0].weight, source_actor.actor[0].weight)
        larger, _ = networks(92)
        with self.assertRaises(ValueError):
            core.load_warm_start(larger, encoder, saved, allow_critic_input_reset=True)
        # Evaluation of a loaded checkpoint without the payload is also supported.
        no_load, enc = networks(90)
        core.load_warm_start(no_load, enc, weights(actor, encoder), allow_critic_input_reset=True)

    def test_nonfinite_checkpoint_rejected(self):
        actor, encoder = networks()
        saved = weights(actor, encoder)
        saved["model_state_dict"]["logstd"][0] = float("nan")
        with self.assertRaises(ValueError):
            core.load_warm_start(actor, encoder, saved)

    def test_actual_ppo_updates_actor_but_preserves_frozen_encoder(self):
        from rsl_rl.algorithm import PPO
        actor, encoder = networks()
        ppo = PPO(2, encoder, actor, num_learning_epochs=1, num_mini_batches=1, learning_rate=1e-4, device="cpu")
        core.freeze_history_encoder(ppo)
        self.assertIsNone(ppo.extra_optimizer)
        before = {k: v.clone() for k, v in encoder.state_dict().items()}
        actor_before = actor.actor[0].weight.clone()
        ppo.init_storage(2, 4, [72], [90], [420], [3], [10])
        with torch.no_grad():
            for _ in range(4):
                obs, history, commands, critic = torch.randn(2, 72), torch.randn(2, 420), torch.zeros(2, 3), torch.randn(2, 87)
                ppo.act(obs, history, commands, critic)
                ppo.process_env_step(torch.tensor([1., -.3]), torch.zeros(2, dtype=torch.long), {}, obs)
            ppo.compute_returns(torch.cat((critic, commands), dim=-1))
        self.assertTrue(all(math.isfinite(x) for x in ppo.update()))
        self.assertFalse(torch.equal(actor_before, actor.actor[0].weight))
        for key, value in encoder.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_native_command_reset_and_clock_are_consistent(self):
        cfg = self.native.StandWalkVelocityCfg()
        cfg.evaluation_case = 3
        env = types.SimpleNamespace(num_envs=2, device="cpu", step_dt=.02, episode_length_buf=torch.zeros(2, dtype=torch.long))
        command = self.native.StandWalkVelocity(cfg, env)
        command._resample_command(torch.tensor([0, 1]))
        torch.testing.assert_close(command.command[:, 0], torch.full((2,), -.25))
        env.episode_length_buf[:] = 400  # 8 seconds: stop portion of walk-stop-walk.
        command._update_command()
        self.assertEqual(command.is_standing_env.tolist(), [True, True])
        env.command_manager = types.SimpleNamespace(get_term=lambda _: command)
        gait = self.native.EpisodeClockGait(types.SimpleNamespace(), env)
        gait.gait_command = torch.tensor([[.8, .5, .5, .2]]).repeat(2, 1)
        gait.gait_indices = torch.zeros(2)
        gait._update_command()
        torch.testing.assert_close(gait.gait_indices, torch.remainder(env.episode_length_buf * .02 * .8, 1.))
        # No phase-zeroing at stand: preserves the original actor's sin/cos clock.
        self.assertGreater(float(gait.gait_indices[0]), 0)
        env.episode_length_buf[0] = 0
        command._resample_command(torch.tensor([0]))
        self.assertEqual(command.is_standing_env.tolist(), [False, True])
        gait.reset(torch.tensor([0]))
        self.assertEqual(float(gait.gait_indices[0]), 0.)
        env.command_manager.get_term = lambda name: command if name == "base_velocity" else gait
        env.episode_length_buf[:] = 101
        reward = self.native.EpisodeClockGaitReward(env)
        phase = reward.compute_contact_targets(gait.gait_command)
        torch.testing.assert_close(phase, torch.remainder(env.episode_length_buf * .02 * .8, 1.))

    def test_stand_still_reward_respects_joint_filter(self):
        path = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/rewards.py"
        node = next(x for x in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(x, ast.FunctionDef) and x.name == "stand_still_for_joints")
        scope = {"torch": torch, "ManagerBasedRLEnv": object,
                 "SceneEntityCfg": lambda name: types.SimpleNamespace(name=name, joint_ids=slice(None))}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        robot = types.SimpleNamespace(data=types.SimpleNamespace(joint_pos=torch.tensor([[.1, .2, 100.]]),
                                                                default_joint_pos=torch.zeros(1, 3)))
        env = types.SimpleNamespace(scene={"robot": robot}, command_manager=types.SimpleNamespace(get_command=lambda _: torch.zeros(1, 3)))
        result = scope["stand_still_for_joints"](env, "base_velocity", types.SimpleNamespace(name="robot", joint_ids=[0, 1]))
        torch.testing.assert_close(result, torch.tensor([.3]))

    def test_metrics_do_not_accept_short_fall_or_missing_standing_samples(self):
        metrics = core.EpisodeMetrics()
        zero = torch.zeros(3)
        for step in range(17):
            metrics.update((step + 1) * .02, zero, zero, 0., zero[:2], 7., .02)
        failed = metrics.result(.34, False, require_standing=True)
        self.assertFalse(failed["passed"])
        self.assertIsNone(failed["standing_linear_rms"])
        metrics = core.EpisodeMetrics()
        for step in range(1000):
            metrics.update((step + 1) * .02, zero, zero, 0., zero[:2], 7., .02)
        self.assertTrue(metrics.result(20., True, require_standing=True)["passed"])

    def test_terminal_telemetry_is_not_replaced_by_reset_state(self):
        class BaseEnv:
            def __init__(self):
                self.episode_length_buf = torch.zeros(1, dtype=torch.long)
                self.command = torch.tensor([[-.25, 0., 0.]])
                self.command_manager = types.SimpleNamespace(get_command=lambda _: self.command)
                data = types.SimpleNamespace(root_lin_vel_b=torch.zeros(1, 3), root_ang_vel_b=torch.zeros(1, 3),
                    root_pos_w=torch.tensor([[0., 0., .85]]), projected_gravity_b=torch.tensor([[0., 0., -1.]]))
                class Scene(dict):
                    env_origins = torch.zeros(1, 3)
                self.scene = Scene(robot=types.SimpleNamespace(data=data))

            def _reset_idx(self, env_ids):
                self.scene["robot"].data.root_lin_vel_b.zero_()
                self.scene["robot"].data.projected_gravity_b[:] = torch.tensor([0., 0., -1.])
                self.episode_length_buf[env_ids] = 0
                self.command.zero_()

            def step(self, action):
                self.episode_length_buf += 1
                self.scene["robot"].data.root_lin_vel_b[0, 0] = 2.
                self.scene["robot"].data.projected_gravity_b[0, 2] = -math.cos(math.radians(40))
                self._reset_idx(torch.tensor([0]))
                return {}, torch.zeros(1), torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {}
        environments = types.ModuleType("isaaclab.envs")
        environments.ManagerBasedRLEnv = BaseEnv
        spec = importlib.util.spec_from_file_location("standwalk_env_test", ROOT /
            "exts/bipedal_locomotion/bipedal_locomotion/tasks/standwalk/env.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"isaaclab": types.ModuleType("isaaclab"), "isaaclab.envs": environments}):
            spec.loader.exec_module(module)
        env = module.StandWalkRLEnv()
        *_, extras = env.step(torch.zeros(1, 10))
        state = extras["standwalk_state"]
        self.assertAlmostEqual(float(state["tilt_deg"][0]), 40., places=4)
        self.assertEqual(float(state["linear_velocity"][0, 0]), 2.)
        self.assertEqual(float(env.scene["robot"].data.root_lin_vel_b[0, 0]), 0.)
        self.assertEqual(float(state["command"][0, 0]), -.25)

    def test_yaw_tracking_failure_cannot_pass_with_zero_planar_error(self):
        metrics = core.EpisodeMetrics()
        for step in range(1000):
            metrics.update((step + 1) * .02, torch.tensor([0., 0., .2]), torch.zeros(3),
                           0., torch.zeros(2), 7., .02)
        result = metrics.result(20., True)
        self.assertEqual(result["linear_tracking_rmse"], 0.)
        self.assertAlmostEqual(result["yaw_tracking_rmse"], .2)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
