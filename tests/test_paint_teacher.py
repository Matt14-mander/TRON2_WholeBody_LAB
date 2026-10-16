"""Teacher contracts and controller lifecycle, without an Isaac Sim runtime.

Manager/asset doubles test the native term's rates, reset isolation and frame
semantics. They do not validate PhysX stability or replace a simulator smoke run.
"""
import importlib.util
import contextlib
import io
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers"))
import paint_teacher_core as core


def checkpoint(path):
    actor = core.mlp([78, 512, 512, 256, 128, 10])
    encoder = core.mlp([420, 256, 128, 3])
    torch.save({"model_state_dict": {f"actor.{k}": v for k, v in actor.state_dict().items()},
                "encoder_state_dict": {f"encoder.{k}": v for k, v in encoder.state_dict().items()}}, path)
    return actor, encoder


def quat_inverse(q, v):
    u = -q[:, 1:]
    uv = torch.cross(u, v, dim=-1)
    return v + 2 * (q[:, :1] * uv + torch.cross(u, uv, dim=-1))


class BaseTerm:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env
        self.num_envs, self.device = env.num_envs, env.device
        self._asset = env.scene["robot"]


class CommandDouble(BaseTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.metrics = {}
        self.time_left = torch.full((env.num_envs,), 5.)

    def compute(self, dt):
        self.time_left -= dt
        self._update_metrics()
        self._update_command()

    def reset(self, env_ids):
        self._resample_command(env_ids)
        return {}


def native_terms():
    managers = types.ModuleType("isaaclab.managers")
    managers.ActionTerm, managers.CommandTerm = BaseTerm, CommandDouble
    managers.ActionTermCfg = managers.CommandTermCfg = type("TermCfg", (), {})
    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    math_module = types.ModuleType("isaaclab.utils.math")
    math_module.quat_apply_inverse = quat_inverse
    packages = {name: types.ModuleType(name) for name in
                ("isaaclab", "bipedal_locomotion", "bipedal_locomotion.controllers")}
    packages.update({"isaaclab.managers": managers, "isaaclab.utils": utils,
                     "isaaclab.utils.math": math_module,
                     "bipedal_locomotion.controllers.paint_teacher_core": core})
    spec = importlib.util.spec_from_file_location("teacher_mdp_contract_test",
        ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/teacher/mdp.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, packages):
        spec.loader.exec_module(module)
    return module


class RobotDouble:
    def __init__(self, count):
        self.names = list(core.LEG_JOINT_NAMES + core.ARM_JOINT_NAMES)
        zero = torch.zeros(count, 16)
        limits = torch.stack((zero - 1, zero + 1), dim=-1)
        quat = torch.zeros(count, 4)
        quat[:, 0] = 1
        gravity = torch.tensor([[0., 0., -1.]]).repeat(count, 1)
        root = torch.tensor([[0., 0., .85]]).repeat(count, 1)
        body = torch.tensor([[[.2, 0., 1.]]]).repeat(count, 1, 1)
        self.data = types.SimpleNamespace(default_joint_pos=zero.clone(), joint_pos=zero.clone(),
            joint_vel=zero.clone(), soft_joint_pos_limits=limits, projected_gravity_b=gravity,
            root_lin_vel_b=torch.zeros(count, 3), root_ang_vel_b=torch.zeros(count, 3),
            root_quat_w=quat, root_pos_w=root, body_pos_w=body, body_quat_w=quat[:, None],
            body_com_lin_vel_w=torch.zeros(count, 1, 3), body_ang_vel_w=torch.zeros(count, 1, 3),
            GRAVITY_VEC_W=gravity)
        self.root_physx_view = types.SimpleNamespace(get_masses=lambda: torch.full((count, 1), .5))

    def find_joints(self, names, preserve_order):
        return [self.names.index(n) for n in names], names

    def find_bodies(self, name, preserve_order):
        return [0], [name]

    def set_joint_position_target(self, value, joint_ids):
        pass

    def set_joint_velocity_target(self, value, joint_ids):
        assert torch.count_nonzero(value) == 0

    set_joint_effort_target = set_joint_velocity_target

    def set_external_force_and_torque(self, force, torque, body_ids, is_global):
        assert body_ids == [0] and not is_global
        self.force, self.torque = force.clone(), torque.clone()


class TeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        (ROOT / ".test-temp").mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT / ".test-temp")
        cls.path = Path(cls.temp.name) / "low.pt"
        cls.actor, cls.encoder = checkpoint(cls.path)
        cls.native = native_terms()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_strict_checkpoint_and_exact_original_actor_order(self):
        frozen = core.FrozenLocomotion(self.path)
        p, h, cmd = torch.randn(2, 42), torch.randn(2, 10, 42), torch.randn(2, 3)
        expected = self.actor(torch.cat((self.encoder(h.flatten(1)), p, torch.zeros(2, 30), cmd), -1))
        torch.testing.assert_close(frozen(p, h, cmd), expected)
        self.assertFalse(frozen.training)
        self.assertTrue(all(not x.requires_grad for x in frozen.parameters()))
        saved = torch.load(self.path, weights_only=False)
        saved["model_state_dict"].pop("actor.0.bias")
        broken = Path(self.temp.name) / "broken.pt"
        torch.save(saved, broken)
        with self.assertRaises(ValueError):
            core.FrozenLocomotion(broken)

    def test_history_is_causal_oldest_first_and_reset_isolated(self):
        history = core.SampledHistory(2, 4, 1, "cpu")
        history.append(torch.tensor([[1.], [10.]]), 0)
        history.append(torch.tensor([[2.], [20.]]), 4)
        history.append(torch.tensor([[99.], [99.]]), 4)
        self.assertEqual(history.data[:, :, 0].tolist(), [[1., 1., 1., 2.], [10., 10., 10., 20.]])
        history.reset(torch.tensor([0]))
        history.append(torch.tensor([[7.], [30.]]), 8)
        self.assertEqual(history.data[:, :, 0].tolist(), [[7., 7., 7., 7.], [10., 10., 20., 30.]])

    def test_admittance_bounded_response_and_release(self):
        velocity = torch.zeros(2, 3)
        force = torch.tensor([[5., 0., .5], [-5., 0., -.5]])
        for _ in range(100):
            velocity = core.admittance_step(velocity, force, .04, (4, 4, .5), (20, 20, 2), (.35, .2, .4))
        torch.testing.assert_close(velocity, torch.tensor([[.25, 0., .25], [-.25, 0., -.25]]))
        released = core.admittance_step(velocity, torch.zeros_like(force), .04, (4, 4, .5), (20, 20, 2), (.35, .2, .4))
        self.assertTrue((released.abs() <= velocity.abs()).all())
        with self.assertRaises(ValueError):
            core.admittance_step(velocity, force, .04, (0, 4, .5), (20, 20, 2), (.35, .2, .4))

    def test_force_envelope_has_zero_release(self):
        phase = torch.tensor([-.1, 0., .05, .5, .95, 1., 1.1])
        torch.testing.assert_close(core.force_envelope(phase, torch.zeros(7, dtype=torch.bool)),
                                   torch.tensor([0., 0., .5, 1., .5, 0., 0.]))

    def make_env(self, low_checkpoint=None):
        robot = RobotDouble(2)
        scene = {"robot": robot}
        class Scene(dict):
            env_origins = torch.zeros(2, 3)
        env = types.SimpleNamespace(num_envs=2, device="cpu", physics_dt=.005, step_dt=.04,
            cfg=types.SimpleNamespace(decimation=8, commands=types.SimpleNamespace(
                interaction=self.native.PayloadInteractionCfg())),
            episode_length_buf=torch.full((2,), 60), common_step_counter=0, _sim_step_counter=0,
            scene=Scene(scene))
        interaction = self.native.PayloadInteraction(env.cfg.commands.interaction, env)
        interaction._command.copy_(torch.tensor([[3., -2., .4], [-1., 1., -.2]]))
        gait = torch.tensor([[1.125, .5, .5, .28]]).repeat(2, 1)
        base = torch.zeros(2, 3)
        commands = {"interaction": interaction.command, "gait_command": gait, "base_velocity": base}
        env.command_manager = types.SimpleNamespace(get_command=commands.__getitem__,
                                                    get_term=lambda name: interaction)
        cfg = self.native.FrozenLocomotionActionCfg()
        cfg.locomotion_checkpoint = str(low_checkpoint or self.path)
        action = self.native.FrozenLocomotionAction(cfg, env)
        env.action_manager = types.SimpleNamespace(get_term=lambda name: action)
        return env, action, interaction

    def test_v3_phase_holds_stance_and_interpolates_low_level_samples(self):
        saved = torch.load(self.path, weights_only=False)
        saved["model_state_dict"]["actor.9.scale"] = torch.tensor(5.)
        bounded_path = Path(self.temp.name) / "bounded_low.pt"
        torch.save(saved, bounded_path)
        env, action, interaction = self.make_env(bounded_path)
        gait = types.SimpleNamespace(gait_indices=torch.tensor([.25, .5]))
        base = types.SimpleNamespace(is_standing_env=torch.zeros(2, dtype=torch.bool))
        terms = {"interaction": interaction, "gait_command": gait, "base_velocity": base}
        env.command_manager.get_term = terms.__getitem__
        self.assertTrue(action.managed_phase)
        torch.testing.assert_close(action.proprioception()[:, 36:38], torch.tensor([[0., 1.], [0., 1.]]))
        action.process_actions(torch.tensor([[0.] * 9, [.5] + [0.] * 8]))
        self.assertEqual(base.is_standing_env.tolist(), [True, False])
        angle = 2 * math.pi * (.5 + .02 * 1.125)
        expected = torch.tensor([[0., 1.], [math.sin(angle), math.cos(angle)]])
        torch.testing.assert_close(action.proprioception(.02)[:, 36:38], expected)

    def test_native_rate_frame_observations_and_reset(self):
        env, action, interaction = self.make_env()
        calls = []
        handle = action.locomotion.actor.register_forward_pre_hook(lambda _, values: calls.append(values[0].clone()))
        action.process_actions(torch.ones(2, 9))
        for _ in range(8):
            env._sim_step_counter += 1
            action.apply_actions()
        env.episode_length_buf += 1
        obs = self.native.teacher_observation(env)
        critic = self.native.critic_observation(env)
        self.assertEqual(len(calls), 2)
        self.assertEqual(obs.shape, (2, 99))
        self.assertEqual(critic.shape, (2, 106))
        self.assertTrue(all(torch.count_nonzero(x[:, 45:75]) == 0 for x in calls))
        robot = env.scene["robot"]
        torch.testing.assert_close(robot.force[:, 0, :2], interaction.command[:, :2])
        torch.testing.assert_close(robot.torque[:, 0, 2], interaction.command[:, 2])
        self.assertEqual(action.arm_history.last_tick.tolist(), [8, 8])
        self.assertTrue((action.arm_history.data[:, -2:, 12:] > 0).all())
        unchanged = action.low_history.data[1].clone()
        action.reset(torch.tensor([0]))
        self.assertEqual(int(action.low_history.last_tick[0]), -1)
        torch.testing.assert_close(action.low_history.data[1], unchanged)
        action.sample_histories(8)
        self.assertTrue((action.arm_history.data[0, :, 12:] == 0).all())
        handle.remove()

    def test_wrench_units_order_and_reset_without_extra_physics_force(self):
        env, action, interaction = self.make_env()
        physical = torch.arange(60, dtype=torch.float32).reshape(2, 5, 6)
        action.set_low_wrench_prediction(physical)
        calls = []
        handle = action.locomotion.actor.register_forward_pre_hook(lambda _, values: calls.append(values[0].clone()))
        action.process_actions(torch.zeros(2, 9))
        env._sim_step_counter += 1
        action.apply_actions()
        expected = (physical * torch.tensor([.01]*3+[.05]*3)).flatten(1)
        torch.testing.assert_close(calls[0][:, 45:75], expected)
        torch.testing.assert_close(env.scene["robot"].force[:, 0, :2], interaction.command[:, :2])
        action.reset(torch.tensor([0]))
        torch.testing.assert_close(action.low_wrench_prediction[0], torch.zeros(30))
        torch.testing.assert_close(action.low_wrench_prediction[1], expected[1])
        with self.assertRaises(ValueError):
            action.set_low_wrench_prediction(torch.zeros(2, 30))
        with self.assertRaises(ValueError):
            action.set_low_wrench_prediction(torch.full((2, 5, 6), float("nan")))
        handle.remove()

    def test_force_reset_warmup_and_reference_do_not_cross_envs(self):
        env, _, interaction = self.make_env()
        interaction.reference.fill_(.1)
        interaction.reset(torch.tensor([0]))
        torch.testing.assert_close(interaction.reference[1], torch.full((3,), .1))
        env.episode_length_buf[0] = 0
        interaction._update_command()
        self.assertEqual(interaction.command[0].tolist(), [0., 0., 0.])
        self.assertEqual(interaction.reference[0].tolist(), [0., 0., 0.])

    def test_warmup_does_not_shorten_first_force_profile(self):
        env, _, interaction = self.make_env()
        env.episode_length_buf[:] = 0
        for _ in range(49):
            interaction.compute(.04)
            env.episode_length_buf += 1
        torch.testing.assert_close(interaction.time_left, torch.full((2,), 5.))
        torch.testing.assert_close(interaction.elapsed, torch.zeros(2))
        env.episode_length_buf[:] = 50
        interaction.compute(.04)
        torch.testing.assert_close(interaction.time_left, torch.full((2,), 4.96))
        torch.testing.assert_close(interaction.elapsed, torch.full((2,), .04))

    def test_follow_reward_uses_payload_frame_com_velocity(self):
        env, _, interaction = self.make_env()
        robot = env.scene["robot"]
        robot.data.body_quat_w[:] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
        robot.data.body_com_lin_vel_w[:, 0] = torch.tensor([0., .2, 0.])
        interaction.reference[:, 0] = .2
        torch.testing.assert_close(self.native.follow_reward(env), torch.ones(2))

    def test_slew_and_soft_limits(self):
        nominal = torch.zeros(2, 6)
        previous = torch.zeros(2, 9)
        actions = torch.full((2, 9), 100.)
        limits = torch.full((2, 6), .01)
        out = core.bounded_targets(actions, previous, nominal, -limits, limits,
                                   torch.tensor([.35, .2, .4]), .25, torch.ones(9), .04)
        torch.testing.assert_close(out[:, :3], torch.full((2, 3), .04))
        torch.testing.assert_close(out[:, 3:], limits)

    def test_repository_ppo_updates_teacher_without_estimator_or_low_policy(self):
        sys.path.insert(0, str(ROOT / "rsl_rl"))
        from rsl_rl.modules import MLP_Encoder, ActorCritic
        from rsl_rl.algorithm import PPO
        env, action, _ = self.make_env()
        low_before = {k: v.clone() for k, v in action.locomotion.state_dict().items()}
        with contextlib.redirect_stdout(io.StringIO()):
            encoder = MLP_Encoder(num_input_dim=1, num_output_dim=0, hidden_dims=[1])
            teacher = ActorCritic(102, 109, 9, actor_hidden_dims=[16], critic_hidden_dims=[16])
        ppo = PPO(2, encoder, teacher, num_learning_epochs=1, num_mini_batches=1, device="cpu")
        self.assertIsNone(ppo.extra_optimizer)
        ppo.init_storage(2, 4, [99], [109], [1], [3], [9])
        before = teacher.actor[0].weight.clone()
        with torch.no_grad():
            for step in range(4):
                obs = self.native.teacher_observation(env)
                critic = self.native.critic_observation(env)
                high = ppo.act(obs, torch.zeros(2, 1), torch.zeros(2, 3), critic)
                ppo.process_env_step(torch.tensor([1., -.2]), torch.zeros(2, dtype=torch.long), {}, obs)
            ppo.compute_returns(torch.cat((critic, torch.zeros(2, 3)), -1))
        losses = ppo.update()
        self.assertTrue(all(math.isfinite(value) for value in losses))
        self.assertFalse(torch.equal(before, teacher.actor[0].weight))
        for key, value in action.locomotion.state_dict().items():
            torch.testing.assert_close(value, low_before[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
