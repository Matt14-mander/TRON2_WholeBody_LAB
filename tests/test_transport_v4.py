"""Transport contracts, native action doubles, frozen-backbone parity and PPO."""
import contextlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "exts/bipedal_locomotion/bipedal_locomotion"
sys.path.insert(0, str(ROOT / "rsl_rl"))


def load_core():
    packages = {}
    for name, path in (("bipedal_locomotion", PACKAGE), ("bipedal_locomotion.controllers", PACKAGE / "controllers")):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        packages[name] = module
    with patch.dict(sys.modules, packages):
        module = importlib.import_module("bipedal_locomotion.controllers.paint_transport_v4")
        module._test_low_core = importlib.import_module("bipedal_locomotion.controllers.locomotion_v4")
        return module


core = load_core()


def native_module():
    class Term:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env
            self.num_envs, self.device = env.num_envs, env.device
            self._asset = env.scene["robot"]
    managers = types.ModuleType("isaaclab.managers")
    managers.ActionTerm = managers.CommandTerm = Term
    managers.ActionTermCfg = managers.CommandTermCfg = object
    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    maths = types.ModuleType("isaaclab.utils.math")
    def apply(q, v):
        return v + 2 * (q[..., :1] * torch.cross(q[..., 1:], v, dim=-1)
            + torch.cross(q[..., 1:], torch.cross(q[..., 1:], v, dim=-1), dim=-1))
    def inverse(q, v):
        conjugate = q.clone()
        conjugate[..., 1:] *= -1
        return apply(conjugate, v)
    maths.quat_apply, maths.quat_apply_inverse = apply, inverse
    names = {"isaaclab": types.ModuleType("isaaclab"), "isaaclab.managers": managers,
        "isaaclab.utils": utils, "isaaclab.utils.math": maths,
        "bipedal_locomotion.controllers.paint_transport_v4": core}
    spec = importlib.util.spec_from_file_location("transport_native_double", PACKAGE / "tasks/transport_v4/mdp.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, names):
        spec.loader.exec_module(module)
    return module


def fake_native():
    mdp = native_module()
    data = types.SimpleNamespace(default_joint_pos=torch.zeros(2, 18), default_joint_vel=torch.zeros(2, 18),
        joint_pos=torch.zeros(2, 18), joint_vel=torch.zeros(2, 18), joint_acc=torch.zeros(2, 18),
        root_ang_vel_b=torch.zeros(2, 3), projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(2, 1),
        root_pos_w=torch.tensor([[0., 0., .85]]).repeat(2, 1),
        root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
        body_quat_w=torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]]).repeat(2, 1, 1),
        soft_joint_pos_limits=torch.tensor([-3., 3.]).repeat(2, 18, 1))
    force_calls = []
    class Robot:
        def find_joints(self, names, **kwargs):
            if names == list(core.LEG_JOINTS): return list(range(10)), list(core.LEG_JOINTS)
            if names == list(core.ARM_JOINTS): return list(range(10, 16)), list(core.ARM_JOINTS)
            return [16, 17], ["gripper1_Joint", "gripper2_Joint"]
        def find_bodies(self, name):
            return ([1], [name]) if name == "paint_payload_Link" else ([0], [name])
        def set_joint_position_target(self, *a, **kw): pass
        def set_joint_velocity_target(self, *a, **kw): pass
        def set_joint_effort_target(self, *a, **kw): pass
        def set_external_force_and_torque(self, f, t, **kw):
            force_calls.append((f.clone(), t.clone(), kw))
    robot = Robot()
    robot.data = data
    class Scene(dict): pass
    scene = Scene(robot=robot)
    scene.sensors = {"height_scanner": types.SimpleNamespace(data=types.SimpleNamespace(ray_hits_w=torch.zeros(2, 81, 3)))}
    reference = torch.tensor([[.1, 0., .1], [.1, 0., .1]])
    env = types.SimpleNamespace(num_envs=2, device="cpu", physics_dt=.005, step_dt=.02,
        _sim_step_counter=0, cfg=types.SimpleNamespace(decimation=4), scene=scene,
        episode_length_buf=torch.zeros(2), command_manager=types.SimpleNamespace(get_command=lambda name: reference))
    class Low:
        manifest = {"velocity_limits": [.35, .2, .4]}
        calls = []
        def __init__(self, *a): pass
        def __call__(self, p, h, c):
            self.calls.append((p.clone(), h.clone(), c.clone()))
            return torch.full((2, 10), .1)
    cfg = types.SimpleNamespace(low_path="", low_format="checkpoint", low_manifest="", base_limits=(.35, .2, .4),
        arm_scale=.1, arm_rate=1., student_noise=False, disturbance_scale=0.)
    with patch.object(mdp, "FrozenBackbone", Low):
        term = mdp.TransportAction(cfg, env)
    env.action_manager = types.SimpleNamespace(get_term=lambda name: term)
    return mdp, env, term, force_calls, reference


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / ".test-temp").mkdir(exist_ok=True)

    def test_histories_zero_repeat_idempotent_and_partial_reset(self):
        h = core.History(2, 8, 18, "cpu")
        row = torch.ones(2, 18)
        h.append(row, 0)
        self.assertTrue(torch.equal(h.data[:, :-1], torch.zeros(2, 7, 18)))
        h.append(row * 4, 0)
        self.assertTrue(torch.equal(h.data[:, -1], row))
        saved = h.data[1].clone()
        h.reset([0])
        h.append(row * 2, 0)
        self.assertTrue(torch.equal(h.data[1], saved))
        low = core.History(2, 10, 36, "cpu", repeat_first=True)
        low.append(torch.ones(2, 36), 0)
        self.assertTrue(torch.equal(low.data, torch.ones(2, 10, 36)))

    def test_guide_profile_and_physical_action_units(self):
        t = torch.tensor([0., .5, 1., 5., 9., 9.5, 10.])
        self.assertTrue(torch.allclose(core.envelope(t, 10.), torch.tensor([0., .5, 1., 1., 1., .5, 0.])))
        raw = torch.zeros(2, 9)
        raw[0, :3] = torch.tensor([.05, -.1, .2])
        raw[1, :3] = torch.tensor([1., -.9, .02])
        out = core.base_command(raw, (.35, .2, .4))
        self.assertTrue(torch.allclose(out, torch.tensor([[.05, -.1, .2], [.35, -.2, 0.]])))
        raw[:, 3:] = 1.
        arm = core.arm_targets(raw, torch.zeros(2, 6), torch.zeros(2, 6),
            torch.tensor([-.01, .01]).repeat(2, 6, 1), .1, 1., .02)
        self.assertTrue(torch.allclose(arm, torch.full((2, 6), .01)))

    def test_tracking_reward_prefers_correct_response(self):
        ref = torch.tensor([[.1, -.1, .2]])
        correct = core.tracking_scores(ref, ref)
        wrong = core.tracking_scores(-ref, ref)
        self.assertEqual(float(correct[0]), 1.)
        self.assertGreater(float(correct[0]), float(wrong[0]))
        self.assertGreater(float(correct[1]), float(wrong[1]))

    def test_tron2_height_band_can_cover_nominal_carry_posture(self):
        mdp, env, _, _, _ = fake_native()
        env.scene["robot"].data.body_com_pos_w = torch.tensor([[[0., 0., 1.39], [0., 0., 1.39]]]).repeat(2, 1, 1)
        env.cfg.payload_height_range = (.44, .64)
        self.assertTrue(torch.equal(mdp.payload_height(env), torch.zeros(2)))
        env.cfg.payload_height_range = (.05, .25)
        self.assertTrue((mdp.payload_height(env) > .08).all())

    def test_interaction_samples_only_at_reset_and_respects_episode_profile(self):
        mdp, env, _, _, _ = fake_native()
        env.num_envs = 2000
        env.episode_length_buf = torch.zeros(2000)
        env.max_episode_length_s = 10.
        cfg = types.SimpleNamespace(max_wrench=(3., 3., .5), zero_probability=.02, case="random")
        command = mdp.Interaction(cfg, env)
        torch.manual_seed(42)
        command._resample_command(torch.arange(2000))
        target = command.target.clone()
        fraction_zero = float((target == 0).all(-1).float().mean())
        self.assertTrue(.005 < fraction_zero < .04)
        command._update_command()
        self.assertTrue(torch.equal(command.command, torch.zeros(2000, 3)))
        env.episode_length_buf[:] = 50  # 1 second = end of ramp
        command._update_command()
        self.assertTrue(torch.allclose(command.command, target))
        env.episode_length_buf[:] = 500
        command._update_command()
        self.assertTrue(torch.equal(command.command, torch.zeros(2000, 3)))
        self.assertTrue(torch.equal(command.target, target))

    def test_terrain_curriculum_uses_wrench_reference_and_ignores_zero_force(self):
        mdp = native_module()
        seen = []
        terrain = types.SimpleNamespace(terrain_levels=torch.zeros(3),
            update_env_origins=lambda ids, up, down: seen.append((up.clone(), down.clone())))
        class Scene(dict): pass
        scene = Scene(robot=types.SimpleNamespace(data=types.SimpleNamespace(
            root_pos_w=torch.tensor([[1., 0., .8], [.1, 0., .8], [1., 0., .8]]))))
        scene.terrain = terrain
        env = types.SimpleNamespace(device="cpu", scene=scene, expected_distance=torch.tensor([1., 1., 0.]),
            reference_displacement=torch.tensor([[1., 0.], [1., 0.], [0., 0.]]),
            start_xy=torch.zeros(3, 2), episode_length_buf=torch.full((3,), 500),
            command_manager=types.SimpleNamespace(get_term=lambda name: types.SimpleNamespace(
                target=torch.tensor([[.1, 0., 0.], [.1, 0., 0.], [0., 0., 0.]]))),
            termination_manager=types.SimpleNamespace(terminated=torch.tensor([False, True, False])))
        mdp.terrain_progress(env, torch.arange(3))
        self.assertEqual(seen[0][0].tolist(), [True, False, False])
        self.assertEqual(seen[0][1].tolist(), [False, True, False])

    def test_teacher_checkpoint_rejects_different_action_contract(self):
        manifest = {"schema": core.SCHEMA, "actor_mode": "teacher", "low_sha256": "abc",
            "observation_shapes": {"policy": [147]}, "control_hz": 50, "arm_pd": [60., 6.],
            "arguments": {"arm_scale": .1, "base_limits": [.35, .2, .4]}}
        core.check_teacher_contract(manifest, manifest)
        with self.assertRaises(ValueError):
            core.check_teacher_contract(dict(manifest, arguments={"arm_scale": .5}), manifest)

    def test_timeout_and_failure_together_cannot_pass_and_all_cases_are_recorded(self):
        spec = importlib.util.spec_from_file_location("transport_entry", ROOT / "scripts/rsl_rl/train_transport_v4.py")
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        command = types.SimpleNamespace(cfg=types.SimpleNamespace(case="random"))
        class Env:
            num_envs = 1
            device = "cpu"
            step_dt = max_episode_length_s = .02
            max_episode_length = 1
            command_manager = types.SimpleNamespace(get_term=lambda name: command)
            def reset(self, seed): pass
            def get_observations(self): return {"policy": torch.zeros(1, 147)}
            def step(self, action):
                state = {"base_velocity": torch.zeros(1, 3), "reference": torch.zeros(1, 3), "base_tilt_deg": torch.zeros(1)}
                return self.get_observations(), torch.zeros(1), torch.tensor([True]), {
                    "transport_state": state, "transport_terminated": torch.tensor([True]),
                    "transport_done_terms": {"fall": torch.tensor([True]), "time_out": torch.tensor([True])},
                    "time_outs": torch.tensor([True]), "transport_rewards": {"termination": torch.tensor([-50.])}}
        env = Env()
        env.unwrapped = env
        args = types.SimpleNamespace(mode="eval", cases=["zero", "forward"], zero_force=False,
            episodes=1, seed=42, controller="hold", realtime=False)
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-temp") as tmp:
            result = script.rollout(env, args, types.SimpleNamespace(is_running=lambda: True), Path(tmp))
            self.assertFalse(result["all_passed"])
            self.assertEqual(result["failures"], 2)
            self.assertEqual(result["timeouts"], 0)
            self.assertEqual(len(result["episodes"]), 2)

            # A stationary robot must not pass a weak reference below 0.1 m/s.
            def stationary_step(action):
                state = {"base_velocity": torch.zeros(1, 3), "reference": torch.tensor([[.075, 0., 0.]]),
                         "base_tilt_deg": torch.zeros(1)}
                return env.get_observations(), torch.zeros(1), torch.tensor([True]), {
                    "transport_state": state, "transport_terminated": torch.tensor([False]),
                    "transport_done_terms": {"time_out": torch.tensor([True])}, "time_outs": torch.tensor([True]),
                    "transport_rewards": {"force_tracking": torch.zeros(1)}}
            env.step = stationary_step
            args.cases = ["forward"]
            result = script.rollout(env, args, types.SimpleNamespace(is_running=lambda: True), Path(tmp))
            self.assertTrue(result["episodes"][0]["completed"])
            self.assertFalse(result["all_passed"])
            self.assertAlmostEqual(result["episodes"][0]["linear_rmse_limit"], .0375, places=5)

            args.controller = "probe"
            args.probe_command = (.04, 0., 0.)
            result = script.rollout(env, args, types.SimpleNamespace(is_running=lambda: True), Path(tmp))
            self.assertEqual(result["tracking_target"], "probe_command")
            self.assertAlmostEqual(result["episodes"][0]["reference_rms_vx_vy_wz"][0], .04, places=5)

    def test_low_inference_once_per_control_and_separate_last_leg_action(self):
        _, env, term, calls, _ = fake_native()
        actions = torch.zeros(2, 9)
        actions[:, 0] = .05
        term.process_actions(actions)
        self.assertEqual(len(term.low.calls), 1)
        self.assertTrue(torch.equal(term.low.calls[0][0][:, 26:36], torch.zeros(2, 10)))
        for _ in range(4): term.apply_actions()
        self.assertEqual(len(term.low.calls), 1)
        self.assertEqual(len(calls), 4)
        env._sim_step_counter += 4
        term.process_actions(actions)
        self.assertTrue(torch.allclose(term.low.calls[-1][0][:, 26:36], torch.full((2, 10), .1)))
        self.assertTrue(torch.allclose(term.leg_target, torch.full((2, 10), .025)))
        before = term.low_history.data[1].clone()
        term.reset(torch.tensor([0]))
        self.assertTrue(torch.equal(term.low_history.data[0], torch.zeros(10, 36)))
        self.assertTrue(torch.equal(term.low_history.data[1], before))

    def test_payload_force_rotates_base_to_world_to_body_every_substep(self):
        _, env, term, calls, _ = fake_native()
        env.scene["robot"].data.root_quat_w[:] = torch.tensor([2 ** -.5, 0., 0., 2 ** -.5])
        term.process_actions(torch.zeros(2, 9))
        term.apply_actions()
        self.assertTrue(torch.allclose(calls[0][0][:, 0], torch.tensor([[0., 4., 0.]]).repeat(2, 1), atol=1e-5))
        self.assertTrue(torch.allclose(calls[0][1][:, 0], torch.tensor([[0., 0., 1.]]).repeat(2, 1)))
        env.scene["robot"].data.body_quat_w[:, 1] = env.scene["robot"].data.root_quat_w
        term.apply_actions()
        self.assertTrue(torch.allclose(calls[1][0][:, 0], torch.tensor([[4., 0., 0.]]).repeat(2, 1), atol=1e-5))

    def test_student_has_no_wrench_and_teacher_gets_current_wrench(self):
        mdp, env, term, _, reference = fake_native()
        student = mdp.student_observation(env).clone()
        teacher = mdp.teacher_observation(env).clone()
        self.assertEqual(tuple(student.shape), (2, 144))
        self.assertEqual(tuple(teacher.shape), (2, 147))
        reference *= -1
        self.assertTrue(torch.equal(mdp.student_observation(env), student))
        self.assertTrue(torch.equal(mdp.teacher_observation(env)[:, -3:], reference))
        self.assertTrue(torch.equal(mdp.teacher_observation(env)[:, :144], teacher[:, :144]))

    def test_student_noise_is_uniform_and_teacher_history_clean(self):
        mdp, env, term, _, _ = fake_native()
        term.cfg.student_noise = True
        student = mdp.student_observation(env).reshape(2, 8, 18)
        teacher = mdp.teacher_observation(env)[:, :144].reshape(2, 8, 18)
        self.assertTrue(torch.equal(teacher, torch.zeros_like(teacher)))
        self.assertLessEqual(float(student[..., :6].abs().max()), .01)
        self.assertLessEqual(float(student[..., 6:12].abs().max()), 1.5)
        self.assertTrue(torch.equal(student[..., 12:], torch.zeros_like(student[..., 12:])))

    def test_checkpoint_and_torchscript_backbone_parity_and_frozen_weights(self):
        low_core = core._test_low_core
        contract, STAGES, ExportedPolicy = low_core.contract, low_core.STAGES, low_core.ExportedPolicy
        actor = core.mlp([123, 16, 8, 10])
        actor.add_module(str(len(actor)), core.CheckpointScaledTanh(5.))
        encoder = core.mlp([360, 12, 3])
        manifest = contract(117, "nominal", (.35, .2, .4), STAGES["nominal"])
        manifest["agent"] = {"policy": {"activation": "elu", "actor_hidden_dims": [16, 8], "squash_action": True,
            "squash_action_scale": 5.}, "encoder": {"activation": "elu", "hidden_dims": [12]}}
        checkpoint = {"v4_contract": manifest,
            "model_state_dict": {"actor." + k: v for k, v in actor.state_dict().items()},
            "encoder_state_dict": {"encoder." + k: v for k, v in encoder.state_dict().items()}}
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-temp") as tmp:
            folder = Path(tmp)
            path = folder / "model.pt"
            torch.save(checkpoint, path)
            model = core.FrozenBackbone(path)
            before = model.fingerprint()
            with (folder / "locomotion_v4.pt").open("wb") as f:
                torch.jit.save(torch.jit.script(ExportedPolicy(actor, encoder, 117)), f)
            (folder / "contract.json").write_text(json.dumps(manifest), encoding="utf-8")
            exported = core.FrozenBackbone(folder / "locomotion_v4.pt", file_format="torchscript")
            p,h,c = torch.randn(3, 117), torch.randn(3, 10, 36), torch.randn(3, 3)
            self.assertTrue(torch.allclose(model(p,h,c), exported(p,h,c), atol=1e-6))
            self.assertFalse(model(p,h,c).requires_grad)
            self.assertTrue(all(not v.requires_grad for v in model.parameters()))
            self.assertEqual(before, model.fingerprint())
        with self.assertRaises(ValueError): core.validate_low_manifest({"schema": "v3"})

    def test_teacher_ppo_updates_without_training_low_or_estimator(self):
        from rsl_rl.modules import ActorCritic, MLP_Encoder
        from rsl_rl.algorithm import PPO
        with contextlib.redirect_stdout(io.StringIO()):
            actor = ActorCritic(147, 177, 9, actor_hidden_dims=[16], critic_hidden_dims=[16],
                                bounded_distribution=True, init_noise_std=.2)
            encoder = MLP_Encoder(1, 0, hidden_dims=[1], output_detach=True)
        algorithm = PPO(4, encoder, actor, device="cpu", num_learning_epochs=1, num_mini_batches=1)
        self.assertIsNone(algorithm.extra_optimizer)
        algorithm.init_storage(4, 4, [147], [177], [1], [0], [9])
        before = actor.actor[0].weight.clone()
        for _ in range(4):
            obs, history, cmd, critic = torch.randn(4, 147), torch.zeros(4, 1), torch.zeros(4, 0), torch.randn(4, 177)
            action = algorithm.act(obs, history, cmd, critic)
            self.assertLessEqual(float(action.abs().max()), 1.)
            algorithm.process_env_step(torch.randn(4), torch.zeros(4, dtype=torch.bool), {}, obs)
        algorithm.compute_returns(critic)
        losses = algorithm.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses))
        self.assertFalse(torch.equal(before, actor.actor[0].weight))


if __name__ == "__main__": unittest.main()
