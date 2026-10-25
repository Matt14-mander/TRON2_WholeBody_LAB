"""Physical load contracts, native term doubles, TorchScript and real CPU PPO."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
CONTROLLERS = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers"
sys.path[:0] = [str(CONTROLLERS), str(ROOT / "rsl_rl")]
import locomotion_v4 as core
import standwalk_core


def native_module():
    class Command:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env
            self.num_envs, self.device = env.num_envs, env.device

    class Action(Command):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._asset = env.scene["robot"]
            self._joint_ids = list(range(10))

        def process_actions(self, actions):
            self.processed = actions

        def apply_actions(self):
            self._asset.leg_calls += 1

    managers = types.ModuleType("isaaclab.managers")
    managers.CommandTerm, managers.CommandTermCfg = Command, object
    actions = types.ModuleType("isaaclab.envs.mdp.actions")
    actions.JointPositionAction, actions.JointPositionActionCfg = Action, object
    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    maths = types.ModuleType("isaaclab.utils.math")
    def inverse(q, v):
        # Hamilton (w,x,y,z), same convention as Isaac.
        xyz = -q[:, 1:]
        return v + 2 * (q[:, :1] * torch.cross(xyz, v, dim=-1)
                         + torch.cross(xyz, torch.cross(xyz, v, dim=-1), dim=-1))
    maths.quat_apply_inverse = inverse
    modules = {name: types.ModuleType(name) for name in ("isaaclab", "isaaclab.envs", "isaaclab.envs.mdp",
        "bipedal_locomotion", "bipedal_locomotion.controllers")}
    modules.update({"isaaclab.managers": managers, "isaaclab.envs.mdp.actions": actions,
        "isaaclab.utils": utils, "isaaclab.utils.math": maths,
        "bipedal_locomotion.controllers.locomotion_v4": core,
        "bipedal_locomotion.controllers.standwalk_core": standwalk_core})
    path = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion_v4/mdp.py"
    spec = importlib.util.spec_from_file_location("v4_native_terms", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class V4Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / ".test-temp").mkdir(exist_ok=True)

    def test_wrench_disk_and_release_envelope(self):
        torch.manual_seed(7)
        w = core.sample_planar_wrench(10000, 15., 3., "cpu")
        self.assertLessEqual(float(w[:, :2].norm(dim=-1).max()), 15.00001)
        self.assertLessEqual(float(w[:, 2].abs().max()), 3.)
        self.assertLess(float(w[:, :2].mean(0).abs().max()), .3)
        self.assertTrue(torch.equal(core.sample_planar_wrench(5, 0., 0., "cpu"), torch.zeros(5, 3)))

    def test_legacy_and_changed_contracts_rejected(self):
        manifest = core.contract(117, "nominal", (.35, .2, .4), core.STAGES["nominal"])
        with self.assertRaises(ValueError):
            core.check_checkpoint({"model_state_dict": {}}, manifest)
        core.check_checkpoint({"v4_contract": manifest}, manifest)
        changed = dict(manifest, policy_width=118)
        with self.assertRaises(ValueError):
            core.check_checkpoint({"v4_contract": changed}, manifest)

    def test_invalid_load_ranges(self):
        for mass, force, torque, limits in (((0., 1.), 1., 1., (1., 1., 1.)),
            ((1., .5), 1., 1., (1., 1., 1.)), ((.1, 1.), -1., 1., (1., 1., 1.)),
            ((.1, 1.), 1., 1., (float("nan"), 1., 1.))):
            with self.assertRaises(ValueError):
                core.validate_ranges(mass, force, torque, limits)

    def test_command_quota_reset_and_transitions(self):
        mdp = native_module()
        env = types.SimpleNamespace(num_envs=100, device="cpu", step_dt=.02,
                                     episode_length_buf=torch.zeros(100))
        cfg = types.SimpleNamespace(limits=(.35, .2, .4), evaluation_case=-1)
        cmd = mdp.VelocityCommand(cfg, env)
        self.assertEqual(torch.bincount(cmd.roles).tolist(), [20, 40, 20, 20])
        ids = torch.arange(100)
        cmd._resample_command(ids)
        before = cmd.roles.clone()
        env.episode_length_buf[:] = 300  # reset during the middle of a transition
        cfg.evaluation_case, cfg.evaluation_velocity = 2, (.2, 0., 0.)
        cmd._resample_command(ids[:1])
        self.assertTrue(torch.equal(cmd.command[0], torch.zeros(3)))
        self.assertTrue(torch.equal(cmd.roles[1:], before[1:]))
        cmd._update_command()
        self.assertAlmostEqual(float(cmd.command[0, 0]), .2, places=5)
        env.episode_length_buf[:] = 700
        cmd._update_command()
        self.assertTrue(cmd.is_standing_env[0])

    def test_payload_inertia_scales_from_defaults_and_only_selected_envs(self):
        mdp = native_module()
        masses = torch.tensor([[3., .5], [3., .5]])
        inertia = torch.ones(2, 2, 9)
        data = types.SimpleNamespace(default_mass=masses.clone(), default_inertia=inertia.clone())
        view = types.SimpleNamespace(get_masses=lambda: masses.clone(), get_inertias=lambda: inertia.clone(),
            set_masses=lambda value, ids: masses.__setitem__(ids, value[ids]),
            set_inertias=lambda value, ids: inertia.__setitem__(ids, value[ids]))
        env = types.SimpleNamespace(device="cpu", scene={"robot": types.SimpleNamespace(data=data, root_physx_view=view)})
        cfg = types.SimpleNamespace(name="robot", body_ids=[1])
        for _ in range(2):
            mdp.randomize_payload(env, torch.tensor([0]), (1., 1.), cfg)
            self.assertEqual(masses[0, 1], 1.)
            self.assertTrue(torch.equal(inertia[0, 1], torch.full((9,), 2.)))
        self.assertTrue(torch.equal(masses[1], data.default_mass[1]))
        self.assertTrue(torch.equal(inertia[1], data.default_inertia[1]))

    def test_force_reapplied_and_rotated_at_each_physics_step(self):
        mdp = native_module()
        calls = []
        quat = torch.tensor([[[1., 0., 0., 0.]]])
        class Robot:
            leg_calls = 0
            data = types.SimpleNamespace(default_joint_pos=torch.zeros(1, 18), body_quat_w=quat)
            def find_joints(self, names, **kwargs):
                return (list(range(10)), list(core.LEG_JOINTS)) if names == list(core.LEG_JOINTS) else (list(range(10, 18)), [])
            def find_bodies(self, names):
                return [0], ["paint_payload_Link"]
            def set_joint_position_target(self, *a, **kw): pass
            def set_joint_velocity_target(self, *a, **kw): pass
            def set_joint_effort_target(self, *a, **kw): pass
            def set_external_force_and_torque(self, force, torque, **kwargs):
                calls.append((force.clone(), torque.clone()))
        robot = Robot()
        env = types.SimpleNamespace(num_envs=1, device="cpu", scene={"robot": robot},
            command_manager=types.SimpleNamespace(get_command=lambda name: torch.tensor([[10., 0., 2.]])))
        action = mdp.LoadedLegAction(None, env)
        action.apply_actions()
        quat[:] = torch.tensor([[[2 ** -.5, 0., 0., 2 ** -.5]]])
        action.apply_actions()
        self.assertEqual(robot.leg_calls, 2)
        self.assertTrue(torch.allclose(calls[0][0][0, 0], torch.tensor([10., 0., 0.])))
        self.assertTrue(torch.allclose(calls[1][0][0, 0], torch.tensor([0., -10., 0.]), atol=1e-5))
        self.assertTrue(torch.allclose(calls[1][1][0, 0], torch.tensor([0., 0., 2.])))

    def test_height_is_relative_to_base_not_ray_origin(self):
        mdp = native_module()
        env = types.SimpleNamespace(scene=types.SimpleNamespace())
        class Scene(dict): pass
        env.scene = Scene(robot=types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=torch.tensor([[0., 0., .85]]))))
        env.scene.sensors = {"height_scanner": types.SimpleNamespace(data=types.SimpleNamespace(ray_hits_w=torch.zeros(1, 81, 3)))}
        self.assertTrue(torch.allclose(mdp.heights(env), torch.full((1, 81), .35)))

    def test_actual_ppo_update_and_scripted_export_parity(self):
        from rsl_rl.modules import ActorCritic, MLP_Encoder
        from rsl_rl.algorithm import PPO
        with contextlib.redirect_stdout(io.StringIO()):
            actor = ActorCritic(129, 131, 10, actor_hidden_dims=[16], critic_hidden_dims=[16],
                                squash_action=True, squash_action_scale=5., init_noise_std=.6)
            encoder = MLP_Encoder(360, 3, hidden_dims=[16], output_detach=True)
        algorithm = PPO(4, encoder, actor, device="cpu", num_learning_epochs=1, num_mini_batches=1,
                        learning_rate=3e-4, max_learning_rate=3e-4, encoder_max_grad_norm=1.)
        algorithm.init_storage(4, 4, [123], [131], [360], [3], [10])
        before = actor.actor[0].weight.detach().clone()
        for _ in range(4):
            obs, history, cmd, critic = torch.randn(4, 123), torch.randn(4, 360), torch.randn(4, 3), torch.randn(4, 128)
            algorithm.act(obs, history, cmd, critic)
            algorithm.process_env_step(torch.randn(4), torch.zeros(4, dtype=torch.bool), {}, obs)
        algorithm.compute_returns(torch.cat((critic, cmd), -1))
        losses = algorithm.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses))
        self.assertFalse(torch.equal(before, actor.actor[0].weight))
        exported = core.ExportedPolicy(actor.actor, encoder.encoder, 123)
        scripted = torch.jit.script(exported)
        expected = actor.act_inference(torch.cat((encoder.encode(history), obs, cmd), -1)).clamp(-5, 5)
        self.assertTrue(torch.allclose(scripted(obs, history, cmd), expected))
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-temp") as tmp:
            path = Path(tmp) / "v4.pt"
            with path.open("wb") as stream:
                torch.jit.save(scripted, stream)
            with path.open("rb") as stream:
                self.assertTrue(torch.allclose(torch.jit.load(stream)(obs, history, cmd), expected))

    def test_evaluation_continues_after_failed_case_and_writes_summary(self):
        import json
        spec = importlib.util.spec_from_file_location("v4_training_entry", ROOT / "scripts/rsl_rl/train_locomotion_v4.py")
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        term = types.SimpleNamespace(cfg=types.SimpleNamespace(), time_left=torch.zeros(1))
        robot = types.SimpleNamespace(find_bodies=lambda name: ([0], [name]),
            root_physx_view=types.SimpleNamespace(get_masses=lambda: torch.tensor([[.5]])))
        class Env:
            device = "cpu"
            step_dt = .02
            max_episode_length = 1
            max_episode_length_s = .02
            scene = {"robot": robot}
            command_manager = types.SimpleNamespace(get_term=lambda name: term)
            resets = 0
            def reset(self, seed): self.resets += 1
            def get_observations(self):
                return {"policy": torch.zeros(1, 117), "obsHistory": torch.zeros(1, 10, 36),
                        "commands": torch.zeros(1, 3)}
            def step(self, action):
                state = {"command": torch.zeros(1, 3), "linear_velocity": torch.zeros(1, 3),
                    "yaw_rate": torch.zeros(1), "position": torch.zeros(1, 3), "tilt_deg": torch.tensor([40.]),
                    "payload_wrench": torch.zeros(1, 3)}
                return self.get_observations(), torch.zeros(1), torch.tensor([True]), {
                    "standwalk_state": state, "time_outs": torch.tensor([False])}
        env = Env()
        env.unwrapped = env
        runner = types.SimpleNamespace(get_inference_policy=lambda **kw: lambda obs: torch.zeros(1, 10),
            get_inference_encoder=lambda **kw: lambda history: torch.zeros(1, 3))
        args = types.SimpleNamespace(eval_cases=["stand", "walk"], episodes=1, seed=42,
                                    velocity_limits=(.35, .2, .4), terrain_level=0)
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-temp") as tmp:
            with patch.dict(sys.modules, {"bipedal_locomotion.controllers.standwalk_core": standwalk_core,
                                         "bipedal_locomotion.controllers.locomotion_v4": core}):
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = script.evaluate(env, runner, args, Path(tmp), types.SimpleNamespace(is_running=lambda: True))
            self.assertEqual(env.resets, 2)
            self.assertFalse(summary["all_passed"])
            self.assertFalse(summary["interrupted"])
            saved = json.loads((Path(tmp) / "evaluation.json").read_text())
            self.assertEqual([row["case"] for row in saved["episodes"]], ["stand", "walk"])


if __name__ == "__main__":
    unittest.main()
