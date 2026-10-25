"""Full-body curriculum, dense unloading, collision clearance, and real step acceptance."""
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_locomotion_v4 import core, biped, native_module


class BipedTests(unittest.TestCase):
    def test_bootstrap_has_real_forward_and_backward_commands(self):
        torch.manual_seed(4)
        roles = biped.command_roles(100, "bootstrap", "cpu")
        self.assertEqual(torch.bincount(roles).tolist(), [5, 95])
        commands = biped.movement_targets(2000, "bootstrap", (.35, .2, .4), "cpu")
        self.assertTrue(bool((commands[:, 0].abs() >= .2).all()))
        self.assertTrue(bool((commands[:, 0].abs() <= .35).all()))
        self.assertEqual(int(torch.count_nonzero(commands[:, 1:])), 0)
        self.assertGreater(int((commands[:, 0] < 0).sum()), 200)
        self.assertGreater(int((commands[:, 0] > 0).sum()), 1000)

    def test_omni_and_transition_cover_axes_and_small_movement(self):
        torch.manual_seed(5)
        for skill, counts in (("omni", [10, 80, 5, 5]), ("transitions", [20, 40, 20, 20])):
            self.assertEqual(torch.bincount(biped.command_roles(100, skill, "cpu")).tolist(), counts)
            c = biped.movement_targets(4000, skill, (.35, .2, .4), "cpu")
            self.assertTrue(bool((c.abs() <= torch.tensor([.35, .2, .4]) + 1e-6).all()))
            for axis in range(3):
                pure = ((c != 0).sum(-1) == 1) & (c[:, axis] != 0)
                self.assertGreater(int(pure.sum()), 200)
                self.assertTrue(bool((c[pure, axis] > 0).any()) and bool((c[pure, axis] < 0).any()))
            if skill == "transitions":
                self.assertGreater(int(((c[:, 0].abs() >= .05) & (c[:, 0].abs() <= .1)).sum()), 100)

    def test_unloading_has_dense_signal_and_is_mass_normalized(self):
        desired = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [1., 0.]])
        load = torch.tensor([[.5, .5], [.6, .4], [1., 0.], [0., 0.]])
        mass = torch.full((4,), 40.)
        cost = biped.load_tracking_cost(load * (40 * 9.81), mass, desired)
        self.assertGreater(float(cost[0]), float(cost[1]))
        self.assertGreater(float(cost[1]), float(cost[2]))
        self.assertGreater(float(cost[3]), float(cost[0]))
        torch.testing.assert_close(cost, biped.load_tracking_cost(load * (80 * 9.81), mass * 2, desired))

    def test_motion_shortfall_rejects_standing_and_wrong_direction(self):
        c = torch.tensor([[.2, 0., 0.], [0., 0., -.2], [0., 0., 0.]])
        torch.testing.assert_close(biped.motion_shortfall(c, torch.zeros(3, 3), torch.zeros(3)), torch.tensor([1., 1., 0.]))
        v = c.clone()
        v[:, 2] = 0
        torch.testing.assert_close(biped.motion_shortfall(c, v, c[:, 2]), torch.zeros(3))
        torch.testing.assert_close(biped.motion_shortfall(c, -v, -c[:, 2]), torch.tensor([2., 2., 0.]))

    def test_tracking_is_tight_along_command_and_tolerates_balance_motion(self):
        c = torch.tensor([[.2,0.,0.], [.2,0.,0.], [.2,0.,0.], [0.,.2,0.]])
        v = torch.tensor([[0.,0.,0.], [.2,0.,0.], [.2,.1,0.], [.1,.2,0.]])
        r = biped.velocity_score(c,v)
        self.assertAlmostEqual(float(r[0]), math.exp(-1), places=6)
        self.assertAlmostEqual(float(r[1]), 1.)
        torch.testing.assert_close(r[2],r[3])
        self.assertGreater(float(r[2]), math.exp(-.1**2/.2**2))

    def test_startup_settle_uses_the_same_observation_and_reward_clock(self):
        t = torch.tensor([0., .1, .25])
        features = core.gait_features(t, biped.GAIT)
        torch.testing.assert_close(features[0],features[1])
        torch.testing.assert_close(features[1],features[2])
        c,h,v = core.gait_targets(t,biped.GAIT)
        torch.testing.assert_close(h,torch.zeros_like(h))
        torch.testing.assert_close(v,torch.zeros_like(v))
        mdp = native_module()
        env = types.SimpleNamespace(num_envs=100, device="cpu", step_dt=.02,
            episode_length_buf=torch.zeros(100))
        cfg = types.SimpleNamespace(limits=(.35,.2,.4),profile="bootstrap",evaluation_case=-1)
        command = mdp.VelocityCommand(cfg,env)
        command._resample_command(torch.arange(100))
        torch.testing.assert_close(command.command,torch.zeros(100,3))
        env.episode_length_buf[:] = 25
        command._update_command()
        torch.testing.assert_close(command.command[command.roles==1],command.target[command.roles==1])

    def test_geometry_uses_collision_sole_not_link_origin(self):
        geometry = biped.sole_geometry(ROOT / "robot_description/tron2/SFYG_TRON2A/urdf/robot.urdf")
        self.assertEqual(len(geometry), 2)
        z = []
        for foot in geometry:
            z.append(min(p[2] - r for p, r in zip(foot["points"], foot["radii"])))
        self.assertAlmostEqual(z[0], -.07374, places=5)
        self.assertAlmostEqual(z[1], z[0], places=7)

    def test_nominal_spawn_removes_the_inherited_twelve_centimeter_drop(self):
        urdf = ROOT / "robot_description/tron2/SFYG_TRON2A/urdf/robot.urdf"
        height = biped.nominal_root_height(urdf, {"proximal_yaw_L_Joint": -3.14159, "proximal_yaw_R_Joint": 3.14159})
        # Cross-check against the measured USD pose after its first .02 s step.
        # The root has already fallen ~2 mm from the authored .85 m by then.
        self.assertAlmostEqual(height, .8479769826 - .1223470792 + .01, delta=1e-5)

    def test_sphere_radius_is_world_vertical_and_cache_invalidates(self):
        mdp = native_module()
        class Scene(dict): pass
        class Robot:
            def find_bodies(self, *args, **kwargs): return [0, 1], ["L", "R"]
            def find_joints(self, *args, **kwargs): return list(range(10)), list(core.LEG_JOINTS)
        robot = Robot()
        robot.data = types.SimpleNamespace(body_pos_w=torch.tensor([[[0., .2, .1], [0., -.2, .1]]]),
            body_quat_w=torch.tensor([[[2**-.5, 2**-.5, 0., 0.]]]).repeat(1,2,1),
            body_lin_vel_w=torch.zeros(1,2,3))
        sensor = types.SimpleNamespace(find_bodies=lambda *a, **kw: ([0,1], ["L","R"]),
            data=types.SimpleNamespace(net_forces_w=torch.tensor([[[0.,0.,200.],[0.,0.,200.]]])))
        scene = Scene(robot=robot)
        scene.sensors = {"contact_forces": sensor, "height_scanner": types.SimpleNamespace(
            data=types.SimpleNamespace(ray_hits_w=torch.zeros(1,81,3)))}
        env = types.SimpleNamespace(num_envs=1, device="cpu", _sim_step_counter=0, scene=scene,
            cfg=types.SimpleNamespace(biped={"sole_geometry": [{"points": [[0.,0.,0.]], "radii": [.1]}] * 2}))
        torch.testing.assert_close(mdp.biped_feet(env)["clearance"], torch.zeros(1,2))
        robot.data.body_pos_w[...,2] += .1
        torch.testing.assert_close(mdp.biped_feet(env)["clearance"], torch.zeros(1,2))
        env._sim_step_counter += 4
        torch.testing.assert_close(mdp.biped_feet(env)["clearance"], torch.full((1,2), .1))
        robot.data.body_pos_w[...,2] += .1
        env._biped_feet_cache = None  # reset, without advancing the physics clock
        torch.testing.assert_close(mdp.biped_feet(env)["clearance"], torch.full((1,2), .2))

    def test_reset_randomizes_only_legs_and_restores_real_arm(self):
        mdp = native_module()
        default = torch.zeros(4,18)
        default[:,10:16] = torch.tensor([0.,1.57,-1.48,0.,0.,0.])
        default[:,16:] = torch.tensor([.05,-.05])
        calls = []
        robot = types.SimpleNamespace(data=types.SimpleNamespace(default_joint_pos=default,
            default_joint_vel=torch.zeros_like(default), joint_pos_limits=torch.tensor([-4.,4.]).repeat(4,18,1)),
            write_joint_state_to_sim=lambda q, dq, **kw: calls.append((q.clone(),dq.clone(),kw)),
            set_joint_position_target=lambda *a, **kw: None)
        env = types.SimpleNamespace(scene={"robot":robot}, num_envs=4, device="cpu",
                                     _biped_air_peak=torch.ones(4,2))
        selected = torch.tensor([1,3])
        cfg = types.SimpleNamespace(name="robot",joint_ids=list(range(10)))
        mdp.reset_biped_joints(env, selected, cfg, offset=.05)
        q, dq, kw = calls[0]
        torch.testing.assert_close(q[:,10:], default[selected,10:])
        self.assertTrue(bool((q[:,:10].abs() <= .05).all()))
        self.assertGreater(int(torch.count_nonzero(q[:,:10])), 0)
        torch.testing.assert_close(dq, torch.zeros_like(dq))
        torch.testing.assert_close(env._biped_air_peak[selected], torch.zeros(2,2))
        torch.testing.assert_close(env._biped_air_peak[[0,2]], torch.ones(2,2))

    def test_contact_switches_without_clearance_cannot_pass(self):
        for real_step in (False, True):
            metric = biped.SteppingMetrics()
            for step in range(1000):
                t = (step + 1) * .02
                desired, height, _ = core.gait_targets(torch.tensor([t]), biped.GAIT)
                contact = desired[0] > .5
                metric.update(t, torch.tensor([.25, 0., 0.]), contact.float() * 300,
                    height[0] if real_step else torch.zeros(2), torch.zeros(2, 3), .02)
            self.assertEqual(metric.result()["stepping_passed"], real_step)
        stationary = biped.SteppingMetrics()
        for step in range(1000):
            stationary.update((step + 1) * .02, torch.tensor([.25, 0., 0.]),
                              torch.full((2,), 200.), torch.zeros(2), torch.zeros(2, 3), .02)
        self.assertFalse(stationary.result()["stepping_passed"])

    def test_biped_contract_rejects_old_standing_actor_and_keeps_height_scan(self):
        old = core.contract(123, "nominal", (.35, .2, .4), core.STAGES["nominal"], core.DEFAULT_GAIT)
        new = core.contract(135, "nominal", (.35, .2, .4), core.STAGES["nominal"], biped.GAIT, {"sole_geometry": []})
        self.assertEqual(new["height_width"], 81)
        self.assertEqual(new["history_width"], 360)
        core.check_checkpoint({"v4_contract": new}, new)
        with self.assertRaises(ValueError):
            core.check_checkpoint({"v4_contract": old}, new)

    def test_promotion_requires_matching_successful_checkpoint_report(self):
        saved = {"training_objective": {"skill_stage": "bootstrap"}}
        report = {"checkpoint_sha256": "abc", "skill_stage": "bootstrap",
            "acceptance_version": "v4_biped_stepping_v1", "interrupted": False,
            "episodes": [{"case": c, "passed": True} for c in ("walk", "forward") for _ in range(10)]}
        (ROOT / ".test-temp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-temp") as temp:
            path = Path(temp) / "evaluation.json"
            path.write_text(json.dumps(report))
            biped.validate_promotion(saved, "omni", path, "abc")
            with self.assertRaises(ValueError):
                biped.validate_promotion(saved, "omni", path, "wrong")
            with self.assertRaises(ValueError):
                biped.validate_promotion(saved, "transitions", path, "abc")
            report["episodes"][0]["passed"] = False
            report["episodes"][1]["passed"] = False
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                biped.validate_promotion(saved, "omni", path, "abc")

    def test_cli_resolves_biped_defaults_and_keeps_baseline(self):
        spec = importlib.util.spec_from_file_location("biped_cli", ROOT / "scripts/rsl_rl/train_locomotion_v4.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for design, profile, std, weight in (("baseline", "mixed", .5, .75), ("biped", "bootstrap", .2, 1.)):
            args = module.parser_for_task().parse_args(["--design", design])
            with patch.dict(sys.modules, {"bipedal_locomotion.controllers.locomotion_v4": core}):
                module.settings_for_args(args)
            self.assertEqual((args.command_profile, args.tracking_std, args.gait_weight), (profile, std, weight))

    def test_real_ppo_and_export_on_new_full_body_input(self):
        from rsl_rl.modules import ActorCritic, MLP_Encoder
        from rsl_rl.algorithm import PPO
        with contextlib.redirect_stdout(io.StringIO()):
            actor = ActorCritic(141, 176, 10, actor_hidden_dims=[16], critic_hidden_dims=[16], init_noise_std=.8,
                                squash_action=True, squash_action_scale=5.)
            encoder = MLP_Encoder(360, 3, hidden_dims=[16], output_detach=True)
        algo = PPO(4, encoder, actor, update_kl_guard=True, num_learning_epochs=2, num_mini_batches=2,
                   desired_kl=.01, learning_rate=1e-4, est_learning_rate=3e-5)
        algo.init_storage(4, 4, [135], [176], [360], [3], [10])
        for _ in range(4):
            p, h, c, v = torch.randn(4,135), torch.randn(4,360), torch.randn(4,3), torch.randn(4,173)
            with torch.no_grad():
                algo.act(p,h,c,v)
                algo.process_env_step(torch.ones(4), torch.zeros(4, dtype=torch.bool), {}, p)
        algo.compute_returns(torch.cat((v,c),-1))
        algo.update()
        self.assertLessEqual(algo.last_update_stats["post_update_probe_kl"], .015 + 1e-6)
        exported = torch.jit.script(core.ExportedPolicy(actor.actor, encoder.encoder, 135))
        with torch.no_grad():
            expected = actor.act_inference(torch.cat((encoder.encode(h),p,c),-1)).clamp(-5,5)
            torch.testing.assert_close(expected, exported(p,h,c))


if __name__ == "__main__":
    unittest.main()
