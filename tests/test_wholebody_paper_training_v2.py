from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WHOLEBODY_CFG = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_solefoot_yg_tron2a_wholebody_env_cfg.py"
WRENCH_COMMAND = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/commands/wrench_sequence_command.py"
WRENCH_CFG = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/commands/commands_cfg.py"
OBSERVATIONS = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/observations.py"
GAIT_COMMAND = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/commands/gait_command.py"
PPO_CFG = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py"
RUNNER = ROOT / "rsl_rl/rsl_rl/runner/on_policy_runner.py"
PPO = ROOT / "rsl_rl/rsl_rl/algorithm/ppo.py"
ACTOR_CRITIC = ROOT / "rsl_rl/rsl_rl/modules/actor_critic.py"
EVENTS = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/events.py"
PAINT_TEACHER_CORE = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers/paint_teacher_core.py"


class WholeBodyPaperTrainingV3Tests(unittest.TestCase):
    def test_changed_python_files_parse(self):
        for path in (
            WHOLEBODY_CFG,
            WRENCH_COMMAND,
            WRENCH_CFG,
            OBSERVATIONS,
            GAIT_COMMAND,
            PPO_CFG,
            RUNNER,
            PPO,
            ACTOR_CRITIC,
            EVENTS,
            PAINT_TEACHER_CORE,
        ):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_training_and_play_use_different_robot_models(self):
        source = WHOLEBODY_CFG.read_text(encoding="utf-8")
        self.assertIn("def _configure_paper_training", source)
        self.assertIn("self.scene.robot = SOLEFOOT_TRON2A_CFG.replace", source)
        self.assertIn("def _configure_whole_body", source)
        self.assertIn("self.scene.robot = SOLEFOOT_YG_TRON2A_WHOLEBODY_CFG.replace", source)

        tree = ast.parse(source)
        calls_by_class: dict[str, set[str]] = {}
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            calls = {
                child.func.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
            }
            calls_by_class[node.name] = calls

        self.assertIn("_configure_paper_training", calls_by_class["SFYG_TRON2A_WholeBodyFlatEnvCfg"])
        self.assertIn("_configure_paper_training", calls_by_class["SFYG_TRON2A_WholeBodyRoughEnvCfg"])
        self.assertIn("_configure_whole_body", calls_by_class["SFYG_TRON2A_WholeBodyFlatEnvCfg_PLAY"])
        self.assertIn("_configure_whole_body", calls_by_class["SFYG_TRON2A_WholeBodyRoughEnvCfg_PLAY"])

    def test_wrench_is_generated_world_and_observed_in_base(self):
        source = WRENCH_COMMAND.read_text(encoding="utf-8")
        self.assertIn("quat_apply_inverse", source)
        self.assertIn("world_sequence", source)
        self.assertIn("is_global=True", source)
        self.assertIn("self._applied_wrench[env_ids] = applied_b", source)

    def test_phase_observation_uses_reward_phase_buffer(self):
        observation_source = OBSERVATIONS.read_text(encoding="utf-8")
        gait_source = GAIT_COMMAND.read_text(encoding="utf-8")
        function = ast.parse(observation_source).body
        gait_phase = next(
            node for node in function if isinstance(node, ast.FunctionDef) and node.name == "get_managed_gait_phase"
        )
        gait_phase_source = ast.get_source_segment(observation_source, gait_phase)
        self.assertIn("command_term.gait_indices", gait_phase_source)
        self.assertNotIn("episode_length_buf", gait_phase_source)
        self.assertIn("if self.cfg.reset_phase_on_resample", gait_source)
        self.assertIn("self.commands.gait_command.reset_phase_on_resample = True", WHOLEBODY_CFG.read_text(encoding="utf-8"))

    def test_new_run_has_separate_experiment_name(self):
        source = PPO_CFG.read_text(encoding="utf-8")
        self.assertIn('experiment_name = "sfyg_tron_2a_wholebody_paper_v3_flat"', source)
        self.assertIn("max_iterations = 20000", source)

    def test_v3_bounds_policy_and_optimizer_updates(self):
        config_source = PPO_CFG.read_text(encoding="utf-8")
        runner_source = RUNNER.read_text(encoding="utf-8")
        ppo_source = PPO.read_text(encoding="utf-8")
        actor_source = ACTOR_CRITIC.read_text(encoding="utf-8")

        self.assertIn("clip_actions = 5.0", config_source)
        self.assertIn("squash_action=True", config_source)
        self.assertIn("squash_action_scale=5.0", config_source)
        self.assertIn("max_learning_rate=3.0e-4", config_source)
        self.assertIn("encoder_max_grad_norm=1.0", config_source)
        self.assertIn("actions = torch.clamp", runner_source)
        self.assertIn("Policy produced NaN or Inf", runner_source)
        self.assertIn('"Policy/action_saturation_fraction"', runner_source)
        self.assertIn('"encoder_optimizer_state_dict"', runner_source)
        self.assertIn("Checkpoint action-bound architecture", runner_source)
        self.assertIn("start_iter = self.current_learning_iteration", runner_source)
        self.assertIn("{locs['it']}/{locs['tot_iter']}", runner_source)
        self.assertIn("self.max_learning_rate", ppo_source)
        self.assertIn("clip_grad_norm_", ppo_source)
        self.assertIn("actor_layers.append(ScaledTanh(squash_action_scale))", actor_source)
        self.assertIn('register_buffer("scale"', actor_source)
        self.assertIn('if "9.scale" in actor', PAINT_TEACHER_CORE.read_text(encoding="utf-8"))

    def test_zero_wrench_is_an_explicit_training_condition(self):
        env_source = WHOLEBODY_CFG.read_text(encoding="utf-8")
        command_cfg_source = WRENCH_CFG.read_text(encoding="utf-8")
        command_source = WRENCH_COMMAND.read_text(encoding="utf-8")

        self.assertIn("zero_wrench_probability=0.25", env_source)
        self.assertIn("zero_wrench_probability: float = 0.0", command_cfg_source)
        self.assertIn("self._zero_wrench_mask", command_source)
        self.assertIn("noisy_command[zero_mask] = 0.0", command_source)
        self.assertIn("unobserved[zero_mask] = 0.0", command_source)

    def test_wholebody_termination_matches_bounded_action_contract(self):
        source = WHOLEBODY_CFG.read_text(encoding="utf-8")
        self.assertIn('action_out_of_limits.params["threshold"] = 5.01', source)
        self.assertIn("~torch.isfinite(actions).all", EVENTS.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
