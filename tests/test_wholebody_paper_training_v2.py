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


class WholeBodyPaperTrainingV2Tests(unittest.TestCase):
    def test_changed_python_files_parse(self):
        for path in (WHOLEBODY_CFG, WRENCH_COMMAND, WRENCH_CFG, OBSERVATIONS, GAIT_COMMAND, PPO_CFG):
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
        self.assertIn('experiment_name = "sfyg_tron_2a_wholebody_paper_v2_flat"', source)
        self.assertIn("max_iterations = 20000", source)


if __name__ == "__main__":
    unittest.main()
