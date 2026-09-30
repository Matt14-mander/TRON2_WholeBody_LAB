"""Keep the offline OCS2 initial observation aligned with SFYG's Isaac reset."""

import ast
from pathlib import Path
import re
import unittest


REPO = Path(__file__).resolve().parents[1]
TASK_INFO = REPO / "ocs2_ws/src/tron2_ocs2/config/task.info"
ROBOT_CFG = (
    REPO
    / "exts/bipedal_locomotion/bipedal_locomotion/assets/config"
    / "solefoot_yg_tron2a_wholebody_cfg.py"
)


class Ocs2InitialHeightTest(unittest.TestCase):
    def test_offline_initial_height_matches_isaac_reset(self):
        task_text = TASK_INFO.read_text(encoding="utf-8")
        match = re.search(r"(?m)^\s*initialBaseHeight\s+([0-9.eE+-]+)\s*$", task_text)
        self.assertIsNotNone(match, "OCS2 task.info must configure initialBaseHeight")
        initial_height = float(match.group(1))

        module = ast.parse(ROBOT_CFG.read_text(encoding="utf-8"))
        reset_positions = (
            ast.literal_eval(keyword.value)
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "InitialStateCfg"
            for keyword in node.keywords
            if keyword.arg == "pos"
        )
        reset_position = next(reset_positions)
        self.assertAlmostEqual(initial_height, reset_position[2])


if __name__ == "__main__":
    unittest.main()
