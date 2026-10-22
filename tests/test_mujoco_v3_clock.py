"""Keep policy phase/history frequency fixed when testing SDK publication rates."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/mujoco/run_wholebody_v3.py"
spec = importlib.util.spec_from_file_location("v3_clock", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ClockTests(unittest.TestCase):
    def test_publish_rate_preserves_policy_rate(self):
        for publish_hz, expected in ((100, 2), (500, 10)):
            cfg = SimpleNamespace(policy_frequency=50., loop_frequency=500., decimation=10)
            module.configure_publish_rate(cfg, publish_hz)
            self.assertEqual(cfg.decimation, expected)
            self.assertEqual(cfg.loop_frequency / cfg.decimation, 50.)
            self.assertEqual(cfg.policy_frequency, 50.)

    def test_invalid_ratio_is_rejected_without_mutation(self):
        for invalid in (0, 25, 99, float("nan")):
            cfg = SimpleNamespace(policy_frequency=50., loop_frequency=500., decimation=10)
            with self.assertRaises(ValueError):
                module.configure_publish_rate(cfg, invalid)
            self.assertEqual(cfg.loop_frequency, 500.)
            self.assertEqual(cfg.decimation, 10)


if __name__ == "__main__":
    unittest.main()
