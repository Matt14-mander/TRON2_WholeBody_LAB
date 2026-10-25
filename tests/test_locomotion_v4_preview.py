"""Continuous preview survives automatic resets and produces a playable video."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))
from play_locomotion_v4 import preview


class PreviewTests(unittest.TestCase):
    def run_preview(self, video):
        command = types.SimpleNamespace(roles=torch.zeros(1, dtype=torch.long), cfg=types.SimpleNamespace())
        class Env:
            device = "cpu"
            step_dt = .02
            steps = 0
            renders = 0
            command_manager = types.SimpleNamespace(get_term=lambda name: command)
            def reset(self, seed):
                assert command.roles.item() == 1
            def get_observations(self):
                return {"policy": torch.zeros(1, 117), "obsHistory": torch.zeros(1, 10, 36),
                        "commands": torch.tensor([[.15, 0., 0.]])}
            def step(self, action):
                self.steps += 1
                state = {"command": torch.tensor([[.15, 0., 0.]]),
                    "linear_velocity": torch.tensor([[.01, 0., 0.]]),
                    "yaw_rate": torch.zeros(1), "position": torch.zeros(1, 3),
                    "tilt_deg": torch.ones(1), "payload_wrench": torch.zeros(1, 3)}
                done = self.steps % 2 == 0
                return self.get_observations(), torch.zeros(1), torch.tensor([done]), {
                    "standwalk_state": state, "time_outs": torch.tensor([done])}
            def render(self):
                self.renders += 1
                return np.full((240, 640, 3), 100 + self.steps, dtype=np.uint8)
        env = Env()
        env.unwrapped = env
        runner = types.SimpleNamespace(get_inference_policy=lambda **kw: lambda obs: torch.zeros(1, 10),
            get_inference_encoder=lambda **kw: lambda history: torch.zeros(1, 3))
        args = types.SimpleNamespace(command_interval=3., base_command=None, seed=142,
                                     seconds=.12, video=video, real_time=False)
        (ROOT / ".test-temp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-temp") as tmp:
            output = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                preview(env, runner, args, output, types.SimpleNamespace(is_running=lambda: True))
            report = json.loads((output / "preview.json").read_text())
            rows = [json.loads(line) for line in (output / "preview_steps.jsonl").read_text().splitlines()]
            self.assertEqual(env.steps, 6)
            self.assertEqual(len(rows), 6)
            self.assertTrue(report["completed"])
            self.assertFalse(report["acceptance_evaluated"])
            self.assertEqual(report["timeout_resets"], 3)
            self.assertEqual(report["failure_resets"], 0)
            if video:
                import imageio.v2 as imageio
                reader = imageio.get_reader(output / "preview.mp4")
                try:
                    self.assertEqual(reader.count_frames(), 6)
                    frame = reader.get_data(0)
                    self.assertGreater(float(frame.mean()), 10.)
                    self.assertGreater(float(frame[:90].std()), 1.)
                finally:
                    reader.close()
                self.assertEqual(env.renders, 46)  # Warmup does not advance physics.
            else:
                self.assertEqual(env.renders, 0)

    def test_continuous_rollout_does_not_stop_at_episode_boundaries(self):
        self.run_preview(False)

    def test_video_encodes_all_frames_across_resets_with_overlay(self):
        self.run_preview(True)


if __name__ == "__main__":
    unittest.main()
