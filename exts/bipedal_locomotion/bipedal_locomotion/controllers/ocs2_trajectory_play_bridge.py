"""Isaac Lab PLAY adapter for an offline OCS2 trajectory."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from .ocs2_play_bridge import Ocs2OutputApplicator
from .ocs2_trajectory import Ocs2Trajectory


class Ocs2TrajectoryPlayBridge(Ocs2OutputApplicator):
    """Replay a validated OCS2 trajectory against simulation time."""

    def __init__(
        self,
        env,
        trajectory_path: str,
        zero_base_command: bool = False,
        zero_wrench: bool = False,
    ):
        super().__init__(env)
        self._trajectory = Ocs2Trajectory.from_csv(trajectory_path)
        self._zero_base_command = zero_base_command
        self._zero_wrench = zero_wrench
        self._start_time: float | None = None
        self._terminal_reported = False
        print(
            f"[INFO] Loaded offline OCS2 trajectory: {trajectory_path}; "
            f"samples={self._trajectory.data.shape[0]}, duration={self._trajectory.duration:.3f}s."
        )
        data = self._trajectory.data
        print(
            "[INFO] Offline OCS2 trajectory bounds: "
            f"max_arm_dq={abs(data[:, 7:13]).max():.3f}rad/s, "
            f"max_arm_tau={abs(data[:, 13:19]).max():.3f}Nm, "
            f"max_base_cmd={abs(data[:, 19:22]).max():.3f}, "
            f"max_wrench={abs(data[:, 22:52]).max():.3f}."
        )

    def update(self, policy_observation: torch.Tensor, command_observation: torch.Tensor) -> bool:
        simulation_time = float(self._env.common_step_counter) * float(self._env.step_dt)
        if self._start_time is None:
            self._start_time = simulation_time
        playback_time = max(0.0, simulation_time - self._start_time)
        solution = self._trajectory.sample(playback_time)
        if self._zero_base_command:
            solution = replace(solution, base_velocity_command=np.zeros(3, dtype=np.float64))
        if self._zero_wrench:
            solution = replace(solution, base_wrench_prediction=np.zeros((5, 6), dtype=np.float64))
        self._apply_solution(solution, policy_observation, command_observation)
        if playback_time >= self._trajectory.duration and not self._terminal_reported:
            print(
                f"[INFO] Offline OCS2 trajectory completed at {self._trajectory.duration:.3f}s; "
                "holding terminal arm position with zero base command."
            )
            self._terminal_reported = True
        return True

    def reset(self) -> None:
        self._start_time = None
        self._terminal_reported = False

    def close(self) -> None:
        pass
