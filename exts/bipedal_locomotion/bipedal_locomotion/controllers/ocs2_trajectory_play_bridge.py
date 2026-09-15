"""Isaac Lab PLAY adapter for an offline OCS2 trajectory."""

from __future__ import annotations

import torch

from .ocs2_play_bridge import Ocs2OutputApplicator
from .ocs2_trajectory import Ocs2Trajectory


class Ocs2TrajectoryPlayBridge(Ocs2OutputApplicator):
    """Replay a validated OCS2 trajectory against simulation time."""

    def __init__(self, env, trajectory_path: str):
        super().__init__(env)
        self._trajectory = Ocs2Trajectory.from_csv(trajectory_path)
        self._start_time: float | None = None
        self._terminal_reported = False
        print(
            f"[INFO] Loaded offline OCS2 trajectory: {trajectory_path}; "
            f"samples={self._trajectory.data.shape[0]}, duration={self._trajectory.duration:.3f}s."
        )

    def update(self, policy_observation: torch.Tensor, command_observation: torch.Tensor) -> bool:
        simulation_time = float(self._env.common_step_counter) * float(self._env.step_dt)
        if self._start_time is None:
            self._start_time = simulation_time
        playback_time = max(0.0, simulation_time - self._start_time)
        solution = self._trajectory.sample(playback_time)
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
