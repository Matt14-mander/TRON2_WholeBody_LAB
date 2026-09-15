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
        terminal_base_command: tuple[float, float, float] | None = None,
        terminal_transition_duration_s: float = 0.5,
        start_delay_s: float = 0.0,
    ):
        super().__init__(env)
        self._trajectory = Ocs2Trajectory.from_csv(trajectory_path)
        self._zero_base_command = zero_base_command
        self._zero_wrench = zero_wrench
        self._terminal_base_command = (
            None
            if terminal_base_command is None
            else np.asarray(terminal_base_command, dtype=np.float64)
        )
        if self._terminal_base_command is not None:
            if self._terminal_base_command.shape != (3,) or not np.all(
                np.isfinite(self._terminal_base_command)
            ):
                raise ValueError("Terminal base command must contain three finite values.")
            if terminal_transition_duration_s <= 0.0:
                raise ValueError("Terminal base-command transition duration must be positive.")
        self._terminal_transition_duration_s = terminal_transition_duration_s
        if not np.isfinite(start_delay_s) or start_delay_s < 0.0:
            raise ValueError("Trajectory start delay must be finite and non-negative.")
        self._start_delay_s = start_delay_s
        self._start_time: float | None = None
        self._terminal_reported = False
        self._current_solution = None
        self._playback_time = 0.0
        self._phase = "warmup" if self._start_delay_s > 0.0 else "motion"
        print(
            f"[INFO] Loaded offline OCS2 trajectory: {trajectory_path}; "
            f"samples={self._trajectory.data.shape[0]}, duration={self._trajectory.duration:.3f}s, "
            f"start_delay={self._start_delay_s:.3f}s."
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
        elapsed_time = max(0.0, simulation_time - self._start_time)
        playback_time = max(0.0, elapsed_time - self._start_delay_s)
        self._playback_time = playback_time
        solution = self._trajectory.sample(playback_time)
        if elapsed_time < self._start_delay_s:
            # Let contacts, policy history, and actuator targets settle before
            # the arm motion begins. Keep the arm at the first trajectory pose
            # and walk with the requested terminal cruise command when given.
            warmup_base_command = (
                solution.base_velocity_command
                if self._terminal_base_command is None
                else self._terminal_base_command
            )
            solution = replace(
                solution,
                arm_velocity=np.zeros(6, dtype=np.float64),
                base_velocity_command=warmup_base_command.copy(),
            )
        if self._terminal_base_command is not None:
            transition_start = max(
                0.0, self._trajectory.duration - self._terminal_transition_duration_s
            )
            alpha = np.clip(
                (playback_time - transition_start) / self._terminal_transition_duration_s,
                0.0,
                1.0,
            )
            # Cubic smoothstep avoids a velocity-command discontinuity at
            # either end of the transition.
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            base_command = (
                (1.0 - alpha) * solution.base_velocity_command
                + alpha * self._terminal_base_command
            )
            solution = replace(solution, base_velocity_command=base_command)
        if self._zero_base_command:
            solution = replace(solution, base_velocity_command=np.zeros(3, dtype=np.float64))
        if self._zero_wrench:
            solution = replace(solution, base_wrench_prediction=np.zeros((5, 6), dtype=np.float64))
        if elapsed_time < self._start_delay_s:
            self._phase = "warmup"
        elif playback_time < self._trajectory.duration:
            self._phase = "motion"
        else:
            self._phase = "terminal"
        self._current_solution = solution
        self._apply_solution(solution, policy_observation, command_observation)
        if playback_time >= self._trajectory.duration and not self._terminal_reported:
            print(
                f"[INFO] Offline OCS2 trajectory completed at {self._trajectory.duration:.3f}s; "
                "holding terminal arm position."
            )
            self._terminal_reported = True
        return True

    def reset(self) -> None:
        self._start_time = None
        self._terminal_reported = False
        self._current_solution = None
        self._playback_time = 0.0
        self._phase = "warmup" if self._start_delay_s > 0.0 else "motion"

    @property
    def current_solution(self):
        """Most recently applied, post-ablation OCS2 sample."""
        return self._current_solution

    @property
    def playback_time(self) -> float:
        return self._playback_time

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def trajectory_duration(self) -> float:
        return self._trajectory.duration

    def close(self) -> None:
        pass
