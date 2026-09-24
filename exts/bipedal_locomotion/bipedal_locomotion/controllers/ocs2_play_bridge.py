"""Isaac Lab PLAY adapter for the native TRON2 OCS2 TCP service."""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from isaaclab.assets import Articulation

from .ocs2_interface import (
    Ocs2AsyncClient,
    Ocs2BridgeError,
    Ocs2MpcObservation,
    Ocs2MpcSolution,
    Ocs2TcpClient,
)


ARM_JOINT_NAMES = tuple(f"arm{index}_Joint" for index in range(1, 7))


class Ocs2OutputApplicator:
    """Shared Isaac Lab output path for live and offline OCS2 controllers."""

    def __init__(self, env):
        if env.num_envs != 1:
            raise ValueError("OCS2 PLAY control currently supports exactly one Isaac Lab environment.")
        self._env = env
        self._robot: Articulation = env.scene["robot"]
        self._arm_joint_ids, arm_joint_names = self._robot.find_joints(
            list(ARM_JOINT_NAMES), preserve_order=True
        )
        if tuple(arm_joint_names) != ARM_JOINT_NAMES:
            raise RuntimeError(f"Unexpected arm joint order: {arm_joint_names}.")
        self._wrench_term = env.command_manager.get_term("arm_wrench")
        if not hasattr(self._wrench_term, "set_external_prediction"):
            raise RuntimeError("arm_wrench command term does not support external OCS2 predictions.")
        self._policy_wrench_slice = self._find_policy_wrench_slice()

    @staticmethod
    def _numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor[0].detach().cpu().numpy().astype(np.float64, copy=True)

    def _find_policy_wrench_slice(self) -> slice:
        manager = self._env.observation_manager
        names = manager._group_obs_term_names["policy"]
        dimensions = manager._group_obs_term_dim["policy"]
        try:
            term_index = names.index("wrench_prediction")
        except ValueError as error:
            raise RuntimeError("Policy observation has no wrench_prediction term.") from error
        widths = [dimension if isinstance(dimension, int) else math.prod(dimension) for dimension in dimensions]
        if widths[term_index] != 30:
            raise RuntimeError(
                f"wrench_prediction observation has width {widths[term_index]}, expected 30."
            )
        start = sum(widths[:term_index])
        return slice(start, start + widths[term_index])

    def _set_arm_targets(self, position: np.ndarray, velocity: np.ndarray, effort: np.ndarray) -> None:
        device = self._env.device
        dtype = self._robot.data.joint_pos.dtype
        self._robot.set_joint_position_target(
            torch.as_tensor(position, device=device, dtype=dtype).unsqueeze(0),
            joint_ids=self._arm_joint_ids,
        )
        self._robot.set_joint_velocity_target(
            torch.as_tensor(velocity, device=device, dtype=dtype).unsqueeze(0),
            joint_ids=self._arm_joint_ids,
        )
        self._robot.set_joint_effort_target(
            torch.as_tensor(effort, device=device, dtype=dtype).unsqueeze(0),
            joint_ids=self._arm_joint_ids,
        )

    def _write_policy_inputs(
        self, solution: Ocs2MpcSolution, policy_observation: torch.Tensor, command_observation: torch.Tensor
    ) -> None:
        device = self._env.device
        dtype = policy_observation.dtype
        wrench = torch.as_tensor(solution.base_wrench_prediction, device=device, dtype=dtype).unsqueeze(0)
        self._wrench_term.set_external_prediction(wrench)
        normalized_wrench = wrench.clone()
        normalized_wrench[..., :3] *= 0.01
        normalized_wrench[..., 3:] *= 0.05
        policy_observation[:, self._policy_wrench_slice] = normalized_wrench.flatten(start_dim=1)

        base_command = torch.as_tensor(
            solution.base_velocity_command, device=device, dtype=command_observation.dtype
        ).unsqueeze(0)
        self._env.command_manager.get_command("base_velocity")[:] = base_command
        if command_observation.shape != base_command.shape:
            raise RuntimeError(
                f"Command observation shape {tuple(command_observation.shape)} does not match {tuple(base_command.shape)}."
            )
        command_observation.copy_(base_command)

    def _fallback(self, policy_observation: torch.Tensor, command_observation: torch.Tensor) -> None:
        current_position = self._numpy(self._robot.data.joint_pos[:, self._arm_joint_ids])
        zeros = np.zeros(6, dtype=np.float64)
        self._set_arm_targets(current_position, zeros, zeros)
        zero_wrench = torch.zeros(1, 5, 6, device=self._env.device, dtype=policy_observation.dtype)
        self._wrench_term.set_external_prediction(zero_wrench)
        policy_observation[:, self._policy_wrench_slice].zero_()
        self._env.command_manager.get_command("base_velocity").zero_()
        command_observation.zero_()

    def _apply_solution(
        self, solution: Ocs2MpcSolution, policy_observation: torch.Tensor, command_observation: torch.Tensor
    ) -> None:
        self._set_arm_targets(
            solution.arm_position, solution.arm_velocity, solution.arm_feedforward_effort
        )
        self._write_policy_inputs(solution, policy_observation, command_observation)


class Ocs2PlayBridge(Ocs2OutputApplicator):
    """Submit low-rate OCS2 work without blocking Isaac Lab's policy/render loop."""

    def __init__(
        self,
        env,
        client: Ocs2TcpClient,
        target_position_world: np.ndarray | None,
        target_quaternion_world: np.ndarray | None,
        max_solution_age_s: float = 0.5,
        update_period_s: float = 0.1,
        target_offset_world: np.ndarray = (0.0, 0.0, 0.0),
        target_arrival_s: float = 1.0,
    ):
        super().__init__(env)
        if max_solution_age_s < 0.0:
            raise ValueError("max_solution_age_s must be non-negative.")
        if update_period_s <= 0.0:
            raise ValueError("update_period_s must be positive.")
        self._target_position_implicit = target_position_world is None
        self._target_quaternion_implicit = target_quaternion_world is None
        self._target_position_world = (
            None if target_position_world is None else np.asarray(target_position_world, dtype=np.float64).copy()
        )
        self._target_quaternion_world = (
            None if target_quaternion_world is None else np.asarray(target_quaternion_world, dtype=np.float64).copy()
        )
        self._target_offset_world = np.asarray(target_offset_world, dtype=np.float64).copy()
        if self._target_offset_world.shape != (3,) or not np.all(np.isfinite(self._target_offset_world)):
            raise ValueError("OCS2 target offset must contain three finite values.")
        if self._target_position_world is not None:
            if self._target_position_world.shape != (3,) or not np.all(np.isfinite(self._target_position_world)):
                raise ValueError("OCS2 target position must contain three finite values.")
            if np.any(self._target_offset_world):
                raise ValueError("OCS2 target offset cannot be combined with an absolute target position.")
        if self._target_quaternion_world is not None:
            if self._target_quaternion_world.shape != (4,) or not np.all(np.isfinite(self._target_quaternion_world)):
                raise ValueError("OCS2 target quaternion must contain four finite values in [w, x, y, z] order.")
            quaternion_norm = np.linalg.norm(self._target_quaternion_world)
            if quaternion_norm < 1e-9:
                raise ValueError("OCS2 target quaternion has zero norm.")
            self._target_quaternion_world /= quaternion_norm
        if not np.isfinite(target_arrival_s) or target_arrival_s < 0.0:
            raise ValueError("OCS2 target arrival time must be finite and non-negative.")
        self._target_arrival_s = target_arrival_s
        self._target_deadline_s: float | None = None
        ee_body_ids, ee_body_names = self._robot.find_bodies(["gripper_base_Link"], preserve_order=True)
        if tuple(ee_body_names) != ("gripper_base_Link",):
            raise RuntimeError(f"Unexpected OCS2 end-effector body: {ee_body_names}.")
        self._ee_body_id = ee_body_ids[0]
        self._max_solution_age_s = max_solution_age_s
        self._update_period_s = update_period_s
        self._async_client = Ocs2AsyncClient(client)
        self._last_submission_time = -math.inf
        self._next_retry_wall_time = -math.inf
        self._last_warning_wall_time = -math.inf
        self._failure_count = 0
        self._reported_error_serial = 0
        self._reported_solution_time: float | None = None
        self._pending_message_printed = False

    def _observation(self) -> Ocs2MpcObservation:
        data = self._robot.data
        simulation_time = float(self._env.common_step_counter) * float(self._env.step_dt)
        if self._target_position_world is None:
            self._target_position_world = (
                self._numpy(data.body_pos_w[:, self._ee_body_id]) + self._target_offset_world
            )
        if self._target_quaternion_world is None:
            self._target_quaternion_world = self._numpy(data.body_quat_w[:, self._ee_body_id])
        if self._target_deadline_s is None:
            self._target_deadline_s = simulation_time + self._target_arrival_s
            print(
                f"[INFO] OCS2 live target: position={self._target_position_world.tolist()}, "
                f"quaternion_wxyz={self._target_quaternion_world.tolist()}, "
                f"arrival={self._target_arrival_s:.3f}s."
            )
        base_twist_body = np.concatenate((self._numpy(data.root_lin_vel_b), self._numpy(data.root_ang_vel_b)))
        return Ocs2MpcObservation(
            time=simulation_time,
            base_position_world=self._numpy(data.root_pos_w),
            base_quaternion_world=self._numpy(data.root_quat_w),
            base_twist_body=base_twist_body,
            arm_position=self._numpy(data.joint_pos[:, self._arm_joint_ids]),
            arm_velocity=self._numpy(data.joint_vel[:, self._arm_joint_ids]),
            end_effector_target_position_world=self._target_position_world,
            end_effector_target_quaternion_world=self._target_quaternion_world,
            end_effector_arrival_time=max(0.0, self._target_deadline_s - simulation_time),
        )

    def update(self, policy_observation: torch.Tensor, command_observation: torch.Tensor) -> bool:
        """Submit work and apply the newest safe result without blocking simulation."""
        observation = self._observation()
        now = time.monotonic()
        if observation.time + 1e-9 < self._last_submission_time:
            self._last_submission_time = -math.inf
        if (
            observation.time - self._last_submission_time + 1e-9 >= self._update_period_s
            and now >= self._next_retry_wall_time
        ):
            self._async_client.submit(observation)
            self._last_submission_time = observation.time

        status = self._async_client.status()
        if status.error_serial > self._reported_error_serial:
            self._reported_error_serial = status.error_serial
            self._failure_count += 1
            # Physics may be paused at a fixed simulation timestamp. Permit a
            # wall-clock-limited retry even though simulation time did not
            # advance since the failed request.
            self._last_submission_time = -math.inf
            self._next_retry_wall_time = now + 1.0
            self._warn(f"OCS2 background solve failed (failure {self._failure_count}): {status.error}")

        solution = status.solution
        if solution is None:
            self._fallback(policy_observation, command_observation)
            if not self._pending_message_printed:
                print(
                    "[INFO] OCS2 solve is pending; physics is paused while WebRTC remains responsive."
                )
                self._pending_message_printed = True
            return False

        solution_age = observation.time - solution.time
        if solution_age < -1e-6 or solution_age > self._max_solution_age_s:
            self._fallback(policy_observation, command_observation)
            self._warn(
                f"OCS2 solution is stale: age={solution_age:.3f}s, "
                f"limit={self._max_solution_age_s:.3f}s"
            )
            return False

        try:
            self._apply_solution(solution, policy_observation, command_observation)
            self._failure_count = 0
            self._pending_message_printed = False
            if solution.time != self._reported_solution_time:
                duration_ms = 1000.0 * status.solve_duration_s if status.solve_duration_s is not None else math.nan
                print(
                    f"[INFO] Applied OCS2 solution: sim_time={solution.time:.3f}s, "
                    f"age={solution_age:.3f}s, solve={duration_ms:.1f}ms"
                )
                if status.solve_duration_s is not None and status.solve_duration_s > self._update_period_s:
                    self._warn(
                        f"OCS2 solve took {status.solve_duration_s:.2f}s, exceeding the "
                        f"{self._update_period_s:.2f}s update period; live control is not real-time."
                    )
                self._reported_solution_time = solution.time
            return True
        except (Ocs2BridgeError, OSError, ValueError, RuntimeError) as error:
            self._failure_count += 1
            self._fallback(policy_observation, command_observation)
            self._warn(f"OCS2 bridge fallback (failure {self._failure_count}): {error}")
            return False

    def _warn(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warning_wall_time >= 1.0:
            print(f"[WARN] {message}")
            self._last_warning_wall_time = now

    def reset(self) -> None:
        if self._target_position_implicit:
            self._target_position_world = None
        if self._target_quaternion_implicit:
            self._target_quaternion_world = None
        self._target_deadline_s = None
        self._last_submission_time = -math.inf
        self._next_retry_wall_time = -math.inf
        self._reported_solution_time = None
        self._pending_message_printed = False
        self._async_client.reset()

    def close(self) -> None:
        self._async_client.close()
