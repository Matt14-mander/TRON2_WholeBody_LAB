"""Paper-style external-wrench sequence generator for locomotion training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import WrenchSequenceCommandCfg


class WrenchSequenceCommand(CommandTerm):
    """Generate a smooth, partially observed base-wrench sequence.

    The flattened command contains ``[w(0.0), ..., w(0.8)]``. Each wrench is
    ordered ``[Fx, Fy, Fz, Tx, Ty, Tz]`` in the current base-link frame.
    """

    cfg: WrenchSequenceCommandCfg

    def __init__(self, cfg: WrenchSequenceCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]
        body_ids, body_names = self._asset.find_bodies(cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one body matching {cfg.body_name!r}, found {body_names}."
            )
        self._body_ids = body_ids
        self._all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._prediction_times = torch.tensor(cfg.prediction_times, device=self.device)
        self._bounds = torch.tensor(
            [*cfg.force_ranges, *cfg.torque_ranges], device=self.device, dtype=torch.float
        )
        if self._bounds.shape != (6, 2):
            raise ValueError("force_ranges and torque_ranges require three (min, max) pairs each.")

        sequence_size = len(cfg.prediction_times) * 6
        self._clean_command = torch.zeros(self.num_envs, sequence_size, device=self.device)
        self._noisy_command = torch.zeros_like(self._clean_command)
        self._applied_wrench = torch.zeros(self.num_envs, 6, device=self.device)
        self._points = torch.zeros(self.num_envs, 3, 6, device=self.device)
        self._beta = torch.ones(self.num_envs, 1, device=self.device)
        self._acceleration_gain = torch.zeros(self.num_envs, 6, device=self.device)
        self._previous_twist = torch.zeros(self.num_envs, 6, device=self.device)
        self.metrics = {}

    @property
    def command(self) -> torch.Tensor:
        return self._noisy_command

    @property
    def clean_command(self) -> torch.Tensor:
        return self._clean_command

    @property
    def applied_wrench(self) -> torch.Tensor:
        return self._applied_wrench

    def _update_metrics(self):
        pass

    def _sample_uniform(self, env_ids: torch.Tensor) -> torch.Tensor:
        low = self._bounds[:, 0]
        high = self._bounds[:, 1]
        return low + torch.rand(len(env_ids), 6, device=self.device) * (high - low)

    @staticmethod
    def _evaluate_quadratic(points: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Evaluate the quadratic through samples at t={0, 1, 2}."""
        times = times.reshape(1, -1, 1)
        l0 = (times - 1.0) * (times - 2.0) * 0.5
        l1 = -times * (times - 2.0)
        l2 = times * (times - 1.0) * 0.5
        return points[:, 0:1] * l0 + points[:, 1:2] * l1 + points[:, 2:3] * l2

    def _resample_command(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        self._points[env_ids, 0] = self._sample_uniform(env_ids)
        self._points[env_ids, 1] = self._sample_uniform(env_ids)
        self._points[env_ids, 2] = self._sample_uniform(env_ids)
        self._beta[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(*self.cfg.beta_range)
        acceleration_limits = torch.tensor(
            [*self.cfg.force_acceleration_gain, *self.cfg.torque_acceleration_gain], device=self.device
        )
        self._acceleration_gain[env_ids] = (
            2.0 * torch.rand(len(env_ids), 6, device=self.device) - 1.0
        ) * acceleration_limits
        self._previous_twist[env_ids, :3] = self._asset.data.root_lin_vel_b[env_ids]
        self._previous_twist[env_ids, 3:] = self._asset.data.root_ang_vel_b[env_ids]
        self._refresh_commands(env_ids)

    def _update_command(self):
        dt = self._env.step_dt
        shift_times = torch.tensor([dt, 1.0 + dt, 2.0 + dt], device=self.device)
        shifted = self._evaluate_quadratic(self._points, shift_times)
        endpoint_span = (self._bounds[:, 1] - self._bounds[:, 0]).reshape(1, 6)
        endpoint_noise = 2.0 * torch.rand(self.num_envs, 6, device=self.device) - 1.0
        endpoint_noise *= self._beta * self.cfg.endpoint_step_scale * endpoint_span
        self._points[:, 0] = shifted[:, 0]
        self._points[:, 1] = shifted[:, 1]
        self._points[:, 2] = torch.clamp(
            shifted[:, 2] + endpoint_noise,
            min=self._bounds[:, 0],
            max=self._bounds[:, 1],
        )
        self._refresh_commands(self._all_env_ids)

    def _refresh_commands(self, env_ids: torch.Tensor):
        clean_sequence = self._evaluate_quadratic(self._points[env_ids], self._prediction_times)
        flattened = clean_sequence.flatten(start_dim=1)
        self._clean_command[env_ids] = flattened
        noise_std = torch.tensor(
            [*self.cfg.force_observation_noise_std, *self.cfg.torque_observation_noise_std],
            device=self.device,
        ).repeat(len(self.cfg.prediction_times))
        self._noisy_command[env_ids] = flattened + torch.randn_like(flattened) * noise_std

        current_twist = torch.cat(
            (self._asset.data.root_lin_vel_b[env_ids], self._asset.data.root_ang_vel_b[env_ids]), dim=-1
        )
        acceleration = (current_twist - self._previous_twist[env_ids]) / self._env.step_dt
        unobserved = self._acceleration_gain[env_ids] * acceleration
        unobserved += torch.randn_like(unobserved) * self.cfg.unobserved_noise_std
        unobserved_limits = torch.tensor(self.cfg.unobserved_wrench_limits, device=self.device)
        unobserved = torch.clamp(unobserved, min=-unobserved_limits, max=unobserved_limits)
        applied = clean_sequence[:, 0] + unobserved
        self._applied_wrench[env_ids] = applied
        self._previous_twist[env_ids] = current_twist
        self._asset.set_external_force_and_torque(
            applied[:, None, :3],
            applied[:, None, 3:],
            body_ids=self._body_ids,
            env_ids=env_ids,
            is_global=False,
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
