"""Commands and stance-only rewards for the independent stand/walk task."""
from __future__ import annotations

import math

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from bipedal_locomotion.controllers.standwalk_core import allocate_case_quota, scheduled_velocity, standing_mask
from bipedal_locomotion.tasks.locomotion import mdp as legacy
from bipedal_locomotion.tasks.locomotion.mdp.commands.gait_command import GaitCommand


class StandWalkVelocity(CommandTerm):
    """Explicit episode schedules, with standing derived from the applied command."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.stage not in ("stand", "transition", "expanded") or cfg.evaluation_case not in (-1, 0, 1, 2, 3):
            raise ValueError("Invalid stand/walk stage.")
        if (len(cfg.walk_velocity) != 3 or any(not math.isfinite(x) for x in cfg.walk_velocity)
                or len(cfg.velocity_limits) != 3 or any(not math.isfinite(x) or x <= 0 for x in cfg.velocity_limits)):
            raise ValueError("Velocity targets must be finite and limits must be positive three-vectors.")
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.target = torch.zeros_like(self._command)
        self.cases = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.case_quota = allocate_case_quota(self.num_envs, cfg.stage, self.device)
        self.is_standing_env = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics = {key: torch.zeros(self.num_envs, device=self.device)
                        for key in ("linear_error", "yaw_error", "standing_fraction")}

    @property
    def command(self):
        return self._command

    def _resample_command(self, env_ids):
        count = len(env_ids)
        cases = self.case_quota[env_ids].clone()
        target = torch.tensor(self.cfg.walk_velocity, device=self.device).repeat(count, 1)
        if self.cfg.stage == "expanded":
            random_target = (2 * torch.rand(count, 3, device=self.device) - 1) * torch.tensor(self.cfg.velocity_limits, device=self.device)
            # Half of moving targets remain at the known baseline to prevent forgetting.
            target = torch.where((torch.rand(count, 1, device=self.device) < .5), target, random_target)
        if self.cfg.evaluation_case >= 0:
            cases.fill_(self.cfg.evaluation_case)
            target[:] = torch.tensor(self.cfg.walk_velocity, device=self.device)
        self.cases[env_ids], self.target[env_ids] = cases, target
        self._command[env_ids] = scheduled_velocity(torch.zeros(count, device=self.device), cases, target,
                                                   self.cfg.transition_s, self.cfg.first_hold_s, self.cfg.second_hold_s)
        self.is_standing_env[env_ids] = standing_mask(self._command[env_ids],
                                                     torch.zeros(count, dtype=torch.bool, device=self.device))

    def _update_command(self):
        time = self._env.episode_length_buf * self._env.step_dt
        self._command.copy_(scheduled_velocity(time, self.cases, self.target, self.cfg.transition_s,
                                               self.cfg.first_hold_s, self.cfg.second_hold_s))
        self.is_standing_env.copy_(standing_mask(self._command, self.is_standing_env))

    def _update_metrics(self):
        data = self._env.scene["robot"].data
        self.metrics["linear_error"].copy_((data.root_lin_vel_b[:, :2] - self._command[:, :2]).norm(dim=-1))
        self.metrics["yaw_error"].copy_((data.root_ang_vel_b[:, 2] - self._command[:, 2]).abs())
        self.metrics["standing_fraction"].copy_(self.is_standing_env.float())


@configclass
class StandWalkVelocityCfg(CommandTermCfg):
    class_type: type = StandWalkVelocity
    resampling_time_range: tuple = (1e9, 1e9)
    stage: str = "stand"
    walk_velocity: tuple = (-.25, 0., 0.)
    velocity_limits: tuple = (.35, .2, .4)
    transition_s: float = 1.
    first_hold_s: float = 5.
    second_hold_s: float = 6.
    evaluation_case: int = -1


class EpisodeClockGait(GaitCommand):
    """Keep legacy sin/cos observation and reward phase on the same episode clock.

    The clock continues in stance. Gait reward is masked by the actual velocity
    command; no frequency=0 or new observation channel is introduced.
    """
    def _update_command(self):
        self.stand_still_flag = self._env.command_manager.get_term("base_velocity").is_standing_env
        self.gait_indices.copy_(torch.remainder(
            self._env.episode_length_buf * self._env.step_dt * self.gait_command[:, 0], 1.))

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device)
        if env_ids is not None:
            ids = ids[env_ids]
        self.gait_indices[ids] = 0
        return super().reset(ids)


class EpisodeClockGaitReward(legacy.GaitReward):
    def compute_contact_targets(self, gait_params):
        # Isaac computes reward before command_manager.compute(). Refresh the
        # buffer in place here, so both contact and foot-height targets use the
        # just-completed interval's episode clock (including the first step).
        self._env.command_manager.get_term(self.command_name)._update_command()
        return super().compute_contact_targets(gait_params)


def standing(env):
    return env.command_manager.get_term("base_velocity").is_standing_env


def standing_speed(env):
    return standing(env) * env.scene["robot"].data.root_lin_vel_b[:, :2].square().sum(-1)


def standing_yaw(env):
    return standing(env) * env.scene["robot"].data.root_ang_vel_b[:, 2].square()


def standing_joint_motion(env, asset_cfg):
    return standing(env) * env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids].square().sum(-1)


def standing_contact_deficit(env, sensor_cfg):
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return standing(env) * (forces < 5.).float().sum(-1)


def walking_joint_orientation(env, asset_cfg):
    # Preserve the old gait pose objective in motion without forcing a rigid
    # straight-legged pose while balancing at zero command.
    return (~standing(env)) * legacy.joint_orientation_l1(env, asset_cfg)


def fall(env, max_tilt_deg=35., min_height=.45):
    data = env.scene["robot"].data
    height = data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (-data.projected_gravity_b[:, 2] < math.cos(math.radians(max_tilt_deg))) | (height < min_height)


def failure(env):
    return env.termination_manager.terminated.float()


def hold_arm_nominal(env, env_ids, asset_cfg):
    robot = env.scene[asset_cfg.name]
    target = robot.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
    zeros = torch.zeros_like(target)
    robot.set_joint_position_target(target, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    robot.set_joint_velocity_target(zeros, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    robot.set_joint_effort_target(zeros, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
