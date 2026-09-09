"""This sub-module contains the reward functions that can be used for LimX Point Foot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import distributions
from typing import TYPE_CHECKING, Optional

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
import isaaclab.utils.math as math_utils
from bipedal_locomotion.utils.math import CubicSpline

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.terminated).float()


def stay_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for staying alive."""
    return torch.ones(env.num_envs, device=env.device)


def foot_landing_vel(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        foot_radius: float,
        about_landing_threshold: float,
) -> torch.Tensor:
    """Penalize high foot landing velocities"""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    z_vels = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 0.1

    foot_heights = torch.clip(
    asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - foot_radius, 0, 1
    )  # TODO: change to the height relative to the vertical projection of the terrain

    about_to_land = (foot_heights < about_landing_threshold) & (~contacts) & (z_vels < 0.0)
    landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
    reward = torch.sum(torch.square(landing_z_vels), dim=1)
    return reward


def joint_powers_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint powers on the articulation using L1-kernel"""

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_power = torch.mul(
        asset.data.applied_torque[:, asset_cfg.joint_ids],
        asset.data.joint_vel[:, asset_cfg.joint_ids],
    )
    return torch.sum(torch.abs(joint_power), dim=1)


def joint_deviation_from_default_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize selected joints deviating from their default (initial) positions using L2 kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_error), dim=1)


def no_fly(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Reward if only one foot is in contact with the ground."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    latest_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]

    contacts = latest_contact_forces > threshold
    single_contact = torch.sum(contacts.float(), dim=1) == 1

    return 1.0 * single_contact


def unbalance_feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize if the feet air time variance exceeds the balance threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    return torch.var(contact_sensor.data.last_air_time[:, sensor_cfg.body_ids], dim=-1)


def unbalance_feet_height(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the variance of feet maximum height using sensor positions."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    feet_positions = contact_sensor.data.pos_w[:, sensor_cfg.body_ids]

    if feet_positions is None:
        return torch.zeros(env.num_envs)

    feet_heights = feet_positions[:, :, 2]
    max_feet_heights = torch.max(feet_heights, dim=-1)[0]
    height_variance = torch.var(max_feet_heights, dim=-1)
    return height_variance


# def feet_distance(
#     env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Penalize if the distance between feet is below a minimum threshold."""

#     asset: Articulation = env.scene[asset_cfg.name]

#     feet_positions = asset.data.joint_pos[sensor_cfg.body_ids]

#     if feet_positions is None:
#         return torch.zeros(env.num_envs)

#     # feet distance on x-y plane
#     feet_distance = torch.norm(feet_positions[0, :2] - feet_positions[1, :2], dim=-1)

#     return torch.clamp(0.1 - feet_distance, min=0.0)


def feet_distance(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                  feet_links_name: list[str]=["foot_[RL]_Link"],
                  min_feet_distance: float = 0.1,
                  max_feet_distance: float = 1.0,)-> torch.Tensor:
    # Penalize base height away from target
    asset: Articulation = env.scene[asset_cfg.name]
    feet_links_idx = asset.find_bodies(feet_links_name)[0]
    feet_pos = asset.data.body_link_pos_w[:,feet_links_idx]
    # feet distance on x-y plane
    feet_distance = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=-1)
    reward = torch.clip(min_feet_distance - feet_distance, 0, 1)
    reward += torch.clip(feet_distance - max_feet_distance, 0, 1)
    return reward

def nominal_foot_position(env: ManagerBasedRLEnv, command_name: str,
                          base_height_target: float,
                           asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Compute the nominal foot position"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_apply_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    feet_center_b = torch.mean(feet_pos_b[:, :, :3], dim=1)
    base_height_error = torch.abs((feet_center_b[:, 2] - env._foot_radius + base_height_target))

    reward = torch.exp(-base_height_error / std**2)
    return reward

def leg_symmetry(env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Reward regulate abad joint position."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_apply_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    leg_symmetry_err = torch.abs(feet_pos_b[:, 0, 1]) - torch.abs(feet_pos_b[:, 1, 1])

    return torch.exp(-leg_symmetry_err ** 2 / std**2)

def same_feet_x_position(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward regulate abad joint position."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_apply_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    feet_x_distance = torch.abs(feet_pos_b[:, 0, 0] - feet_pos_b[:, 1, 0])
    # return torch.exp(-feet_x_distance / 0.2)
    return feet_x_distance

def contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    asset: Articulation = env.scene[asset_cfg.name]
    robot_links_mass = asset.root_physx_view.get_masses()
    robot_mass=torch.sum(robot_links_mass,dim=-1,keepdim=True)
    robot_mass = robot_mass.to(env.device)

    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - robot_mass*9.8
    # compute the penalty
    return torch.sum(violation.clip(min=0.0), dim=1)

def keep_ankle_pitch_zero_in_air(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=["ankle_[LR]_Link"]),
    force_threshold: float = 2.0,
    pitch_scale: float = 0.2
) -> torch.Tensor:
    """Reward for keeping ankle pitch angle close to zero when foot is in the air.
    
    Args:
        env: The environment object.
        asset_cfg: Configuration for the robot asset containing DOF positions.
        sensor_cfg: Configuration for the contact force sensor.
        force_threshold: Threshold value for contact detection (in Newtons).
        pitch_scale: Scaling factor for the exponential reward.
        
    Returns:
        The computed reward tensor.
    """
    asset = env.scene[asset_cfg.name]
    ankle_pitch_left_idx = asset.find_joints(["ankle_pitch_L_Joint"])[0]
    ankle_pitch_right_idx = asset.find_joints(["ankle_pitch_R_Joint"])[0]
    contact_forces_history = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    current_contact = torch.norm(contact_forces_history[:, -1], dim=-1) > force_threshold
    last_contact = torch.norm(contact_forces_history[:, -2], dim=-1) > force_threshold
    contact_filt = torch.logical_or(current_contact, last_contact)
    ankle_pitch_left = torch.abs(asset.data.joint_pos[:, ankle_pitch_left_idx]).squeeze(-1) * ~contact_filt[:, 0]
    ankle_pitch_right = torch.abs(asset.data.joint_pos[:, ankle_pitch_right_idx]).squeeze(-1) * ~contact_filt[:, 1]
    weighted_ankle_pitch = ankle_pitch_left + ankle_pitch_right
    return torch.exp(-weighted_ankle_pitch / pitch_scale)

def no_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize if both feet are not in contact with the ground.
    """

    # Access the contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get the latest contact forces in the z direction (upward direction)
    latest_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]  # shape: (env_num, 2)

    # Determine if each foot is in contact
    contacts = latest_contact_forces > 1.0  # Returns a boolean tensor where True indicates contact

    return (torch.sum(contacts.float(), dim=1) == 0).float()

def distance_aligned(env: ManagerBasedRLEnv,
                     asset_cfg: SceneEntityCfg,
                     min_dist: float,
                     max_dist: float,
                     desired_dist: float,
                     std: float,) -> torch.Tensor:
    """Reward for maintaining desired feet distance, aligned to heading direction."""
    asset: RigidObject = env.scene[asset_cfg.name]

    left_idx = asset_cfg.body_ids[0]
    right_idx = asset_cfg.body_ids[1]
    base_quat = asset.data.root_quat_w
    heading_aligned = math_utils.yaw_quat(base_quat)

    left_pos = math_utils.quat_apply_inverse(heading_aligned, asset.data.body_pos_w[:, left_idx])
    right_pos = math_utils.quat_apply_inverse(heading_aligned, asset.data.body_pos_w[:, right_idx])

    distance_y = torch.abs(left_pos[:, 1] - right_pos[:, 1])

    too_close = torch.clamp(min_dist - distance_y, min=0.0)
    too_far = torch.clamp(distance_y - max_dist, min=0.0)
    deviation_from_desired = torch.abs(distance_y - desired_dist)

    error = too_close + too_far + deviation_from_desired
    reward = torch.exp(-error ** 2 / std ** 2)
    return reward


def stand_still(
    env, lin_threshold: float = 0.05, ang_threshold: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    penalizing linear and angular motion when command velocities are near zero.
    """

    asset = env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_w[:, :2]
    base_ang_vel = asset.data.root_ang_vel_w[:, -1]

    commands = env.command_manager.get_command("base_velocity")

    lin_commands = commands[:, :2]
    ang_commands = commands[:, 2]

    reward_lin = torch.sum(
        torch.abs(base_lin_vel) * (torch.norm(lin_commands, dim=1, keepdim=True) < lin_threshold), dim=-1
    )

    reward_ang = torch.abs(base_ang_vel) * (torch.abs(ang_commands) < ang_threshold)

    total_reward = reward_lin + reward_ang
    return total_reward

def stand_still_for_joints(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    dof_error = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

    cmd = env.command_manager.get_command(command_name)
    mask = (torch.norm(cmd[:, :2], dim=1) < 0.05) & (torch.abs(cmd[:, 2]) < 0.05)
    return dof_error * mask

def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)
    

# def feet_regulation(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     desired_body_height: float = 0.65,
# ) -> torch.Tensor:
#     """Penalize if the feet are not in contact with the ground.

#     Args:
#         env: The environment object.
#         sensor_cfg: The configuration of the contact sensor.
#         desired_body_height: The desired body height used for normalization.

#     Returns:
#         A tensor representing the feet regulation penalty for each environment.
#     """

#     asset: Articulation = env.scene[asset_cfg.name]

#     feet_positions_z = asset.data.joint_pos[sensor_cfg.body_ids, 2]

#     feet_vel_xy = asset.data.joint_vel[sensor_cfg.body_ids, :2]

#     vel_norms_xy = torch.norm(feet_vel_xy, dim=-1)

#     exp_term = torch.exp(-feet_positions_z / (0.025 * desired_body_height))

#     r_fr = torch.sum(vel_norms_xy**2 * exp_term, dim=-1)

#     return r_fr

def feet_regulation(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    foot_radius: float,
    base_height_target: float,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_height = torch.clip(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - foot_radius, 0, 1
    )  # TODO: change to the height relative to the vertical projection of the terrain
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

    height_scale = torch.exp(-feet_height / base_height_target)
    reward = torch.sum(height_scale * torch.square(torch.norm(feet_vel_xy, dim=-1)), dim=1)
    return reward


def base_height_rough_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        Currently, it assumes a flat terrain, i.e. the target height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    height = asset.data.root_pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[:, :, 2]
    # sensor.data.ray_hits_w can be inf, so we clip it to avoid NaN
    height = torch.nan_to_num(height, nan=target_height, posinf=target_height, neginf=target_height)
    return torch.square(height.mean(dim=1) - target_height)

def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 3 * forces_z, dim=1).float()
    return reward

def base_com_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.abs(asset.data.root_pos_w[:, 2] - adjusted_target_height)


# class GaitReward(ManagerTermBase):
#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
#         """Initialize the term.

#         Args:
#             cfg: The configuration of the reward.
#             env: The RL environment instance.
#         """
#         super().__init__(cfg, env)

#         self.sensor_cfg = cfg.params["sensor_cfg"]
#         self.asset_cfg = cfg.params["asset_cfg"]

#         # extract the used quantities (to enable type-hinting)
#         self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
#         self.asset: Articulation = env.scene[self.asset_cfg.name]

#         # Store configuration parameters
#         self.force_scale = float(cfg.params["tracking_contacts_shaped_force"])
#         self.vel_scale = float(cfg.params["tracking_contacts_shaped_vel"])
#         self.force_sigma = cfg.params["gait_force_sigma"]
#         self.vel_sigma = cfg.params["gait_vel_sigma"]
#         self.kappa_gait_probs = cfg.params["kappa_gait_probs"]
#         self.command_name = cfg.params["command_name"]
#         self.dt = env.step_dt

#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         tracking_contacts_shaped_force,
#         tracking_contacts_shaped_vel,
#         gait_force_sigma,
#         gait_vel_sigma,
#         kappa_gait_probs,
#         command_name,
#         sensor_cfg,
#         asset_cfg,
#     ) -> torch.Tensor:
#         """Compute the reward.

#         The reward combines force-based and velocity-based terms to encourage desired gait patterns.

#         Args:
#             env: The RL environment instance.

#         Returns:
#             The reward value.
#         """

#         gait_params = env.command_manager.get_command(self.command_name)

#         # Update contact targets
#         desired_contact_states = self.compute_contact_targets(gait_params)

#         # Force-based reward
#         foot_forces = torch.norm(self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1)
#         force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

#         # Velocity-based reward
#         foot_velocities = torch.norm(self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids], dim=-1)
#         velocity_reward = self._compute_velocity_reward(foot_velocities, desired_contact_states)

#         # Combine rewards
#         total_reward = force_reward + velocity_reward
#         return total_reward

#     def compute_contact_targets(self, gait_params):
#         """Calculate desired contact states for the current timestep."""
#         frequencies = gait_params[:, 0]
#         offsets = gait_params[:, 1]
#         durations = torch.cat(
#             [
#                 gait_params[:, 2].view(self.num_envs, 1),
#                 gait_params[:, 2].view(self.num_envs, 1),
#             ],
#             dim=1,
#         )

#         assert torch.all(frequencies > 0), "Frequencies must be positive"
#         assert torch.all((offsets >= 0) & (offsets <= 1)), "Offsets must be between 0 and 1"
#         assert torch.all((durations > 0) & (durations < 1)), "Durations must be between 0 and 1"

#         gait_indices = torch.remainder(self._env.episode_length_buf * self.dt * frequencies, 1.0)

#         # Calculate foot indices
#         foot_indices = torch.remainder(
#             torch.cat(
#                 [gait_indices.view(self.num_envs, 1), (gait_indices + offsets + 1).view(self.num_envs, 1)],
#                 dim=1,
#             ),
#             1.0,
#         )

#         # Determine stance and swing phases
#         stance_idxs = foot_indices < durations
#         swing_idxs = foot_indices > durations

#         # Adjust foot indices based on phase
#         foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (0.5 / durations[stance_idxs])
#         foot_indices[swing_idxs] = 0.5 + (torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]) * (
#             0.5 / (1 - durations[swing_idxs])
#         )

#         # Calculate desired contact states using von mises distribution
#         smoothing_cdf_start = distributions.normal.Normal(0, self.kappa_gait_probs).cdf
#         desired_contact_states = smoothing_cdf_start(foot_indices) * (
#             1 - smoothing_cdf_start(foot_indices - 0.5)
#         ) + smoothing_cdf_start(foot_indices - 1) * (1 - smoothing_cdf_start(foot_indices - 1.5))

#         return desired_contact_states

#     def _compute_force_reward(self, forces: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
#         """Compute force-based reward component."""
#         reward = torch.zeros_like(forces[:, 0])
#         if self.force_scale < 0:  # Negative scale means penalize unwanted contact
#             for i in range(forces.shape[1]):
#                 reward += (1 - desired_contacts[:, i]) * (1 - torch.exp(-forces[:, i] ** 2 / self.force_sigma))
#         else:  # Positive scale means reward desired contact
#             for i in range(forces.shape[1]):
#                 reward += (1 - desired_contacts[:, i]) * torch.exp(-forces[:, i] ** 2 / self.force_sigma)

#         return (reward / forces.shape[1]) * self.force_scale

#     def _compute_velocity_reward(self, velocities: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
#         """Compute velocity-based reward component."""
#         reward = torch.zeros_like(velocities[:, 0])
#         if self.vel_scale < 0:  # Negative scale means penalize movement during contact
#             for i in range(velocities.shape[1]):
#                 reward += desired_contacts[:, i] * (1 - torch.exp(-velocities[:, i] ** 2 / self.vel_sigma))
#         else:  # Positive scale means reward movement during swing
#             for i in range(velocities.shape[1]):
#                 reward += desired_contacts[:, i] * torch.exp(-velocities[:, i] ** 2 / self.vel_sigma)

#         return (reward / velocities.shape[1]) * self.vel_scale

def stand_still_reg(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                    exclude_joints_name: list[str] = [r"J\d"]) -> torch.Tensor:
    """Regulate the stand still"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    exclude_joints_idx = asset.find_joints(exclude_joints_name)[0]
    all_joints_idx = range(asset.num_joints)
    vel_idx_exclude_arm = [i for i in all_joints_idx if i not in exclude_joints_idx]

    joint_vel = asset.data.joint_vel[:, vel_idx_exclude_arm]

    # stand still env , set to zero
    is_standing_env = env.command_manager.get_term("base_velocity").is_standing_env # type: ignore

    not_standing_env_ids = (~is_standing_env).nonzero(as_tuple=False).flatten()

    reward = torch.sum(torch.abs(joint_vel), dim=1)

    reward[not_standing_env_ids] = 0.0

    return reward


class TorquesSmoothnessPenaltyWrapper:
    """
    A wrapper class for calculating torques smoothness penalty.
    """
    def __init__(self):
        self.prev_torques = None
        self.__name__ = "torques_smoothness_penalty"

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """Penalize large instantaneous changes in the torques output"""

        asset: Articulation = env.scene[asset_cfg.name]
        torques = asset.data.applied_torque[:, asset_cfg.joint_ids].clone()
        if self.prev_torques is None:
            self.prev_torques = torques
            return torch.zeros(torques.shape[0], device=torques.device)
        reward = torch.sum(torch.square(torques - self.prev_torques), dim=1)
        self.prev_torques = torques
        return reward

torques_smoothness_penalty = TorquesSmoothnessPenaltyWrapper()


def body_orientation_yaw_exp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_yaw: float = 0.0) -> torch.Tensor:
    # yaw
    asset: RigidObject = env.scene[asset_cfg.name]
    num_body = len(asset_cfg.body_ids)
    base_quat = asset.data.root_quat_w
    inverse_base_quat = math_utils.quat_inv(base_quat).unsqueeze(1).expand(-1, num_body, -1) #[env, body_num, 4]
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :] #[env, body_num, 4]
    body_quat_b = math_utils.quat_mul(inverse_base_quat, body_quat_w).flatten(0,1) #[env * body_num, 4]
    r, p, y = math_utils.euler_xyz_from_quat(body_quat_b.squeeze(1))
    
    # Calculate yaw error relative to target_yaw and normalize
    yaw_error = normalize_angle(y - target_yaw)
    
    quat_mismatch = torch.exp(-torch.abs(yaw_error).reshape(-1, num_body) * 10) #[env, body_num], yaw
    return torch.mean(quat_mismatch, dim=1) #[env]


def joint_deviation_from_default_l1(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*pitch.*"]),
) -> torch.Tensor:
    """Penalize selected joints deviating from their default (initial) positions using L1 kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.abs(joint_error), dim=1)

    # Reduce penalty when lateral (y) or yaw command is active.
    commands = env.command_manager.get_command(command_name)
    y_cmd = torch.abs(commands[:, 1]) if commands.shape[1] > 1 else torch.zeros_like(penalty)
    yaw_cmd = torch.abs(commands[:, 2]) if commands.shape[1] > 2 else torch.zeros_like(penalty)
    is_lateral_or_yaw_active = (y_cmd > command_threshold) | (yaw_cmd > command_threshold)
    penalty = torch.where(is_lateral_or_yaw_active, penalty * 0.01, penalty)
    return penalty

def base_projection_at_feet_midpoint(
    env: ManagerBasedRLEnv, 
    std: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="wheel_.*")
) -> torch.Tensor:
    """奖励基座中心投影位于双足连线的中点。

    该奖励计算基座在 XY 平面的投影与双足 XY 中点之间的距离，并使用指数核。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 1. 获取双足在世界系下的 XY 位置 (形状: num_envs, 2, 2)
    # 假设 feet_cfg.body_ids 包含左脚和右脚的索引
    feet_pos_w = asset.data.body_pos_w[:, feet_cfg.body_ids, :2]

    # 2. 计算双足连线的中点 XY
    midpoint_xy = torch.mean(feet_pos_w, dim=1)

    # 3. 获取基座在世界系下的 XY 位置
    base_xy = asset.data.root_pos_w[:, :2]

    # 4. 计算偏差距离的平方
    error_sq = torch.sum(torch.square(base_xy - midpoint_xy), dim=1)

    # 5. 返回指数奖励
    return torch.exp(-error_sq / std**2)

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps, triggered only when feet first contact the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0
    last_contacts = contact_sensor.data.net_forces_w_history[:, 1, sensor_cfg.body_ids, 2] > 1.0
    contact_filt = torch.logical_or(contact, last_contacts)

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids] & contact_filt
    air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((air_time - 0.5) * first_contact, dim=1)

    command = env.command_manager.get_command(command_name)
    reward *= torch.norm(command[:, :2], dim=1) > 0.1
    return reward


def joint_orientation_l1_symmetric(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # 双向惩罚
    reward = torch.sum(torch.abs(joint_error), dim=-1)
    return reward


def joint_orientation_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    knee_joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(knee_joint_error) * (knee_joint_error > 0), dim=-1)
    return reward

def weighted_joint_deviation_l1(env: ManagerBasedRLEnv, 
                                deviation_weight: dict[str, float],
                                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    weighted_joint_deviation = torch.zeros_like(asset.data.joint_pos)

    for joint_name, w in deviation_weight.items():
        joint_idx = asset.find_joints(joint_name)[0]
        weighted_joint_deviation[:, joint_idx] = (
            torch.abs(angle[:, joint_idx]) * w
        )
    return torch.sum(weighted_joint_deviation, dim=1)


def weighted_joint_power_l1(
    env: ManagerBasedRLEnv,
    power_weight: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint power applied on the articulation using L1 kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    weighted_power = torch.zeros_like(asset.data.applied_torque)

    for joint_name, w in power_weight.items():
        joint_idx = asset.find_joints(joint_name)[0]
        weighted_power[:, joint_idx] = (
            torch.abs(asset.data.applied_torque[:, joint_idx] * asset.data.joint_vel[:, joint_idx]) * w
        )

    return torch.sum(weighted_power, dim=1)


def feet_angle_slide(env: ManagerBasedRLEnv, 
                     command_name: str,
                     sensor_cfg: SceneEntityCfg, 
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the angler velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    feet_ang = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, 2]
    command = env.command_manager.get_command(command_name)
    feet_ang_reward = torch.sum(torch.abs(feet_ang) * contacts, dim=1)
    reward = torch.where(torch.abs(command[:, 2]) > 0.1, feet_ang_reward, feet_ang_reward * 0.1)
    return reward


def feet_orientation_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward feet being oriented vertically when in contact with the ground."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    left_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    left_projected_gravity = math_utils.quat_apply_inverse(left_quat, asset.data.GRAVITY_VEC_W)
    right_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[1], :]
    right_projected_gravity = math_utils.quat_apply_inverse(right_quat, asset.data.GRAVITY_VEC_W)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1

    return (
        torch.sum(torch.square(left_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 0]
        + torch.sum(torch.square(right_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 1]
    )


class ActionSmoothnessPenaltyWrapper:
    """
    A wrapper class for calculating action smoothness penalty.

    The main purposes of this wrapper are:
    1. To maintain state across multiple calls (prev_action and prev_prev_action).
    2. To calculate a smoothness penalty based on the current, previous, and
       two-steps-ago actions.
    3. To provide a serializable interface compatible with IsaacLab's YAML
       configuration system.
    """

    def __init__(self):
        self.prev_prev_action = None
        self.prev_action = None
        self.__name__ = "action_smoothness_penalty"

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Penalize large instantaneous changes in the network action output"""
        current_action = env.action_manager.action.clone()

        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # Update actions for next call
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        startup_env_musk = env.episode_length_buf < 3
        penalty[startup_env_musk] = 0

        return penalty


action_smoothness_penalty = ActionSmoothnessPenaltyWrapper()

class GaitReward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]

        # extract the used quantities (to enable type-hinting)
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # Store configuration parameters
        self.force_scale = float(cfg.params["tracking_contacts_shaped_force"])
        self.vel_scale = float(cfg.params["tracking_contacts_shaped_vel"])
        self.height_scale = float(cfg.params["tracking_contacts_shaped_height"])
        self.force_sigma = cfg.params["gait_force_sigma"]
        self.vel_sigma = cfg.params["gait_vel_sigma"]
        self.height_sigma = cfg.params["gait_height_sigma"]
        self.touch_down_vel = float(cfg.params["touch_down_vel"])
        self.kappa_gait_probs = cfg.params["kappa_gait_probs"]
        self.command_name = cfg.params["command_name"]
        self.dt = env.step_dt
        self.use_reference_motion = cfg.params["use_reference_motion"]
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        tracking_contacts_shaped_force,
        tracking_contacts_shaped_vel,
        tracking_contacts_shaped_height,
        gait_force_sigma,
        gait_vel_sigma,
        gait_height_sigma,
        touch_down_vel,
        kappa_gait_probs,
        command_name,
        sensor_cfg,
        asset_cfg,
        use_reference_motion,
    ) -> torch.Tensor:
        """Compute the reward.

        The reward combines force-based and velocity-based terms to encourage desired gait patterns.

        Args:
            env: The RL environment instance.

        Returns:
            The reward value.
        """

        gait_params = env.command_manager.get_command(self.command_name) # type: ignore
        gait_indices = env.command_manager.get_term(self.command_name).gait_indices # type: ignore

        # Update contact targets
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Update foot height targets
        self.compute_desired_foot_height(gait_params, gait_indices)

        # Force-based reward
        foot_forces = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1
        )
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

        total_reward = force_reward

        # Velocity-based reward
        if self.vel_scale != 0:
            foot_velocities = self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids]
            velocity_reward = self._compute_velocity_reward(
                foot_velocities, self.des_foot_velocity_z, desired_contact_states
            )
            total_reward += velocity_reward

        # Height-based reward
        ## @Todo: 这里要算上地形高度
        if self.height_scale != 0:
            foot_heights = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, 2]
            height_reward = self._compute_height_reward(foot_heights, self.des_foot_height, desired_contact_states)
            total_reward += height_reward

        # stand still env , set to zero
        is_standing_env = env.command_manager.get_term("base_velocity").is_standing_env # type: ignore
        # total_reward = torch.where(is_plane, total_reward, total_reward/10)

        no_gait_env_ids = is_standing_env.nonzero(as_tuple=False).flatten()

        total_reward[no_gait_env_ids] = 0.0
        return total_reward

    def compute_contact_targets(self, gait_params):
        """Calculate desired contact states for the current timestep."""
        frequencies = gait_params[:, 0]
        offsets = gait_params[:, 1]
        durations = torch.cat(
            [
                gait_params[:, 2].view(self.num_envs, 1),
                gait_params[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )

        assert torch.all(frequencies > 0), "Frequencies must be positive"
        assert torch.all(
            (offsets >= 0) & (offsets <= 1)
        ), "Offsets must be between 0 and 1"
        assert torch.all(
            (durations > 0) & (durations < 1)
        ), "Durations must be between 0 and 1"

        # gait_indices = torch.remainder(
        #     self._env.episode_length_buf * self.dt * frequencies, 1.0
        # )

        # use gait indices from command
        command_term = self._env.command_manager.get_term("gait_command") # type: ignore
        gait_indices = command_term.gait_indices # type: ignore

        # Calculate foot indices
        foot_indices = torch.remainder(
            torch.cat(
                [
                    gait_indices.view(self.num_envs, 1),
                    (gait_indices + offsets + 1).view(self.num_envs, 1),
                ],
                dim=1,
            ),
            1.0,
        )

        # Determine stance and swing phases
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        # Adjust foot indices based on phase
        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (
            0.5 / durations[stance_idxs]
        )
        foot_indices[swing_idxs] = 0.5 + (
            torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]
        ) * (0.5 / (1 - durations[swing_idxs]))

        # Calculate desired contact states using von mises distribution
        smoothing_cdf_start = torch.distributions.normal.Normal(
            0, self.kappa_gait_probs
        ).cdf
        desired_contact_states = smoothing_cdf_start(foot_indices) * (
            1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (
            1 - smoothing_cdf_start(foot_indices - 1.5)
        )

        return desired_contact_states


    def compute_desired_foot_height(self, gait_params, gait_indices):
        """Calculate desired foot height for the current timestep."""
        frequencies = gait_params[:, 0]
        mask_0 = (gait_indices < 0.25) & (gait_indices >= 0.0)  # lift up
        mask_1 = (gait_indices < 0.5) & (gait_indices >= 0.25)  # touch down
        mask_2 = (gait_indices < 0.75) & (gait_indices >= 0.5)  # lift up
        mask_3 = (gait_indices <= 1.0) & (gait_indices >= 0.75)  # touch down
        swing_start_time = torch.zeros(self.num_envs, device=self.device)
        swing_start_time[mask_1] = 0.25 / frequencies[mask_1]
        swing_start_time[mask_2] = 0.5 / frequencies[mask_2]
        swing_start_time[mask_3] = 0.75 / frequencies[mask_3]
        swing_end_time = swing_start_time + 0.25 / frequencies
        swing_start_pos = torch.ones(self.num_envs, device=self.device)
        swing_start_pos[mask_0] = 0.0
        swing_start_pos[mask_2] = 0.0
        swing_end_pos = torch.ones(self.num_envs, device=self.device)
        swing_end_pos[mask_1] = 0.0
        swing_end_pos[mask_3] = 0.0
        swing_end_vel = torch.ones(self.num_envs, device=self.device)
        swing_end_vel[mask_0] = 0.0
        swing_end_vel[mask_2] = 0.0
        swing_end_vel[mask_1] = self.touch_down_vel
        swing_end_vel[mask_3] = self.touch_down_vel

        # generate desire foot z trajectory
        swing_height = gait_params[:, 3]
        # self.des_foot_height = 0.5 * swing_height * (1 - torch.cos(4 * np.pi * self.gait_indices))
        # self.des_foot_velocity_z = 2 * np.pi * swing_height * frequencies * torch.sin(
        #     4 * np.pi * self.gait_indices)

        start = {'time': swing_start_time, 'position': swing_start_pos * swing_height,
                 'velocity': torch.zeros(self.num_envs, device=self.device)}
        end = {'time': swing_end_time, 'position': swing_end_pos * swing_height,
               'velocity': swing_end_vel}
        cubic_spline = CubicSpline(start, end)
        self.des_foot_height = cubic_spline.position(gait_indices / frequencies)
        self.des_foot_velocity_z = cubic_spline.velocity(gait_indices / frequencies)

    def _compute_force_reward(
        self, forces: torch.Tensor, desired_contacts: torch.Tensor
    ) -> torch.Tensor:
        """Compute force-based reward component."""
        reward = torch.zeros_like(forces[:, 0])
        if self.force_scale < 0:  # Negative scale means penalize unwanted contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * (
                    1 - torch.exp(-forces[:, i] ** 2 / self.force_sigma)
                )
        else:  # Positive scale means reward desired contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * torch.exp(
                    -forces[:, i] ** 2 / self.force_sigma
                )

        return (reward / forces.shape[1]) * self.force_scale

    def _compute_velocity_reward(
        self, foot_velocities: torch.Tensor, des_foot_velocities_z: torch.Tensor, desired_contacts: torch.Tensor
    ) -> torch.Tensor:
        """Compute velocity-based reward component."""
        foot_velocity_norm = torch.norm(foot_velocities, dim=-1)
        reward = torch.zeros_like(foot_velocity_norm[:, 0])
        if self.vel_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(foot_velocity_norm.shape[1]):
                reward += desired_contacts[:, i] * (
                    1 - torch.exp(-foot_velocity_norm[:, i] ** 2 / self.vel_sigma)
                )
                if self.use_reference_motion:
                    swing_phase = 1 - desired_contacts[:, i]
                    reward += swing_phase * (1 - torch.exp(
                        -((foot_velocities[:, i, 2] - des_foot_velocities_z) ** 2)
                        / self.vel_sigma)
                    )
        else:  # Positive scale means reward movement during swing
            for i in range(foot_velocity_norm.shape[1]):
                reward += desired_contacts[:, i] * torch.exp(
                    -foot_velocity_norm[:, i] ** 2 / self.vel_sigma
                )
                if self.use_reference_motion:
                    swing_phase = 1 - desired_contacts[:, i]
                    reward += swing_phase * torch.exp(
                        -((foot_velocities[:, i, 2] - des_foot_velocities_z) ** 2)
                        / self.vel_sigma
                    )

        return (reward / foot_velocity_norm.shape[1]) * self.vel_scale
    
    def _compute_height_reward(
        self, foot_heights: torch.Tensor, des_foot_height:torch.Tensor, desired_contacts: torch.Tensor
    ) -> torch.Tensor:
        """Compute height-based reward component."""
        reward = torch.zeros_like(foot_heights[:, 0])
        if self.height_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(foot_heights.shape[1]):
                if self.use_reference_motion:
                    swing_phase = 1 - desired_contacts[:, i]
                    # if self.cfg.terrain.mesh_type == "plane":
                    reward += swing_phase * (
                            1 - torch.exp(
                        -(foot_heights[:, i] - des_foot_height) ** 2 / self.height_sigma)
                    )
                stand_phase = desired_contacts[:, i]
                reward += stand_phase * (1 - torch.exp(-(foot_heights[:, i]) ** 2 / self.height_sigma))
        else:  # Positive scale means reward movement during swing
            for i in range(foot_heights.shape[1]):
                if self.use_reference_motion:
                    swing_phase = 1 - desired_contacts[:, i]
                    # if self.cfg.terrain.mesh_type == "plane":
                    reward += swing_phase * torch.exp(
                        -(foot_heights[:, i] - des_foot_height) ** 2 / self.height_sigma
                    )
                stand_phase = desired_contacts[:, i]
                reward += stand_phase * torch.exp(-(foot_heights[:, i]) ** 2 / self.height_sigma)
        
        return (reward / foot_heights.shape[1]) * self.height_scale



class ActionSmoothnessPenalty(ManagerTermBase):
    """
    A reward term for penalizing large instantaneous changes in the network action output.
    This penalty encourages smoother actions over time.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.prev_prev_action = None
        self.prev_action = None
        # self.__name__ = "action_smoothness_penalty"

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        # Get the current action from the environment's action manager
        current_action = env.action_manager.action.clone()

        # If this is the first call, initialize the previous actions
        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        # Compute the smoothness penalty
        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # Update the previous actions for the next call
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        # Apply a condition to ignore penalty during the first few episodes
        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0

        # Return the penalty scaled by the configured weight
        return penalty
