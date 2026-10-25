"""Preserve pre-reset terminal observations and rollout standing coverage."""
import torch
from bipedal_locomotion.tasks.standwalk.env import StandWalkRLEnv


class PaintLocomotionV4Env(StandWalkRLEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._biped_sums = torch.zeros(10, device=self.device)
        self._biped_samples = 0
        self._previous_biped_contact = None
        self._previous_biped_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _reset_idx(self, env_ids):
        self._biped_feet_cache = None
        super()._reset_idx(env_ids)
        self._biped_feet_cache = None

    def pop_biped_metrics(self):
        if not self.cfg.biped:
            return {}
        a = self._biped_sums.tolist()
        metrics = {"Motion/command_speed_mean": a[0] / max(a[2], 1),
            "Motion/actual_speed_along_command_mean": a[1] / max(a[2], 1),
            "Motion/linear_progress_ratio": a[1] / max(a[0], 1e-6),
            "Gait/moving_single_support_fraction": a[3] / max(a[4], 1),
            "Actuator/leg_saturation_fraction": a[5] / max(a[6], 1),
            "Actuator/arm_tracking_rmse": (a[7] / max(a[8], 1)) ** .5,
            "Gait/contact_releases_per_env_second": a[9] / max(self._biped_samples * self.step_dt, 1e-6)}
        self._biped_sums.zero_()
        self._biped_samples = 0
        return metrics

    def step(self, action):
        wrench = self.command_manager.get_command("payload_wrench").clone()
        result = super().step(action)
        result[-1]["standwalk_state"]["payload_wrench"] = wrench
        if self.cfg.biped:
            state = result[-1]["standwalk_state"]
            done = result[2] | result[3]
            command = state["command"]
            speed = command[:, :2].norm(dim=-1)
            linear = speed > .02
            moving = linear | (command[:, 2].abs() > .03)
            along = (state["linear_velocity"][:, :2] * command[:, :2]).sum(-1) / speed.clamp_min(.02)
            contact = state["foot_contact_force_z"] > 5.
            releases = torch.zeros((), device=self.device)
            if self._previous_biped_contact is not None:
                releases = (self._previous_biped_contact & ~contact & moving[:, None]
                            & self._previous_biped_valid[:, None] & ~done[:, None]).sum()
            self._previous_biped_contact = contact.clone()
            self._previous_biped_valid = ~done.bool()
            torque = state["leg_torque_fraction"]
            arm_error = state["arm_tracking_error"]
            self._biped_sums += torch.stack((speed[linear].sum(), along[linear].sum(), linear.sum(),
                ((contact.sum(-1) == 1) & moving).sum(), moving.sum(), (torque >= .95).sum(),
                torque.new_tensor(torque.numel()), arm_error.square().sum(),
                arm_error.new_tensor(arm_error.numel()), releases))
            self._biped_samples += self.num_envs
        return result

    def _state(self):
        state = super()._state()
        state["terrain_height"] = self.scene.sensors["height_scanner"].data.ray_hits_w[..., 2].median(dim=1).values
        state["height_above_terrain"] = self.scene["robot"].data.root_pos_w[:, 2] - state["terrain_height"]
        if not hasattr(self, "_gait_foot_ids"):
            names = ["ankle_pitch_L_Link", "ankle_pitch_R_Link"]
            self._gait_foot_ids = self.scene["robot"].find_bodies(names, preserve_order=True)[0]
            self._gait_sensor_ids = self.scene.sensors["contact_forces"].find_bodies(names, preserve_order=True)[0]
        ids, sensor_ids = self._gait_foot_ids, self._gait_sensor_ids
        state["foot_contact_force_z"] = self.scene.sensors["contact_forces"].data.net_forces_w[:, sensor_ids, 2].clone()
        state["foot_link_height_w"] = self.scene["robot"].data.body_pos_w[:, ids, 2].clone()
        if self.cfg.locomotion_gait is not None:
            from .mdp import gait_observation
            state["gait"] = gait_observation(self)
        if self.cfg.biped:
            from .mdp import biped_feet, biped_arm_observation, current_masses
            feet = biped_feet(self)
            robot = self.scene["robot"]
            biped_arm_observation(self)  # resolve the same ordered six arm joints
            leg, arm = self._biped_leg_ids, self._biped_arm_ids
            data = robot.data
            mass = current_masses(self)
            state.update({"foot_sole_clearance": feet["clearance"].clone(),
                "foot_velocity_world": feet["velocity"].clone(),
                "leg_position": data.joint_pos[:, leg].clone(),
                "leg_target": data.joint_pos_target[:, leg].clone(),
                "leg_torque_fraction": (data.applied_torque[:, leg].abs() / data.joint_effort_limits[:, leg].clamp_min(1e-6)).clone(),
                "arm_tracking_error": (data.joint_pos[:, arm] - data.default_joint_pos[:, arm]).clone(),
                "foot_geometry_xy_bounds": torch.stack((feet["collision_world"][..., :2].flatten(1, 2).min(1).values,
                    feet["collision_world"][..., :2].flatten(1, 2).max(1).values), 1) - self.scene.env_origins[:, None, :2],
                "wholebody_com": ((data.body_com_pos_w * mass[:, :, None]).sum(1) / mass.sum(-1, keepdim=True) - self.scene.env_origins).clone()})
        return state
