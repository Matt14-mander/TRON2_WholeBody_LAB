"""Terminal samples before auto-reset; reward terms and reference terrain progress."""
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply
from .mdp import controller


class TransportV4Env(ManagerBasedRLEnv):
    def __init__(self, *args, **kwargs):
        self._terminal_sample = None
        super().__init__(*args, **kwargs)
        self.expected_distance = torch.zeros(self.num_envs, device=self.device)
        self.reference_displacement = torch.zeros(self.num_envs, 2, device=self.device)
        self.start_xy = self.scene["robot"].data.root_pos_w[:, :2].clone()

    def state(self):
        term = controller(self)
        data = term._asset.data
        heights = self.scene.sensors["height_scanner"].data.ray_hits_w[..., 2]
        return {"base_velocity": torch.cat((data.root_lin_vel_b[:, :2], data.root_ang_vel_b[:, 2:3]), -1).clone(),
            "base_position": data.root_pos_w.clone(),
            "base_quaternion": data.root_quat_w.clone(),
            "base_tilt_deg": torch.rad2deg(torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1, 1))),
            "base_height": data.root_pos_w[:, 2] - heights.median(dim=1).values,
            "arm_position": data.joint_pos[:, term.arm_ids].clone(),
            "arm_velocity": data.joint_vel[:, term.arm_ids].clone(),
            "arm_target": term.executed_arm.clone(),
            "base_command": term.processed_actions[:, :3].clone(),
            "leg_action": term.last_leg.clone(),
            "payload_position": data.body_com_pos_w[:, term.payload_ids[0]].clone(),
            "force_world": term.force_world.clone(), "torque_world": term.torque_world.clone()}

    def _reset_idx(self, env_ids):
        active = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(active):
            state = self.state()
            self._terminal_sample = (active.clone(), {key: value[active].clone() for key, value in state.items()})
        super()._reset_idx(env_ids)
        if hasattr(self, "expected_distance"):
            self.expected_distance[env_ids] = 0.
            self.reference_displacement[env_ids] = 0.
            self.start_xy[env_ids] = self.scene["robot"].data.root_pos_w[env_ids, :2]

    def step(self, action):
        self._terminal_sample = None
        reference = self.command_manager.get_command("interaction").clone()
        self.expected_distance += reference[:, :2].norm(dim=-1) * self.step_dt
        planar = torch.zeros_like(reference)
        planar[:, :2] = reference[:, :2]
        self.reference_displacement += quat_apply(self.scene["robot"].data.root_quat_w, planar)[:, :2] * self.step_dt
        obs, reward, terminated, truncated, extras = super().step(action)
        state = self.state()
        if self._terminal_sample is not None:
            ids, sample = self._terminal_sample
            for key in state:
                state[key][ids] = sample[key]
        state["reference"] = reference  # preceding interval, never the next/reset force
        extras["transport_state"] = state
        extras["transport_terminated"] = terminated.clone()
        extras["transport_done_terms"] = {name: self.termination_manager.get_term(name).clone()
                                          for name in self.termination_manager.active_terms}
        extras["transport_rewards"] = {name: self.reward_manager._step_reward[:, i].clone() * self.step_dt
                                       for i, name in enumerate(self.reward_manager.active_terms)}
        # Also make teacher privilege explicit for future distillation algorithms.
        extras["teacher_obs"] = obs["teacher"]
        debug = getattr(self, "_transport_debug", None)
        if debug is not None:
            debug.update(state, extras["transport_rewards"], self.common_step_counter)
        return obs, reward, terminated, truncated, extras
