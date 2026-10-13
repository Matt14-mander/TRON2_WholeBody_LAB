"""Keep terminal diagnostics BEFORE Isaac's automatic reset changes robot state."""
import torch
from isaaclab.envs import ManagerBasedRLEnv


class StandWalkRLEnv(ManagerBasedRLEnv):
    def __init__(self, *args, **kwargs):
        self._terminal_sample = None
        super().__init__(*args, **kwargs)

    def _state(self):
        data = self.scene["robot"].data
        return {"linear_velocity": data.root_lin_vel_b.clone(),
                "yaw_rate": data.root_ang_vel_b[:, 2].clone(),
                "position": (data.root_pos_w - self.scene.env_origins).clone(),
                "tilt_deg": torch.rad2deg(torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1, 1)))}

    def _reset_idx(self, env_ids):
        active = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(active):
            state = self._state()
            self._terminal_sample = (active.clone(), {key: value[active].clone() for key, value in state.items()})
        super()._reset_idx(env_ids)

    def step(self, action):
        self._terminal_sample = None
        command = self.command_manager.get_command("base_velocity").clone()
        obs, reward, terminated, truncated, extras = super().step(action)
        state = self._state()
        if self._terminal_sample is not None:
            ids, sample = self._terminal_sample
            for key in state:
                state[key][ids] = sample[key]
        state["command"] = command
        extras["standwalk_state"] = state
        return obs, reward, terminated, truncated, extras
