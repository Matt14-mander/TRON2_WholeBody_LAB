"""Preserve pre-reset terminal observations and rollout standing coverage."""
from bipedal_locomotion.tasks.standwalk.env import StandWalkRLEnv


class PaintLocomotionV4Env(StandWalkRLEnv):
    def step(self, action):
        wrench = self.command_manager.get_command("payload_wrench").clone()
        result = super().step(action)
        result[-1]["standwalk_state"]["payload_wrench"] = wrench
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
        return state
