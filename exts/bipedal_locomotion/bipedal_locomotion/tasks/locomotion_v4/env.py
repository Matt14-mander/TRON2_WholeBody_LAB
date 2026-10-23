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
        return state
