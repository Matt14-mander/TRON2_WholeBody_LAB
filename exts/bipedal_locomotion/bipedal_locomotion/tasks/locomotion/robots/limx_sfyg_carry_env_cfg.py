"""Small physical adapter over the initial SFYG flat task; no reward/PPO edits."""
import torch
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import SceneEntityCfg, EventTermCfg
from isaaclab.utils import configclass
from .limx_solefoot_yg_tron2a_env_cfg import SFYG_TRON2A_BlindFlatEnvCfg

CARRY_POSE = {"arm1_Joint": 0., "arm2_Joint": 1.5707963,
              "arm3_Joint": -1.4835299, "arm4_Joint": 0.,
              "arm5_Joint": 0., "arm6_Joint": 0.}


class CarryLegAction(JointPositionAction):
    """Original unbounded leg targets; reissue the eight held upper-body targets."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.hold_ids, names = self._asset.find_joints(
            ["arm[1-6]_Joint", "gripper[12]_Joint"])
        if len(names) != 8 or self.action_dim != 10:
            raise ValueError("Expected complete SFYG, eight held joints and ten leg actions")
        self.hold = self._asset.data.default_joint_pos[:, self.hold_ids].clone()

    def apply_actions(self):
        super().apply_actions()
        self._asset.set_joint_position_target(self.hold, joint_ids=self.hold_ids)
        self._asset.set_joint_velocity_target(torch.zeros_like(self.hold), joint_ids=self.hold_ids)
        self._asset.set_joint_effort_target(torch.zeros_like(self.hold), joint_ids=self.hold_ids)


@configclass
class CarryLegActionCfg(JointPositionActionCfg):
    class_type: type = CarryLegAction


@configclass
class SFYGCarryFlatEnvCfg(SFYG_TRON2A_BlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.joint_pos.update(CARRY_POSE)
        previous = self.actions.joint_pos
        self.actions.joint_pos = CarryLegActionCfg(asset_name="robot",
            joint_names=previous.joint_names, preserve_order=True,
            scale=previous.scale, use_default_offset=True)
        # The physical payload is fixed in each run. No force generator, pushes,
        # mass randomization or new observations/rewards are introduced here.
        self.events.push_robot = None
        self.events.reset_arm = EventTermCfg(
            func=reset_held_joints, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot",
                joint_names=["arm[1-6]_Joint", "gripper[12]_Joint"])})


def reset_held_joints(env, env_ids, asset_cfg):
    robot = env.scene[asset_cfg.name]
    q = robot.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
    dq = robot.data.default_joint_vel[env_ids][:, asset_cfg.joint_ids]
    robot.write_joint_state_to_sim(q, dq, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    robot.set_joint_position_target(q, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
