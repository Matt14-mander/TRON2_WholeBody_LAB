"""Independent zero-wrench WholeBody stand/walk fine-tuning environment."""
from copy import deepcopy

from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm, SceneEntityCfg
from isaaclab.utils import configclass

from bipedal_locomotion.controllers.ocs2_reset import configure_nominal_ocs2_reset
from bipedal_locomotion.controllers.paint_teacher_core import ARM_JOINT_NAMES, LEG_JOINT_NAMES
from bipedal_locomotion.tasks.locomotion.robots.limx_solefoot_yg_tron2a_wholebody_env_cfg import SFYG_TRON2A_WholeBodyFlatEnvCfg
from . import mdp


@configclass
class StandWalkCommandsCfg:
    # Update velocity/standing BEFORE the gait command reads the standing mask.
    base_velocity = mdp.StandWalkVelocityCfg()
    gait_command = None
    arm_wrench = None


@configclass
class StandWalkEnvCfg(SFYG_TRON2A_WholeBodyFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        old = self.commands
        self.commands = StandWalkCommandsCfg()
        self.commands.gait_command = deepcopy(old.gait_command)
        self.commands.gait_command.class_type = mdp.EpisodeClockGait
        self.commands.arm_wrench = deepcopy(old.arm_wrench)
        self._disable_training_wrench_for_play()
        self.scene.num_envs = 512
        self.episode_length_s = 20.
        # First stage matches the already-tested collection baseline.
        self.observations.policy.enable_corruption = False
        self.observations.obsHistory.enable_corruption = False
        configure_nominal_ocs2_reset(self, SceneEntityCfg("robot"))
        self.events.hold_arm = EventTerm(func=mdp.hold_arm_nominal, mode="reset", params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True)})
        for name in ("push_robot", "add_base_mass", "randomize_rigid_body_mass_inertia",
                     "robot_physics_material", "randomize_actuator_gains", "robot_center_of_mass"):
            if hasattr(self.events, name):
                setattr(self.events, name, None)
        for actuator in self.scene.robot.actuators.values():
            if hasattr(actuator, "min_delay"):
                actuator.min_delay = actuator.max_delay = 0
        self.scene.robot.actuators["arm_mpc"].stiffness = 60.
        self.scene.robot.actuators["arm_mpc"].damping = 6.
        legs = SceneEntityCfg("robot", joint_names=list(LEG_JOINT_NAMES), preserve_order=True)
        # Avoid straight-leg/default-pose pressure while learning to balance.
        self.rewards.stand_still = None
        self.rewards.gait_reward.func = mdp.EpisodeClockGaitReward
        self.rewards.knee_joint_orientation.func = mdp.walking_joint_orientation
        self.rewards.standing_speed = RewTerm(func=mdp.standing_speed, weight=-4.)
        self.rewards.standing_yaw = RewTerm(func=mdp.standing_yaw, weight=-.5)
        self.rewards.standing_joint_motion = RewTerm(func=mdp.standing_joint_motion, weight=-.02,
                                                    params={"asset_cfg": legs})
        self.rewards.standing_contact_deficit = RewTerm(func=mdp.standing_contact_deficit, weight=-.5,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_[RL]_Link")})
        self.rewards.failure = RewTerm(func=mdp.failure, weight=-250.)  # -5 at 50 Hz.
        self.terminations.fall = DoneTerm(func=mdp.fall)
        self.curriculum.terrain_levels = None
