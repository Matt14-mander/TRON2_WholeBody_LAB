"""Flat PAINT teacher: privileged interaction force, frozen WholeBody locomotion."""
from copy import deepcopy

from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, SceneEntityCfg
from isaaclab.utils import configclass

from bipedal_locomotion.controllers.ocs2_reset import configure_nominal_ocs2_reset
from bipedal_locomotion.controllers.paint_teacher_core import ARM_JOINT_NAMES
from bipedal_locomotion.tasks.locomotion import mdp as locomotion_mdp
from bipedal_locomotion.tasks.locomotion.robots.limx_solefoot_yg_tron2a_wholebody_env_cfg import (
    SFYG_TRON2A_WholeBodyFlatEnvCfg,
)
from bipedal_locomotion.tasks.locomotion.cfg.SF_TRON2A.limx_base_env_cfg import CommandsCfg
from . import mdp


@configclass
class TeacherActionsCfg:
    teacher = mdp.FrozenLocomotionActionCfg()


@configclass
class TeacherCommandsCfg(CommandsCfg):
    interaction = mdp.PayloadInteractionCfg()


@configclass
class TeacherObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        state = ObsTerm(func=mdp.teacher_observation)
        enable_corruption = False
        concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        state = ObsTerm(func=mdp.critic_observation)
        enable_corruption = False
        concatenate_terms = True

    @configclass
    class CommandsCfg(ObsGroup):
        # Stateful admittance reference (payload local vx/vy/omega-z).
        reference = ObsTerm(func=mdp.reference_observation)
        enable_corruption = False
        concatenate_terms = True

    @configclass
    class HistoryCfg(ObsGroup):
        unused = ObsTerm(func=mdp.runner_history_placeholder)
        enable_corruption = False
        concatenate_terms = True

    policy = PolicyCfg()
    critic = CriticCfg()
    commands = CommandsCfg()
    obsHistory = HistoryCfg()


@configclass
class TeacherRewardsCfg:
    follow = RewTerm(func=mdp.follow_reward, weight=4.)
    payload_tilt = RewTerm(func=mdp.payload_tilt, weight=-3.)
    payload_height = RewTerm(func=mdp.payload_height, weight=-2.)
    payload_vertical_motion = RewTerm(func=mdp.payload_vertical_motion, weight=-1.)
    base_tilt = RewTerm(func=locomotion_mdp.flat_orientation_l2, weight=-2.)
    base_vertical_motion = RewTerm(func=locomotion_mdp.lin_vel_z_l2, weight=-.5)
    base_roll_pitch_velocity = RewTerm(func=locomotion_mdp.ang_vel_xy_l2, weight=-.05)
    target_rate = RewTerm(func=mdp.target_rate, weight=-.1)
    arm_recentering = RewTerm(func=mdp.arm_recentering, weight=-.2)
    arm_joint_limits = RewTerm(func=locomotion_mdp.joint_pos_limits, weight=-2.,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True)})
    arm_velocity = RewTerm(func=locomotion_mdp.joint_vel_l2, weight=-.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True)})
    arm_effort = RewTerm(func=locomotion_mdp.joint_torques_l2, weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True)})
    failure = RewTerm(func=mdp.teacher_failure, weight=-100.)


@configclass
class TeacherTerminationsCfg:
    time_out = DoneTerm(func=locomotion_mdp.time_out, time_out=True)
    fall = DoneTerm(func=mdp.tilted_or_low)
    payload_overtilt = DoneTerm(func=mdp.payload_overtilt)
    base_contact = DoneTerm(func=locomotion_mdp.illegal_contact, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_Link", "proximal_pitch_[RL]_Link"]),
        "threshold": 1.,
    })


@configclass
class PaintTeacherFlatEnvCfg(SFYG_TRON2A_WholeBodyFlatEnvCfg):
    def __post_init__(self):
        # Run legacy asset configuration first, before replacing its MDP terms.
        super().__post_init__()
        gait = deepcopy(self.commands.gait_command)
        self.commands = TeacherCommandsCfg()
        self.commands.gait_command = gait
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.
        self.commands.base_velocity.rel_standing_envs = 0.
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.resampling_time_range = (1e9, 1e9)
        self.commands.base_velocity.ranges.lin_vel_x = (0., 0.)
        self.commands.base_velocity.ranges.lin_vel_y = (0., 0.)
        self.commands.base_velocity.ranges.ang_vel_z = (0., 0.)
        # No arm_wrench generator: the articulated payload already transfers force.
        self.actions = TeacherActionsCfg()
        self.observations = TeacherObservationsCfg()
        self.rewards = TeacherRewardsCfg()
        self.terminations = TeacherTerminationsCfg()
        self.curriculum.terrain_levels = None
        configure_nominal_ocs2_reset(self, SceneEntityCfg("robot"))
        self.decimation = 8
        self.sim.dt = .005
        self.sim.render_interval = 8
        self.episode_length_s = 20.
        self.scene.num_envs = 1024
        for name in ("push_robot", "add_base_mass", "randomize_rigid_body_mass_inertia",
                     "robot_physics_material", "randomize_actuator_gains", "robot_center_of_mass"):
            if hasattr(self.events, name):
                setattr(self.events, name, None)
        # Match the existing PD data-collection baseline; randomization is later.
        for actuator in self.scene.robot.actuators.values():
            if hasattr(actuator, "min_delay"):
                actuator.min_delay = actuator.max_delay = 0
        self.scene.robot.actuators["arm_mpc"].stiffness = 60.
        self.scene.robot.actuators["arm_mpc"].damping = 6.


@configclass
class PaintTeacherFlatEnvCfg_PLAY(PaintTeacherFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.interaction.initial_scale = 1.
        self.commands.interaction.ramp_steps = 1
