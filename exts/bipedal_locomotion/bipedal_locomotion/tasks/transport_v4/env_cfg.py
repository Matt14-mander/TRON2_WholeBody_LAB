"""Shared PAINT teacher/student transport scene on the frozen v4 backbone."""
from types import SimpleNamespace
from dataclasses import replace
from isaaclab.managers import (ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, EventTermCfg as EventTerm,
    CurriculumTermCfg as CurrTerm, SceneEntityCfg)
from isaaclab.utils import configclass

from bipedal_locomotion.tasks.locomotion import mdp as legacy
from bipedal_locomotion.tasks.locomotion_v4 import mdp as low_mdp
from bipedal_locomotion.tasks.locomotion_v4.env_cfg import (
    PaintLocomotionV4EnvCfg, CommandsCfg as LowCommandsCfg,
    ObservationsCfg as LowObservationsCfg, configure_stage,
)
from bipedal_locomotion.controllers.locomotion_v4 import STAGES
from . import mdp


@configclass
class ActionsCfg:
    transport = mdp.TransportActionCfg()


@configclass
class CommandsCfg:
    interaction = mdp.InteractionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class Policy(ObsGroup):
        state = ObsTerm(func=mdp.policy_observation)
        enable_corruption = False
        concatenate_terms = True

    @configclass
    class Teacher(ObsGroup):
        state = ObsTerm(func=mdp.teacher_observation)
        enable_corruption = False
        concatenate_terms = True

    @configclass
    class Student(ObsGroup):
        state = ObsTerm(func=mdp.student_observation)
        enable_corruption = False
        concatenate_terms = True

    @configclass
    class Critic(ObsGroup):
        state = ObsTerm(func=mdp.critic_observation)
        enable_corruption = False
        concatenate_terms = True

    @configclass
    class Empty(ObsGroup):
        state = ObsTerm(func=mdp.empty_commands)
        enable_corruption = False
        concatenate_terms = True

    @configclass
    class History(ObsGroup):
        state = ObsTerm(func=mdp.placeholder)
        enable_corruption = False
        concatenate_terms = True

    policy, teacher, student, critic = Policy(), Teacher(), Student(), Critic()
    commands, obsHistory = Empty(), History()


@configclass
class RewardsCfg:
    # Positive exponential tracking scores: the guide table's negative signs
    # conflict with its exp(-error) formulas for a reward-maximizing PPO.
    force_tracking = RewTerm(func=mdp.force_tracking, weight=.02)
    torque_tracking = RewTerm(func=mdp.torque_tracking, weight=.01)
    payload_height = RewTerm(func=mdp.payload_height, weight=-.003)
    joint_torques = RewTerm(func=legacy.joint_torques_l2, weight=-2e-9)
    dof_motion = RewTerm(func=mdp.dof_penalty, weight=-2e-4)
    action = RewTerm(func=mdp.action_penalty, weight=-2e-4)
    arm_posture = RewTerm(func=mdp.arm_posture, weight=-.003)
    termination = RewTerm(func=mdp.failed, weight=-2500.)  # -50 per failure at dt=.02


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=legacy.time_out, time_out=True)
    base_contact = DoneTerm(func=legacy.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces",
            body_names=["base_Link", "proximal_pitch_[RL]_Link"]), "threshold": 1.})
    fall = DoneTerm(func=low_mdp.fall)
    arm_payload_contact = DoneTerm(func=legacy.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces",
            body_names=["arm[1-6]_Link", "gripper.*_Link", "paint_payload_Link"]), "threshold": 1.})


@configclass
class TransportV4EnvCfg(PaintLocomotionV4EnvCfg):
    actor_mode: str = "teacher"
    payload_height_range: tuple = (.05, .25)  # entry point calibrates for TRON2 nominal posture

    def __post_init__(self):
        super().__post_init__()
        self.actions, self.commands = ActionsCfg(), CommandsCfg()
        self.observations = ObservationsCfg()
        self.rewards, self.terminations = RewardsCfg(), TerminationsCfg()
        self.episode_length_s = 10.
        self.scene.num_envs = 256
        # Keep low-level grid, offsets and timing exactly the same; newer API only.
        if hasattr(self.scene.height_scanner, "ray_alignment"):
            self.scene.height_scanner.attach_yaw_only = None
            self.scene.height_scanner.ray_alignment = "yaw"
        self.viewer.eye, self.viewer.lookat = (3., 3., 2.4), (0., 0., .8)


@configclass
class TransportV4StudentEnvCfg(TransportV4EnvCfg):
    actor_mode: str = "student"


def configure_domain(cfg, terrain, randomized, mass_range, evaluation=False):
    # Share the exact v4 terrain generator and payload inertia implementation.
    # The temporary low-level groups are configuration adapters only; no unused
    # velocity/wrench command managers are created in the transport environment.
    settings = replace(STAGES["robust" if randomized else "nominal"],
                       payload_mass=tuple(mass_range), push_velocity=0., terrain=terrain)
    proxy = SimpleNamespace(commands=LowCommandsCfg(), observations=LowObservationsCfg(),
        events=cfg.events, scene=cfg.scene, curriculum=cfg.curriculum, seed=cfg.seed)
    configure_stage(proxy, settings, cfg.actions.transport.base_limits, evaluation=evaluation)
    if randomized:
        cfg.events.add_base_mass = EventTerm(func=legacy.randomize_rigid_body_mass,
            mode="reset" if evaluation else "startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base_Link"]),
                "mass_distribution_params": (-2., 5.), "operation": "add", "recompute_inertia": True})
        cfg.events.robot_physics_material.params["static_friction_range"] = (.05, 1.5)
        cfg.events.robot_physics_material.params["dynamic_friction_range"] = (.05, 1.5)
        cfg.events.reset_robot_base.params["pose_range"] = {"x": (-1., 1.), "y": (-1., 1.), "yaw": (-3.14159, 3.14159)}
        cfg.events.reset_robot_base.params["velocity_range"] = {axis: (-.5, .5)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_progress) if terrain == "rough" and not evaluation else None
