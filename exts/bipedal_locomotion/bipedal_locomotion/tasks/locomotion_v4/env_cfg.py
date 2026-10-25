"""PAINT guide replacement: articulated arm + fixed payload, no future wrench."""
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm, SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import (TerrainGeneratorCfg, MeshPlaneTerrainCfg,
    HfPyramidSlopedTerrainCfg, MeshPyramidStairsTerrainCfg, MeshRandomGridTerrainCfg)
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

from bipedal_locomotion.assets.config.solefoot_yg_tron2a_wholebody_cfg import SOLEFOOT_YG_TRON2A_WHOLEBODY_CFG
from bipedal_locomotion.controllers.locomotion_v4 import LEG_JOINTS
from bipedal_locomotion.controllers.ocs2_reset import configure_nominal_ocs2_reset
from bipedal_locomotion.tasks.locomotion import mdp as legacy
from bipedal_locomotion.tasks.locomotion.robots.limx_solefoot_tron2a_env_cfg import SF_TRON2A_BaseEnvCfg
from bipedal_locomotion.tasks.standwalk import mdp as stance
from . import mdp


def legs():
    return SceneEntityCfg("robot", joint_names=list(LEG_JOINTS), preserve_order=True)


@configclass
class ProprioCfg(ObsGroup):
    angular_velocity = ObsTerm(func=legacy.base_ang_vel, scale=.25,
                              noise=GaussianNoise(mean=0., std=.05))
    gravity = ObsTerm(func=legacy.projected_gravity, noise=GaussianNoise(mean=0., std=.025))
    joint_pos = ObsTerm(func=legacy.joint_pos_rel, params={"asset_cfg": legs()},
                        noise=GaussianNoise(mean=0., std=.01))
    joint_vel = ObsTerm(func=legacy.joint_vel_rel, params={"asset_cfg": legs()}, scale=.05,
                        noise=GaussianNoise(mean=0., std=1.5))
    last_action = ObsTerm(func=legacy.last_action)
    concatenate_terms = True
    enable_corruption = False


@configclass
class PolicyCfg(ProprioCfg):
    heights = ObsTerm(func=mdp.heights, clip=(-1., 1.), noise=GaussianNoise(mean=0., std=.01))


@configclass
class HistoryCfg(ProprioCfg):
    history_length = 10
    flatten_history_dim = False


@configclass
class CriticCfg(ObsGroup):
    # Supervised velocity estimator target MUST occupy the first three columns.
    velocity = ObsTerm(func=legacy.base_lin_vel)
    angular_velocity = ObsTerm(func=legacy.base_ang_vel, scale=.25)
    gravity = ObsTerm(func=legacy.projected_gravity)
    joint_pos = ObsTerm(func=legacy.joint_pos_rel, params={"asset_cfg": legs()})
    joint_vel = ObsTerm(func=legacy.joint_vel_rel, params={"asset_cfg": legs()}, scale=.05)
    last_action = ObsTerm(func=legacy.last_action)
    heights = ObsTerm(func=mdp.heights, clip=(-1., 1.))
    mass = ObsTerm(func=mdp.current_masses)
    wrench = ObsTerm(func=legacy.generated_commands, params={"command_name": "payload_wrench"}, scale=.05)
    enable_corruption = False
    concatenate_terms = True


@configclass
class ObservationsCfg:
    policy = PolicyCfg()
    critic = CriticCfg()
    obsHistory = HistoryCfg()

    @configclass
    class CommandsCfg(ObsGroup):
        velocity = ObsTerm(func=legacy.generated_commands, params={"command_name": "base_velocity"})
        enable_corruption = False
        concatenate_terms = True

    commands = CommandsCfg()


@configclass
class CommandsCfg:
    base_velocity = mdp.VelocityCommandCfg()
    payload_wrench = mdp.PayloadWrenchCfg()


@configclass
class ActionsCfg:
    legs = mdp.LoadedLegActionCfg(asset_name="robot", joint_names=list(LEG_JOINTS),
                                 preserve_order=True, scale=.25, use_default_offset=True)


@configclass
class PaintLocomotionV4EnvCfg(SF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        self.scene.height_scanner = RayCasterCfg(prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            offset=RayCasterCfg.OffsetCfg(pos=(0., 0., 20.)), attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=.1, size=(.8, .8)),
            debug_vis=False, mesh_prim_paths=["/World/ground"])
        # New Isaac Lab checks the deprecated flag at every sensor update.
        # Clear it while retaining the same yaw alignment; older releases keep
        # using attach_yaw_only=True.
        if hasattr(self.scene.height_scanner, "ray_alignment"):
            self.scene.height_scanner.attach_yaw_only = None
            self.scene.height_scanner.ray_alignment = "yaw"
        super().__post_init__()
        self.scene.robot = SOLEFOOT_YG_TRON2A_WHOLEBODY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 1024
        self.scene.env_spacing = 3.
        self.scene.terrain.physics_material.restitution = 0.
        self.scene.terrain.visual_material = None
        self.scene.light.spawn.texture_file = None
        for actuator in self.scene.robot.actuators.values():
            if hasattr(actuator, "min_delay"):
                actuator.min_delay = actuator.max_delay = 0
        self.actions, self.commands, self.observations = ActionsCfg(), CommandsCfg(), ObservationsCfg()
        configure_nominal_ocs2_reset(self, SceneEntityCfg("robot"))
        # Reuse basic locomotion objectives, discard inherited rigid gait/pose pressure.
        for name in ("gait_reward", "stand_still", "knee_joint_orientation", "orientation_exp_knee",
                     "feet_air_time", "action_out_of_limits"):
            if hasattr(self.rewards, name):
                setattr(self.rewards, name, None)
        for name in ("dof_torques_l2", "dof_vel_l2", "dof_acc_l2", "dof_pos_limits", "dof_vel_limits"):
            getattr(self.rewards, name).params["asset_cfg"] = legs()
        self.rewards.track_lin_vel_xy.weight = 2.
        self.rewards.track_ang_vel_z.weight = 1.
        self.rewards.base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.rewards.feet_air_time = RewTerm(func=mdp.single_support, weight=.5,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_[RL]_Link")})
        self.rewards.standing_speed = RewTerm(func=stance.standing_speed, weight=-4.)
        self.rewards.standing_yaw = RewTerm(func=stance.standing_yaw, weight=-.5)
        self.rewards.standing_joint_motion = RewTerm(func=stance.standing_joint_motion, weight=-.01,
            params={"asset_cfg": legs()})
        self.rewards.standing_contact_deficit = RewTerm(func=stance.standing_contact_deficit, weight=-.5,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_[RL]_Link")})
        self.rewards.failure = RewTerm(func=stance.failure, weight=-250.)
        self.terminations.action_out_of_limits = None
        self.terminations.fall = DoneTerm(func=mdp.fall)
        self.terminations.arm_payload_contact = DoneTerm(func=legacy.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces",
                body_names=["arm[1-6]_Link", "gripper.*_Link", "paint_payload_Link"]), "threshold": 1.})
        self.events.payload_mass = EventTerm(func=mdp.randomize_payload, mode="reset",
            params={"mass_range": (.5, .5),
                    "asset_cfg": SceneEntityCfg("robot", body_names=["paint_payload_Link"])})
        self.curriculum.terrain_levels = None
        self.decimation, self.sim.dt = 4, .005
        self.sim.render_interval = 4
        self.sim.physx.enable_external_forces_every_iteration = True
        self.scene.height_scanner.update_period = .02
        self.episode_length_s = 20.


def configure_stage(cfg, settings, limits, evaluation=False):
    """Applied BEFORE env construction; the same I/O is retained for every stage."""
    cfg.commands.base_velocity.limits = tuple(limits)
    cfg.commands.payload_wrench.force_n = settings.force_n
    cfg.commands.payload_wrench.torque_nm = settings.yaw_torque_nm
    cfg.events.payload_mass.params["mass_range"] = tuple(settings.payload_mass)
    for group in (cfg.observations.policy, cfg.observations.obsHistory):
        group.enable_corruption = settings.noise and not evaluation
    if settings.base_mass_scale != (1., 1.):
        cfg.events.add_base_mass = EventTerm(func=legacy.randomize_rigid_body_mass, mode="reset" if evaluation else "startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base_Link"]),
                    "mass_distribution_params": settings.base_mass_scale, "operation": "scale",
                    "recompute_inertia": True})
    cfg.events.robot_physics_material = EventTerm(func=legacy.randomize_rigid_body_material, mode="reset" if evaluation else "startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": settings.friction, "dynamic_friction_range": settings.friction,
                "restitution_range": (0., 0.), "num_buckets": 64, "make_consistent": True})
    if settings.push_velocity > 0:
        cfg.events.push_robot = EventTerm(func=legacy.push_by_setting_velocity, mode="interval",
            interval_range_s=(5., 10.), params={"velocity_range": {
                "x": (-settings.push_velocity, settings.push_velocity),
                "y": (-settings.push_velocity, settings.push_velocity)}})
    if settings.terrain == "rough":
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(size=(8., 8.), border_width=20.,
            num_rows=10, num_cols=12, horizontal_scale=.1, vertical_scale=.005,
            seed=cfg.seed, curriculum=True, difficulty_range=(0., 1.), use_cache=False,
            sub_terrains={
                "flat": MeshPlaneTerrainCfg(proportion=.25),
                "slope": HfPyramidSlopedTerrainCfg(proportion=.25, slope_range=(0., .15), platform_width=2.),
                "stairs": MeshPyramidStairsTerrainCfg(proportion=.25, step_height_range=(.01, .08),
                    step_width=.3, platform_width=2., border_width=1., holes=False),
                "obstacles": MeshRandomGridTerrainCfg(proportion=.25, grid_width=.3,
                    grid_height_range=(.01, .06), platform_width=2.),
            })
        if not evaluation:
            from isaaclab.managers import CurriculumTermCfg
            cfg.curriculum.terrain_levels = CurriculumTermCfg(func=legacy.terrain_levels_vel)
