"""Leg-only control, real payload forces and explicit stop/start commands."""
import math
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from bipedal_locomotion.controllers.locomotion_v4 import (
    LEG_JOINTS, sample_planar_wrench, gait_features, gait_targets, gait_score,
)
from bipedal_locomotion.controllers.standwalk_core import scheduled_velocity, standing_mask
from bipedal_locomotion.controllers import locomotion_biped as biped


class VelocityCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Fixed env quotas prevent short standing failures from disappearing from rollout.
        if getattr(cfg, "profile", "mixed") in biped.SKILLS:
            self.roles = biped.command_roles(self.num_envs, cfg.profile, self.device)
        elif getattr(cfg, "profile", "mixed") == "walk_first":
            roles = torch.arange(self.num_envs, device=self.device) % 20
            self.roles = torch.where(roles < 2, 0, torch.where(roles < 18, 1, torch.where(roles < 19, 2, 3)))
        else:
            roles = torch.arange(self.num_envs, device=self.device) % 10
            self.roles = torch.where(roles < 2, 0, torch.where(roles < 6, 1, torch.where(roles < 8, 2, 3)))
        self.roles = self.roles[torch.randperm(self.num_envs, device=self.device)]
        self.target = torch.zeros(self.num_envs, 3, device=self.device)
        self._command = torch.zeros_like(self.target)
        self.is_standing_env = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics = {key: torch.zeros(self.num_envs, device=self.device)
                        for key in ("linear_error", "yaw_error", "standing_fraction")}

    @property
    def command(self):
        return self._command

    def _resample_command(self, env_ids):
        n = len(env_ids)
        target = (2 * torch.rand(n, 3, device=self.device) - 1) * torch.tensor(self.cfg.limits, device=self.device)
        # Half are axis-aligned: explicitly train both lateral directions and pure yaw.
        axes = torch.randint(0, 3, (n,), device=self.device)
        mask = torch.nn.functional.one_hot(axes, 3).bool()
        target = torch.where((torch.rand(n, 1, device=self.device) < .5) | mask, target, 0.)
        if getattr(self.cfg, "profile", "mixed") in biped.SKILLS:
            target = biped.movement_targets(n, self.cfg.profile, self.cfg.limits, self.device)
        if getattr(self.cfg, "profile", "mixed") == "walk_first":
            # A separate experiment: exclude tiny nonzero movement targets.
            limits = target.new_tensor(self.cfg.limits)
            minimum = torch.minimum(target.new_tensor((.15, .10, .15)), limits)
            magnitude = minimum + (target.abs() / limits) * (limits - minimum)
            target = torch.where(target != 0., target.sign() * magnitude, 0.)
        if self.cfg.evaluation_case >= 0:
            self.roles[env_ids] = self.cfg.evaluation_case
            target[:] = torch.tensor(self.cfg.evaluation_velocity, device=self.device)
        self.target[env_ids] = target
        initial = scheduled_velocity(torch.zeros(n, device=self.device), self.roles[env_ids], target)
        if getattr(self.cfg, "profile", "mixed") in biped.SKILLS:
            initial.zero_()
        self._command[env_ids] = initial
        self.is_standing_env[env_ids] = standing_mask(initial,
            torch.zeros(n, device=self.device, dtype=torch.bool))

    def _update_command(self):
        time = self._env.episode_length_buf * self._env.step_dt
        command = scheduled_velocity(time, self.roles, self.target, transition_s=1.)
        if getattr(self.cfg, "profile", "mixed") in biped.SKILLS:
            u = (time / .5).clamp(0., 1.)
            command *= (u.square() * (3 - 2 * u))[:, None]
        # Match high-level clipping and deadband in training and deployment.
        command[:, :2] = torch.where(command[:, :2].norm(dim=-1, keepdim=True) <= .02, 0., command[:, :2])
        command[:, 2] = torch.where(command[:, 2].abs() <= .03, 0., command[:, 2])
        self._command.copy_(command)
        self.is_standing_env.copy_(standing_mask(command, self.is_standing_env))

    def _update_metrics(self):
        data = self._env.scene["robot"].data
        self.metrics["linear_error"].copy_((data.root_lin_vel_b[:, :2] - self.command[:, :2]).norm(dim=-1))
        self.metrics["yaw_error"].copy_((data.root_ang_vel_b[:, 2] - self.command[:, 2]).abs())
        self.metrics["standing_fraction"].copy_(self.is_standing_env.float())


@configclass
class VelocityCommandCfg(CommandTermCfg):
    class_type: type = VelocityCommand
    resampling_time_range: tuple = (5., 8.)
    limits: tuple = (.35, .2, .4)
    profile: str = "mixed"
    evaluation_case: int = -1
    evaluation_velocity: tuple = (0., 0., 0.)


class PayloadWrench(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics = {"force_norm": torch.zeros(self.num_envs, device=self.device)}

    @property
    def command(self):
        return self._command

    def _resample_command(self, env_ids):
        wrench = sample_planar_wrench(len(env_ids), self.cfg.force_n, self.cfg.torque_nm, self.device)
        # Include release intervals, not just continuous pulls.
        wrench[torch.rand(len(env_ids), device=self.device) < self.cfg.release_probability] = 0.
        self._command[env_ids] = wrench

    def _update_command(self):
        pass

    def _update_metrics(self):
        self.metrics["force_norm"].copy_(self.command[:, :2].norm(dim=-1))


@configclass
class PayloadWrenchCfg(CommandTermCfg):
    class_type: type = PayloadWrench
    resampling_time_range: tuple = (1., 3.)
    force_n: float = 0.
    torque_nm: float = 0.
    release_probability: float = .25


class LoadedLegAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        ids, names = self._asset.find_joints(list(LEG_JOINTS), preserve_order=True)
        if tuple(names) != LEG_JOINTS or list(self._joint_ids) != list(ids):
            raise ValueError("v4 requires the exact ordered ten leg joints.")
        self.hold_ids, _ = self._asset.find_joints(["arm[1-6]_Joint", "gripper[12]_Joint"])
        self.payload_ids, names = self._asset.find_bodies("paint_payload_Link")
        if len(names) != 1:
            raise ValueError("Launch train_locomotion_v4.py to prepare the physical payload before spawning.")
        self.nominal = self._asset.data.default_joint_pos[:, self.hold_ids].clone()

    def process_actions(self, actions):
        super().process_actions(actions.clamp(-5., 5.))

    def apply_actions(self):
        super().apply_actions()
        self._asset.set_joint_position_target(self.nominal, joint_ids=self.hold_ids)
        self._asset.set_joint_velocity_target(torch.zeros_like(self.nominal), joint_ids=self.hold_ids)
        self._asset.set_joint_effort_target(torch.zeros_like(self.nominal), joint_ids=self.hold_ids)
        planar = self._env.command_manager.get_command("payload_wrench")
        force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        torque = torch.zeros_like(force)
        force[:, 0, :2], torque[:, 0, 2] = planar[:, :2], planar[:, 2]
        # Recompute frame transform EACH physics step as the payload rotates.
        quat = self._asset.data.body_quat_w[:, self.payload_ids[0]]
        force[:, 0] = quat_apply_inverse(quat, force[:, 0])
        torque[:, 0] = quat_apply_inverse(quat, torque[:, 0])
        self._asset.set_external_force_and_torque(force, torque, body_ids=self.payload_ids, is_global=False)


@configclass
class LoadedLegActionCfg(JointPositionActionCfg):
    class_type: type = LoadedLegAction


def randomize_payload(env, env_ids, mass_range, asset_cfg):
    """Scale mass AND inertia from authored defaults, never cumulatively."""
    robot = env.scene[asset_cfg.name]
    ids = torch.as_tensor(env_ids, device="cpu", dtype=torch.long)
    body = asset_cfg.body_ids[0]
    masses = robot.root_physx_view.get_masses()
    inertias = robot.root_physx_view.get_inertias()
    sample = torch.empty(len(ids)).uniform_(*mass_range)
    default_mass = robot.data.default_mass.cpu()[ids, body]
    default_inertia = robot.data.default_inertia.cpu()[ids, body]
    masses[ids, body] = sample
    inertias[ids, body] = default_inertia * (sample / default_mass)[:, None]
    robot.root_physx_view.set_masses(masses, ids)
    robot.root_physx_view.set_inertias(inertias, ids)
    env._v4_masses = masses.to(env.device)


def current_masses(env):
    if not hasattr(env, "_v4_masses"):
        env._v4_masses = env.scene["robot"].root_physx_view.get_masses().to(env.device)
    return env._v4_masses


def heights(env):
    # Ray origin is 20 m above the base: use BASE z, not sensor-origin z.
    hits = env.scene.sensors["height_scanner"].data.ray_hits_w[..., 2]
    if not torch.isfinite(hits).all():
        raise RuntimeError("v4 height scan missed the terrain; check terrain mesh and ray coverage.")
    return env.scene["robot"].data.root_pos_w[:, 2:3] - hits - .5


def gait_observation(env):
    # Read the episode clock directly: rewards run before command updates in Isaac.
    return gait_features(env.episode_length_buf * env.step_dt, env.cfg.locomotion_gait)


def biped_arm_observation(env):
    robot = env.scene["robot"]
    if not hasattr(env, "_biped_arm_ids"):
        env._biped_arm_ids = robot.find_joints(list(biped.ARM_JOINTS), preserve_order=True)[0]
    ids = env._biped_arm_ids
    return torch.cat((robot.data.joint_pos[:, ids], robot.data.joint_vel[:, ids] * .1), -1)


def biped_feet(env):
    """Foot collisions transformed to world, cached within one control tick.

    Nearest local scan height is adequate on nominal flat terrain. Detailed
    per-corner terrain contact on rough terrain requires its own acceptance.
    Reset invalidates this cache even when the physics tick did not advance.
    """
    tick = env._sim_step_counter
    cached = getattr(env, "_biped_feet_cache", None)
    if cached is not None and cached[0] == tick:
        return cached[1]
    robot = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    if not hasattr(env, "_biped_geometry"):
        names = ["ankle_pitch_L_Link", "ankle_pitch_R_Link"]
        env._biped_body_ids = robot.find_bodies(names, preserve_order=True)[0]
        env._biped_contact_ids = sensor.find_bodies(names, preserve_order=True)[0]
        env._biped_leg_ids = robot.find_joints(list(LEG_JOINTS), preserve_order=True)[0]
        geometry = env.cfg.biped["sole_geometry"]
        env._biped_geometry = (
            torch.tensor([foot["points"] for foot in geometry], device=env.device),
            torch.tensor([foot["radii"] for foot in geometry], device=env.device))
    ids = env._biped_body_ids
    points, radii = env._biped_geometry
    pos, quat = robot.data.body_pos_w[:, ids], robot.data.body_quat_w[:, ids]
    shape = (env.num_envs, 2, points.shape[1])
    rotated = quat_apply(quat[:, :, None].expand(*shape, 4).reshape(-1, 4),
                         points[None].expand(*shape, 3).reshape(-1, 3)).reshape(*shape, 3)
    collision_world = pos[:, :, None] + rotated
    hits = env.scene.sensors["height_scanner"].data.ray_hits_w
    nearest = (pos[:, :, None, :2] - hits[:, None, :, :2]).square().sum(-1).argmin(-1)
    ground = hits[..., 2].gather(1, nearest)
    clearance = (collision_world[..., 2] - radii[None]).min(-1).values - ground
    value = {"force_z": sensor.data.net_forces_w[:, env._biped_contact_ids, 2],
             "clearance": clearance, "velocity": robot.data.body_lin_vel_w[:, ids],
             "collision_world": collision_world, "ground": ground,
             "contact": sensor.data.net_forces_w[:, env._biped_contact_ids, 2] > 5.}
    env._biped_feet_cache = (tick, value)
    return value


def biped_load_observation(env):
    feet = biped_feet(env)
    return feet["force_z"] / (current_masses(env).sum(-1, keepdim=True) * 9.81)


def reset_biped_joints(env, env_ids, asset_cfg, offset=.05):
    robot = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    q = robot.data.default_joint_pos[env_ids].clone()
    dq = robot.data.default_joint_vel[env_ids].clone()
    ids = asset_cfg.joint_ids
    q[:, ids] += (2 * torch.rand(len(env_ids), len(ids), device=env.device) - 1) * offset
    limits = robot.data.joint_pos_limits[env_ids][:, ids]
    q[:, ids] = q[:, ids].maximum(limits[..., 0]).minimum(limits[..., 1])
    robot.write_joint_state_to_sim(q, dq, env_ids=env_ids)
    robot.set_joint_position_target(q, env_ids=env_ids)
    if hasattr(env, "_biped_air_peak"):
        env._biped_air_peak[env_ids] = 0.
    env._biped_feet_cache = None


def biped_load_cost(env):
    contact = gait_targets(env.episode_length_buf * env.step_dt, env.cfg.locomotion_gait)[0]
    return biped.load_tracking_cost(biped_feet(env)["force_z"], current_masses(env).sum(-1), contact) * ~env.command_manager.get_term("base_velocity").is_standing_env


def biped_clearance_cost(env):
    contact, height, _ = gait_targets(env.episode_length_buf * env.step_dt, env.cfg.locomotion_gait)
    error = (biped_feet(env)["clearance"].clamp_min(0.) - height) / env.cfg.locomotion_gait["swing_height"]
    return ((1 - contact) * error.square().clamp(max=4.)).mean(-1) * ~env.command_manager.get_term("base_velocity").is_standing_env


def biped_stall_cost(env):
    data = env.scene["robot"].data
    return biped.motion_shortfall(env.command_manager.get_command("base_velocity"),
                                  data.root_lin_vel_b, data.root_ang_vel_b[:, 2])


def biped_track_linear(env, std=.2):
    return biped.velocity_score(env.command_manager.get_command("base_velocity"),
                                 env.scene["robot"].data.root_lin_vel_b, std)


def biped_airborne_cost(env):
    return (~biped_feet(env)["contact"]).all(-1).float()


def biped_touchdown(env):
    """A small event reward, requiring real clearance and correctly timed landing."""
    feet = biped_feet(env)
    if not hasattr(env, "_biped_air_peak"):
        env._biped_air_peak = torch.zeros(env.num_envs, 2, device=env.device)
    peak = env._biped_air_peak
    peak.copy_(torch.maximum(peak, feet["clearance"]))
    sensor = env.scene.sensors["contact_forces"]
    ids = env._biped_contact_ids
    first = sensor.compute_first_contact(env.step_dt)[:, ids]
    duration = sensor.data.last_air_time[:, ids]
    desired = gait_targets(env.episode_length_buf * env.step_dt, env.cfg.locomotion_gait)[0]
    valid = first & (duration >= .08) & (duration <= 1.) & (peak >= .015) & (desired > .5)
    peak[feet["contact"]] = 0.
    moving = ~env.command_manager.get_term("base_velocity").is_standing_env
    # RewardManager multiplies by dt: this contributes .1 per valid touchdown.
    return .1 * valid.float().sum(-1) * moving / env.step_dt


def restored_gait(env, sensor_cfg, asset_cfg):
    sensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]
    feet = robot.data.body_pos_w[:, asset_cfg.body_ids]
    hits = env.scene.sensors["height_scanner"].data.ray_hits_w
    # Preserve the legacy foot-link height objective on flat ground, but remove
    # the world terrain elevation. This is link height, not calibrated sole clearance.
    nearest = (feet[:, :, None, :2] - hits[:, None, :, :2]).square().sum(-1).argmin(-1)
    ground = hits[..., 2].gather(1, nearest)
    reward = gait_score(
        sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1),
        robot.data.body_lin_vel_w[:, asset_cfg.body_ids], feet[..., 2] - ground,
        gait_targets(env.episode_length_buf * env.step_dt, env.cfg.locomotion_gait))
    return reward * ~env.command_manager.get_term("base_velocity").is_standing_env


def single_support(env, sensor_cfg):
    """Encourage short single-support intervals including pure yaw / slow commands."""
    sensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 5.
    time = torch.where(contact, sensor.data.current_contact_time[:, sensor_cfg.body_ids],
                       sensor.data.current_air_time[:, sensor_cfg.body_ids])
    reward = time.min(dim=1).values.clamp(max=.35) * (contact.sum(-1) == 1)
    return reward * ~env.command_manager.get_term("base_velocity").is_standing_env


def fall(env):
    data = env.scene["robot"].data
    terrain = env.scene.sensors["height_scanner"].data.ray_hits_w[..., 2].median(dim=1).values
    return (-data.projected_gravity_b[:, 2] < math.cos(math.radians(35.))) | (data.root_pos_w[:, 2] - terrain < .45)
