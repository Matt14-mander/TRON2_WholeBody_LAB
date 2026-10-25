"""Leg-only control, real payload forces and explicit stop/start commands."""
import math
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from bipedal_locomotion.controllers.locomotion_v4 import (
    LEG_JOINTS, sample_planar_wrench, gait_features, gait_targets, gait_score,
)
from bipedal_locomotion.controllers.standwalk_core import scheduled_velocity, standing_mask


class VelocityCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Fixed env quotas prevent short standing failures from disappearing from rollout.
        if getattr(cfg, "profile", "mixed") == "walk_first":
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
        self._command[env_ids] = initial
        self.is_standing_env[env_ids] = standing_mask(initial,
            torch.zeros(n, device=self.device, dtype=torch.bool))

    def _update_command(self):
        time = self._env.episode_length_buf * self._env.step_dt
        command = scheduled_velocity(time, self.roles, self.target, transition_s=1.)
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
