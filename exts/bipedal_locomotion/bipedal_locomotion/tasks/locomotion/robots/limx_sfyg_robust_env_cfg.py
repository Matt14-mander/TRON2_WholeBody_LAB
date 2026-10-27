"""Physical-only randomization over the working SFYG carry task."""
from dataclasses import asdict
import torch
from isaaclab.managers import EventTermCfg, ObservationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from bipedal_locomotion.controllers.payload_robustness import (
    RobustSettings, ForceSchedule, nominal_mask, scale_payload_inertia)
from .limx_sfyg_carry_env_cfg import SFYGCarryFlatEnvCfg, CarryLegAction, CarryLegActionCfg


class RobustState:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.cfg = RobustSettings(**env.cfg.robustness).validate()
        self.nominal = nominal_mask(env.num_envs, self.cfg.nominal_fraction, env.device)
        self.schedule = ForceSchedule(self.nominal, self.cfg)
        self.payload_ids, _ = self.robot.find_bodies("paint_payload_Link")
        if len(self.payload_ids) != 1:
            raise ValueError("Expected exactly one physical payload")
        self.payload = self.payload_ids[0]
        self.materials = self.robot.root_physx_view.get_material_properties().clone()
        if not bool((self.materials[..., :2] > 0).any()):
            raise ValueError("Authored friction is zero; multiplicative randomization would be ineffective")
        self.frictions = torch.ones(env.num_envs, device=env.device)
        self.masses = self.robot.root_physx_view.get_masses().to(env.device)
        self.last_local = torch.zeros(env.num_envs, 1, 3, device=env.device)
        self.sums = torch.zeros(6, device=env.device)
        self.samples = 0
        self.peak_force = torch.zeros((), device=env.device)

    def indices(self, ids):
        if ids is None:
            ids = torch.arange(self.env.num_envs, device=self.env.device)
        return ids, ids.to(device="cpu", dtype=torch.long)

    def observe(self, command):
        velocity = self.robot.data.root_lin_vel_b[:, :2]
        speed = command[:, :2].norm(dim=-1)
        moving = speed > .02
        along = (velocity * command[:, :2]).sum(-1) / speed.clamp_min(.02)
        force = self.schedule.current.norm(dim=-1)
        self.sums += torch.stack((force.sum(), (force > .01).sum(),
            speed[moving].sum(), along[moving].sum(),
            self.masses[:, self.payload].sum(), self.frictions.sum()))
        self.peak_force.copy_(torch.maximum(self.peak_force, force.max()))
        self.samples += self.env.num_envs

    def pop_metrics(self):
        a, n = self.sums.tolist(), max(1, self.samples)
        result = {"Robust/force_norm_mean_n": a[0]/n, "Robust/force_active_fraction": a[1]/n,
            "Robust/force_norm_max_n": float(self.peak_force),
            "Motion/linear_progress_ratio": a[3]/max(a[2],1e-6),
            "Robust/payload_mass_mean_kg": a[4]/n, "Robust/friction_scale_mean": a[5]/n}
        self.sums.zero_()
        self.peak_force.zero_()
        self.samples = 0
        return result


def robust_state(env):
    if not hasattr(env, "_sfyg_robust"):
        env._sfyg_robust = RobustState(env)
    return env._sfyg_robust


def randomize_payload_mass(env, env_ids):
    s = robust_state(env)
    ids, cpu = s.indices(env_ids)
    mass = torch.empty(len(cpu)).uniform_(*s.cfg.mass_range)
    mass[s.nominal[ids].cpu()] = s.cfg.nominal_mass_kg
    robot, body = s.robot, s.payload
    masses = robot.root_physx_view.get_masses()
    inertias = robot.root_physx_view.get_inertias()
    masses[cpu, body] = mass
    inertias[cpu, body] = scale_payload_inertia(robot.data.default_inertia.cpu()[cpu, body],
                                              mass, robot.data.default_mass.cpu()[cpu, body])
    robot.root_physx_view.set_masses(masses, cpu)
    robot.root_physx_view.set_inertias(inertias, cpu)
    s.masses[ids] = masses[cpu].to(env.device)


def randomize_friction(env, env_ids):
    """Startup buckets relative to authored material; restitution is unchanged."""
    s = robust_state(env)
    ids, cpu = s.indices(env_ids)
    buckets = torch.linspace(*s.cfg.friction_scale_range, s.cfg.material_buckets)
    scales = buckets[torch.randint(s.cfg.material_buckets, (len(cpu),))]
    scales[s.nominal[ids].cpu()] = 1.
    material = s.robot.root_physx_view.get_material_properties()
    material[cpu] = s.materials[cpu]
    material[cpu, :, :2] = s.materials[cpu, :, :2] * scales[:, None, None]
    s.robot.root_physx_view.set_material_properties(material, cpu)
    s.frictions[ids] = scales.to(env.device)


def actual_masses(env):
    # Same width/order as original robot_mass; actual randomized mass is visible
    # only to the critic. No applied-force ground truth is added to the actor.
    return robust_state(env).masses


class RobustCarryLegAction(CarryLegAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.state = robust_state(env)

    def apply_actions(self):
        super().apply_actions()
        s = self.state
        world = s.schedule.advance(self._env.physics_dt)
        quat = self._asset.data.body_quat_w[:, s.payload]
        s.last_local[:, 0] = quat_apply_inverse(quat, world)
        self._asset.set_external_force_and_torque(s.last_local, torch.zeros_like(s.last_local),
            body_ids=s.payload_ids, is_global=False)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        s = self.state
        ids, _ = s.indices(env_ids)
        s.schedule.reset(ids)
        s.last_local[ids] = 0.
        # Event reset happens before ActionManager.reset. Clear the simulator's
        # force buffer too; resetting only our command would leak the old load.
        self._asset.set_external_force_and_torque(s.last_local[ids], torch.zeros_like(s.last_local[ids]),
            body_ids=s.payload_ids, env_ids=ids, is_global=False)


@configclass
class RobustCarryLegActionCfg(CarryLegActionCfg):
    class_type: type = RobustCarryLegAction


@configclass
class SFYGCarryRobustEnvCfg(SFYGCarryFlatEnvCfg):
    robustness: dict = asdict(RobustSettings())

    def __post_init__(self):
        super().__post_init__()
        previous = self.actions.joint_pos
        self.actions.joint_pos = RobustCarryLegActionCfg(asset_name="robot",
            joint_names=previous.joint_names, preserve_order=True, scale=previous.scale, use_default_offset=True)
        self.events.payload_mass = EventTermCfg(func=randomize_payload_mass, mode="reset")
        self.events.robust_friction = EventTermCfg(func=randomize_friction, mode="startup")
        self.observations.critic.robot_mass = ObservationTermCfg(func=actual_masses)
        self.sim.physx.enable_external_forces_every_iteration = True
