"""Shared transport environment MDP; teacher privilege never enters student obs."""
import math
import torch
from isaaclab.managers import ActionTerm, ActionTermCfg, CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from bipedal_locomotion.controllers.paint_transport_v4 import (
    ARM_JOINTS, LEG_JOINTS, WRENCH_SCALE, History, FrozenBackbone,
    envelope, base_command, arm_targets, tracking_scores,
)


class Interaction(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.target = torch.zeros(self.num_envs, 3, device=self.device)
        self._command = torch.zeros_like(self.target)
        self.metrics = {"normalized_force": torch.zeros(self.num_envs, device=self.device)}

    @property
    def command(self):
        return self._command

    def _resample_command(self, env_ids):
        n = len(env_ids)
        scale = torch.tensor(WRENCH_SCALE, device=self.device)
        target = (2 * torch.rand(n, 3, device=self.device) - 1) * torch.tensor(self.cfg.max_wrench, device=self.device) / scale
        if self.cfg.case != "random":
            signs = {"zero": (0, 0, 0), "forward": (1, 0, 0), "backward": (-1, 0, 0),
                     "left": (0, 1, 0), "right": (0, -1, 0), "yaw_left": (0, 0, 1), "yaw_right": (0, 0, -1)}
            target[:] = torch.tensor(signs[self.cfg.case], device=self.device) * torch.tensor(self.cfg.max_wrench, device=self.device) / scale
        elif n:
            target[torch.rand(n, device=self.device) < self.cfg.zero_probability] = 0.
        self.target[env_ids] = target
        self._command[env_ids] = 0.

    def _update_command(self):
        age = self._env.episode_length_buf * self._env.step_dt
        self._command.copy_(self.target * envelope(age, self._env.max_episode_length_s)[:, None])

    def _update_metrics(self):
        self.metrics["normalized_force"].copy_(self.command[:, :2].norm(dim=-1))


@configclass
class InteractionCfg(CommandTermCfg):
    class_type: type = Interaction
    resampling_time_range: tuple = (1e9, 1e9)
    max_wrench: tuple = (3., 3., .5)
    zero_probability: float = .02
    case: str = "random"


class TransportAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not math.isclose(env.physics_dt, .005) or env.cfg.decimation != 4:
            raise ValueError("Transport v4 requires 200 Hz physics and 50 Hz controls.")
        self.leg_ids, leg_names = self._asset.find_joints(list(LEG_JOINTS), preserve_order=True)
        self.arm_ids, arm_names = self._asset.find_joints(list(ARM_JOINTS), preserve_order=True)
        self.gripper_ids, _ = self._asset.find_joints(["gripper[12]_Joint"])
        self.payload_ids, _ = self._asset.find_bodies("paint_payload_Link")
        self.ee_ids, _ = self._asset.find_bodies("gripper_base_Link")
        if tuple(leg_names) != LEG_JOINTS or tuple(arm_names) != ARM_JOINTS or len(self.payload_ids) != 1 or len(self.ee_ids) != 1:
            raise ValueError("Transport requires the declared full robot and separate payload.")
        self.low = FrozenBackbone(cfg.low_path, self.device, cfg.low_format, cfg.low_manifest or None)
        trained_limits = self.low.manifest["velocity_limits"]
        if any(limit > trained + 1e-7 for limit, trained in zip(cfg.base_limits, trained_limits)):
            raise ValueError("Transport command limits exceed frozen v4 training limits.")
        self.nominal = self._asset.data.default_joint_pos[:, self.arm_ids].clone()
        self.soft_limits = self._asset.data.soft_joint_pos_limits[:, self.arm_ids].clone()
        if ((self.nominal < self.soft_limits[..., 0]) | (self.nominal > self.soft_limits[..., 1])).any():
            raise ValueError("Nominal arm posture violates soft joint limits.")
        self._raw = torch.zeros(self.num_envs, 9, device=self.device)
        self._processed = torch.cat((torch.zeros(self.num_envs, 3, device=self.device), self.nominal), -1)
        self.previous = self._processed.clone()
        self.delta = torch.zeros_like(self._processed)
        self.last_leg = torch.zeros(self.num_envs, 10, device=self.device)
        self.leg_target = self._asset.data.default_joint_pos[:, self.leg_ids].clone()
        self.executed_arm = self.nominal.clone()
        self.low_history = History(self.num_envs, 10, 36, self.device, repeat_first=True)
        self.clean_history = History(self.num_envs, 8, 18, self.device)
        self.student_history = History(self.num_envs, 8, 18, self.device)
        self.force_world = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.torque_world = torch.zeros_like(self.force_world)
        self._substep = 0

    @property
    def action_dim(self): return 9

    @property
    def raw_actions(self): return self._raw

    @property
    def processed_actions(self): return self._processed

    def sample_arm_history(self):
        tick = self._env._sim_step_counter
        # Observation getters can be called repeatedly at the same tick.
        if bool((self.clean_history.tick == tick).all()):
            return
        data = self._asset.data
        row = torch.cat((data.joint_pos[:, self.arm_ids] - self.nominal,
                         data.joint_vel[:, self.arm_ids], self.executed_arm - self.nominal), -1)
        self.clean_history.append(row, tick)
        noisy = row.clone()
        if self.cfg.student_noise:
            noisy[:, :6] += (2 * torch.rand_like(noisy[:, :6]) - 1) * .01
            noisy[:, 6:12] += (2 * torch.rand_like(noisy[:, 6:12]) - 1) * 1.5
        self.student_history.append(noisy, tick)

    @torch.no_grad()
    def process_actions(self, actions):
        if not torch.isfinite(actions).all() or (actions.abs() > 1.00001).any():
            raise ValueError("High-level actions must be finite and in [-1,1].")
        self._raw.copy_(actions)
        self.previous.copy_(self._processed)
        self._processed[:, :3] = base_command(actions, self.cfg.base_limits)
        self._processed[:, 3:] = arm_targets(actions, self.previous[:, 3:], self.nominal,
            self.soft_limits, self.cfg.arm_scale, self.cfg.arm_rate, self._env.step_dt)
        self.delta.copy_(self._processed - self.previous)
        data = self._asset.data
        proprio = torch.cat((data.root_ang_vel_b * .25, data.projected_gravity_b,
            data.joint_pos[:, self.leg_ids] - data.default_joint_pos[:, self.leg_ids],
            (data.joint_vel[:, self.leg_ids] - data.default_joint_vel[:, self.leg_ids]) * .05,
            self.last_leg), -1)
        self.low_history.append(proprio, self._env._sim_step_counter)
        hits = self._env.scene.sensors["height_scanner"].data.ray_hits_w[..., 2]
        if not torch.isfinite(hits).all():
            raise RuntimeError("Transport height scan missed terrain.")
        heights = (data.root_pos_w[:, 2:3] - hits - .5).clamp(-1., 1.)
        leg = self.low(torch.cat((proprio, heights), -1), self.low_history.data, self._processed[:, :3])
        if not torch.isfinite(leg).all():
            raise FloatingPointError("Frozen v4 returned nonfinite actions.")
        self.last_leg.copy_(leg)
        self.leg_target.copy_(data.default_joint_pos[:, self.leg_ids] + .25 * leg)
        self._substep = 0

    def apply_actions(self):
        self._asset.set_joint_position_target(self.leg_target, joint_ids=self.leg_ids)
        self._asset.set_joint_position_target(self._processed[:, 3:], joint_ids=self.arm_ids)
        self._asset.set_joint_velocity_target(torch.zeros_like(self.nominal), joint_ids=self.arm_ids)
        self._asset.set_joint_effort_target(torch.zeros_like(self.nominal), joint_ids=self.arm_ids)
        self.executed_arm.copy_(self._processed[:, 3:])
        self._asset.set_joint_position_target(self._asset.data.default_joint_pos[:, self.gripper_ids], joint_ids=self.gripper_ids)
        data = self._asset.data
        c = self._env.command_manager.get_command("interaction")
        force, torque = torch.zeros_like(self.force_world), torch.zeros_like(self.torque_world)
        local_force, local_torque = torch.zeros_like(c), torch.zeros_like(c)
        local_force[:, :2], local_torque[:, 2] = c[:, :2] * 40., c[:, 2] * 10.
        force[:, 0] = quat_apply(data.root_quat_w, local_force)
        torque[:, 0] = quat_apply(data.root_quat_w, local_torque)
        t = self._env.episode_length_buf * self._env.step_dt + self._substep * self._env.physics_dt
        scale = self.cfg.disturbance_scale
        force[:, 1, 2] = scale * 2 * torch.sin(2 * math.pi * t)
        torque[:, 1, 0] = scale * torch.sin(2 * math.pi * t + math.pi / 4)
        torque[:, 1, 1] = scale * torch.sin(2 * math.pi * t + math.pi / 2)
        self.force_world.copy_(force)
        self.torque_world.copy_(torque)
        ids = self.payload_ids + self.ee_ids
        quats = data.body_quat_w[:, ids]
        local_f = quat_apply_inverse(quats.reshape(-1, 4), force.reshape(-1, 3)).reshape_as(force)
        local_t = quat_apply_inverse(quats.reshape(-1, 4), torque.reshape(-1, 3)).reshape_as(torque)
        self._asset.set_external_force_and_torque(local_f, local_t, body_ids=ids, is_global=False)
        self._substep += 1

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.
        self._processed[ids, :3] = 0.
        self._processed[ids, 3:] = self.nominal[ids]
        self.previous[ids] = self._processed[ids]
        self.delta[ids] = 0.
        self.last_leg[ids] = 0.
        self.leg_target[ids] = self._asset.data.default_joint_pos[ids][:, self.leg_ids]
        self.executed_arm[ids] = self.nominal[ids]
        self.force_world[ids], self.torque_world[ids] = 0., 0.
        for history in (self.low_history, self.clean_history, self.student_history):
            history.reset(ids)


@configclass
class TransportActionCfg(ActionTermCfg):
    class_type: type = TransportAction
    asset_name: str = "robot"
    low_path: str = ""
    low_format: str = "checkpoint"
    low_manifest: str = ""
    base_limits: tuple = (.35, .2, .4)
    arm_scale: float = .1
    arm_rate: float = 1.
    disturbance_scale: float = 0.
    student_noise: bool = True


def controller(env): return env.action_manager.get_term("transport")


def student_observation(env):
    term = controller(env)
    term.sample_arm_history()
    return term.student_history.data.flatten(1)


def teacher_observation(env):
    term = controller(env)
    term.sample_arm_history()
    return torch.cat((term.clean_history.data.flatten(1), env.command_manager.get_command("interaction")), -1)


def policy_observation(env):
    return teacher_observation(env) if env.cfg.actor_mode == "teacher" else student_observation(env)


def critic_observation(env):
    term = controller(env)
    data = term._asset.data
    states = [teacher_observation(env)]
    for ids in (term.ee_ids, term.payload_ids):
        body = ids[0]
        q = data.root_quat_w
        states.extend((quat_apply_inverse(q, data.body_com_pos_w[:, body] - data.root_pos_w),
            quat_apply_inverse(q, data.body_com_lin_vel_w[:, body]),
            quat_apply_inverse(q, data.body_com_ang_vel_w[:, body]),
            quat_apply_inverse(q, data.body_com_lin_acc_w[:, body]),
            quat_apply_inverse(q, data.body_com_ang_acc_w[:, body])))
    return torch.cat(states, -1)


def empty_commands(env): return torch.zeros(env.num_envs, 0, device=env.device)


def placeholder(env): return torch.zeros(env.num_envs, 1, device=env.device)


def force_tracking(env):
    data = env.scene["robot"].data
    actual = torch.cat((data.root_lin_vel_b[:, :2], data.root_ang_vel_b[:, 2:3]), -1)
    return tracking_scores(actual, env.command_manager.get_command("interaction"))[0]


def torque_tracking(env):
    data = env.scene["robot"].data
    actual = torch.cat((data.root_lin_vel_b[:, :2], data.root_ang_vel_b[:, 2:3]), -1)
    return tracking_scores(actual, env.command_manager.get_command("interaction"))[1]


def payload_height(env):
    term = controller(env)
    data = term._asset.data
    height = quat_apply_inverse(data.root_quat_w,
        data.body_com_pos_w[:, term.payload_ids[0]] - data.root_pos_w)[:, 2]
    lower, upper = env.cfg.payload_height_range
    return ((lower - height).clamp_min(0) + (height - upper).clamp_min(0)).square()


def arm_posture(env):
    term = controller(env)
    return (term._asset.data.joint_pos[:, term.arm_ids] - term.nominal).square().sum(-1)


def action_penalty(env):
    term = controller(env)
    # Physical base commands, relative arm targets; normalize arm by its declared range.
    current = torch.cat((term.processed_actions[:, :3],
        (term.processed_actions[:, 3:] - term.nominal) / term.cfg.arm_scale), -1)
    previous = torch.cat((term.previous[:, :3], (term.previous[:, 3:] - term.nominal) / term.cfg.arm_scale), -1)
    return (current - previous).square().sum(-1) + .5 * current.square().sum(-1)


def dof_penalty(env):
    data = env.scene["robot"].data
    return (data.joint_vel.square() + .00025 * data.joint_acc.square()).sum(-1)


def failed(env): return env.termination_manager.terminated.float()


def terrain_progress(env, env_ids):
    terrain = env.scene.terrain
    if not hasattr(env, "expected_distance"):
        return torch.tensor(0., device=env.device)
    ids = torch.as_tensor(env_ids, device=env.device)
    expected = env.expected_distance[ids]
    desired = env.reference_displacement[ids]
    displacement = env.scene["robot"].data.root_pos_w[ids, :2] - env.start_xy[ids]
    distance = (displacement * desired).sum(-1) / desired.norm(dim=-1).clamp_min(1e-6)
    moving = (expected > .1) & (desired.norm(dim=-1) > .05) & (env.episode_length_buf[ids] > 0)
    failure = env.termination_manager.terminated[ids]
    up = moving & ~failure & (distance > .8 * expected)
    moving_target = env.command_manager.get_term("interaction").target[ids, :2].norm(dim=-1) > .01
    down = (moving_target & failure & (env.episode_length_buf[ids] > 0)) | (moving & (distance < .5 * expected))
    terrain.update_env_origins(ids, up, down)
    return terrain.terrain_levels.float().mean()
