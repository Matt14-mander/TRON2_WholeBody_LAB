"""Native manager terms for privileged force-following with frozen locomotion."""
from __future__ import annotations

import math

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg, CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from bipedal_locomotion.controllers.paint_teacher_core import (
    ARM_JOINT_NAMES, LEG_JOINT_NAMES, FrozenLocomotion, SampledHistory,
    admittance_step, bounded_targets, force_envelope,
)


class PayloadInteraction(CommandTerm):
    """Piecewise held, known external-on-payload COM force; no force prediction.

    The command exposed after step/reset is the force applied in the NEXT high
    control interval. The teacher observes it before choosing that action.
    Reference velocity is in payload local axes, including local-z rotation.
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if (not 0 <= cfg.zero_probability <= 1 or not 0 <= cfg.mixed_probability <= 1
                or not 0 < cfg.initial_scale <= 1 or cfg.ramp_steps < 1
                or not 0 < cfg.active_duration[0] <= cfg.active_duration[1] < cfg.resampling_time_range[0]
                or not math.isfinite(cfg.warmup_s) or cfg.warmup_s < 0
                or any(not math.isfinite(x) or x <= 0 for x in cfg.max_wrench)):
            raise ValueError("Invalid teacher interaction amplitudes, probabilities, timing or curriculum.")
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.reference = torch.zeros_like(self._command)
        self.amplitude = torch.zeros_like(self._command)
        self.elapsed = torch.zeros(self.num_envs, device=self.device)
        self.duration = torch.ones_like(self.elapsed)
        self.sine = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics = {key: torch.zeros_like(self.elapsed) for key in ("force_norm", "curriculum_scale")}

    @property
    def command(self):
        return self._command

    def _resample_command(self, env_ids):
        count = len(env_ids)
        amplitude = 2 * torch.rand(count, 3, device=self.device) - 1
        # Single-axis episodes first; mixed-direction episodes enter gradually.
        stage = min(1., self._env.common_step_counter / max(1, self.cfg.ramp_steps))
        axis = torch.randint(3, (count,), device=self.device)
        single = torch.nn.functional.one_hot(axis, 3).to(amplitude.dtype)
        mixed = torch.rand(count, 1, device=self.device) < self.cfg.mixed_probability * stage
        amplitude *= torch.where(mixed, torch.ones_like(single), single)
        zero = torch.rand(count, device=self.device) < self.cfg.zero_probability
        amplitude[zero] = 0
        scale = self.cfg.initial_scale + (1 - self.cfg.initial_scale) * stage
        self.amplitude[env_ids] = amplitude * torch.tensor(self.cfg.max_wrench, device=self.device) * scale
        self.duration[env_ids] = torch.empty(count, device=self.device).uniform_(*self.cfg.active_duration)
        self.elapsed[env_ids] = 0
        self.sine[env_ids] = torch.rand(count, device=self.device) < .5

    def _update_command(self):
        envelope = force_envelope(self.elapsed / self.duration, self.sine)
        settled = self._env.episode_length_buf * self._env.step_dt >= self.cfg.warmup_s
        self._command.copy_(self.amplitude * envelope[:, None] * settled[:, None])
        self.reference.copy_(admittance_step(self.reference, self._command, self._env.step_dt,
                                             self.cfg.virtual_mass, self.cfg.damping, self.cfg.velocity_limits))
        # Hold the resampled profile at t=0 until the per-environment warmup ends.
        self.elapsed += self._env.step_dt * settled

    def _update_metrics(self):
        self.metrics["force_norm"].copy_(torch.linalg.vector_norm(self._command[:, :2], dim=-1))
        stage = min(1., self._env.common_step_counter / max(1, self.cfg.ramp_steps))
        self.metrics["curriculum_scale"].fill_(self.cfg.initial_scale + (1 - self.cfg.initial_scale) * stage)

    def compute(self, dt):
        warmup = self._env.episode_length_buf * self._env.step_dt < self.cfg.warmup_s
        # Preserve the full first pulse/recovery period after settling.
        self.time_left += dt * warmup
        super().compute(dt)

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device)
        if env_ids is not None:
            ids = ids[env_ids]
        self.reference[ids] = 0
        self._command[ids] = 0
        return super().reset(ids)


@configclass
class PayloadInteractionCfg(CommandTermCfg):
    class_type: type = PayloadInteraction
    resampling_time_range: tuple = (5., 7.)
    max_wrench: tuple = (5., 5., .5)
    warmup_s: float = 2.
    active_duration: tuple = (2., 4.)
    zero_probability: float = .2
    mixed_probability: float = .5
    initial_scale: float = .2
    ramp_steps: int = 25000
    virtual_mass: tuple = (4., 4., .5)
    damping: tuple = (20., 20., 2.)
    velocity_limits: tuple = (.35, .2, .4)


class FrozenLocomotionAction(ActionTerm):
    """Nine high-level outputs; actor+encoder inference at 50 Hz, physics 200 Hz."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not math.isclose(env.physics_dt, .005, abs_tol=1e-9) or env.cfg.decimation != 8:
            raise ValueError("Teacher v1 requires 200 Hz physics / 25 Hz high-level control.")
        self.arm_ids, arm_names = self._asset.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
        self.leg_ids, leg_names = self._asset.find_joints(list(LEG_JOINT_NAMES), preserve_order=True)
        self.payload_ids, payload_names = self._asset.find_bodies(cfg.payload_body_name, preserve_order=True)
        if tuple(arm_names) != ARM_JOINT_NAMES or tuple(leg_names) != LEG_JOINT_NAMES or len(payload_names) != 1:
            raise ValueError("Teacher requires the ordered six-arm/ten-leg joints and one attached payload link.")
        imported_mass = self._asset.root_physx_view.get_masses()[:, self.payload_ids[0]]
        if not torch.allclose(imported_mass, torch.full_like(imported_mass, cfg.payload_mass), rtol=.05, atol=1e-5):
            raise ValueError("Imported payload mass does not match the configured attached payload.")
        self.locomotion = FrozenLocomotion(cfg.locomotion_checkpoint, self.device)
        self.managed_phase = hasattr(self.locomotion.actor[-1], "scale")
        self.nominal = self._asset.data.default_joint_pos[:, self.arm_ids].clone()
        self.limits = self._asset.data.soft_joint_pos_limits[:, self.arm_ids].clone()
        if ((self.nominal < self.limits[..., 0]) | (self.nominal > self.limits[..., 1])).any():
            raise ValueError("Nominal arm pose violates soft joint limits.")
        self._raw = torch.zeros(self.num_envs, 9, device=self.device)
        self._processed = torch.cat((torch.zeros(self.num_envs, 3, device=self.device), self.nominal), dim=-1)
        self.previous = self._processed.clone()
        self.delta = torch.zeros_like(self._processed)
        self.last_leg_action = torch.zeros(self.num_envs, 10, device=self.device)
        self.low_wrench_prediction = torch.zeros(self.num_envs, 30, device=self.device)
        self.executed_arm_target = self.nominal.clone()
        self.leg_target = self._asset.data.default_joint_pos[:, self.leg_ids].clone()
        self.low_history = SampledHistory(self.num_envs, 10, 42, self.device)
        self.arm_history = SampledHistory(self.num_envs, 4, 18, self.device)
        self.payload_gravity_reference = torch.zeros(self.num_envs, 3, device=self.device)
        self.payload_gravity_reference[:, 2] = -1
        self.payload_height_reference = torch.zeros(self.num_envs, device=self.device)
        self.velocity_scale = torch.tensor(cfg.velocity_scale, device=self.device)
        self.slew_rate = torch.tensor(cfg.slew_rate, device=self.device)
        self._substep = 0
        self.support_model = None
        self.support_check = None
        self.arm_feedforward = torch.zeros_like(self.nominal)

    @torch.no_grad()
    def update_support(self):
        # Lazy setup: native reset joint/body states must be available before
        # comparing the URDF frames with the imported articulation.
        if self.support_model is None:
            from pathlib import Path
            from bipedal_locomotion.controllers.paint_wrench_preview import AddedBodyModel
            from bipedal_locomotion.controllers.paint_teacher_support import BatchedSupport
            root = Path(__file__).resolve().parents[5]
            description = root / "robot_description/tron2"
            model = AddedBodyModel(description / "SFYG_TRON2A/urdf/robot_rl.urdf",
                description / "SF_TRON2A/urdf/robot_rl.urdf", self.cfg.payload_mass)
            self.support_model = BatchedSupport(model, self._asset.joint_names, ARM_JOINT_NAMES, self.device)
        data = self._asset.data
        wrench, effort, poses = self.support_model.compute(data.joint_pos,
            data.projected_gravity_b * 9.81, self._env.command_manager.get_command("interaction"))
        if self.support_check is None:
            ids = [self._asset.body_names.index(n) for n in self.support_model.names]
            native_mass = self._asset.root_physx_view.get_masses()[0, ids].to(self.device)
            native_com = quat_apply_inverse(data.root_quat_w[0].expand(len(ids), 4),
                data.body_com_pos_w[0, ids] - data.root_pos_w[0])
            errors = (native_com - poses[0, :, :3, 3]).norm(dim=-1)
            mass_errors = (native_mass - self.support_model.mass).abs()
            valid = bool(((mass_errors <= torch.maximum(native_mass * .05,
                torch.full_like(native_mass, .001))) & (errors <= .03)).all())
            self.support_check = {"valid": valid, "bodies": self.support_model.names,
                "com_error_m": errors.tolist(), "mass_error_kg": mass_errors.tolist()}
            if not valid:
                raise RuntimeError(f"Teacher support inertial/frame validation failed: {self.support_check}")
        if not torch.isfinite(wrench).all() or not torch.isfinite(effort).all():
            raise RuntimeError("Teacher support produced NaN/Inf")
        # A held current estimate fills the five samples. This is explicitly
        # quasi-static support, not the arm-plan inverse-dynamics preview.
        self.set_low_wrench_prediction(wrench[:, None].expand(-1, 5, -1))
        age = self._env.episode_length_buf * self._env.step_dt + self._substep * self._env.physics_dt
        ramp = (age / self.cfg.gravity_ramp_s).clamp(0, 1)
        limit = data.joint_effort_limits[:, self.arm_ids].clamp(max=self.cfg.gravity_effort_limit)
        self.arm_feedforward.copy_(effort.clamp(-limit, limit) * ramp[:, None])

    @property
    def action_dim(self):
        return 9

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._processed

    def proprioception(self, offset_s=0.):
        data = self._asset.data
        gait = self._env.command_manager.get_command("gait_command")
        if self.managed_phase:
            index = self._env.command_manager.get_term("gait_command").gait_indices
            moving = self._processed[:, :3].norm(dim=-1) > 1e-3
            phase = 2 * torch.pi * torch.where(moving,
                torch.remainder(index + offset_s * gait[:, 0], 1.), torch.zeros_like(index))
        else:
            phase = 2 * torch.pi * torch.remainder(
                (self._env.episode_length_buf * self._env.step_dt + offset_s) * gait[:, 0], 1.)
        return torch.cat((data.root_ang_vel_b.clamp(-100, 100) * .25,
                          data.projected_gravity_b.clamp(-100, 100),
                          (data.joint_pos[:, self.leg_ids] - data.default_joint_pos[:, self.leg_ids]).clamp(-100, 100),
                          data.joint_vel[:, self.leg_ids].clamp(-100, 100) * .05,
                          self.last_leg_action.clamp(-100, 100),
                          torch.stack((phase.sin(), phase.cos()), dim=-1), gait), dim=-1)

    def sample_histories(self, tick, offset_s=0.):
        first = self.arm_history.last_tick < 0
        if first.any():
            data = self._asset.data
            body = self.payload_ids[0]
            self.payload_gravity_reference[first] = quat_apply_inverse(data.body_quat_w[first, body], data.GRAVITY_VEC_W[first])
            self.payload_height_reference[first] = data.body_pos_w[first, body, 2] - self._env.scene.env_origins[first, 2]
        self.low_history.append(self.proprioception(offset_s), tick)
        data = self._asset.data
        row = torch.cat((data.joint_pos[:, self.arm_ids], data.joint_vel[:, self.arm_ids],
                         self.executed_arm_target), dim=-1)
        self.arm_history.append(row, tick)

    def process_actions(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError("Teacher action contains NaN/Inf.")
        if self.cfg.require_bounded_actions and (actions.abs() > 1 + 1e-6).any():
            raise ValueError("Teacher v2 requires bounded policy samples, not hard-clipped unbounded actions")
        # Capture the preceding executed arm target before accepting a new target.
        self.sample_histories(self._env._sim_step_counter)
        self._raw.copy_(actions)
        self.previous.copy_(self._processed)
        self._processed.copy_(bounded_targets(actions, self.previous, self.nominal,
                                              self.limits[..., 0], self.limits[..., 1],
                                              self.velocity_scale, self.cfg.arm_scale,
                                              self.slew_rate, self._env.step_dt, self.cfg.base_deadzone))
        self.delta.copy_(self._processed - self.previous)
        self._env.command_manager.get_command("base_velocity").copy_(self._processed[:, :3])
        # Initial settling is a stand/hold interval, not a chance to exploit reset motion.
        warmup = self._env.episode_length_buf * self._env.step_dt < self._env.cfg.commands.interaction.warmup_s
        self._processed[warmup, :3] = 0
        self._processed[warmup, 3:] = self.nominal[warmup]
        self.delta[warmup] = 0
        self._env.command_manager.get_command("base_velocity").copy_(self._processed[:, :3])
        if self.managed_phase:
            # The native GaitCommand advances after each 40ms teacher step;
            # proprioception interpolates its second 20ms low-level sample.
            self._env.command_manager.get_term("base_velocity").is_standing_env.copy_(
                self._processed[:, :3].norm(dim=-1) <= 1e-3)
        self._substep = 0

    @torch.no_grad()
    def set_low_wrench_prediction(self, prediction):
        """Arm-on-base N/Nm, about base_Link origin, all samples in current base axes.

        Observation only: this never applies an additional force to the asset.
        """
        prediction = torch.as_tensor(prediction, device=self.device, dtype=self._processed.dtype)
        if prediction.shape != (self.num_envs, 5, 6) or not torch.isfinite(prediction).all():
            raise ValueError("Expected finite physical base-wrench sequence (N,5,6).")
        scale = prediction.new_tensor((.01, .01, .01, .05, .05, .05))
        self.low_wrench_prediction.copy_((prediction * scale).flatten(1).clamp(-10, 10))

    @torch.no_grad()
    def apply_actions(self):
        if self._substep % 4 == 0:
            if self.cfg.native_support:
                self.update_support()
            tick = self._env._sim_step_counter - 1
            self.sample_histories(tick, self._substep * self._env.physics_dt)
            low_action = self.locomotion(self.proprioception(self._substep * self._env.physics_dt),
                                         self.low_history.data, self._processed[:, :3], self.low_wrench_prediction)
            if not torch.isfinite(low_action).all():
                raise RuntimeError("Frozen locomotion produced a non-finite action.")
            self.last_leg_action.copy_(low_action)
            self.leg_target.copy_(self._asset.data.default_joint_pos[:, self.leg_ids] + .25 * low_action)
        self._asset.set_joint_position_target(self.leg_target, joint_ids=self.leg_ids)
        self._asset.set_joint_position_target(self._processed[:, 3:], joint_ids=self.arm_ids)
        zeros = torch.zeros_like(self.nominal)
        self._asset.set_joint_velocity_target(zeros, joint_ids=self.arm_ids)
        self._asset.set_joint_effort_target(self.arm_feedforward, joint_ids=self.arm_ids)
        self.executed_arm_target.copy_(self._processed[:, 3:])
        planar = self._env.command_manager.get_command("interaction")
        force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        torque = torch.zeros_like(force)
        force[:, 0, :2] = planar[:, :2]
        torque[:, 0, 2] = planar[:, 2]
        self._asset.set_external_force_and_torque(force, torque, body_ids=self.payload_ids, is_global=False)
        self._substep += 1

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self._processed[ids, :3] = 0
        self._processed[ids, 3:] = self.nominal[ids]
        self.previous[ids] = self._processed[ids]
        self.delta[ids] = 0
        self.last_leg_action[ids] = 0
        self.low_wrench_prediction[ids] = 0
        self.arm_feedforward[ids] = 0
        self.executed_arm_target[ids] = self.nominal[ids]
        self.leg_target[ids] = self._asset.data.default_joint_pos[ids][:, self.leg_ids]
        self.low_history.reset(ids)
        self.arm_history.reset(ids)


@configclass
class FrozenLocomotionActionCfg(ActionTermCfg):
    class_type: type = FrozenLocomotionAction
    asset_name: str = "robot"
    locomotion_checkpoint: str = ""
    payload_mass: float = .5
    payload_body_name: str = "paint_payload_Link"
    # The dedicated training entry enables this; legacy support audits retain
    # their explicitly supplied preview and zero effort for reproducibility.
    native_support: bool = False
    base_deadzone: float = 0.  # legacy audits; dedicated teacher entry sets .15
    require_bounded_actions: bool = False
    gravity_ramp_s: float = 2.
    gravity_effort_limit: float = 30.
    velocity_scale: tuple = (.35, .2, .4)
    arm_scale: float = .25
    # Base acceleration limits, followed by arm target slew limits (rad/s).
    slew_rate: tuple = (.6, .4, .8, .5, .5, .5, .5, .5, .5)


def controller(env):
    return env.action_manager.get_term("teacher")


def payload_state(env):
    term = controller(env)
    data = term._asset.data
    body = term.payload_ids[0]
    quat = data.body_quat_w[:, body]
    lin = quat_apply_inverse(quat, data.body_com_lin_vel_w[:, body])
    ang = quat_apply_inverse(quat, data.body_ang_vel_w[:, body])
    gravity = quat_apply_inverse(quat, data.GRAVITY_VEC_W)
    return lin, ang, gravity


def teacher_observation(env):
    term = controller(env)
    term.sample_histories(env._sim_step_counter)
    data = term._asset.data
    force = env.command_manager.get_command("interaction") / torch.tensor((5., 5., .5), device=env.device)
    relative_pos = quat_apply_inverse(data.root_quat_w, data.body_pos_w[:, term.payload_ids[0]] - data.root_pos_w)
    _, _, gravity = payload_state(env)
    history = term.arm_history.data.clone()
    history[..., :6] -= term.nominal[:, None]
    history[..., 6:12] *= .1
    history[..., 12:] -= term.nominal[:, None]
    applied = torch.cat((term.processed_actions[:, :3] / term.velocity_scale,
                         (term.processed_actions[:, 3:] - term.nominal) / term.cfg.arm_scale), dim=-1)
    return torch.cat((force, data.root_lin_vel_b, data.root_ang_vel_b * .25, data.projected_gravity_b,
                      history.flatten(1), relative_pos, gravity, applied), dim=-1)


def critic_observation(env):
    lin, ang, _ = payload_state(env)
    term = controller(env)
    height = term._asset.data.body_pos_w[:, term.payload_ids[0], 2:3] - env.scene.env_origins[:, 2:3]
    return torch.cat((teacher_observation(env), lin, ang, height), dim=-1)


def reference_observation(env):
    return env.command_manager.get_term("interaction").reference


def runner_history_placeholder(env):
    # Repository PPO requires an obsHistory group even with a zero-width encoder.
    # Real arm history is already in policy, sampled at 50 Hz, with no estimator.
    return torch.zeros(env.num_envs, 1, device=env.device)


def follow_reward(env):
    lin, ang, _ = payload_state(env)
    velocity = torch.cat((lin[:, :2], ang[:, 2:3]), dim=-1)
    reference = reference_observation(env)
    error = (velocity - reference) / torch.tensor((.2, .2, .3), device=env.device)
    return torch.exp(-error.square().sum(-1))


def payload_tilt(env):
    # Stabilize the nominal carry orientation; it need not be world-horizontal.
    gravity = payload_state(env)[2]
    return (gravity - controller(env).payload_gravity_reference).square().sum(-1)


def payload_height(env):
    term = controller(env)
    height = term._asset.data.body_pos_w[:, term.payload_ids[0], 2] - env.scene.env_origins[:, 2]
    return (height - term.payload_height_reference).square()


def payload_vertical_motion(env):
    term = controller(env)
    return term._asset.data.body_com_lin_vel_w[:, term.payload_ids[0], 2].square()


def target_rate(env):
    term = controller(env)
    scale = torch.cat((term.velocity_scale, torch.full((6,), term.cfg.arm_scale, device=env.device)))
    return (term.delta / scale).square().sum(-1)


def arm_recentering(env):
    term = controller(env)
    released = torch.linalg.vector_norm(env.command_manager.get_command("interaction"), dim=-1) < .05
    return released * (term._asset.data.joint_pos[:, term.arm_ids] - term.nominal).square().sum(-1)


def target_magnitude(env):
    term = controller(env)
    normalized = torch.cat((term.processed_actions[:, :3] / term.velocity_scale,
        (term.processed_actions[:, 3:] - term.nominal) / term.cfg.arm_scale), -1)
    return normalized.square().mean(-1)


def released_hold(env):
    """Penalize real base motion and requested motion only after reference decays."""
    term = controller(env)
    force = env.command_manager.get_command("interaction")
    quiet = ((force[:, :2].abs().amax(-1) < .05) & (force[:, 2].abs() < .005)
             & (reference_observation(env).abs().amax(-1) < .01))
    data = term._asset.data
    actual = torch.cat((data.root_lin_vel_b[:, :2] / .05, data.root_ang_vel_b[:, 2:3] / .1), -1)
    requested = term.processed_actions[:, :3] / term.velocity_scale
    return quiet * (actual.square().clamp(max=25).mean(-1) + requested.square().mean(-1))


def tilted_or_low(env, max_tilt_deg=35., min_height=.45):
    data = env.scene["robot"].data
    height = data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (-data.projected_gravity_b[:, 2] < math.cos(math.radians(max_tilt_deg))) | (height < min_height)


def payload_overtilt(env, max_tilt_deg=45.):
    alignment = (payload_state(env)[2] * controller(env).payload_gravity_reference).sum(-1)
    return alignment < math.cos(math.radians(max_tilt_deg))


def teacher_failure(env):
    return env.termination_manager.terminated.float()
