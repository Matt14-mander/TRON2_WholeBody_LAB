"""Scripted PAINT support acceptance; no teacher PPO and no intent estimator.

Uses the actual teacher action term and a physical fixed payload. Native COM
accelerations are sampled before each 200Hz physics interval, plus terminal state.
Low-level input can be zero, static gravity, or a scripted inverse-dynamics preview.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rsl_rl"))
sys.path.insert(0, str(ROOT / "exts/bipedal_locomotion"))
CASES = ("hold", "forward", "backward", "left", "right", "yaw_left", "yaw_right",
         "start_stop", "arm_hold", "arm_walk", "force_hold", "force_walk", "coupled")


def stage(args, name):
    args._diagnostic_stage = name
    print(f"[PAINT STAGE] {name}", flush=True)


def record_failure(args, error):
    """Persist the original exception before Isaac/Kit shutdown can hide it."""
    if getattr(args, "_failure_recorded", False):
        return
    details = dict(stage=getattr(args, "_diagnostic_stage", "startup"),
        exception=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    print("[PAINT ERROR] " + json.dumps(details), file=sys.stderr, flush=True)
    output = getattr(args, "_acceptance_output", None)
    if output:
        try:
            Path(output, "failure.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        except OSError as write_error:
            print(f"[PAINT ERROR] Cannot save failure.json: {write_error}", file=sys.stderr, flush=True)
    args._failure_recorded = True


def smooth(u):
    u = max(0., min(1., u))
    return u ** 3 * (10. - 15. * u + 6. * u * u)


def targets(t, case, velocity, arm_delta):
    fixed = {"forward": (.15, 0., 0.), "backward": (-.25, 0., 0.),
             "left": (0., .1, 0.), "right": (0., -.1, 0.),
             "yaw_left": (0., 0., .2), "yaw_right": (0., 0., -.2)}
    moving = case in fixed or case in ("start_stop", "arm_walk", "force_walk", "coupled")
    v = fixed.get(case, velocity)
    scale = smooth(t - 3.) if moving else 0.
    if case == "start_stop":
        scale *= 1. - smooth(t - 12.)
    arm = smooth((t - 7.) / 3.) * (1. - smooth((t - 12.) / 3.))
    if case not in ("arm_hold", "arm_walk", "coupled"):
        arm = 0.
    return [x * scale for x in v], [x * arm for x in arm_delta]


def pulse(t):
    return smooth((t - 10.) / .5) * smooth((12. - t) / .5)


def summarize(rows, event_end, hold_s):
    import numpy as np
    selected = [r for r in rows if r["time_s"] >= 2.]
    if not selected:
        selected = rows
    if not selected:
        return {"samples": 0, "recovery": None}
    t = np.array([r["time_s"] for r in selected])
    tilt = np.array([r["payload_tilt_from_reset_deg"] for r in selected])
    accel = np.linalg.norm([r["payload_acc_world"] for r in selected], axis=1)
    error = np.array([r["base_velocity"] for r in selected]) - np.array([r["base_command"] for r in selected])
    speed_error = np.linalg.norm(error[:, :2], axis=1)
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    report = dict(samples=len(rows), base_tilt_peak_deg=max(r["base_tilt_deg"] for r in rows),
        payload_tilt_peak_world_deg=max(r["payload_tilt_world_deg"] for r in rows),
        payload_tilt_peak_from_reset_deg=max(r["payload_tilt_from_reset_deg"] for r in rows),
        payload_tilt_rms_from_reset_deg=rms(tilt), payload_acc_rms_m_s2=rms(accel),
        payload_acc_peak_m_s2=max(float(np.linalg.norm(r["payload_acc_world"])) for r in rows),
        base_xy_tracking_rmse=rms(speed_error),
        arm_tracking_peak_rad=max(max(abs(a-b) for a,b in zip(r["arm_q"],r["arm_target"])) for r in rows),
        force_active_samples=sum(any(abs(x) > 1e-8 for x in r["force_payload_local"]) for r in rows),
        low_action_abs_max=max(r["low_action_abs_max"] for r in rows), recovery=None)
    report["base_velocity_mean"] = np.mean([r["base_velocity"] for r in selected], axis=0).tolist()
    report["base_velocity_std"] = np.std([r["base_velocity"] for r in selected], axis=0).tolist()
    report["wrench_outside_repo_bounds_fraction"] = float(np.mean([
        r.get("wrench_outside_repo_training_bounds", 0.) for r in selected]))
    baseline = (t >= 4.5) & (t < 6.5)
    if event_end is not None and baseline.sum() >= 20 and t[-1] >= event_end + hold_s:
        # Diagnostic restoration of pre-event RMS, not a task-specific pass/fail.
        limits = [rms(tilt[baseline]) + 2., rms(accel[baseline]) + 1., rms(speed_error[baseline]) + .05]
        dt = float(np.median(np.diff(t)))
        n = max(2, int(round(hold_s / dt)))
        if len(t) >= n:
            values = [np.sqrt(np.maximum(0., np.convolve(x*x, np.ones(n)/n, mode="valid")))
                      for x in (tilt, accel, speed_error)]
            good = t[:len(t)-n+1] >= event_end
            for value, limit in zip(values, limits):
                good &= value <= limit
            indices = np.flatnonzero(good)
            report["recovery"] = dict(event_end_s=event_end, window_s=hold_s,
                limits=dict(payload_tilt_rms_deg=limits[0], payload_acc_rms_m_s2=limits[1],
                            base_xy_tracking_rmse=limits[2]), recovered=bool(len(indices)),
                confirmation_latency_s=float(t[indices[0]+n-1]-event_end) if len(indices) else None,
                note="RMS returned to pre-event baseline plus diagnostic margins; not a formal acceptance gate")
    return report


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--cases", nargs="+", choices=CASES, default=["hold"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--duration", type=float, default=30.)
    p.add_argument("--arm_pose", choices=("nominal", "tucked"), default="nominal")
    p.add_argument("--arm_joints", nargs=6, type=float,
                   help="Override nominal arm1..arm6 joint angles in radians with the actual PAINT hold pose")
    p.add_argument("--payload_mass", type=float, default=.5)
    p.add_argument("--payload_mode", choices=("attached", "none"), default="attached")
    p.add_argument("--low_wrench", choices=("zero", "gravity", "plan"), default="zero")
    p.add_argument("--payload_dimensions", nargs=3, type=float, default=[.20, .12, .08])
    p.add_argument("--payload_offset", nargs=3, type=float, default=[.15, 0., 0.])
    p.add_argument("--velocity", nargs=3, type=float, default=[.15, 0., 0.])
    p.add_argument("--arm_delta", nargs=6, type=float, default=[0., .10, 0., -.05, 0., 0.])
    p.add_argument("--force", nargs=3, type=float, default=[5., 0., 0.], help="Payload-COM local Fx/Fy/Mz (N,N,Nm)")
    p.add_argument("--recovery_window", type=float, default=1.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir")
    return p


def main(args, app):
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils import configclass
    from isaaclab_tasks.utils import parse_env_cfg
    from bipedal_locomotion.controllers.paint_payload import create_payload_asset, create_unloaded_robot_asset
    from bipedal_locomotion.tasks.teacher import mdp
    from bipedal_locomotion.controllers.paint_teacher_core import admittance_step
    from bipedal_locomotion.controllers.paint_wrench_preview import AddedBodyModel
    from bipedal_locomotion.tasks.locomotion.mdp.commands.commands_cfg import WrenchSequenceCommandCfg

    numeric = [args.duration, args.payload_mass, args.recovery_window, *args.payload_dimensions,
               *args.payload_offset, *args.velocity, *args.arm_delta, *args.force, *(args.arm_joints or [])]
    if (not all(math.isfinite(v) for v in numeric) or args.duration < 22. or args.episodes < 1
            or args.payload_mass <= 0 or min(args.payload_dimensions) <= 0 or args.recovery_window <= 0):
        raise ValueError("Use finite values, positive payload/geometry/window, duration>=22s and episodes>=1")
    if len(set(args.cases)) != len(args.cases):
        raise ValueError("Duplicate test cases")
    if any(c in args.cases for c in ("force_hold", "force_walk", "coupled")) and not any(args.force):
        raise ValueError("Force test requires a nonzero --force")
    if args.payload_mode == "none" and any(c in args.cases for c in ("force_hold", "force_walk", "coupled")):
        raise ValueError("Use attached payload mode for force acceptance")
    if args.low_wrench != "zero" and any(c in args.cases for c in ("force_hold", "force_walk", "coupled")):
        raise ValueError("Gravity/plan previews currently cover hold and arm-motion cases, not interaction forces")
    source = Path(args.checkpoint_path).expanduser().resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    output = Path(args.output_dir or ROOT / "logs/paint" /
                  datetime.now().strftime("paint_support_%Y%m%d_%H%M%S_%f")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    args._acceptance_output = str(output)
    task = "Isaac-Limx-SFYG-TRON2A-Paint-Teacher-Flat-Play-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    cfg.seed, cfg.episode_length_s = args.seed, args.duration
    cfg.actions.teacher.locomotion_checkpoint = str(source)
    cfg.actions.teacher.payload_mass = args.payload_mass
    if args.arm_pose == "tucked":
        cfg.scene.robot.init_state.joint_pos.update({"arm2_Joint": .15, "arm3_Joint": -.15})
    if args.arm_joints is not None:
        cfg.scene.robot.init_state.joint_pos.update(
            {f"arm{i+1}_Joint": q for i,q in enumerate(args.arm_joints)})
    if args.payload_mode == "attached":
        cfg.scene.robot.spawn.usd_path = create_payload_asset(cfg.scene.robot.spawn.usd_path,
            output / "robot_payload.usdc", args.payload_mass, args.payload_dimensions, args.payload_offset)
    else:
        cfg.scene.robot.spawn.usd_path = create_unloaded_robot_asset(
            cfg.scene.robot.spawn.usd_path, output / "robot_unloaded.usdc")
        cfg.actions.teacher.payload_body_name = "gripper_base_Link"

    class KnownPulse(mdp.PayloadInteraction):
        def _resample_command(self, env_ids):
            self.amplitude[env_ids] = 0
            self.elapsed[env_ids] = 0

        def _update_command(self):
            t = float(self._env.episode_length_buf[0]) * self._env.step_dt
            self._command[:] = torch.tensor(self.cfg.test_force, device=self.device) * pulse(t)
            self.reference.copy_(admittance_step(self.reference, self._command, self._env.step_dt,
                self.cfg.virtual_mass, self.cfg.damping, self.cfg.velocity_limits))

    @configclass
    class KnownPulseCfg(mdp.PayloadInteractionCfg):
        class_type = KnownPulse
        resampling_time_range = (1e9, 1e9)
        test_force = (0., 0., 0.)

    cfg.commands.interaction = KnownPulseCfg()
    model = AddedBodyModel(ROOT / "robot_description/tron2/SFYG_TRON2A/urdf/robot.urdf",
        ROOT / "robot_description/tron2/SF_TRON2A/urdf/robot.urdf",
        args.payload_mass if args.payload_mode == "attached" else 0.,
        args.payload_dimensions, args.payload_offset)
    bounds_cfg = WrenchSequenceCommandCfg()
    training_bounds = [*bounds_cfg.force_ranges, *bounds_cfg.torque_ranges]

    class AuditAction(mdp.FrozenLocomotionAction):
        def __init__(self, action_cfg, raw_env):
            if args.payload_mode == "none":
                asset = raw_env.scene[action_cfg.asset_name]
                ids, _ = asset.find_bodies("gripper_base_Link", preserve_order=True)
                # No added body: track the real gripper and validate its existing mass.
                action_cfg.payload_mass = float(asset.root_physx_view.get_masses()[0, ids[0]])
            super().__init__(action_cfg, raw_env)
            self.preview_case = "hold"
            self.physical_preview = torch.zeros(1, 5, 6, device=self.device)
            self.preview_time = 0.
            self.model_check = None

        def update_preview(self, t):
            import numpy as np
            self.preview_time = t
            if args.low_wrench == "zero":
                self.physical_preview.zero_()
            else:
                if self.model_check is None:
                    stage(args, "preview: read measured joints and gravity")
                q = dict(zip(self._asset.joint_names, self._asset.data.joint_pos[0].cpu().tolist()))
                gravity = self._asset.data.projected_gravity_b[0].cpu().numpy() * np.linalg.norm(cfg.sim.gravity)
                if self.model_check is None:
                    stage(args, "preview: read native body masses and COM positions")
                    # Compare URDF mass/COM with the actual spawned USD before
                    # trusting its wrench. This also catches merged/missing links.
                    names = self._asset.body_names
                    states = model.states(q)
                    masses = self._asset.root_physx_view.get_masses()[0].cpu().numpy()
                    com_b = mdp.quat_apply_inverse(
                        self._asset.data.root_quat_w[0].expand(len(names), -1),
                        self._asset.data.body_com_pos_w[0]-self._asset.data.root_pos_w[0]).cpu().numpy()
                    checks = []
                    stage(args, "preview: compare URDF and USD inertias/frames")
                    for name, (mass, _, _) in model.bodies.items():
                        if name not in names:
                            checks.append(dict(body=name, missing=True)); continue
                        i = names.index(name)
                        checks.append(dict(body=name, mass_error_kg=float(abs(masses[i]-mass)),
                            com_error_m=float(np.linalg.norm(com_b[i]-states[name][:3, 3]))))
                    valid = all(not r.get("missing") and r["mass_error_kg"] <= max(.001, model.bodies[r["body"]][0]*.05)
                                and r["com_error_m"] <= .03 for r in checks)
                    self.model_check = dict(valid=valid, checks=checks,
                        tolerance="mass 5% or 1g; COM 3cm at initial pose")
                    (output / "wrench_model_check.json").write_text(json.dumps(self.model_check, indent=2), encoding="utf-8")
                    if not valid:
                        raise ValueError("URDF/USD inertial or frame mismatch; inspect wrench_model_check.json before interpreting predictions")
                    stage(args, "preview: inertial/frame consistency passed")
                if args.low_wrench == "gravity":
                    physical = np.repeat(model.gravity_wrench(q, gravity)[None], 5, axis=0)
                else:
                    # Constant present tracking offset; future derivatives come
                    # from the scripted arm plan. Base pose held at current pose.
                    now = targets(t, self.preview_case, args.velocity, args.arm_delta)[1]
                    def position_at(future):
                        delta = targets(future, self.preview_case, args.velocity, args.arm_delta)[1]
                        out = dict(q)
                        for i in range(6):
                            out[f"arm{i+1}_Joint"] += delta[i]-now[i]
                        return out
                    physical = model.preview(position_at, t, gravity)
                self.physical_preview.copy_(torch.as_tensor(physical, device=self.device).unsqueeze(0))
            self.set_low_wrench_prediction(self.physical_preview)

        def apply_actions(self):
            t = float(self._env.episode_length_buf[0]) * self._env.step_dt + self._substep * self._env.physics_dt
            if self._substep % 4 == 0:
                self.update_preview(t)
            callback = getattr(self._env, "_acceptance_capture", None)
            if callback is not None:
                callback(self._env, float(self._env.episode_length_buf[0]) * self._env.step_dt
                         + self._substep * self._env.physics_dt)
            super().apply_actions()

    class AuditEnv(ManagerBasedRLEnv):
        def __init__(self, *a, **kw):
            self._acceptance_capture = None
            super().__init__(*a, **kw)

        def _reset_idx(self, ids):
            if self._acceptance_capture is not None and len(ids) and self.episode_length_buf[0] > 0:
                self._acceptance_capture(self, float(self.episode_length_buf[0]) * self.step_dt)
            super()._reset_idx(ids)

    cfg.actions.teacher.class_type = AuditAction
    stage(args, "create environment")
    env = AuditEnv(cfg)
    results = []
    try:
        term = mdp.controller(env)
        if not term.managed_phase:
            raise ValueError("Use the bounded v3 low-level checkpoint for this acceptance run")
        if any(abs(v) > float(s) for v,s in zip(args.velocity, term.velocity_scale)):
            raise ValueError("--velocity exceeds teacher action bounds")
        if max(abs(v) for v in args.arm_delta) > cfg.actions.teacher.arm_scale:
            raise ValueError("--arm_delta exceeds teacher action bounds")
        full_target = term.nominal + torch.tensor(args.arm_delta, device=env.device)
        if any(c in args.cases for c in ("arm_hold", "arm_walk", "coupled")) and (
                (full_target < term.limits[..., 0]) | (full_target > term.limits[..., 1])).any():
            raise ValueError("Scripted arm target violates soft limits")
        before = {k:v.clone() for k,v in term.locomotion.state_dict().items()}
        (output / "contract.json").write_text(json.dumps(dict(args=vars(args), task=task,
            checkpoint_sha256=source_hash, nominal_arm=term.nominal[0].tolist(),
            low_wrench=args.low_wrench, teacher_optimizer=False, intent_estimator=False,
            wrench_model=dict(source=model.source, bodies={n:b[0] for n,b in model.bodies.items()},
                prediction_times_s=[0., .2, .4, .6, .8], about="base_Link origin", axes="current base",
                sign="added subsystem ON base", normalization=[.01, .01, .01, .05, .05, .05],
                training_bounds_current_repo=training_bounds,
                bounds_note="Current repository defaults, not recovered checkpoint training parameters; physical predictions are not clipped to these bounds",
                limitations="URDF inertias; base acceleration ignored; future arm follows script with constant measured position offset; no MPC and no arm feedforward"),
            added_payload_mass_kg=args.payload_mass if args.payload_mode == "attached" else 0.,
            tracked_body=cfg.actions.teacher.payload_body_name,
            torque_note="Isaac actuator buffers; implicit PD estimates are not measured joint reaction torques",
            force_contract="external on payload COM, payload-local Fx/Fy/Mz",
            sample_rate_hz=1/env.physics_dt, acceleration="native world-frame COM linear acceleration",
            event_schedule="walk ramp 3-4s; arm out 7-10, hold 10-12, return 12-15; force 10-12s",
            env=cfg.to_dict()), indent=2, default=str), encoding="utf-8")
        with (output / "episodes.jsonl").open("w", encoding="utf-8") as reports:
            for case in args.cases:
                for trial in range(args.episodes):
                    if not app.is_running():
                        break
                    env._acceptance_capture = None
                    interaction = env.command_manager.get_term("interaction")
                    interaction.cfg.test_force = tuple(args.force) if case in ("force_hold", "force_walk", "coupled") else (0., 0., 0.)
                    stage(args, f"{case} trial={trial}: reset")
                    env.reset(seed=args.seed + trial)
                    term.preview_case = case
                    stage(args, f"{case} trial={trial}: initialize {args.low_wrench} preview")
                    term.update_preview(0.)
                    stage(args, f"{case} trial={trial}: preview ready")
                    if trial == 0:
                        print(f"[WRENCH INPUT] mode={args.low_wrench} first={term.physical_preview[0, 0].tolist()}", flush=True)
                    term._asset.set_joint_position_target(term._asset.data.default_joint_pos)
                    rows = []

                    def capture(raw, t):
                        data = term._asset.data
                        body = term.payload_ids[0]
                        gravity = mdp.payload_state(raw)[2]
                        payload_tilt = torch.rad2deg(torch.acos((gravity * term.payload_gravity_reference).sum(-1).clamp(-1, 1)))
                        base_tilt = torch.rad2deg(torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1, 1)))
                        row = dict(time_s=t, base_tilt_deg=float(base_tilt[0]),
                            payload_tilt_from_reset_deg=float(payload_tilt[0]),
                            payload_tilt_world_deg=float(torch.rad2deg(torch.acos((-gravity[0, 2]).clamp(-1, 1)))),
                            payload_position_world=data.body_pos_w[0, body].tolist(),
                            payload_velocity_world=data.body_com_lin_vel_w[0, body].tolist(),
                            payload_acc_world=data.body_com_lin_acc_w[0, body].tolist(),
                            base_velocity=data.root_lin_vel_b[0].tolist(),
                            base_position_world=data.root_pos_w[0].tolist(),
                            base_command=term.processed_actions[0, :3].tolist(),
                            arm_q=data.joint_pos[0, term.arm_ids].tolist(),
                            arm_target=term.processed_actions[0, 3:].tolist(),
                            force_payload_local=interaction.command[0].tolist(),
                            low_wrench_physical=term.physical_preview[0].flatten().tolist(),
                            low_wrench_normalized=term.low_wrench_prediction[0].tolist(),
                            low_wrench_time_s=term.preview_time,
                            low_action_abs_max=float(term.last_leg_action.abs().max()))
                        for name in ("computed_torque", "applied_torque", "joint_effort_limits"):
                            value = getattr(data, name, None)
                            if value is not None:
                                row["arm_" + name] = value[0, term.arm_ids].tolist()
                        bounds = term.physical_preview.new_tensor(training_bounds)
                        row["wrench_outside_repo_training_bounds"] = float(((term.physical_preview < bounds[:, 0]) |
                            (term.physical_preview > bounds[:, 1])).any())
                        if any(not math.isfinite(x) for value in row.values()
                               for x in (value if isinstance(value, list) else [value])):
                            raise RuntimeError("Non-finite support telemetry")
                        rows.append(row)
                        stream.write(json.dumps(row) + "\n")

                    completed = False
                    with (output / f"{case}_trial{trial}.jsonl").open("w", encoding="utf-8") as stream:
                        env._acceptance_capture = capture
                        for step in range(env.max_episode_length):
                            if not app.is_running():
                                break
                            v, delta = targets(step * env.step_dt, case, args.velocity, args.arm_delta)
                            action = torch.tensor([v + delta], device=env.device)
                            action[:, :3] /= term.velocity_scale
                            action[:, 3:] /= cfg.actions.teacher.arm_scale
                            _, _, failed, timeout, _ = env.step(action)
                            if bool(failed[0]) or bool(timeout[0]):
                                completed = bool(timeout[0]) and not bool(failed[0])
                                break
                        env._acceptance_capture = None
                    event_end = 15. if case in ("arm_hold", "arm_walk", "coupled") else (
                        12. if case in ("force_hold", "force_walk") else None)
                    row = dict(case=case, trial=trial, completed=completed, low_wrench=args.low_wrench,
                        **summarize(rows, event_end, args.recovery_window),
                        termination={name:bool(env.termination_manager.get_term(name)[0])
                                     for name in env.termination_manager.active_terms})
                    results.append(row)
                    reports.write(json.dumps(row) + "\n")
                    reports.flush()
                    print("[PAINT SUPPORT] " + json.dumps(row), flush=True)
        frozen = all(torch.equal(before[k], v) for k,v in term.locomotion.state_dict().items())
        (output / "report.json").write_text(json.dumps(dict(results=results, frozen_weights_verified=frozen), indent=2), encoding="utf-8")
        if not frozen:
            raise RuntimeError("Frozen locomotion weights changed")
        print(f"[PAINT SUPPORT] complete; frozen weights verified; output={output}", flush=True)
    except BaseException as error:
        record_failure(args, error)
        raise
    finally:
        env._acceptance_capture = None
        env.close()


def run_with_diagnostics(args, app, run=main):
    status = 0
    try:
        run(args, app)
    except BaseException as error:
        record_failure(args, error)
        status = 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        try:
            app.close()
        except BaseException as error:
            # Preserve a failure even if Kit shutdown requests sys.exit(0).
            if not (isinstance(error, SystemExit) and error.code in (None, 0)):
                record_failure(args, error)
                status = status or 1
    return status


if __name__ == "__main__":
    parser = make_parser()
    if "--help" in sys.argv or "-h" in sys.argv:
        parser.parse_args()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    sys.exit(run_with_diagnostics(args, app))
