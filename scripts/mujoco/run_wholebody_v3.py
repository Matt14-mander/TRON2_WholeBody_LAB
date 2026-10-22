"""Export bounded v3 models and run a local SFYG MuJoCo transfer diagnostic.

The original deployment sources and models remain unchanged. The control mode
patches only its process's legacy observation function to match managed v3 phase.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
FOLDED_ARM = {f"arm{i}_Joint": 0. for i in range(1, 7)}


def configure_publish_rate(config, publish_hz):
    """Preserve the 50Hz policy while changing SDK target republication only."""
    policy_hz = float(config.policy_frequency)
    ratio = publish_hz / policy_hz
    if (not math.isfinite(publish_hz) or not math.isclose(policy_hz, 50., abs_tol=1e-6)
            or ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-6)):
        raise ValueError("SDK publication must be an integer multiple of the 50Hz policy")
    config.loop_frequency = float(publish_hz)
    config.decimation = int(round(ratio))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simulate(args):
    root = Path(args.sim_root).expanduser().resolve()
    if not (root / "simulator.py").is_file():
        raise FileNotFoundError(root / "simulator.py")
    sys.path.insert(0, str(root))
    from tron2_sim.variants import sfyg
    if args.arm_pose == "folded":
        # Change only initial joint positions; preserve mass, PD and effort limits.
        sfyg.INITIAL_ARM_AND_GRIPPER = {**sfyg.INITIAL_ARM_AND_GRIPPER, **FOLDED_ARM}
    os.environ["ROBOT_TYPE"] = "SFYG_TRON2A"
    simulator = importlib.import_module("simulator")
    sys.argv = [str(root / "simulator.py")]
    if args.headless:
        sys.argv.append("--headless")
    if args.payload_mass is not None:
        sys.argv += ["--payload-mass", str(args.payload_mass),
                     "--payload-dimensions", *map(str, args.payload_dimensions),
                     "--payload-offset", *map(str, args.payload_offset)]
    if args.external_wrench is not None:
        sys.argv += ["--external-wrench", *map(str, args.external_wrench),
                     "--wrench-body", args.wrench_body, "--wrench-frame", args.wrench_frame,
                     "--wrench-profile", args.wrench_profile,
                     "--wrench-start", str(args.wrench_start),
                     "--wrench-duration", str(args.wrench_duration),
                     "--wrench-ramp-time", str(args.wrench_ramp_time),
                     "--wrench-frequency", str(args.wrench_frequency)]
    print(f"[V3 SIM] arm_pose={args.arm_pose}; initial arm="
          f"{[sfyg.INITIAL_ARM_AND_GRIPPER[name] for name in FOLDED_ARM]}", flush=True)
    print(f"[V3 SIM] payload mass={args.payload_mass} dimensions={args.payload_dimensions} offset={args.payload_offset}", flush=True)
    simulator.main()


def export(args):
    import numpy as np
    import torch
    import onnxruntime as ort

    # This module is torch-only; bypass package __init__ (which starts Isaac imports).
    source_module = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers/paint_teacher_core.py"
    spec = importlib.util.spec_from_file_location("v3_export_core", source_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = Path(args.checkpoint_path).expanduser().resolve()
    model = module.FrozenLocomotion(checkpoint, device="cpu")
    bound = getattr(model.actor[-1], "scale", None)
    if bound is None or not torch.isfinite(bound).all() or abs(float(bound) - 5.) > 1e-6:
        raise ValueError("Expected a v3 checkpoint containing the 5*tanh(x/5) actor bound")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                    actor_bound=5., phase="managed_stance", onnx_sha256={})
    for name, network, width, out_width in (("policy", model.actor, 78, 10),
                                             ("encoder", model.encoder, 420, 3)):
        path = output / f"{name}.onnx"
        torch.onnx.export(network, torch.zeros(width), str(path), opset_version=17,
                          input_names=["input"], output_names=["output"], dynamo=False)
        session = ort.InferenceSession(str(path), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        if session.get_inputs()[0].shape != [width] or session.get_outputs()[0].shape != [out_width]:
            raise RuntimeError(f"Unexpected {name} ONNX dimensions")
        with torch.no_grad():
            for _ in range(10):
                sample = torch.randn(width)
                expected = network(sample).numpy()
                actual = session.run(None, {"input": sample.numpy()})[0]
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
                if name == "policy" and np.max(np.abs(actual)) > 5.00001:
                    raise RuntimeError("Exported policy violated its v3 action bound")
        manifest["onnx_sha256"][name] = digest(path)
        print(f"[EXPORT] {name} {width}->{out_width}; PyTorch/ONNX agreement verified: {path}", flush=True)
    # Created only after both ONNX files pass validation.
    (output / "v3_source.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def control(args):
    import numpy as np

    arm_requested = args.arm_test_joint is not None or args.arm_test_offsets is not None
    deploy = Path(args.deploy_root).expanduser().resolve()
    root = Path(args.model_root).expanduser().resolve()
    model_dir = root / "SFYG_TRON2A"
    manifest = json.loads((model_dir / "v3_source.json").read_text(encoding="utf-8"))
    if manifest.get("actor_bound") != 5. or manifest.get("phase") != "managed_stance":
        raise ValueError("Use artifacts produced by this script's v3 export mode")
    for name in ("encoder", "policy"):
        if digest(model_dir / f"{name}.onnx") != manifest["onnx_sha256"][name]:
            raise ValueError(f"{name}.onnx does not match the v3 export manifest")
    sys.path.insert(0, str(deploy))
    deployment = importlib.import_module("controllers.SFYGController")
    from controllers.sfyg_contract import projected_gravity_from_wxyz
    from controllers.sfyg_contract import JointSpaceArmTest
    from controllers.sfyg_contract import WRENCH_SCALE
    from controllers.sfyg_contract import SFYGPolicyConfig
    from limxsdk.robot import Robot, RobotType
    if not np.allclose(WRENCH_SCALE, [.01, .01, .01, .05, .05, .05]):
        raise ValueError("Deployment wrench normalization differs from the v3 training contract")

    original = deployment.build_proprio_observation
    gait_index = 0.

    def v3_observation(config, q, dq, gyro, quaternion, last_action, command, elapsed):
        nonlocal gait_index
        observation = original(config, q, dq, gyro, quaternion, last_action, command, elapsed)
        # V3 GaitCommand holds stance at zero and advances only while moving.
        if np.linalg.norm(command) <= 1e-3:
            gait_index = 0.
        else:
            gait_index = (gait_index + config.gait[0] / config.policy_frequency) % 1.
        angle = 2 * math.pi * gait_index
        observation[36:38] = (math.sin(angle), math.cos(angle))
        return observation

    deployment.build_proprio_observation = v3_observation
    preview = None
    preview_path = None
    arm_gravity = None
    if args.low_wrench != "zero" or args.arm_gravity_compensation:
        sim_root = Path(args.sim_root).expanduser().resolve()
        candidates = [sim_root / "robot-description" / folder / "SFYG_TRON2A/xml/robot.xml"
                      for folder in ("tron2a", "tron2")]
        preview_path = next((p for p in candidates if p.is_file()), None)
        if preview_path is None:
            raise FileNotFoundError(f"SFYG MuJoCo XML not found under {sim_root}/robot-description")
        sys.path.insert(0, str(sim_root))
        sys.path.insert(0, str(ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers"))
        from tron2_sim.model_loader import load_mujoco_model, load_payload_model
        from paint_wrench_preview import MuJoCoAddedBodyModel, LatestPreviewWorker, current_frame_preview
        shadow = (load_payload_model(preview_path, args.payload_mass, args.payload_dimensions, args.payload_offset)
                  if args.payload_mass is not None else load_mujoco_model(preview_path))
        preview = MuJoCoAddedBodyModel(shadow, ROOT / "robot_description/tron2/SF_TRON2A/urdf/robot_rl.urdf")
        config_check = SFYGPolicyConfig(str(model_dir / "params.yaml"))
        mapping = preview.resolve_joint_names(config_check.wire_joint_names)
        print("[V3 PREVIEW JOINTS] " + json.dumps(mapping), flush=True)
        initial = preview.gravity_wrench(dict(zip(config_check.wire_joint_names, config_check.default_q)),
                                        np.array([0., 0., -np.linalg.norm(shadow.opt.gravity)]))
        print(f"[V3 PREVIEW] source={preview_path}; mode={args.low_wrench}; "
              f"added_mass={sum(b[0] for b in preview.bodies.values()):.6f}kg; "
              f"payload_mass={args.payload_mass}; private model; arm gravity feedforward={args.arm_gravity_compensation}", flush=True)
        print(f"[V3 PREVIEW CHECK] joints={len(mapping)} gravity={initial.tolist()}", flush=True)
        if args.arm_gravity_compensation:
            from arm_gravity_compensation import ArmGravityCompensation
            arm_gravity = ArmGravityCompensation(shadow, mapping, config_check.wire_joint_names[10:16],
                args.arm_gravity_max_effort, args.arm_gravity_ramp_time)
            initial_positions = dict(zip(config_check.wire_joint_names, config_check.default_q))
            if args.arm_pose == "folded":
                initial_positions.update(FOLDED_ARM)
            initial_effort = arm_gravity.compute(initial_positions, [0., 0., -np.linalg.norm(shadow.opt.gravity)])
            print(f"[V3 ARM GRAVITY CHECK] raw_Nm={initial_effort.tolist()} "
                  f"limits_Nm={arm_gravity.limits.tolist()}; ramp={arm_gravity.ramp_time}s", flush=True)
        if args.preview_only:
            print("[V3 PREVIEW CHECK] passed; no SDK connection started", flush=True)
            return

    def compute_preview(request):
        positions, g, test = request["positions"], request["gravity"], request["arm_test"]
        if args.low_wrench == "gravity" or test is None:
            return np.repeat(preview.gravity_wrench(positions, g)[None], 5, axis=0)
        t, command, names = request["time"], request["command"], request["arm_names"]
        now = test.sample(t, command).arm_position
        def position_at(future):
            delta = test.sample(max(0., future), command).arm_position-now
            out = dict(positions)
            for i, name in enumerate(names):
                out[name] += delta[i]
            return out
        return preview.preview(position_at, t, g)

    class DiagnosticController(deployment.SFYGController):
        def __init__(self, *a, **kw):
            self.policy_calls = 0
            self.peak_tilt = 0.
            self.rl_started = None
            self.arm_sequence_started = False
            self.arm_sequence_completed = False
            self.arm_sequence_aborted = False
            self.inference_starts = []
            self.preview_cost_ms = []
            self.inference_cost_ms = []
            self.preview_worker = None
            self.worker_cost_ms = []
            self.prediction_age_ms = []
            self.arm_recorder = None
            self._record_policy_publish = False
            self._record_phase = "hold"
            self._gravity_context = None
            super().__init__(*a, **kw)
            if preview is not None:
                print("[V3 PREVIEW JOINTS] " + json.dumps(
                    preview.resolve_joint_names(self.config.wire_joint_names)), flush=True)
            self.config.action_clip = 5.
            if args.arm_pose == "folded":
                if list(self.config.wire_joint_names[10:16]) != list(FOLDED_ARM):
                    raise ValueError("Unexpected SFYG arm joint order")
                self.config.default_q[10:16] = list(FOLDED_ARM.values())
            # Build AFTER the pose override: otherwise a folded test would
            # silently reuse the legacy extended arm origin during its motion.
            if arm_requested:
                self.arm_test = JointSpaceArmTest(
                    self.config.default_q[10:16].copy(),
                    args.arm_test_joint, args.arm_test_delta,
                    offsets=args.arm_test_offsets,
                    start_delay=args.arm_test_start_delay,
                    move_duration=args.arm_test_move_duration,
                    hold_duration=args.arm_test_hold_duration)
            if self.trajectory is not None:
                if not np.allclose(self.trajectory.sample(0.).arm_position,
                                   self.config.default_q[10:16], atol=.05):
                    raise ValueError("OCS2 CSV initial arm pose differs from configured pose; align the trajectory first")
                if args.duration < args.ocs2_start_delay + self.trajectory.duration + 1.:
                    raise ValueError("duration must cover OCS2 startup, full trajectory and at least 1s terminal hold")
            if (not np.allclose(self.config.action_scale, .25)
                    or not np.allclose(self.config.gait, [.8, .5, .5, .2])
                    or abs(self.config.policy_frequency - 50.) > 1e-6):
                raise ValueError("Expected action scale .25, gait .8/.5/.5/.2 and 50Hz policy")
            configure_publish_rate(self.config, args.publish_frequency)
            print(f"[V3 CLOCK] publish={self.config.loop_frequency:g}Hz "
                  f"policy={self.config.policy_frequency:g}Hz decimation={self.config.decimation}; "
                  "MuJoCo MIT PD still runs at its physics rate", flush=True)
            if args.record_dir:
                from arm_telemetry import ArmTelemetry
                self.arm_recorder = ArmTelemetry(args.record_dir, dict(
                    joints=list(self.config.wire_joint_names[10:16]),
                    arguments=vars(args), checkpoint=manifest["checkpoint"],
                    publish_hz=self.config.loop_frequency,
                    policy_hz=self.config.policy_frequency,
                    arm_gravity_compensation=args.arm_gravity_compensation,
                    gravity_effort_limits=None if arm_gravity is None else arm_gravity.limits.tolist(),
                    position_error="measured minus commanded, rad; velocity rad/s; effort Nm",
                    feedback_source="SDK RobotState.tau; SFYG simulator publishes actuator_force (unit gear)",
                    estimate="Kp*(q_target-q)+Kd*(dq_target-dq)+feedforward; latest received state, not physics-synchronous",
                    sampling="One row per policy target publication; feedback predates outgoing command. Not physics-rate peaks or joint reaction torque.",
                    torque_limits="Not inferred from SDK feedback; saturation is not certified by this log."))
                print(f"[V3 ARM RECORD] {self.arm_recorder.directory}", flush=True)

        def _publish_policy_targets(self, targets, solution):
            self._record_phase = ("aborted" if arm_requested and self.arm_test is None else
                self.arm_test.phase(solution.time) if self.arm_test is not None else
                "trajectory" if self.trajectory is not None else "hold")
            if arm_gravity is not None:
                q, _, quaternion, _ = self._snapshot()
                positions = dict(zip(self.config.wire_joint_names, q))
                gravity = projected_gravity_from_wxyz(quaternion) * np.linalg.norm(arm_gravity.model.opt.gravity)
                raw = arm_gravity.compute(positions, gravity)
                elapsed = max(0., time.monotonic()-self.rl_started) if self.rl_started is not None else 0.
                applied, weight = arm_gravity.command(raw, elapsed)
                self._gravity_context = dict(gravity_raw_effort=raw.tolist(),
                    gravity_applied_effort=applied.tolist(), gravity_weight=weight,
                    gravity_clipped=(np.abs(raw) > arm_gravity.limits).tolist())
                # Replace the solution passed to publication, not just inference.
                solution = replace(solution, arm_effort=np.asarray(solution.arm_effort)+applied)
            self._record_policy_publish = True
            try:
                return super()._publish_policy_targets(targets, solution)
            finally:
                self._record_policy_publish = False

        def _publish(self, q, dq=None, tau=None, kp=None, kd=None):
            record = None
            if self._record_policy_publish and self.arm_recorder is not None and self.rl_started is not None:
                now = time.monotonic()
                with self._state_lock:
                    state, stamp, count = self.state, self._state_stamp, self._state_count
                # Callback already orders q/dq, but leaves tau in SDK motor_names order.
                feedback = getattr(state, "tau", None)
                try:
                    feedback = (self._ordered_state(feedback, getattr(state, "motor_names", None))
                                if feedback is not None else None)
                except (ValueError, IndexError, TypeError):
                    feedback = None
                record = dict(time_s=now-self.rl_started, phase=self._record_phase,
                    q=np.asarray(state.q)[10:16], dq=np.asarray(state.dq)[10:16],
                    target=np.asarray(q)[10:16],
                    velocity_target=np.zeros(6) if dq is None else np.asarray(dq)[10:16],
                    effort=np.zeros(6) if tau is None else np.asarray(tau)[10:16],
                    kp=self.config.kp[10:16] if kp is None else np.asarray(kp)[10:16],
                    kd=self.config.kd[10:16] if kd is None else np.asarray(kd)[10:16],
                    feedback=None if feedback is None else np.asarray(feedback)[10:16],
                    state_age_s=now-stamp, state_count=count,
                    diagnostics=self._gravity_context)
            result = super()._publish(q, dq=dq, tau=tau, kp=kp, kd=kd)
            if record is not None:
                self.arm_recorder.append(**record)
            return result

        def _move_to_default(self, start_q):
            if args.arm_pose == "folded" and not np.allclose(
                    np.asarray(start_q)[10:16], self.config.default_q[10:16], atol=.05):
                raise ValueError("Arm initial pose mismatch: restart sim with --arm-pose folded")
            return super()._move_to_default(start_q)

        def _infer(self, proprio, solution):
            infer_start = time.monotonic()
            self.inference_starts.append(infer_start)
            if self.rl_started is None:
                self.rl_started = time.monotonic()
            q, _, quaternion, _ = self._snapshot()
            if arm_requested:
                if self.arm_test is None:
                    self.arm_sequence_aborted = True
                else:
                    stage = self.arm_test.phase(solution.time)
                    self.arm_sequence_started |= stage in ("outbound", "hold", "return")
                    self.arm_sequence_completed |= stage == "complete" and self.arm_sequence_started
            gravity = projected_gravity_from_wxyz(quaternion)
            preview_start = time.monotonic()
            if preview is not None and args.low_wrench != "zero":
                positions = dict(zip(self.config.wire_joint_names, q))
                g = gravity * np.linalg.norm(preview.model.opt.gravity)
                request = dict(positions=positions, gravity=g, arm_test=self.arm_test,
                    time=solution.time, command=solution.base_command.copy(),
                    arm_names=tuple(self.config.wire_joint_names[10:16]),
                    quaternion=np.array(quaternion, copy=True), stamp=time.monotonic())
                if args.preview_execution == "async":
                    if self.preview_worker is None:
                        seed = np.repeat(preview.gravity_wrench(positions, g)[None], 5, axis=0)
                        self.preview_worker = LatestPreviewWorker(compute_preview, seed, request)
                    self.preview_worker.submit(request)
                    (raw, source, cost_ms), age = self.preview_worker.read(time.monotonic(), args.preview_max_age)
                    if source["arm_test"] is not request["arm_test"]:
                        raise RuntimeError("Arm trajectory changed; stale plan preview rejected")
                    prediction = current_frame_preview(raw, source["quaternion"], quaternion, age)
                    self.worker_cost_ms.append(cost_ms)
                    self.prediction_age_ms.append(age*1000)
                else:
                    prediction = compute_preview(request)
                solution = replace(solution, wrench_prediction=prediction)
            self.preview_cost_ms.append((time.monotonic()-preview_start)*1000.)
            tilt = float(np.degrees(np.arccos(np.clip(-gravity[2], -1., 1.))))
            self.peak_tilt = max(self.peak_tilt, tilt)
            if not math.isfinite(tilt) or tilt > 35.:
                raise RuntimeError(f"Tilt limit exceeded: {tilt:.2f} deg")
            action = super()._infer(proprio, solution)
            self.inference_cost_ms.append((time.monotonic()-infer_start)*1000.)
            if self.policy_calls % 50 == 0:
                print(f"[V3 RL] t={time.monotonic() - self.rl_started:.2f}s "
                      f"calls={self.policy_calls} tilt={tilt:.2f}deg "
                      f"action_max={np.max(np.abs(action)):.3f} "
                      f"arm_q={np.round(q[10:16], 3).tolist()} "
                      f"arm_target={np.round(solution.arm_position, 3).tolist()} "
                      f"wrench0={np.round(solution.wrench_prediction[0], 3).tolist()} "
                      f"preview_ms={self.preview_cost_ms[-1]:.2f} infer_ms={self.inference_cost_ms[-1]:.2f}", flush=True)
            self.policy_calls += 1
            return action

        def timing_report(self):
            intervals = np.diff(self.inference_starts)
            return dict(target_hz=self.config.policy_frequency,
                configured_publish_hz=self.config.loop_frequency,
                observed_hz=float(1/intervals.mean()) if len(intervals) else None,
                interval_p95_ms=float(np.percentile(intervals, 95)*1000) if len(intervals) else None,
                preview_p95_ms=float(np.percentile(self.preview_cost_ms, 95)) if self.preview_cost_ms else None,
                worker_compute_p95_ms=float(np.percentile(self.worker_cost_ms, 95)) if self.worker_cost_ms else None,
                prediction_age_p95_ms=float(np.percentile(self.prediction_age_ms, 95)) if self.prediction_age_ms else None,
                infer_p95_ms=float(np.percentile(self.inference_cost_ms, 95)) if self.inference_cost_ms else None)

    controller = None
    completed = False
    error = None
    try:
        robot = Robot(RobotType.Tron2)
        if not robot.init("127.0.0.1"):
            raise RuntimeError("SDK init failed")
        controller = DiagnosticController(str(root), robot, "SFYG_TRON2A",
            start_controller=True, base_command=args.command,
            ocs2_trajectory=args.ocs2_trajectory,
            ocs2_start_delay=args.ocs2_start_delay if args.ocs2_trajectory else 0.,
            safe_walk_command=args.safe_walk_command,
            ocs2_terminal_command=args.safe_walk_command if args.ocs2_trajectory else None)
        print(f"[V3] checkpoint={manifest['checkpoint']} command={args.command}; "
              f"articulated SFYG, predicted wrench={'OCS2 CSV' if args.ocs2_trajectory else args.low_wrench}, "
              "managed stance phase", flush=True)
        print(f"[V3] arm_pose={args.arm_pose} target={controller.config.default_q[10:16].tolist()} "
              f"kp={controller.config.kp[10:16].tolist()} kd={controller.config.kd[10:16].tolist()}", flush=True)
        completed = controller.run(duration=args.duration)
    except KeyboardInterrupt:
        error = "interrupted"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if controller is not None and controller.preview_worker is not None:
            controller.preview_worker.close()
        recorder = getattr(controller, "arm_recorder", None)
        if recorder is not None:
            try:
                recorder.save(dict(completed=completed, error=error,
                    arm_sequence_completed=controller.arm_sequence_completed,
                    arm_sequence_aborted=controller.arm_sequence_aborted,
                    timing=controller.timing_report()))
                print(f"[V3 ARM RECORD] saved {len(recorder.rows)} samples to {recorder.directory}", flush=True)
            except Exception as exc:
                error = f"Arm record save failed: {type(exc).__name__}: {exc}; control error={error}"
                completed = False
        print("[V3 RESULT] " + json.dumps(dict(completed=completed, error=error,
            requested_test_completed=bool(completed and (not arm_requested or (
                getattr(controller, "arm_sequence_completed", False) and not getattr(controller, "arm_sequence_aborted", False)))),
            timing=controller.timing_report() if controller is not None else None,
            arm_pose=args.arm_pose,
            low_wrench=args.low_wrench, payload_mass=args.payload_mass,
            preview_execution=args.preview_execution,
            arm_gravity_compensation=args.arm_gravity_compensation,
            record_dir=str(recorder.directory) if recorder is not None else None,
            ocs2_trajectory=args.ocs2_trajectory,
            arm_test_requested=arm_requested,
            arm_sequence_started=getattr(controller, "arm_sequence_started", False),
            arm_sequence_completed=getattr(controller, "arm_sequence_completed", False),
            arm_sequence_aborted=getattr(controller, "arm_sequence_aborted", False),
            policy_calls=getattr(controller, "policy_calls", 0),
            peak_tilt_deg=getattr(controller, "peak_tilt", None))), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        # Loopback MuJoCo only: avoid the SDK's interpreter teardown race.
        os._exit(0 if completed else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--checkpoint_path", required=True)
    exp.add_argument("--output_dir", required=True)
    sim = sub.add_parser("sim")
    sim.add_argument("--sim-root", required=True)
    sim.add_argument("--arm-pose", choices=("configured", "folded"), default="configured")
    sim.add_argument("--headless", action="store_true")
    sim.add_argument("--external-wrench", nargs=6, type=float)
    sim.add_argument("--wrench-body", default="gripper_base_Link")
    sim.add_argument("--wrench-frame", choices=("world", "body"), default="world")
    sim.add_argument("--wrench-profile", choices=("step", "ramp", "sine"), default="ramp")
    sim.add_argument("--wrench-start", type=float, default=7.)
    sim.add_argument("--wrench-duration", type=float, default=2.)
    sim.add_argument("--wrench-ramp-time", type=float, default=.5)
    sim.add_argument("--wrench-frequency", type=float, default=1.)
    ctl = sub.add_parser("control")
    ctl.add_argument("--deploy-root", required=True)
    ctl.add_argument("--model-root", required=True, help="Parent of an SFYG_TRON2A model folder")
    ctl.add_argument("--command", nargs=3, type=float, default=[0., 0., 0.])
    ctl.add_argument("--duration", type=float, default=60.)
    ctl.add_argument("--arm-pose", choices=("configured", "folded"), default="configured")
    ctl.add_argument("--low-wrench", choices=("zero", "gravity", "plan"), default="zero")
    ctl.add_argument("--sim-root", help="Same simulator root as terminal A; required for gravity/plan")
    ctl.add_argument("--preview-only", action="store_true", help="Validate model, 18 joints and gravity without connecting to SDK")
    ctl.add_argument("--record-dir", help="New directory for buffered arm errors, torque telemetry and phase summaries")
    ctl.add_argument("--arm-gravity-compensation", action="store_true", help="Add model-based gravity effort to actual arm SDK commands (simulation only)")
    ctl.add_argument("--arm-gravity-ramp-time", type=float, default=2.)
    ctl.add_argument("--arm-gravity-max-effort", type=float, default=30., help="Symmetric per-joint gravity feedforward cap, additionally bounded by native motor limits (Nm)")
    ctl.add_argument("--preview-execution", choices=("async", "sync"), default="async")
    ctl.add_argument("--preview-max-age", type=float, default=.10, help="Stop if asynchronous prediction is older than this many seconds")
    ctl.add_argument("--publish-frequency", type=int, choices=(100, 500), default=500,
                     help="SDK target publication rate; policy stays 50Hz. Use 100 for a larger per-tick compute budget")
    motion = ctl.add_mutually_exclusive_group()
    motion.add_argument("--arm-test-joint", choices=tuple(f"arm{i}" for i in range(1, 7)))
    motion.add_argument("--arm-test-offsets", nargs=6, type=float)
    motion.add_argument("--ocs2-trajectory")
    ctl.add_argument("--ocs2-start-delay", type=float, default=3.)
    ctl.add_argument("--safe-walk-command", nargs=3, type=float)
    ctl.add_argument("--arm-test-delta", type=float)
    ctl.add_argument("--arm-test-start-delay", type=float, default=5.)
    ctl.add_argument("--arm-test-move-duration", type=float, default=3.)
    ctl.add_argument("--arm-test-hold-duration", type=float, default=2.)
    for p in (sim, ctl):
        p.add_argument("--payload-mass", type=float, help="Same physical payload parameters in both terminals; omit for no payload")
        p.add_argument("--payload-dimensions", nargs=3, type=float, default=[.20, .12, .08])
        p.add_argument("--payload-offset", nargs=3, type=float, default=[.15, 0., 0.])
    args = parser.parse_args()
    if args.mode in ("sim", "control"):
        values = [*args.payload_dimensions, *args.payload_offset]
        if (not all(math.isfinite(v) for v in values) or min(args.payload_dimensions) <= 0
                or (args.payload_mass is not None and (not math.isfinite(args.payload_mass) or args.payload_mass <= 0))):
            parser.error("payload mass/dimensions must be positive; offset finite")
    if args.mode == "sim" and args.external_wrench is not None:
        if (any(not math.isfinite(v) for v in args.external_wrench)
                or not math.isfinite(args.wrench_start) or args.wrench_start < 0
                or any(not math.isfinite(v) or v <= 0 for v in (
                    args.wrench_duration, args.wrench_ramp_time, args.wrench_frequency))):
            parser.error("wrench must be finite; start nonnegative; duration/ramp/frequency positive")
    if args.mode == "control":
        if not math.isfinite(args.preview_max_age) or args.preview_max_age <= 0:
            parser.error("--preview-max-age must be finite and positive")
        if args.arm_gravity_compensation:
            if any(not math.isfinite(x) or x <= 0 for x in
                   (args.arm_gravity_ramp_time, args.arm_gravity_max_effort)):
                parser.error("Gravity compensation ramp time and effort cap must be positive and finite")
            if args.ocs2_trajectory:
                parser.error("Do not add arm gravity compensation to OCS2 effort; it may already include gravity")
        if args.preview_only and args.low_wrench == "zero" and not args.arm_gravity_compensation:
            parser.error("--preview-only requires --low-wrench gravity or plan")
        if args.low_wrench != "zero" or args.arm_gravity_compensation:
            if args.sim_root is None or not Path(args.sim_root).expanduser().is_dir():
                parser.error("gravity/plan requires an existing --sim-root")
            if args.ocs2_trajectory:
                parser.error("Use either the OCS2 CSV wrench or --low-wrench gravity/plan")
        if args.ocs2_trajectory is not None:
            if not Path(args.ocs2_trajectory).expanduser().is_file():
                parser.error("OCS2 CSV does not exist; provide its real absolute path")
            if args.arm_pose != "configured":
                parser.error("Legacy absolute OCS2 CSV replay requires --arm-pose configured")
            if any(args.command):
                parser.error("Do not combine --command with OCS2 CSV; use --safe-walk-command")
            if (args.safe_walk_command is None
                    or any(not math.isfinite(v) or abs(v) > limit for v, limit in
                           zip(args.safe_walk_command, (1., .5, 1.5)))
                    or math.hypot(*args.safe_walk_command[:2]) < .02):
                parser.error("OCS2 replay requires a verified nonzero --safe-walk-command")
            if not math.isfinite(args.ocs2_start_delay) or args.ocs2_start_delay < 0:
                parser.error("OCS2 start delay must be finite and nonnegative")
        elif args.safe_walk_command is not None:
            parser.error("--safe-walk-command requires --ocs2-trajectory")
        if (args.arm_test_joint is None) != (args.arm_test_delta is None):
            parser.error("provide both --arm-test-joint and --arm-test-delta")
        if args.arm_test_offsets is not None:
            offsets = args.arm_test_offsets
            if (any(not math.isfinite(v) for v in offsets)
                    or sum(abs(v) > 1e-9 for v in offsets) != 2
                    or max(abs(v) for v in offsets) > .10
                    or math.sqrt(sum(v * v for v in offsets)) > .12):
                parser.error("offsets require exactly two active joints, |each|<=.10 and norm<=.12")
        if args.arm_test_joint is not None or args.arm_test_offsets is not None:
            values = (args.arm_test_start_delay,
                      args.arm_test_move_duration, args.arm_test_hold_duration)
            if (any(not math.isfinite(v) for v in values)
                    or (args.arm_test_delta is not None and
                        (not math.isfinite(args.arm_test_delta) or not 0 < abs(args.arm_test_delta) <= .15))
                    or args.arm_test_start_delay < 2. or args.arm_test_move_duration < 1.5
                    or args.arm_test_hold_duration < .5):
                parser.error("arm test requires |delta|<=.15, delay>=2, move>=1.5, hold>=.5 (finite)")
            if args.duration < args.arm_test_start_delay + 2 * args.arm_test_move_duration + args.arm_test_hold_duration + 1:
                parser.error("duration must cover the full arm sequence and at least 1s after return")
        if not math.isfinite(args.duration) or args.duration <= 0:
            parser.error("duration must be finite and positive")
        if any(not math.isfinite(v) or abs(v) > limit for v, limit in zip(args.command, (1., .5, 1.5))):
            parser.error("commands must be finite and within training ranges")
    {"export": export, "sim": simulate, "control": control}[args.mode](args)


if __name__ == "__main__":
    main()
