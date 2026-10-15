"""Local MuJoCo diagnostic for the ORIGINAL 48D SFYG locked-arm policy.

Does not modify deployment controllers, model directories or simulator sources.
The sim subcommand changes initial pose/arm effort limits in memory only.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

JOINTS = [f"{name}_{side}_Joint" for side in ("L", "R") for name in
          ("proximal_pitch", "proximal_roll", "proximal_yaw", "knee", "ankle_pitch")]
JOINTS += [f"arm{i}_Joint" for i in range(1, 7)] + ["gripper1_Joint", "gripper2_Joint"]
POSE = [0., 0., -3.14159, 0., 0., 0., 0., 3.14159, 0., 0.] + [0.] * 6 + [.05, -.05]


def schedule(t, target):
    if t < 20:
        return "initial_stand", [0.] * 3
    if t < 21:
        return "start", [v * (t - 20) for v in target]
    if t < 36:
        return "walk", list(target)
    if t < 37:
        return "stop", [v * (37 - t) for v in target]
    if t < 57:
        return "stopped_stand", [0.] * 3
    if t < 58:
        return "restart", [v * (t - 57) for v in target]
    return "walk_again", list(target)


def export(args):
    import hashlib
    import numpy as np
    import onnxruntime as ort
    import torch
    from torch import nn

    source = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    output = Path(args.model_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Use a new model directory to preserve existing artifacts")

    def mlp(sizes):
        layers = []
        for index, (a, b) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(a, b))
            if index < len(sizes) - 2:
                layers.append(nn.ELU())
        return nn.Sequential(*layers).eval()

    actor = mlp([48, 512, 256, 128, 10])
    encoder = mlp([420, 256, 128, 3])
    actor.load_state_dict({k.removeprefix("actor."): v
                           for k, v in checkpoint["model_state_dict"].items()
                           if k.startswith("actor.")}, strict=True)
    encoder.load_state_dict({k.removeprefix("encoder."): v
                             for k, v in checkpoint["encoder_state_dict"].items()
                             if k.startswith("encoder.")}, strict=True)
    output.mkdir(parents=True)
    torch.manual_seed(42)
    for name, network, width in (("encoder", encoder, 420), ("policy", actor, 48)):
        path = output / f"{name}.onnx"
        torch.onnx.export(network, torch.zeros(width), str(path), opset_version=17,
                          input_names=["input"], output_names=["output"], dynamo=False)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            for _ in range(5):
                sample = torch.randn(width)
                actual = session.run(None, {"input": sample.numpy()})[0]
                np.testing.assert_allclose(actual, network(sample).numpy(), atol=1e-5, rtol=1e-4)
        print(f"[EXPORT] {name}: verified PyTorch/ONNX agreement; {path}", flush=True)
    manifest = dict(checkpoint=str(source), source_iteration=checkpoint.get("iter"),
                    checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    encoder="420->3", policy="48->10")
    (output / "source.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def simulate(args):
    root = Path(args.sim_root).expanduser().resolve()
    if not (root / "simulator.py").is_file():
        raise FileNotFoundError(root / "simulator.py")
    sys.path.insert(0, str(root))
    from tron2_sim.variants import sfyg
    sfyg.INITIAL_ARM_AND_GRIPPER.update(dict(zip(JOINTS, POSE)))
    sfyg.TRAINING_EFFORT_LIMIT.update({name: 200. for name in JOINTS[10:]})
    os.environ["ROBOT_TYPE"] = "SFYG_TRON2A"
    import simulator
    sys.argv = [str(root / "simulator.py"), "--sfyg-support-port", str(args.support_port)]
    if args.headless:
        sys.argv.append("--headless")
    print("[BASELINE SIM] arm q=0, effort limit=200; standard SFYG physics timestep", flush=True)
    simulator.main()


def control(args):
    model = Path(args.model_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    for name in ("encoder.onnx", "policy.onnx"):
        if not (model / name).is_file():
            raise FileNotFoundError(f"Missing {model / name}; complete the export subcommand first")
    if output.exists():
        raise FileExistsError(f"Use a new output directory; already exists: {output}")

    import hashlib
    import threading
    from types import SimpleNamespace
    import numpy as np
    import onnxruntime as ort

    deploy = Path(args.deploy_root).expanduser().resolve()
    sys.path.insert(0, str(deploy))
    from controllers.sfyg_contract import build_proprio_observation, projected_gravity_from_wxyz
    from controllers.sfyg_support_telemetry import SimSupportReceiver
    from limxsdk.robot import Robot, RobotType
    from limxsdk import datatypes

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    sessions = []
    hashes = {}
    for name, size, out in (("encoder", 420, 3), ("policy", 48, 10)):
        path = model / f"{name}.onnx"
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        session = ort.InferenceSession(str(path), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        if (len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1
                or session.get_inputs()[0].shape[-1] != size
                or session.get_outputs()[0].shape[-1] != out):
            raise ValueError(f"{name}: expected {size}->{out}")
        sessions.append(session)
    encoder, policy = sessions
    default = np.asarray(POSE)
    kp = np.array([159.67, 159.67, 53.22, 159.67, 53.22] * 2 + [800.] * 8)
    kd = np.array([10.16, 10.16, 3.39, 10.16, 3.39] * 2 + [40.] * 8)
    config = SimpleNamespace(default_q=default, gait=np.array([.8, .5, .5, .20]),
                             ang_vel_scale=.25, dof_pos_scale=1., dof_vel_scale=.05,
                             observation_clip=100.)
    contract = dict(args=vars(args), sha256=hashes, joint_names=JOINTS, default_q=POSE,
                    kp=kp.tolist(), kd=kd.tolist(), phase="continuous episode clock",
                    physics_note="Isaac implicit PD and SDK explicit PD can differ")
    # Invalid/missing models must not leave an empty output directory behind.
    output.mkdir(parents=True, exist_ok=False)
    (output / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")

    lock = threading.Lock()
    latest = {}

    def state_callback(state):
        try:
            names = list(getattr(state, "motor_names", []) or [])
            q, dq = np.asarray(state.q), np.asarray(state.dq)
            if names:
                indices = [names.index(name) for name in JOINTS]
                q, dq = q[indices], dq[indices]
            if q.shape != (18,) or dq.shape != (18,) or not np.isfinite(np.r_[q, dq]).all():
                raise ValueError("Expected finite named 18D joint state")
            with lock:
                latest["state"] = (q.copy(), dq.copy(), time.monotonic())
        except (ValueError, IndexError) as exc:
            with lock:
                latest["state_error"] = str(exc)

    def imu_callback(imu):
        with lock:
            latest["imu"] = (np.asarray(imu.quat[:4]).copy(),
                             np.asarray(imu.gyro[:3]).copy(), time.monotonic())

    def snapshot():
        with lock:
            state, imu = latest.get("state"), latest.get("imu")
        if state is None or imu is None:
            raise RuntimeError(f"Waiting for state/IMU: {latest.get('state_error')}")
        if time.monotonic() - min(state[2], imu[2]) > .5:
            raise RuntimeError("Stale state/IMU")
        return state[0], state[1], imu[0], imu[1]

    robot = Robot(RobotType.Tron2)
    support = SimSupportReceiver(args.support_port)
    initialized = False
    completed = False
    error = None
    phase = "initialization"
    calls = 0
    peak = 0.
    standing = {name: [] for name in ("initial_stand", "stopped_stand")}
    try:
        if not robot.init("127.0.0.1"):
            raise RuntimeError("SDK init failed")
        initialized = True
        robot.subscribeRobotState(state_callback)
        robot.subscribeImuData(imu_callback)
        deadline = time.monotonic() + 10
        while True:
            try:
                q, _, _, _ = snapshot()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.01)
        if np.max(np.abs(q - default)) > .1:
            raise RuntimeError("Initial pose mismatch: restart with this script's sim subcommand")

        def publish(target, damping_only=False):
            command = datatypes.RobotCmd()
            command.motor_names = JOINTS
            command.mode = [0.] * 18
            command.q = target.tolist()
            command.dq = command.tau = [0.] * 18
            command.Kp = ([0.] * 18 if damping_only else kp.tolist())
            command.Kd = kd.tolist()
            robot.publishRobotCmd(command)

        def infer(session, value):
            info = session.get_inputs()[0]
            value = np.asarray(value, dtype=np.float32)
            if len(info.shape) == 2:
                value = value[None]
            return np.asarray(session.run(None, {info.name: value})[0]).reshape(-1)

        last_action = np.zeros(10)
        history = None
        targets = default.copy()
        # No fixed-PD startup: the first published command is from ONNX.
        started = next_tick = time.monotonic()
        next_inference = started
        previous_phase = None
        print("[BASELINE] encoder 420->3, policy 48->10; arm q=0 PD=800/40", flush=True)
        with (output / "steps.jsonl").open("w", encoding="utf-8") as log:
            while time.monotonic() - started < 68:
                now = time.monotonic()
                q, dq, quat, gyro = snapshot()
                if now >= next_inference:
                    elapsed = now - started
                    phase, command = schedule(elapsed, args.command)
                    tilt = float(np.degrees(np.arccos(np.clip(
                        -projected_gravity_from_wxyz(quat)[2], -1., 1.))))
                    peak = max(peak, tilt)
                    if not np.isfinite(tilt) or tilt > 35:
                        raise RuntimeError(f"Tilt limit: {tilt:.2f} deg in {phase}")
                    proprio = build_proprio_observation(
                        config, q, dq, gyro, quat, last_action, command, elapsed)
                    history = (np.tile(proprio, (10, 1)) if history is None else
                               np.concatenate((history[1:], proprio[None]), axis=0))
                    latent = infer(encoder, history.reshape(-1))
                    action = infer(policy, np.r_[latent, proprio, command])
                    if action.shape != (10,) or not np.isfinite(action).all():
                        raise RuntimeError("Invalid ONNX action")
                    last_action = np.clip(action, -100., 100.)
                    targets = default.copy()
                    targets[:10] += .25 * last_action
                    sample = support.poll()
                    fresh = sample is not None and now - sample.stamp < .5
                    speed = sample.base_speed_xy if fresh else None
                    if phase in standing and speed is not None:
                        standing[phase].append(speed)
                    row = dict(time_s=elapsed, stage=phase, command=command, tilt_deg=tilt,
                               arm_q=q[10:16].tolist(), arm_dq=dq[10:16].tolist(),
                               gyro=gyro.tolist(), base_speed_xy=speed,
                               left_normal_force=sample.left_normal_force if fresh else None,
                               right_normal_force=sample.right_normal_force if fresh else None)
                    log.write(json.dumps(row) + "\n")
                    if calls % 50 == 0 or phase != previous_phase:
                        print("[RL] " + json.dumps(row), flush=True)
                    previous_phase = phase
                    calls += 1
                    next_inference += .02
                    if next_inference < now:
                        next_inference = now + .02
                publish(targets)
                next_tick += .002
                if next_tick < time.monotonic():
                    next_tick = time.monotonic()
                time.sleep(max(0., next_tick - time.monotonic()))
        completed = True
    except KeyboardInterrupt:
        error = "interrupted"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if initialized:
            try:
                q, _, _, _ = snapshot()
                publish(q, damping_only=True)
            except Exception:
                pass
        support.close()
        result = dict(completed=completed, error=error, last_stage=phase,
                      peak_tilt_deg=peak, policy_calls=calls,
                      standing_speed_rms={k: math.sqrt(sum(v * v for v in values) / len(values))
                                          if values else None for k, values in standing.items()})
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("[RESULT] " + json.dumps(result), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if completed else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    exp = modes.add_parser("export")
    exp.add_argument("--checkpoint", required=True)
    exp.add_argument("--model-dir", required=True)
    sim = modes.add_parser("sim")
    sim.add_argument("--sim-root", required=True)
    sim.add_argument("--headless", action="store_true")
    sim.add_argument("--support-port", type=int, default=55401)
    ctl = modes.add_parser("control")
    ctl.add_argument("--deploy-root", required=True)
    ctl.add_argument("--model-dir", required=True)
    ctl.add_argument("--output-dir", required=True)
    ctl.add_argument("--command", nargs=3, type=float, default=[-.25, 0., 0.])
    ctl.add_argument("--support-port", type=int, default=55401)
    args = parser.parse_args()
    if args.mode != "export" and not 1024 <= args.support_port <= 65535:
        parser.error("support port must be between 1024 and 65535")
    if args.mode == "control" and any(not math.isfinite(v) or abs(v) > limit
                                      for v, limit in zip(args.command, (1., .5, 1.5))):
        parser.error("command must be finite and within SFYG training ranges")
    {"export": export, "sim": simulate, "control": control}[args.mode](args)


if __name__ == "__main__":
    main()
