"""Export bounded v3 models and run a local SFYG MuJoCo transfer diagnostic.

The original deployment sources and models remain unchanged. The control mode
patches only its process's legacy observation function to match managed v3 phase.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
FOLDED_ARM = {f"arm{i}_Joint": 0. for i in range(1, 7)}


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
    print(f"[V3 SIM] arm_pose={args.arm_pose}; initial arm="
          f"{[sfyg.INITIAL_ARM_AND_GRIPPER[name] for name in FOLDED_ARM]}", flush=True)
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
    from limxsdk.robot import Robot, RobotType

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

    class DiagnosticController(deployment.SFYGController):
        def __init__(self, *a, **kw):
            self.policy_calls = 0
            self.peak_tilt = 0.
            super().__init__(*a, **kw)
            self.config.action_clip = 5.
            if args.arm_pose == "folded":
                if list(self.config.wire_joint_names[10:16]) != list(FOLDED_ARM):
                    raise ValueError("Unexpected SFYG arm joint order")
                self.config.default_q[10:16] = list(FOLDED_ARM.values())
            if (not np.allclose(self.config.action_scale, .25)
                    or not np.allclose(self.config.gait, [.8, .5, .5, .2])
                    or abs(self.config.policy_frequency - 50.) > 1e-6):
                raise ValueError("Expected action scale .25, gait .8/.5/.5/.2 and 50Hz policy")

        def _move_to_default(self, start_q):
            if args.arm_pose == "folded" and not np.allclose(
                    np.asarray(start_q)[10:16], self.config.default_q[10:16], atol=.05):
                raise ValueError("Arm initial pose mismatch: restart sim with --arm-pose folded")
            return super()._move_to_default(start_q)

        def _infer(self, proprio, solution):
            q, _, quaternion, _ = self._snapshot()
            gravity = projected_gravity_from_wxyz(quaternion)
            tilt = float(np.degrees(np.arccos(np.clip(-gravity[2], -1., 1.))))
            self.peak_tilt = max(self.peak_tilt, tilt)
            if not math.isfinite(tilt) or tilt > 35.:
                raise RuntimeError(f"Tilt limit exceeded: {tilt:.2f} deg")
            action = super()._infer(proprio, solution)
            if self.policy_calls % 50 == 0:
                print(f"[V3 RL] calls={self.policy_calls} tilt={tilt:.2f}deg "
                      f"action_max={np.max(np.abs(action)):.3f} "
                      f"arm_q={np.round(q[10:16], 3).tolist()}", flush=True)
            self.policy_calls += 1
            return action

    controller = None
    completed = False
    error = None
    try:
        robot = Robot(RobotType.Tron2)
        if not robot.init("127.0.0.1"):
            raise RuntimeError("SDK init failed")
        controller = DiagnosticController(str(root), robot, "SFYG_TRON2A",
                                          start_controller=True, base_command=args.command)
        print(f"[V3] checkpoint={manifest['checkpoint']} command={args.command}; "
              "articulated SFYG, predicted wrench=0, managed stance phase", flush=True)
        print(f"[V3] arm_pose={args.arm_pose} target={controller.config.default_q[10:16].tolist()} "
              f"kp={controller.config.kp[10:16].tolist()} kd={controller.config.kd[10:16].tolist()}", flush=True)
        completed = controller.run(duration=args.duration)
    except KeyboardInterrupt:
        error = "interrupted"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        print("[V3 RESULT] " + json.dumps(dict(completed=completed, error=error,
            arm_pose=args.arm_pose,
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
    ctl = sub.add_parser("control")
    ctl.add_argument("--deploy-root", required=True)
    ctl.add_argument("--model-root", required=True, help="Parent of an SFYG_TRON2A model folder")
    ctl.add_argument("--command", nargs=3, type=float, default=[0., 0., 0.])
    ctl.add_argument("--duration", type=float, default=60.)
    ctl.add_argument("--arm-pose", choices=("configured", "folded"), default="configured")
    args = parser.parse_args()
    if args.mode == "control":
        if not math.isfinite(args.duration) or args.duration <= 0:
            parser.error("duration must be finite and positive")
        if any(not math.isfinite(v) or abs(v) > limit for v, limit in zip(args.command, (1., .5, 1.5))):
            parser.error("commands must be finite and within training ranges")
    {"export": export, "sim": simulate, "control": control}[args.mode](args)


if __name__ == "__main__":
    main()
