import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST_TEMP = ROOT / ".test-temp"
TEST_TEMP.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers"))
from paint_intent import ARM_JOINT_NAMES, CONDITIONS, SCHEMA, IntentHistory, InteractionProtocol, make_history_samples, collection_conditions
from paint_run_status import PaintRunStatus


class PaintIntentTests(unittest.TestCase):
    def test_support_baselines_do_not_apply_force_without_payload(self):
        self.assertEqual(collection_conditions("none", ["zero"]), ("zero",))
        self.assertEqual(collection_conditions("attached"), CONDITIONS)
        with self.assertRaises(ValueError):
            collection_conditions("none")
        with self.assertRaises(ValueError):
            collection_conditions("none", ["fx_pos"])
        with self.assertRaises(ValueError):
            collection_conditions("attached", ["zero", "zero"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
    def test_gait_buffer_can_be_reset_after_collection_step(self):
        import ast
        import torch

        class GaitBuffer:
            def __init__(self):
                self.indices = torch.zeros(1)

            def step(self):
                self.indices[torch.tensor([], dtype=torch.long)] = 0
                self.indices = torch.remainder(self.indices + .016, 1.)

            def reset(self):
                self.indices[:] = 0

        # Reproduce the actual tensor lifetime across a rollout and forced reset.
        broken = GaitBuffer()
        with torch.inference_mode():
            broken.step()
        with self.assertRaises(RuntimeError):
            broken.reset()
        working = GaitBuffer()
        with torch.no_grad():
            working.step()
        working.reset()
        self.assertFalse(torch.is_inference(working.indices))
        self.assertEqual(float(working.indices[0]), 0.)

        # Ensure the real collection entrypoint does not regress by wrapping
        # mutable simulator steps in InferenceMode again. Pure export may use it.
        tree = ast.parse((ROOT / "scripts/rsl_rl/collect_paint_pd.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.With) and any(isinstance(item.context_expr, ast.Call) and
               isinstance(item.context_expr.func, ast.Attribute) and item.context_expr.func.attr == "inference_mode" for item in node.items):
                self.assertFalse(any(isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and
                                     isinstance(child.func.value, ast.Name) and child.func.value.id == "env" and child.func.attr == "step"
                                     for child in ast.walk(node)))

    def test_simulator_failure_status_persists_before_cleanup(self):
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory, redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
            status = PaintRunStatus(directory)
            status.mark("physics_step", episode_id="group_0000_zero", step=0)
            incomplete = json.loads((Path(directory) / "run_status.json").read_text())
            self.assertFalse(incomplete["complete"])
            try:
                raise RuntimeError("test simulator step failure")
            except RuntimeError as error:
                status.fail(error)
            saved = json.loads((Path(directory) / "run_status.json").read_text())
            self.assertEqual(saved["state"], "failed")
            self.assertEqual(saved["stage"], "physics_step")
            self.assertEqual(saved["error_type"], "RuntimeError")
            self.assertIn("test simulator step failure", errors.getvalue())

    def test_protocol_zero_and_sign_symmetry(self):
        protocol = InteractionProtocol()
        for time in (0., 2., 2.1, 3., 3.9, 4., 5.):
            np.testing.assert_array_equal(protocol.wrench("zero", time), np.zeros(6))
            for axis in ("fx", "fy", "mz"):
                np.testing.assert_allclose(protocol.wrench(axis + "_pos", time), -protocol.wrench(axis + "_neg", time))
        self.assertEqual(protocol.wrench("fx_pos", 3.)[0], 5.)
        self.assertEqual(protocol.wrench("mz_neg", 3.)[5], -.5)
        np.testing.assert_array_equal(protocol.wrench("fx_pos", 4.), np.zeros(6))

    def test_response_alignment_and_online_history(self):
        q = np.arange(42).reshape(7, 6).astype(float)
        dq, command = q * .1, q + 100
        wrench = np.arange(21).reshape(7, 3)
        x, y, indices = make_history_samples(q, dq, command, wrench, np.ones(7, bool), 4)
        online = IntentHistory(4)
        predictions = []
        for i in range(7):
            features = online.update(q[i], dq[i], command[i])
            if features is not None:
                predictions.append(features)
        np.testing.assert_array_equal(x, predictions)
        np.testing.assert_array_equal(indices, [3, 4, 5, 6])
        np.testing.assert_array_equal(y, wrench[indices])
        np.testing.assert_array_equal(x[0, -6:], command[3])
        online.reset()
        self.assertIsNone(online.update(q[0], dq[0], command[0]))

    def test_reset_response_breaks_windows(self):
        q = np.zeros((9, 6))
        valid = np.ones(9, bool)
        valid[4] = False
        _, _, indices = make_history_samples(q, q, q, np.zeros((9, 3)), valid, 4)
        np.testing.assert_array_equal(indices, [3, 8])

    def test_nonfinite_data_rejected(self):
        q = np.zeros((4, 6))
        q[0, 0] = np.nan
        with self.assertRaises(ValueError):
            make_history_samples(q, q, q, np.zeros((4, 3)), np.ones(4, bool))
        with self.assertRaises(ValueError):
            InteractionProtocol(force=float("nan"))

    @unittest.skipUnless(importlib.util.find_spec("mujoco") and importlib.util.find_spec("torch"), "MuJoCo/PyTorch unavailable")
    def test_mujoco_seven_conditions_have_physical_responses(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            output = Path(directory) / "pairs"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/paint/run_mujoco.py"), "--output_dir", str(output),
                                     "--warmup", ".1", "--duration", ".2", "--recovery", ".1", "--pose_jitter", "0"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest), 7)
            self.assertTrue(all(r["accepted"] for r in manifest))
            with np.load(output / "episodes/group_0000_zero.npz") as zero, np.load(output / "episodes/group_0000_fx_pos.npz") as force:
                np.testing.assert_array_equal(zero["applied_position"], force["applied_position"])
                np.testing.assert_array_equal(zero["interaction_wrench"], np.zeros((20, 3)))
                self.assertGreater(np.max(np.abs(force["arm_position"] - zero["arm_position"])), 1e-5)
                self.assertGreater(force["interaction_wrench"][:, 0].max(), 0)
            result = subprocess.run([sys.executable, str(ROOT / "scripts/paint/analyze_pd_pairs.py"), str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("pxr"), "OpenUSD unavailable")
    def test_unloaded_asset_preserves_robot_physics_and_topology(self):
        from paint_payload import create_unloaded_robot_asset
        from pxr import Usd, UsdPhysics
        source = ROOT / "robot_description/tron2/SFYG_TRON2A/usd/robot.usd"

        def physics(stage):
            result = {}
            for prim in stage.Traverse():
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint):
                    result[str(prim.GetPath())] = {attr.GetName(): str(attr.Get()) for attr in prim.GetAttributes()
                        if attr.GetName().startswith(("physics:", "physx", "xformOp"))}
            return result

        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            output = create_unloaded_robot_asset(source, Path(directory) / "unloaded.usdc")
            prepared = Usd.Stage.Open(output)
            self.assertEqual(physics(Usd.Stage.Open(str(source))), physics(prepared))
            self.assertFalse(any(p.GetName() == "paint_payload_Link" for p in prepared.Traverse()))

    @unittest.skipUnless(importlib.util.find_spec("pxr"), "OpenUSD unavailable")
    def test_original_usd_payload_attachment(self):
        from paint_payload import create_payload_asset
        from pxr import Usd, UsdGeom, UsdPhysics
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            path = create_payload_asset(ROOT / "robot_description/tron2/SFYG_TRON2A/usd/robot.usd", Path(directory) / "payload.usdc",
                                        .5, (.2, .12, .08), (.15, 0., 0.))
            stage = Usd.Stage.Open(path)
            payload = stage.GetPrimAtPath("/bipedal_robot/paint_payload_Link")
            attachment = stage.GetPrimAtPath("/bipedal_robot/gripper_base_Link")
            self.assertTrue(payload.HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertEqual(UsdPhysics.MassAPI(payload).GetMassAttr().Get(), .5)
            cache = UsdGeom.XformCache()
            relative = cache.GetLocalToWorldTransform(payload) * cache.GetLocalToWorldTransform(attachment).GetInverse()
            np.testing.assert_allclose(relative.ExtractTranslation(), [.15, 0, 0], atol=1e-7)
            joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath("/bipedal_robot/paint_payload_fixed_joint"))
            self.assertEqual(str(joint.GetBody1Rel().GetTargets()[0]), str(payload.GetPath()))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
    def test_training_export_and_shadow_observer(self):
        # Synthetic mechanics ONLY: checks the data/train/export pipeline, not robot accuracy.
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            root = Path(directory)
            dataset = root / "dataset"
            (dataset / "episodes").mkdir(parents=True)
            contract = {"schema": SCHEMA, "joint_names": list(ARM_JOINT_NAMES), "label_frame": "payload_local", "label_point": "payload_COM", "step_dt": .02}
            (dataset / "contract.json").write_text(json.dumps(contract))
            manifest = []
            for group in range(6):
                for condition in CONDITIONS:
                    count = 30
                    y = np.zeros((count, 3), dtype=np.float32)
                    if condition != "zero":
                        axis, sign = condition.split("_")
                        y[8:20, {"fx": 0, "fy": 1, "mz": 2}[axis]] = (1 if sign == "pos" else -1)
                    q = np.tile(np.arange(6) * .01 + group * .001, (count, 1)).astype(np.float32)
                    q[:, :3] += y * .02
                    dq = np.zeros_like(q)
                    command = np.tile(np.arange(6) * .01, (count, 1)).astype(np.float32)
                    filename = f"episodes/{group}_{condition}.npz"
                    np.savez(dataset / filename, arm_position=q, arm_velocity=dq, applied_position=command,
                             interaction_wrench=y, response_valid=np.ones(count, bool), meta_schema=SCHEMA, meta_accepted=True)
                    manifest.append({"group_id": f"seed_{group}", "condition": condition, "accepted": True, "file": filename})
            (dataset / "manifest.json").write_text(json.dumps(manifest))
            model_dir = root / "model"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/paint/train_intent_estimator.py"), "--datasets", str(dataset),
                                     "--output_dir", str(model_dir), "--epochs", "3"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            splits = json.loads((model_dir / "splits.json").read_text())
            self.assertFalse(set(splits["train"]) & set(splits["test"]))
            self.assertFalse(set(splits["validation"]) & set(splits["test"]))
            metrics = json.loads((model_dir / "metrics.json").read_text())
            self.assertTrue(np.isfinite(metrics["test"]["mae_N_N_Nm"]).all())
            telemetry = "\n".join(json.dumps({"time": i * .02, "q": [0]*6, "dq": [0]*6, "previous_applied_position": [0]*6,
                                                "reset": i == 5}) for i in range(9))
            result = subprocess.run([sys.executable, str(ROOT / "scripts/paint/observe_intent.py"), "--model_dir", str(model_dir)],
                                    input=telemetry, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertFalse(rows[2]["valid"])
            self.assertTrue(rows[3]["valid"])
            self.assertFalse(rows[5]["valid"])
            self.assertTrue(rows[8]["valid"])


if __name__ == "__main__":
    unittest.main()
