"""Author a separate USD copy with a fixed, inertial payload before spawning.

Uses only OpenUSD APIs; import after AppLauncher starts Isaac Sim.
The source robot asset is never edited.
"""
from pathlib import Path
import json
import numpy as np


def _open_flattened_robot(source):
    from pxr import Usd
    source = Path(source).resolve()
    print(f"[ASSET] open_start source={source}", flush=True)
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"Cannot open source USD: {source}")
    print("[ASSET] flatten_start", flush=True)
    stage = Usd.Stage.Open(stage.Flatten())
    print("[ASSET] flatten_complete", flush=True)
    return stage


def create_unloaded_robot_asset(source, output):
    """Use the same flattened composition as payload mode, with no physics edits."""
    stage = _open_flattened_robot(source)
    output = Path(output).resolve()
    if output == Path(source).resolve():
        raise ValueError("Prepared asset must not overwrite the source robot.")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ASSET] export_start output={output}", flush=True)
    if not stage.GetRootLayer().Export(str(output)):
        raise RuntimeError("Failed to export unloaded robot asset.")
    print("[ASSET] export_complete payload=none", flush=True)
    output.with_suffix(".json").write_text(json.dumps({"source": str(Path(source).resolve()), "output": str(output),
        "payload_mode": "none", "composition": "flattened", "added_mass_kg": 0.0, "added_joints": 0}, indent=2) + "\n", encoding="utf-8")
    return str(output)


def create_payload_asset(source, output, mass, dimensions, offset):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    values = np.array([mass, *dimensions, *offset], dtype=float)
    if not np.isfinite(values).all() or mass <= 0 or len(dimensions) != 3 or len(offset) != 3 or np.any(np.array(dimensions) <= 0):
        raise ValueError("Payload requires positive mass/dimensions and a finite three-vector offset.")
    stage = _open_flattened_robot(source)
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise ValueError("Payload dimensions assume a robot USD authored in meters.")
    matches = [p for p in stage.Traverse() if p.GetName() == "gripper_base_Link" and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one rigid gripper_base_Link, found {len(matches)}.")
    attachment = matches[0]
    parent = attachment.GetParent()
    if attachment.IsInstanceProxy() or parent.IsInstance():
        raise RuntimeError("Payload authoring requires editable, non-instance link prims.")
    body_path = parent.GetPath().AppendChild("paint_payload_Link")
    body = UsdGeom.Xform.Define(stage, body_path)
    cache = UsdGeom.XformCache()
    offset_matrix = Gf.Matrix4d(1.0)
    offset_matrix.SetTranslate(Gf.Vec3d(*offset))
    local_matrix = offset_matrix * cache.GetLocalToWorldTransform(attachment) * cache.GetLocalToWorldTransform(parent).GetInverse()
    body.AddTransformOp().Set(local_matrix)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass_api.CreateMassAttr(float(mass))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
    x, y, z = dimensions
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(mass * (y*y + z*z) / 12, mass * (x*x + z*z) / 12, mass * (x*x + y*y) / 12))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1))
    cube = UsdGeom.Cube.Define(stage, body_path.AppendChild("collision"))
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*dimensions))
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.55, 0.85)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    joint = UsdPhysics.FixedJoint.Define(stage, parent.GetPath().AppendChild("paint_payload_fixed_joint"))
    joint.CreateBody0Rel().SetTargets([attachment.GetPath()])
    joint.CreateBody1Rel().SetTargets([body_path])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*offset))
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateCollisionEnabledAttr(False)
    # Adjacent fixed links must not collide with one another (including fingers).
    filtered = UsdPhysics.FilteredPairsAPI.Apply(body.GetPrim())
    filtered.CreateFilteredPairsRel().SetTargets([
        p.GetPath() for p in stage.Traverse()
        if p.GetName() in ("gripper_base_Link", "gripper1_Link", "gripper2_Link")
    ])
    output = Path(output).resolve()
    if output == Path(source).resolve():
        raise ValueError("Prepared asset must not overwrite the source robot.")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ASSET] export_start output={output}", flush=True)
    if not stage.GetRootLayer().Export(str(output)):
        raise RuntimeError("Failed to export payload asset.")
    print("[ASSET] export_complete payload=attached", flush=True)
    report = {"source": str(Path(source).resolve()), "output": str(output), "payload_body": "paint_payload_Link",
              "attachment": str(attachment.GetPath()), "mass_kg": mass, "dimensions_m": list(dimensions),
              "offset_in_gripper_m": list(offset), "coupling": "fixed_joint"}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return str(output)
