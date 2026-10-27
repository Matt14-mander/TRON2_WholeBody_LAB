"""Refresh the small authored-USD reference; deployment itself does not need pxr."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "robot_description/tron2/SFYG_TRON2A/usd"
REFERENCE = Path(__file__).with_suffix(".json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refresh():
    from pxr import Usd, UsdPhysics
    def vector(v):
        return [float(x) for x in v]
    def quat(v):
        return [float(v.GetReal()), *vector(v.GetImaginary())]
    # Some Windows OpenUSD wheels cannot resolve non-ASCII absolute paths.
    xml = ET.parse(ASSET.parent / "xml/robot.xml")
    with tempfile.TemporaryDirectory(prefix="sfyg_usd_") as folder:
        shutil.copytree(ASSET, Path(folder) / "usd")
        stage = Usd.Stage.Open(str(Path(folder) / "usd/robot.usd"))
        bodies, joints = {}, {}
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                mass = UsdPhysics.MassAPI(prim)
                com = vector(mass.GetCenterOfMassAttr().Get())
                com_source = "authored_usd"
                if not all(math.isfinite(v) for v in com):
                    # USD's -inf sentinel requests automatic CoM. These two
                    # camera frames have no collision geometry; retain their
                    # original MJCF origin CoM and report this assumption.
                    node = xml.find(f".//body[@name='{prim.GetName()}']/inertial")
                    com = [float(v) for v in node.get("pos", "0 0 0").split()]
                    com_source = "usd_automatic_no_collider_mjcf_origin"
                bodies[prim.GetName()] = dict(mass=float(mass.GetMassAttr().Get()),
                    com=com, com_source=com_source,
                    inertia=vector(mass.GetDiagonalInertiaAttr().Get()),
                    principal_axes=quat(mass.GetPrincipalAxesAttr().Get()))
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                p0, p1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
                if not p0 or not p1:
                    continue
                item = dict(parent=p0[0].name, child=p1[0].name, type=prim.GetTypeName(),
                    pos0=vector(joint.GetLocalPos0Attr().Get()), pos1=vector(joint.GetLocalPos1Attr().Get()),
                    rot0=quat(joint.GetLocalRot0Attr().Get()), rot1=quat(joint.GetLocalRot1Attr().Get()))
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    typed = UsdPhysics.RevoluteJoint(prim)
                    item.update(axis=typed.GetAxisAttr().Get() or "X", lower=float(typed.GetLowerLimitAttr().Get()),
                                upper=float(typed.GetUpperLimitAttr().Get()))
                elif prim.IsA(UsdPhysics.PrismaticJoint):
                    typed = UsdPhysics.PrismaticJoint(prim)
                    item.update(axis=typed.GetAxisAttr().Get() or "X", lower=float(typed.GetLowerLimitAttr().Get()),
                                upper=float(typed.GetUpperLimitAttr().Get()))
                joints[prim.GetName()] = item
        layers = {}
        for layer in stage.GetUsedLayers():
            if layer.realPath:
                relative = Path(layer.realPath).relative_to(Path(folder) / "usd")
                original = ASSET / relative
                layers[original.relative_to(ROOT).as_posix()] = sha256(original)
    if len(bodies) != 29 or len(joints) != 28:
        raise ValueError(f"Unexpected SFYG topology: {len(bodies)} bodies, {len(joints)} joints")
    return dict(schema="sfyg_authored_usd_v1", sources=layers, bodies=bodies, joints=joints,
                total_mass_kg=sum(b["mass"] for b in bodies.values()))


def load_reference():
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    for name, expected in reference["sources"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"USD reference is stale: {name}; refresh and review the asset changes")
    return reference


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()
    if args.refresh:
        REFERENCE.write_text(json.dumps(refresh(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result = load_reference()
    print(f"[USD] bodies={len(result['bodies'])} mass={result['total_mass_kg']:.6f} kg; hashes verified")
