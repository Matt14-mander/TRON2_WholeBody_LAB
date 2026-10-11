"""Make a separate benchmark URDF with conservative mesh-enclosing boxes.

Visuals, inertias, joints and existing primitive collisions are retained. This
is a computational candidate, not an automatically approved online geometry.
"""
import argparse
import json
from pathlib import Path
import struct
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

import numpy as np


def vector(text, default):
    result = np.asarray([float(x) for x in text.split()] if text else default)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("Expected a finite xyz/rpy/scale vector of length 3.")
    return result


def rotation(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.asarray([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                       [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                       [-sp, cp * sr, cp * cr]])


def mesh_path(filename, source):
    prefix = "package://bipedal_robot/meshes/"
    if filename.startswith(prefix):
        return (source.parent.parent / "meshes" / filename[len(prefix):]).resolve()
    if filename.startswith("file://"):
        path = unquote(urlparse(filename).path)
        if len(path) >= 3 and path[0] == "/" and path[2] == ":":
            path = path[1:]  # Windows file:///E:/... in portable tests.
        return Path(path).resolve()
    if "://" in filename:
        raise ValueError(f"Unsupported mesh URI: {filename}")
    return (source.parent / filename).resolve()


def binary_stl_bounds(path, scale):
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"Truncated binary STL: {path}")
    count = struct.unpack_from("<I", raw, 80)[0]
    if count == 0 or len(raw) != 84 + count * 50:
        raise ValueError(f"Expected a nonempty binary STL with exact facet count: {path}")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    vertices = np.frombuffer(raw, dtype=dtype, offset=84, count=count)["vertices"].reshape(-1, 3)
    vertices = vertices.astype(float) * scale
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"Non-finite STL vertices: {path}")
    return vertices.min(axis=0), vertices.max(axis=0), count


def generate(source, output, padding=0.005):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Output must differ from the source robot URDF.")
    if not np.isfinite(padding) or padding <= 0:
        raise ValueError("Padding must be positive and finite.")
    tree = ET.parse(source)
    root = tree.getroot()
    records = []
    for link in root.findall("link"):
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            mesh = None if geometry is None else geometry.find("mesh")
            if mesh is None:
                continue
            path = mesh_path(mesh.attrib["filename"], source)
            lower, upper, count = binary_stl_bounds(path, vector(mesh.get("scale"), [1, 1, 1]))
            center = (lower + upper) / 2
            size = upper - lower + 2 * padding
            origin = collision.find("origin")
            if origin is None:
                origin = ET.Element("origin")
                collision.insert(0, origin)
            xyz = vector(origin.get("xyz"), [0, 0, 0])
            rpy = vector(origin.get("rpy"), [0, 0, 0])
            # A mesh-local center must be rotated before shifting its origin.
            origin.set("xyz", " ".join(format(v, ".17g") for v in xyz + rotation(rpy) @ center))
            geometry.remove(mesh)
            ET.SubElement(geometry, "box", size=" ".join(format(v, ".17g") for v in size))
            records.append({"link": link.get("name"), "mesh": str(path), "triangles": count,
                            "box_size": size.tolist(), "mesh_local_center": center.tolist()})
    if not records:
        raise ValueError("No collision meshes found; nothing to compare.")
    # The generated URDF lives outside the original urdf/meshes directory.
    for mesh in root.iter("mesh"):
        mesh.set("filename", mesh_path(mesh.attrib["filename"], source).as_uri())
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    report = {"source": str(source), "output": str(output), "padding_m": padding,
              "geometry": "conservative_mesh_enclosing_boxes; benchmark_candidate",
              "mesh_collisions_replaced": len(records), "links": records}
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--padding", type=float, default=0.005)
    args = parser.parse_args()
    report = generate(args.urdf, args.output, args.padding)
    print(f"Generated {args.output}: {report['mesh_collisions_replaced']} enclosing mesh boxes, padding={args.padding} m")


if __name__ == "__main__":
    main()
