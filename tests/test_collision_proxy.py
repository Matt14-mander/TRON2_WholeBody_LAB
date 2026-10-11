import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("collision_proxy", REPO / "scripts/ocs2/make_collision_proxy_urdf.py")
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)


class CollisionProxyTest(unittest.TestCase):
    def test_enclosure_transformed_origin_and_preserved_model(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            mesh = folder / "shape.stl"
            vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 2, 1]], dtype=float)
            mesh.write_bytes(b"test".ljust(80, b"\0") + struct.pack("<I", 1) +
                             struct.pack("<12fH", 0, 0, 1, *vertices.flatten(), 0))
            original = folder / "source.urdf"
            original.write_text('''<robot name="test"><link name="base">
              <inertial><mass value="2"/></inertial>
              <visual><geometry><mesh filename="shape.stl"/></geometry></visual>
              <collision><origin xyz="0.1 0.2 0.3" rpy="0 0 1.5707963267948966"/>
                <geometry><mesh filename="shape.stl" scale="2 1 1"/></geometry></collision>
              <collision><geometry><sphere radius="0.1"/></geometry></collision>
              </link><joint name="arm1" type="revolute"><limit lower="-1" upper="1"/></joint></robot>''')
            before = original.read_bytes()
            output = folder / "out" / "proxy.urdf"
            report = proxy.generate(original, output)
            self.assertEqual(original.read_bytes(), before)
            self.assertEqual(report["mesh_collisions_replaced"], 1)
            root = ET.parse(output).getroot()
            collision = root.find("link/collision")
            np.testing.assert_allclose(proxy.vector(collision.find("origin").get("xyz"), []), [-0.9, 1.2, 0.8])
            size = proxy.vector(collision.find("geometry/box").get("size"), [])
            center = np.asarray(report["links"][0]["mesh_local_center"])
            scaled = vertices * [2, 1, 1]
            self.assertTrue(np.all(np.abs(scaled - center) <= size / 2 - 0.0049))
            self.assertEqual(root.find("link/inertial/mass").get("value"), "2")
            self.assertEqual(root.find("joint/limit").get("lower"), "-1")
            self.assertIsNone(collision.find("geometry/sphere"))
            self.assertEqual(len(root.findall("link/collision/geometry/sphere")), 1)
            self.assertEqual(proxy.mesh_path(root.find("link/visual/geometry/mesh").get("filename"), original), mesh.resolve())

    def test_invalid_mesh_and_in_place_output_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.stl"
            path.write_bytes(b"x" * 84)
            with self.assertRaises(ValueError):
                proxy.binary_stl_bounds(path, np.ones(3))
            with self.assertRaises(ValueError):
                proxy.generate(path, path)


if __name__ == "__main__":
    unittest.main()
