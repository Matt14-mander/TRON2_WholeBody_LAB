"""Physical signs, frames and joint gravity torques, without Isaac Sim."""
from pathlib import Path
import sys
import unittest
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers"))
from paint_wrench_preview import AddedBodyModel
from paint_teacher_support import BatchedSupport
from paint_teacher_core import ARM_JOINT_NAMES, LEG_JOINT_NAMES


class SupportTest(unittest.TestCase):
    def setUp(self):
        folder = ROOT / "robot_description/tron2"
        self.model = AddedBodyModel(folder / "SFYG_TRON2A/urdf/robot_rl.urdf",
            folder / "SF_TRON2A/urdf/robot_rl.urdf", .5)
        self.names = list(LEG_JOINT_NAMES + ARM_JOINT_NAMES) + ["gripper1_Joint", "gripper2_Joint"]
        self.support = BatchedSupport(self.model, self.names, ARM_JOINT_NAMES, "cpu")
        self.q = torch.zeros(3, 18)
        self.q[:, 11] = torch.tensor([1.57, 1.2, 1.8])
        self.q[:, 12] = -1.48
        self.q[1, 10] = .2
        self.q[2, 16:] = .01
        self.g = torch.tensor([[0., 0., -9.81], [1., 0., -9.75], [0., .3, -9.80]])

    def test_gravity_matches_independent_numpy_and_potential_gradient(self):
        wrench, effort, poses = self.support.compute(self.q, self.g, torch.zeros(3, 3))
        for row in range(3):
            q = dict(zip(self.names, self.q[row].tolist()))
            gravity = self.g[row].numpy()
            np.testing.assert_allclose(wrench[row], self.model.gravity_wrench(q, gravity), atol=2e-5)
            states = self.model.states(q)
            for i, name in enumerate(self.support.names):
                np.testing.assert_allclose(poses[row, i], states[name], atol=4e-7)
            def potential(q):
                states = self.model.states(q)
                return -sum(body[0] * np.dot(gravity, states[name][:3, 3]) for name, body in self.model.bodies.items())
            for i, name in enumerate(ARM_JOINT_NAMES):
                plus, minus = dict(q), dict(q)
                plus[name] += 1e-5; minus[name] -= 1e-5
                expected = (potential(plus) - potential(minus)) / 2e-5
                self.assertAlmostEqual(float(effort[row, i]), expected, delta=2e-5)

    def test_external_force_moment_and_gravity_only_feedforward(self):
        zero, effort, poses = self.support.compute(self.q, self.g, torch.zeros(3, 3))
        command = torch.tensor([[2., -3., .4]]).expand(3, -1)
        wrench, with_external, _ = self.support.compute(self.q, self.g, command)
        payload = poses[:, self.support.names.index("paint_payload_Link")]
        force = (payload[:, :3, :3] @ torch.tensor([2., -3., 0.])).numpy()
        torque = (payload[:, :3, :3] @ torch.tensor([0., 0., .4])).numpy()
        expected = np.c_[force, torque + np.cross(payload[:, :3, 3].numpy(), force)]
        np.testing.assert_allclose(wrench-zero, expected, atol=5e-6)
        torch.testing.assert_close(with_external, effort, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
