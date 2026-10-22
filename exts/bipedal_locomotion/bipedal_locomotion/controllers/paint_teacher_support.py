"""Batched, causal quasi-static support model for teacher training.

Gravity and known external-on-payload forces become an observation for the
frozen base policy. Arm feedforward cancels gravity only. No dynamics preview,
future force oracle, or additional physical base force is introduced here.
"""
import torch


class BatchedSupport:
    def __init__(self, model, joint_names, arm_names, device):
        self.model = model
        self.names = list(model.bodies)
        self.arm_names = tuple(arm_names)
        self.device = device
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        self.mass = tensor([model.bodies[n][0] for n in self.names])
        self.inertials = {n: tensor(model.bodies[n][1]) for n in self.names}
        parent_joint = {j[3]: j for j in model.joints}
        needed = set()
        self.ancestors = {}
        for body in self.names:
            current, ancestors = body, set()
            while current != "base_Link":
                j = parent_joint[current]
                ancestors.add(j[0]); needed.add(j[0]); current = j[2]
            self.ancestors[body] = ancestors
        indices = {n: i for i, n in enumerate(joint_names)}
        self.joints = []
        pending = [j for j in model.joints if j[0] in needed]
        available = {"base_Link"}
        while pending:
            remaining = []
            for name, kind, parent, child, pose, axis in pending:
                if parent not in available:
                    remaining.append((name, kind, parent, child, pose, axis)); continue
                if kind != "fixed" and name not in indices:
                    raise ValueError(f"Support model joint missing: {name}")
                axis = tensor(axis); axis = axis / axis.norm()
                x, y, z = axis.unbind()
                zero = x * 0
                skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
                self.joints.append((name, kind, parent, child, tensor(pose), axis, skew,
                                    indices.get(name)))
                available.add(child)
            if len(remaining) == len(pending):
                raise ValueError("Support model is not rooted at base_Link")
            pending = remaining
        self.descendants = tensor([[float(j in self.ancestors[b]) for b in self.names]
                                   for j in self.arm_names])

    def kinematics(self, q):
        eye = torch.eye(4, device=q.device, dtype=q.dtype)
        transforms = {"base_Link": eye.expand(len(q), 4, 4)}
        axes, origins = {}, {}
        for name, kind, parent, child, pose, axis, skew, index in self.joints:
            frame = transforms[parent] @ pose
            if name in self.arm_names:
                axes[name] = (frame[:, :3, :3] @ axis).squeeze(-1)
                origins[name] = frame[:, :3, 3]
            if kind == "fixed":
                transforms[child] = frame
                continue
            motion = eye.expand(len(q), 4, 4).clone()
            angle = q[:, index]
            if kind in ("revolute", "continuous"):
                motion[:, :3, :3] += angle.sin()[:, None, None] * skew + (1-angle.cos())[:, None, None] * (skew @ skew)
            elif kind == "prismatic":
                motion[:, :3, 3] = angle[:, None] * axis
            else:
                raise ValueError(f"Unsupported support joint: {kind}")
            transforms[child] = frame @ motion
        poses = torch.stack([transforms[n] @ self.inertials[n] for n in self.names], dim=1)
        return poses, torch.stack([axes[n] for n in self.arm_names], 1), torch.stack([origins[n] for n in self.arm_names], 1)

    @torch.no_grad()
    def compute(self, q, gravity_base, external_payload):
        poses, axes, origins = self.kinematics(q)
        com = poses[:, :, :3, 3]
        force = self.mass[None, :, None] * gravity_base[:, None]
        wrench = torch.cat((force.sum(1), torch.cross(com, force, dim=-1).sum(1)), -1)
        moments = torch.cross(com[:, None] - origins[:, :, None], force[:, None], dim=-1)
        effort = -(moments * axes[:, :, None] * self.descendants[None, :, :, None]).sum((2, 3))
        payload = poses[:, self.names.index("paint_payload_Link")]
        local_force = torch.zeros_like(gravity_base)
        local_force[:, :2] = external_payload[:, :2]
        local_torque = torch.zeros_like(local_force)
        local_torque[:, 2] = external_payload[:, 2]
        applied_force = (payload[:, :3, :3] @ local_force[..., None]).squeeze(-1)
        applied_torque = (payload[:, :3, :3] @ local_torque[..., None]).squeeze(-1)
        wrench[:, :3] += applied_force
        wrench[:, 3:] += applied_torque + torch.cross(payload[:, :3, 3], applied_force, dim=-1)
        return wrench, effort, poses
