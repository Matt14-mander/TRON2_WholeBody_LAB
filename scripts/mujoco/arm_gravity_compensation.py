"""Gravity-only arm feedforward from a private copy of simulator kinematics."""
import numpy as np


class ArmGravityCompensation:
    def __init__(self, model, mapping, arm_names, max_effort=30., ramp_time=2.):
        import mujoco
        self.mj, self.model = mujoco, model
        self.data = mujoco.MjData(model)  # Never shared with the preview worker.
        self.base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
        if self.base < 0:
            raise ValueError("Gravity compensation requires base_Link")
        if np.any(model.body_gravcomp):
            raise ValueError("Model already contains native gravity compensation")
        if not np.isfinite([max_effort, ramp_time]).all() or min(max_effort, ramp_time) <= 0:
            raise ValueError("Gravity effort limit and ramp time must be positive and finite")
        self.ramp_time = ramp_time
        self.addresses = {}
        self.dofs, self.limits = [], []
        for wire, native in mapping.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, native)
            self.addresses[wire] = int(model.jnt_qposadr[jid])
            if wire in arm_names:
                # Arm SDK torque is actuator force; require direct unit-gear motors.
                acts = [a for a in range(model.nu) if int(model.actuator_trntype[a]) == 0
                        and int(model.actuator_trnid[a, 0]) == jid]
                if len(acts) != 1 or not np.allclose(model.actuator_gear[acts[0]], [1, 0, 0, 0, 0, 0]):
                    raise ValueError(f"Gravity compensation requires one unit-gear joint motor: {wire}")
        for name in arm_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mapping[name])
            self.dofs.append(int(model.jnt_dofadr[jid]))
            aid = next(a for a in range(model.nu) if int(model.actuator_trntype[a]) == 0
                       and int(model.actuator_trnid[a, 0]) == jid)
            limit = float(max_effort)
            for limited, bounds in ((model.actuator_ctrllimited[aid], model.actuator_ctrlrange[aid]),
                                    (model.actuator_forcelimited[aid], model.actuator_forcerange[aid])):
                if limited:
                    limit = min(limit, float(min(-bounds[0], bounds[1])))
            if limit <= 0:
                raise ValueError(f"Motor range must contain both torque signs: {name}")
            self.limits.append(limit)
        self.limits = np.asarray(self.limits)
        self.jac = np.zeros((3, model.nv))

    def compute(self, positions, gravity_base):
        if set(positions) != set(self.addresses):
            raise ValueError("Gravity input must include every mapped measured joint")
        if not np.isfinite([*positions.values(), *gravity_base]).all():
            raise ValueError("Nonfinite measured gravity or joint positions")
        mj, model, data = self.mj, self.model, self.data
        mj.mj_resetData(model, data)
        for name, value in positions.items():
            data.qpos[self.addresses[name]] = value
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        gravity_world = data.xmat[self.base].reshape(3, 3) @ np.asarray(gravity_base)
        effort = np.zeros(model.nv)
        for body in range(1, model.nbody):
            mj.mj_jacBodyCom(model, data, self.jac, None, body)
            # Negative generalized gravity is the actuator effort needed to hold.
            effort -= self.jac.T @ (model.body_mass[body] * gravity_world)
        return effort[self.dofs]

    def command(self, raw, elapsed):
        raw = np.asarray(raw, dtype=float)
        if raw.shape != self.limits.shape or not np.isfinite(raw).all():
            raise ValueError("Invalid gravity compensation torque")
        x = float(np.clip(elapsed / self.ramp_time, 0., 1.))
        weight = x*x*(3.-2.*x)
        return np.clip(raw, -self.limits, self.limits) * weight, weight
