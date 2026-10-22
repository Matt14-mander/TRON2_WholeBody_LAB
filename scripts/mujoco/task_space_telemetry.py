"""Simulator-state task-space recording and offline alignment with arm logs."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np


def clock_contract():
    return dict(host=platform.node(), wall_minus_monotonic_s=time.time()-time.monotonic())


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1., 1.))))


class TaskSpaceRecorder:
    name = "v3_task_space_recorder"
    slider_control = False

    def __init__(self, directory, up_axis=(0, 0, 1), metadata=None, record_policy_io=False):
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        axis = np.asarray(up_axis, dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
            raise ValueError("Payload up axis must be a finite nonzero vector")
        self.up_axis = axis / np.linalg.norm(axis)
        self.rows, self.attached, self.error = [], False, None
        self.record_policy_io = record_policy_io
        self.segment, self.next_time, self.previous = 0, 0., {}
        self.contract = dict(clock=clock_contract(), arguments=metadata,
            payload_up_axis=self.up_axis.tolist(), sample_hz=100,
            source="Native simulator qpos/qvel; consistent FK/Jacobians in private MjData. Live data is never modified.",
            end_effector="gripper_base_Link body origin; no extra TCP offset",
            payload="paint_payload_Link COM; null when absent",
            reference="Received arm position targets, measured other joints and current base pose. Not an independent world-space trajectory.",
            acceleration="Finite difference of world linear velocity over simulator time; net kinematic acceleration, no gravity added. 100Hz sampled, not physics-rate peaks.")
        self._write_contract()

    def _write_contract(self):
        (self.directory / "task_contract.json").write_text(
            json.dumps(self.contract, indent=2, ensure_ascii=False), encoding="utf-8")

    def attach(self, core):
        import mujoco
        self.mj, self.model = mujoco, core.model
        self.actual, self.target = mujoco.MjData(core.model), mujoco.MjData(core.model)
        self.ids = {key: mujoco.mj_name2id(core.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for key, name in dict(base="base_Link", ee="gripper_base_Link", payload="paint_payload_Link").items()}
        if min(self.ids["base"], self.ids["ee"]) < 0:
            raise ValueError("Task recorder requires base_Link and gripper_base_Link")
        self.channel = next(ch for ch in core.channels
                            if all(f"arm{i}_Joint" in ch.spec.joints for i in range(1, 7)))
        self.arm_indices = [self.channel.spec.joints.index(f"arm{i}_Joint") for i in range(1, 7)]
        self.jac = np.zeros((3, core.model.nv))
        self.contract["payload_present"] = self.ids["payload"] >= 0
        self.contract["wire_joint_names"] = list(self.channel.spec.joints)
        if self.record_policy_io:
            if self.channel.imu_ids is None:
                raise ValueError("Policy IO audit requires native IMU sensors")
            self.contract["native_imu_sensors"] = [mujoco.mj_id2name(core.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                                                   for i in self.channel.imu_ids]
        self.attached = True
        self._write_contract()
        print(f"[V3 TASK RECORD] simulator state -> {self.directory}", flush=True)
        return True

    def owned_actuators(self):
        return []

    def on_control(self, core, dt):
        pass

    def on_manual(self, core):
        pass

    def on_render(self, core, render_data):
        pass

    def on_reset(self, core):
        self.segment += 1
        self.next_time, self.previous = 0., {}

    def on_publish(self, core):
        if self.error is not None:
            return
        try:
            if core.manual_mode or core.paused or not self.channel.has_command:
                self.previous = {}
                return
            t = float(core.data.time)
            if t < self.next_time - 1e-9:
                return
            self.next_time = t + .01
            self.capture(core.data, self.channel.cmd, time.monotonic())
        except Exception as exc:
            # Recording faults must not change control. They invalidate analysis.
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[V3 TASK RECORD ERROR] {self.error}", flush=True)

    def capture(self, live, command, stamp):
        mj, model, actual, target = self.mj, self.model, self.actual, self.target
        actual.qpos[:] = live.qpos
        actual.qvel[:] = live.qvel
        target.qpos[:] = live.qpos
        for i in self.arm_indices:
            target.qpos[self.channel.map[i][0]] = command["q"][i]
        for data in (actual, target):
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)
        base = self.ids["base"]
        rb = actual.xmat[base].reshape(3, 3)
        pb = actual.xpos[base]
        row = dict(monotonic_s=float(stamp), sim_time_s=float(live.time), segment=self.segment,
            base_position_world_m=pb.tolist(), base_rotation_world=rb.reshape(-1).tolist(),
            base_tilt_deg=angle(rb[:, 2], np.array([0, 0, 1.])))
        if self.record_policy_io:
            quat, gyro, _ = self.channel._read_imu(live)
            base_quat = np.zeros(4)
            mj.mju_mat2Quat(base_quat, np.ascontiguousarray(rb.reshape(-1)))
            jac_rotation = np.zeros((3, model.nv))
            mj.mj_jacBody(model, actual, None, jac_rotation, base)
            row["native_io"] = dict(q=[float(live.qpos[q]) for q, _, _ in self.channel.map],
                dq=[float(live.qvel[v]) for _, v, _ in self.channel.map], received_q=list(command["q"]),
                imu_quaternion_wxyz=list(quat), imu_gyro=list(gyro),
                base_quaternion_wxyz=base_quat.tolist(),
                base_gyro_body=(rb.T@jac_rotation@actual.qvel).tolist())
        for key in ("ee", "payload"):
            bid = self.ids[key]
            if bid < 0:
                row[key] = None
                continue
            p = actual.xipos[bid] if key == "payload" else actual.xpos[bid]
            pt = target.xipos[bid] if key == "payload" else target.xpos[bid]
            r = actual.xmat[bid].reshape(3, 3)
            rt = target.xmat[bid].reshape(3, 3)
            mj.mj_jac(model, actual, self.jac, None, p, bid)
            velocity = self.jac @ actual.qvel
            previous = self.previous.get(key)
            dt = float(live.time)-previous[0] if previous else 0.
            acceleration = (velocity-previous[1])/dt if 0 < dt <= .03 else None
            self.previous[key] = (float(live.time), velocity.copy())
            qw, qb, qtw, qtb = (np.zeros(4) for _ in range(4))
            for quat, matrix in ((qw, r), (qb, rb.T@r), (qtw, rt), (qtb, rb.T@rt)):
                mj.mju_mat2Quat(quat, np.ascontiguousarray(matrix.reshape(-1)))
            row[key] = dict(position_world_m=p.tolist(), position_base_m=(rb.T@(p-pb)).tolist(),
                target_position_world_m=pt.tolist(), target_position_base_m=(rb.T@(pt-pb)).tolist(),
                quaternion_world_wxyz=qw.tolist(), quaternion_base_wxyz=qb.tolist(),
                target_quaternion_world_wxyz=qtw.tolist(), target_quaternion_base_wxyz=qtb.tolist(),
                position_error_world_m=(p-pt).tolist(), position_error_base_m=(rb.T@(p-pt)).tolist(),
                orientation_error_rad=float(np.arccos(np.clip((np.trace(rt.T@r)-1)/2, -1, 1))),
                velocity_world_m_s=velocity.tolist(),
                acceleration_world_m_s2=None if acceleration is None else acceleration.tolist(),
                up_world=(r@self.up_axis).tolist(),
                tilt_world_deg=angle(r@self.up_axis, np.array([0, 0, 1.])))
        self.rows.append(row)

    def save(self):
        self.contract.update(attached=self.attached, error=self.error, samples=len(self.rows))
        self._write_contract()
        with (self.directory / "task_steps.jsonl").open("w", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
        print(f"[V3 TASK RECORD] saved {len(self.rows)} samples to {self.directory}", flush=True)


def statistics(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return None
    norms = np.linalg.norm(x, axis=1) if x.ndim == 2 else np.abs(x)
    return dict(mean=x.mean(axis=0).tolist(), rms=np.sqrt(np.mean(x*x, axis=0)).tolist(),
                std=x.std(axis=0).tolist(), peak_abs=np.max(np.abs(x), axis=0).tolist(),
                norm_rms=float(np.sqrt(np.mean(norms*norms))),
                norm_peak=float(norms.max()), norm_p95=float(np.percentile(norms, 95)))


def summarize(rows):
    result = dict(samples=len(rows), base_tilt_deg=statistics([r["base_tilt_deg"] for r in rows]))
    for key in ("ee", "payload"):
        points = [r[key] for r in rows if r.get(key) is not None]
        result[key] = {field: statistics([p[field] for p in points if p.get(field) is not None])
            for field in ("position_error_base_m", "position_error_world_m", "orientation_error_rad",
                          "velocity_world_m_s", "acceleration_world_m_s2", "tilt_world_deg",
                          "normal_change_from_first_control_deg", "world_displacement_from_start_m")}
    return result


def analyze(sim_directory, arm_directory, output):
    sim_directory, arm_directory, output = map(Path, (sim_directory, arm_directory, output))
    sc = json.loads((sim_directory / "task_contract.json").read_text(encoding="utf-8"))
    ac = json.loads((arm_directory / "contract.json").read_text(encoding="utf-8"))
    if not sc.get("attached") or sc.get("error"):
        raise ValueError(f"Simulator recorder invalid: {sc.get('error')}")
    if sc["clock"]["host"] != ac["clock"]["host"] or abs(
            sc["clock"]["wall_minus_monotonic_s"]-ac["clock"]["wall_minus_monotonic_s"]) > 1.:
        raise ValueError("Logs must use the same host/monotonic clock session")
    arms = [json.loads(x) for x in (arm_directory / "arm_steps.jsonl").read_text().splitlines()]
    if not arms or any(r.get("monotonic_s") is None for r in arms):
        raise ValueError("Need a new control rollout with monotonic timestamps")
    stamps = np.array([r["monotonic_s"] for r in arms])
    if np.any(np.diff(stamps) <= 0):
        raise ValueError("Controller timestamps must increase")
    selected, baseline, eligible = [], {}, 0
    with (sim_directory / "task_steps.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            stamp = row["monotonic_s"]
            if not stamps[0] <= stamp <= stamps[-1]:
                continue
            eligible += 1
            i = int(np.searchsorted(stamps, stamp, side="right")-1)
            if stamp-stamps[i] > .03:
                continue
            row.update(time_s=stamp-stamps[0], phase=arms[i]["phase"],
                       phase_alignment_age_ms=(stamp-stamps[i])*1000)
            for key in ("ee", "payload"):
                point = row.get(key)
                if point is None:
                    continue
                if key not in baseline:
                    baseline[key] = (np.array(point["up_world"]), np.array(point["position_world_m"]))
                point["normal_change_from_first_control_deg"] = angle(point["up_world"], baseline[key][0])
                point["world_displacement_from_start_m"] = (np.array(point["position_world_m"])-baseline[key][1]).tolist()
            selected.append(row)
    if not selected or len({r["segment"] for r in selected}) != 1:
        raise ValueError("Need overlapping data from one simulator reset segment")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "task_aligned.jsonl").open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    report = dict(definitions=sc, arm_outcome=json.loads((arm_directory / "arm_report.json").read_text())["outcome"],
        alignment="Latest preceding control publication on the same host, <=30ms; phase approximate, target comes from received simulator command.",
        alignment_coverage=dict(sim_samples_in_control_interval=eligible, aligned_samples=len(selected),
            discarded_for_control_gap=eligible-len(selected),
            phase_age_p95_ms=float(np.percentile([r["phase_alignment_age_ms"] for r in selected], 95))),
        all=summarize(selected), after_2s=summarize([r for r in selected if r["time_s"] >= 2]),
        last_5s=summarize([r for r in selected if r["time_s"] >= max(2., selected[-1]["time_s"]-5)]),
        phases={phase: summarize([r for r in selected if r["phase"] == phase])
                for phase in sorted({r["phase"] for r in selected})})
    (output / "task_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Task-space report: {output / 'task_report.json'} ({len(selected)} aligned samples)")
    print("Arm outcome:", report["arm_outcome"])
    print("Alignment:", report["alignment_coverage"])
    for name, group in {"after_2s": report["after_2s"], **report["phases"]}.items():
        print(f"\n=== {name}: {group['samples']} samples ===")
        for point in ("ee", "payload"):
            data = group[point]
            fields = ("position_error_base_m", "orientation_error_rad", "acceleration_world_m_s2",
                      "tilt_world_deg", "normal_change_from_first_control_deg")
            print(point, {key: {metric: round(data[key][metric], 6)
                for metric in ("norm_rms", "norm_p95", "norm_peak")}
                for key in fields if data[key] is not None})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-record-dir", required=True)
    parser.add_argument("--arm-record-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    analyze(args.sim_record_dir, args.arm_record_dir, args.output_dir)
