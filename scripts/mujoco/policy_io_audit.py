"""Passive ONNX I/O capture and independent offline deployment-contract audit."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class SessionTap:
    def __init__(self, session, recorder, name):
        self.session, self.recorder, self.name = session, recorder, name

    def __getattr__(self, name):
        return getattr(self.session, name)

    def run(self, output_names, inputs, *args, **kwargs):
        outputs = self.session.run(output_names, inputs, *args, **kwargs)
        if self.recorder.pending is not None:
            self.recorder.pending[self.name] = dict(
                input=np.array(next(iter(inputs.values())), copy=True),
                output=np.array(outputs[0], copy=True))
        return outputs


class PolicyIORecorder:
    def __init__(self, directory, contract):
        self.directory = Path(directory)
        self.contract, self.rows, self.pending = serial(contract), [], None

    def begin(self, source, proprio, solution):
        self.pending = dict(infer_stamp=time.monotonic(), source=serial(source),
            proprio=np.array(proprio, copy=True),
            wrench_prediction=np.array(solution.wrench_prediction, copy=True),
            command=np.array(solution.base_command, copy=True),
            arm_position=np.array(solution.arm_position, copy=True))
        self.rows.append(self.pending)

    def publish(self, targets):
        if self.pending is not None:
            self.pending.update(publish_stamp=time.monotonic(), published_targets=np.array(targets, copy=True))
            self.pending = None

    def save(self, outcome):
        contract = dict(self.contract, outcome=outcome, samples=len(self.rows))
        (self.directory/"policy_io_contract.json").write_text(
            json.dumps(serial(contract), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        with (self.directory/"policy_io_steps.jsonl").open("w", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(json.dumps(serial(row), allow_nan=False)+"\n")


def gravity(q):
    w, x, y, z = np.asarray(q)/np.linalg.norm(q)
    return np.array([2*(w*y-x*z), -2*(y*z+w*x), 2*(x*x+y*y)-1])


def difference(a, b):
    delta = np.asarray(a)-np.asarray(b)
    return dict(max_abs=float(np.max(np.abs(delta))), rms=float(np.sqrt(np.mean(delta*delta))))


def quaternion_error_deg(a, b):
    qa, qb = np.asarray(a), np.asarray(b)
    cosine = abs(float(np.dot(qa/np.linalg.norm(qa), qb/np.linalg.norm(qb))))
    return float(np.degrees(2*np.arccos(np.clip(cosine, 0, 1))))


def audit_rows(rows, contract):
    complete = [r for r in rows if all(k in r for k in ("encoder", "policy", "published_targets"))]
    if not complete:
        raise ValueError("No complete inference/publication records")
    errors = {k: [] for k in ("proprio", "history", "policy_input", "published_leg_targets",
                              "previous_action", "managed_phase", "command")}
    expected_history, phase_index, previous_action = None, 0., np.zeros(10)
    defaults, scale = np.array(contract["default_q"]), np.array(contract["action_scale"])
    for row in complete:
        s = row["source"]
        phase_index = 0. if np.linalg.norm(s["command"]) <= 1e-3 else (phase_index+contract["gait"][0]/50.) % 1.
        phase = [np.sin(2*np.pi*phase_index), np.cos(2*np.pi*phase_index)]
        expected = np.concatenate((np.asarray(s["gyro"])*contract["ang_vel_scale"], gravity(s["quaternion"]),
            (np.asarray(s["q"])[:10]-defaults[:10])*contract["dof_pos_scale"],
            np.asarray(s["dq"])[:10]*contract["dof_vel_scale"], s["last_action"], phase, contract["gait"]))
        expected = np.clip(expected, -contract["observation_clip"], contract["observation_clip"]).astype(np.float32)
        proprio = np.array(row["proprio"])
        expected_history = np.tile(proprio, 10) if expected_history is None else np.concatenate((expected_history[42:], proprio))
        encoder = np.array(row["encoder"]["output"]).reshape(-1)
        expected_policy = np.concatenate((encoder, proprio,
            (np.array(row["wrench_prediction"])*[.01, .01, .01, .05, .05, .05]).reshape(-1), row["command"]))
        raw_action = np.array(row["policy"]["output"]).reshape(-1)
        clipped_action = np.clip(raw_action, -5., 5.)
        comparisons = dict(proprio=(proprio, expected), history=(np.array(row["encoder"]["input"]).reshape(-1), expected_history),
            policy_input=(np.array(row["policy"]["input"]).reshape(-1), expected_policy),
            published_leg_targets=(np.array(row["published_targets"])[:10], defaults[:10]+scale*clipped_action),
            previous_action=(s["last_action"], previous_action), managed_phase=(proprio[36:38], phase),
            command=(np.array(row["policy"]["input"]).reshape(-1)[75:78], contract["requested_command"]))
        for key, (actual, target) in comparisons.items():
            errors[key].append(difference(actual, target)["max_abs"])
        previous_action = (np.array(row["published_targets"])[:10]-defaults[:10])/scale
    checks = {k: dict(max_abs=float(max(v)), pass_check=bool(max(v) <= 1e-5)) for k, v in errors.items()}
    actual_parameters = np.r_[contract["ang_vel_scale"], contract["dof_pos_scale"], contract["dof_vel_scale"],
                              contract["gait"], scale, defaults[:10]]
    expected_parameters = np.r_[.25, 1., .05, [.8, .5, .5, .2], np.full(10, .25),
                                [0, 0, -3.14159, 0, 0, 0, 0, 3.14159, 0, 0]]
    parameter_error = difference(actual_parameters, expected_parameters)["max_abs"]
    checks["v3_parameters"] = dict(max_abs=parameter_error, pass_check=parameter_error <= 1e-5)
    return dict(complete_frames=len(complete), incomplete_frames=len(rows)-len(complete),
        sequential_checks_valid=len(complete) == len(rows), checks=checks,
        contract_pass=len(complete) == len(rows) and all(v["pass_check"] for v in checks.values())), complete


def native_comparison(rows, native_rows, contract, native_contract):
    if contract["clock"]["host"] != native_contract["clock"]["host"] or abs(
            contract["clock"]["wall_minus_monotonic_s"]-native_contract["clock"]["wall_minus_monotonic_s"]) > 1.:
        raise ValueError("Native/SDK logs must share a host and monotonic clock session")
    samples = [r for r in native_rows if r.get("native_io") is not None]
    if len(samples) < 3:
        raise ValueError("Simulator log needs --record-policy-io")
    if len({r["segment"] for r in samples}) != 1:
        raise ValueError("Native I/O audit does not support resets within the recording")
    names = native_contract["wire_joint_names"]
    order = [names.index(name) for name in contract["wire_joint_names"]]
    stamps = np.array([r["monotonic_s"] for r in samples])
    if np.any(np.diff(stamps) <= 0):
        raise ValueError("Native timestamps must increase")
    # Pick nearest samples only within 20 ms. Differences retain transport and
    # 100Hz sampling error, so these are diagnostics, not exact tensor checks.
    metrics = {key: [] for key in ("joint_q_rad", "joint_dq_rad_s", "imu_gyro_rad_s", "imu_orientation_deg",
        "sdk_gyro_vs_base_body_rad_s", "sdk_orientation_vs_base_deg", "native_imu_vs_base_deg",
        "published_target_to_received_rad")}
    ages = []
    for row in rows:
        src = row["source"]
        stamp = src["stamp"]
        i = int(np.argmin(np.abs(stamps-stamp)))
        if abs(stamps[i]-stamp) > .02:
            continue
        native = samples[i]["native_io"]
        ages.append(abs(stamps[i]-stamp)*1000)
        metrics["joint_q_rad"].append(np.asarray(src["q"])-np.asarray(native["q"])[order])
        metrics["joint_dq_rad_s"].append(np.asarray(src["dq"])-np.asarray(native["dq"])[order])
        metrics["imu_gyro_rad_s"].append(np.asarray(src["gyro"])-np.asarray(native["imu_gyro"]))
        metrics["imu_orientation_deg"].append(quaternion_error_deg(src["quaternion"], native["imu_quaternion_wxyz"]))
        metrics["sdk_gyro_vs_base_body_rad_s"].append(np.asarray(src["gyro"])-np.asarray(native["base_gyro_body"]))
        metrics["sdk_orientation_vs_base_deg"].append(quaternion_error_deg(src["quaternion"], native["base_quaternion_wxyz"]))
        metrics["native_imu_vs_base_deg"].append(quaternion_error_deg(native["imu_quaternion_wxyz"], native["base_quaternion_wxyz"]))
        # Allow normal command transport delay: choose the best matching received
        # target in [publish, publish+30ms], reporting error without a PASS label.
        lo, hi = np.searchsorted(stamps, [row["publish_stamp"], row["publish_stamp"]+.03])
        matching = [s["native_io"]["received_q"] for s in samples[lo:hi]]
        if matching:
            errors = np.asarray(matching)[:, order]-np.asarray(row["published_targets"])
            metrics["published_target_to_received_rad"].append(errors[np.argmin(np.linalg.norm(errors, axis=1))])
    if not ages:
        raise ValueError("No native/SDK time overlap")
    return dict(samples=len(ages), nearest_sample_age_p95_ms=float(np.percentile(ages, 95)),
        note="Nearest 100Hz simulator sample; includes timing/transport error. Received-target match uses up to 30ms forward window. Not exact timestamp matching or an automatic sensor PASS.",
        differences={key: dict(mean=np.mean(v, axis=0).tolist(),
                               rms_per_component=np.sqrt(np.mean(np.asarray(v)**2, axis=0)).tolist(),
                               rms=float(np.sqrt(np.mean(np.asarray(v)**2))),
                               max_abs=float(np.max(np.abs(v)))) if v else None for key, v in metrics.items()})


def replay(complete, checkpoint):
    import torch
    root = Path(__file__).resolve().parents[2]
    path = root/"exts/bipedal_locomotion/bipedal_locomotion/controllers/paint_teacher_core.py"
    spec = importlib.util.spec_from_file_location("io_frozen_core", path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    frozen = core.FrozenLocomotion(checkpoint, device="cpu")
    results = {}
    with torch.no_grad():
        for name, network in (("encoder", frozen.encoder), ("policy", frozen.actor)):
            inputs = torch.tensor(np.array([np.asarray(r[name]["input"]).reshape(-1) for r in complete]), dtype=torch.float32)
            expected = np.array([np.asarray(r[name]["output"]).reshape(-1) for r in complete])
            actual = np.concatenate([network(batch).numpy() for batch in inputs.split(256)])
            results[name] = dict(**difference(actual, expected), pass_check=bool(np.max(np.abs(actual-expected)) <= 1e-4))
    return results


def analyze(control_dir, sim_dir, output, checkpoint=None):
    control_dir, sim_dir, output = map(Path, (control_dir, sim_dir, output))
    contract = json.loads((control_dir/"policy_io_contract.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in (control_dir/"policy_io_steps.jsonl").read_text().splitlines()]
    report, complete = audit_rows(rows, contract)
    sc = json.loads((sim_dir/"task_contract.json").read_text(encoding="utf-8"))
    if not sc.get("attached") or sc.get("error"):
        raise ValueError("Simulator recording invalid")
    native = [json.loads(x) for x in (sim_dir/"task_steps.jsonl").read_text().splitlines()]
    report.update(outcome=contract["outcome"],
        native_diagnostics=native_comparison(complete, native, contract, sc),
        pytorch_replay=replay(complete, checkpoint or contract["manifest"]["checkpoint"]))
    output.mkdir(parents=True, exist_ok=False)
    (output/"io_audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {output/'io_audit_report.json'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--sim-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", help="Override source .pt location for PyTorch replay")
    args = parser.parse_args()
    analyze(args.control_dir, args.sim_dir, args.output_dir, args.checkpoint)
