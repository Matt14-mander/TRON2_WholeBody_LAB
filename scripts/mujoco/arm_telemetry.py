"""Buffered SDK-rate arm diagnostics; no file IO in the control loop."""
import json
from pathlib import Path

import numpy as np


class ArmTelemetry:
    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = metadata
        self.rows = []
        (self.directory / "contract.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    def append(self, *, time_s, phase, q, dq, target, velocity_target, effort,
               kp, kd, feedback, state_age_s, state_count, diagnostics=None):
        q, dq, target, velocity_target, effort, kp, kd = (
            np.array(x, dtype=float, copy=True) for x in
            (q, dq, target, velocity_target, effort, kp, kd))
        position_error = q - target
        velocity_error = dq - velocity_target
        pd = -kp * position_error - kd * velocity_error
        row = dict(time_s=float(time_s), phase=phase,
                   state_age_s=float(state_age_s), state_count=int(state_count))
        for key, value in dict(q=q, dq=dq, target=target,
                velocity_target=velocity_target, position_error=position_error,
                velocity_error=velocity_error, kp=kp, kd=kd,
                feedforward_effort=effort, estimated_pd_effort=pd,
                estimated_total_effort=pd + effort, actuator_feedback=feedback).items():
            row[key] = None if value is None else np.asarray(value, dtype=float).tolist()
        self.rows.append(row)
        if diagnostics is not None:
            row.update(diagnostics)

    @staticmethod
    def summarize(rows):
        result = dict(samples=len(rows))
        for key in ("position_error", "velocity_error", "feedforward_effort",
                    "estimated_pd_effort", "estimated_total_effort", "actuator_feedback",
                    "gravity_raw_effort", "gravity_applied_effort"):
            valid = [r[key] for r in rows if r.get(key) is not None]
            if not valid:
                result[key] = None
                continue
            values = np.asarray(valid)
            result[key] = dict(samples=len(valid), mean=values.mean(axis=0).tolist(),
                rms=np.sqrt(np.mean(values**2, axis=0)).tolist(),
                peak_abs=np.max(np.abs(values), axis=0).tolist())
        if rows:
            result["state_age_p95_ms"] = float(np.percentile(
                [r["state_age_s"] for r in rows], 95) * 1000)
            result["unique_state_samples"] = len({r["state_count"] for r in rows})
            clipped = [r["gravity_clipped"] for r in rows if "gravity_clipped" in r]
            result["gravity_clipped_fraction"] = np.mean(clipped, axis=0).tolist() if clipped else None
        return result

    def save(self, outcome):
        # Save on normal completion, guard abort and interruption, before SDK os._exit.
        with (self.directory / "arm_steps.jsonl").open("w", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        report = dict(outcome=outcome, joints=self.metadata["joints"],
            all=self.summarize(self.rows),
            after_2s=self.summarize([r for r in self.rows if r["time_s"] >= 2.]),
            last_5s=self.summarize([r for r in self.rows if
                r["time_s"] >= max(2., self.rows[-1]["time_s"] - 5.)]) if self.rows else self.summarize([]),
            phases={phase: self.summarize([r for r in self.rows if r["phase"] == phase])
                    for phase in sorted({r["phase"] for r in self.rows})})
        (self.directory / "arm_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return report
