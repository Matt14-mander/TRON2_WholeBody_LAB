"""Measure cold/continuous TCP solves from a captured SOLVE observation JSON.

Run on the solver host (NumPy only). Use a diagnostic state from Isaac Lab;
this measures latency, not tracking or walking stability.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

module_path = Path(__file__).parents[2] / "exts/bipedal_locomotion/bipedal_locomotion/controllers/ocs2_interface.py"
spec = importlib.util.spec_from_file_location("benchmark_ocs2_interface", module_path)
interface = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = interface
spec.loader.exec_module(interface)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", required=True, help="JSON with Ocs2MpcObservation field names.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--period", type=float, default=0.1)
    args = parser.parse_args()
    if args.samples < 2 or not np.isfinite(args.period) or args.period <= 0:
        parser.error("At least two samples and a finite positive period are required.")
    data = json.loads(Path(args.observation).read_text(encoding="utf-8"))
    client = interface.Ocs2TcpClient(args.host, args.port, args.timeout)
    durations = []
    warm = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.reset()
        with output.open("w", encoding="utf-8") as stream:
            for index in range(args.samples):
                observation_data = {name: np.asarray(value) if isinstance(value, list) else value for name, value in data.items()}
                observation_data["time"] = float(data["time"]) + index * args.period
                observation = interface.Ocs2MpcObservation(**observation_data)
                started = time.monotonic()
                try:
                    solution = client.solve(observation)
                    duration = time.monotonic() - started
                    durations.append(duration)
                    if index > 0:
                        warm.append(duration)
                    record = {"sample": index, "cold": index == 0, "observation_sim_time": observation.time,
                              "solve_duration_s": duration, "deadline_missed": duration > args.period,
                              "solution_sim_time": solution.time, "base_velocity_command": solution.base_velocity_command.tolist()}
                except (interface.Ocs2BridgeError, ValueError) as error:
                    record = {"sample": index, "error": str(error), "solve_duration_s": time.monotonic() - started}
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(json.dumps(record), flush=True)
                remaining = args.period - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
        if warm:
            print(json.dumps({"warm_p50_s": float(np.percentile(warm, 50)),
                              "warm_p95_s": float(np.percentile(warm, 95)),
                              "warm_max_s": max(warm), "period_s": args.period}))
    finally:
        client.close()


if __name__ == "__main__":
    main()
