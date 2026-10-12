"""Run exported intent estimation on JSONL telemetry; never commands hardware.

Each input line: {time, q:[6], dq:[6], previous_applied_position:[6], reset?:bool}.
Time is in seconds. Input joint order/units must match model metadata.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exts/bipedal_locomotion/bipedal_locomotion/controllers"))
from paint_intent import IntentHistory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input", default="-", help="JSONL file, or - for stdin.")
    args = parser.parse_args()
    directory = Path(args.model_dir)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    with (directory / "intent_estimator.ts").open("rb") as stream:
        model = torch.jit.load(stream, map_location="cpu").eval()
    history = IntentHistory(metadata["history"])
    stream = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")
    last_time = None
    try:
        for line in stream:
            item = json.loads(line)
            time = float(item["time"])
            if not np.isfinite(time):
                raise ValueError("Nonfinite telemetry time.")
            reset = bool(item.get("reset", False))
            if reset or (last_time is not None and not np.isclose(time - last_time, metadata["step_dt"], rtol=0.1, atol=1e-4)):
                history.reset()
            features = history.update(item["q"], item["dq"], item["previous_applied_position"])
            last_time = time
            prediction = None
            if features is not None:
                with torch.inference_mode():
                    prediction = model(torch.from_numpy(features[None])).numpy()[0].tolist()
            print(json.dumps({"time": time, "valid": prediction is not None, "interaction_wrench_payload_local": prediction}), flush=True)
    finally:
        if stream is not sys.stdin:
            stream.close()


if __name__ == "__main__":
    main()
