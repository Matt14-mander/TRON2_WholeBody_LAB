"""Durable collection status written before risky simulator calls."""
import json
import os
from pathlib import Path
import traceback


class PaintRunStatus:
    def __init__(self, output):
        self.path = Path(output) / "run_status.json"
        self.data = {"complete": False, "state": "running", "stage": "initializing", "accepted": 0, "total": 0}
        self.mark("initializing")

    def mark(self, stage, **fields):
        self.data.update(fields)
        self.data["stage"] = stage
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)
        print(f"[PAINT_STAGE] {stage} episode={self.data.get('episode_id', '-')} step={self.data.get('step', '-')}", flush=True)

    def fail(self, error):
        # Print while Isaac Sim is still alive: cleanup may terminate the process
        # before Python's usual unhandled-exception printer is reached.
        traceback.print_exception(type(error), error, error.__traceback__)
        self.mark(self.data["stage"], complete=False, state="failed", error_type=type(error).__name__, error=str(error))
