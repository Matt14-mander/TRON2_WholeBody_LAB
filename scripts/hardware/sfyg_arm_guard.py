"""Same-host arm heartbeat gate; never treats a preflight checkbox as feedback."""
import json
import math
import time


class ArmGuard:
    def __init__(self, path, policy_sha):
        self.path, self.policy_sha, self.session = path, policy_sha, None

    def check(self, now=None):
        now = time.monotonic() if now is None else now
        data = json.loads(self.path.read_text(encoding="utf-8"))
        age = now - float(data["monotonic_s"])
        if not math.isfinite(age) or not 0 <= age <= .25:
            raise RuntimeError("Arm heartbeat stale or clock invalid")
        if (data.get("schema") != "sfyg_arm_hold_v1" or data.get("state") != "HOLD"
                or data.get("execute") is not True or data.get("ready") is not True
                or data.get("policy_sha256") != self.policy_sha):
            raise RuntimeError("Arm is not actively holding verified carry pose")
        session = data.get("session")
        if not isinstance(session, str) or not session:
            raise RuntimeError("Invalid arm session")
        if self.session is not None and session != self.session:
            raise RuntimeError("Arm controller restarted; restart leg lifecycle")
        self.session = session
