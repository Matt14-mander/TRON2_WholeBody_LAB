"""Offline audit of a shadow log. Never approves physical mapping from static data."""
import argparse
import json
from pathlib import Path
import numpy as np
from shadow_sfyg_locomotion import Coordinates, LEG_NAMES


def audit(log, coordinates, bundle):
    rows = [json.loads(s) for s in log.read_text().splitlines() if s.strip()]
    if not rows:
        raise ValueError("Empty shadow log")
    c = Coordinates(coordinates)
    cfg = json.loads((bundle / "deployment.json").read_text())
    if cfg["leg_names"] != LEG_NAMES:
        raise ValueError("Unexpected policy joint order")
    converted = []
    alternatives = {k: [] for k in ("wxyz", "xyzw")}
    for r in rows:
        # Historical field name was misleading: the values are raw SDK order.
        joint = (0, 0, r["sdk_q"], r["sdk_dq"])
        raw = r["sdk_quat_raw"] if "sdk_quat_raw" in r else r["sdk_quat_wxyz"]
        imu = (0, 0, raw, r["sdk_gyro"])
        converted.append(c.convert(joint, imu))
        for order in alternatives:
            alternate = Coordinates(coordinates)
            alternate.quaternion_order = order
            alternatives[order].append(alternate.convert(joint, imu)[3])
    q = np.asarray([v[0] for v in converted])
    dq = np.asarray([v[1] for v in converted])
    gravity = np.asarray([v[3] for v in converted])
    default = np.asarray(cfg["defaults"][:10])
    result = dict(hardware_mapping_verified=False, frames=len(rows),
                  quaternion_order=c.quaternion_order,
                  gravity_mean=gravity.mean(0).tolist(),
                  gravity_by_order={k: np.mean(v, axis=0).tolist() for k, v in alternatives.items()},
                  joints=[dict(name=n, sdk_index=c.order[i], sign=float(c.sign[i]),
                               zero_rad=float(c.zero[i]), mean_q=float(q[:, i].mean()),
                               max_abs_dq=float(abs(dq[:, i]).max()),
                               max_prepare_error=float(abs(q[:, i]-default[i]).max()))
                          for i, n in enumerate(LEG_NAMES)])
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=Path, required=True)
    p.add_argument("--coordinates", type=Path, required=True)
    p.add_argument("--bundle_dir", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.steps, args.coordinates, args.bundle_dir), indent=2))
