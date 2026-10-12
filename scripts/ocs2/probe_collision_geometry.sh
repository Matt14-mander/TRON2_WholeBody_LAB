#!/usr/bin/env bash
# Compare one measured state with mesh and proxy geometry, without running MPC.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: bash scripts/ocs2/probe_collision_geometry.sh BENCH_RESULT_DIR [OUTPUT_DIR]" >&2
  exit 2
fi
BENCH_DIR="$1"
OUTPUT_DIR="${2:-logs/ocs2_benchmark/collision_probe_$(date -u +%Y%m%d_%H%M%S)_$$}"
TASK_INFO="$BENCH_DIR/analytic_ilqr_8iter_mu100_acc19.info"
for input in "$TASK_INFO" "$BENCH_DIR/observation.json" "$BENCH_DIR/collision_proxy.urdf"; do
  if [[ ! -f "$input" ]]; then echo "ERROR: missing $input" >&2; exit 2; fi
done
for model in mesh aabb; do
  if [[ -e "$OUTPUT_DIR/$model.json" || -e "$OUTPUT_DIR/$model.profile.log" ]]; then
    echo "ERROR: refusing to overwrite $OUTPUT_DIR/$model results." >&2
    exit 2
  fi
done
PACKAGE_PREFIX="$(ros2 pkg prefix tron2_ocs2)"
BENCHMARK_BIN="$PACKAGE_PREFIX/lib/tron2_ocs2/tron2_ocs2_benchmark"
if [[ ! -x "$BENCHMARK_BIN" ]]; then echo "ERROR: missing $BENCHMARK_BIN" >&2; exit 2; fi
for model in mesh aabb; do
  ROBOT_URDF="$REPO_DIR/robot_description/tron2/SFYG_TRON2A/urdf/robot.urdf"
  if [[ "$model" == aabb ]]; then ROBOT_URDF="$BENCH_DIR/collision_proxy.urdf"; fi
  mkdir -p "$OUTPUT_DIR/codegen/$model"
  echo "Inspecting $model: same measured observation, no MPC optimization."
  TRON2_OCS2_COLLISION_ONLY=1 "$BENCHMARK_BIN" "$TASK_INFO" "$ROBOT_URDF" \
    "$OUTPUT_DIR/codegen/$model" 2 0.1 "$BENCH_DIR/observation.json" \
    2> >(tee "$OUTPUT_DIR/$model.profile.log" >&2) | tee "$OUTPUT_DIR/$model.json"
  python - "$OUTPUT_DIR/$model.json" <<'PY'
import json
from pathlib import Path
import sys
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
if len(rows) != 1 or rows[0].get("event") != "collision_inspection":
    raise SystemExit("Expected one collision inspection. Rebuild/source the new native binary.")
PY
done
echo "Collision inspection results: $OUTPUT_DIR"
