#!/usr/bin/env bash
# Run computational comparisons without Isaac Lab or a TCP bridge.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"
OUTPUT_DIR="${MPC_BENCH_OUTPUT:-logs/ocs2_benchmark}"
SAMPLES="${MPC_BENCH_SAMPLES:-20}"
PERIOD="${MPC_BENCH_PERIOD:-0.1}"
PREPARE_ARGS=(--output-dir "$OUTPUT_DIR")
OBSERVATION_ARGS=()
if [[ $# -gt 1 ]]; then
  echo "usage: bash scripts/ocs2/run_mpc_benchmarks.sh [LIVE_DIAGNOSTIC_JSONL]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  PREPARE_ARGS+=(--diagnostic "$1" --step "${MPC_BENCH_STEP:-100}")
  OBSERVATION_ARGS+=("$OUTPUT_DIR/observation.json")
fi
python "$SCRIPT_DIR/prepare_mpc_benchmarks.py" "${PREPARE_ARGS[@]}"
mkdir -p "$OUTPUT_DIR/codegen"
RESULTS=()
for variant in ad_slq analytic_slq_default_init analytic_slq analytic_ilqr analytic_ilqr_rk4 analytic_ilqr_rk4_1thread; do
  echo "Benchmark $variant: samples=$SAMPLES period=$PERIOD (solver only)"
  RESULTS+=("$OUTPUT_DIR/$variant.jsonl")
  if ! ros2 run tron2_ocs2 tron2_ocs2_benchmark \
      "$OUTPUT_DIR/$variant.info" \
      "$REPO_DIR/robot_description/tron2/SFYG_TRON2A/urdf/robot.urdf" \
      "$OUTPUT_DIR/codegen" "$SAMPLES" "$PERIOD" "${OBSERVATION_ARGS[@]}" \
      >"$OUTPUT_DIR/$variant.jsonl" 2>"$OUTPUT_DIR/$variant.profile.log"; then
    echo "FAILED $variant: inspect $OUTPUT_DIR/$variant.profile.log" >&2
  fi
done
python "$SCRIPT_DIR/summarize_mpc_benchmarks.py" "${RESULTS[@]}" \
  --expected-samples "$SAMPLES" --period "$PERIOD" | tee "$OUTPUT_DIR/summary.txt"
