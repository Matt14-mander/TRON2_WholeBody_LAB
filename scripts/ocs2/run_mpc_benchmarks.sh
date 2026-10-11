#!/usr/bin/env bash
# Run computational comparisons without Isaac Lab or a TCP bridge.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"
OUTPUT_DIR="${MPC_BENCH_OUTPUT:-logs/ocs2_benchmark/run_$(date -u +%Y%m%d_%H%M%S)_$$}"
SAMPLES="${MPC_BENCH_SAMPLES:-20}"
PERIOD="${MPC_BENCH_PERIOD:-0.1}"
read -r -a VARIANTS <<< "${MPC_BENCH_VARIANTS:-ad_slq analytic_slq_default_init analytic_slq analytic_ilqr analytic_ilqr_rk4 analytic_ilqr_rk4_1thread}"
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
  echo "ERROR: no benchmark variants selected." >&2
  exit 2
fi
for variant in "${VARIANTS[@]}"; do
  case "$variant" in
    ad_slq|analytic_slq_default_init|analytic_slq|analytic_ilqr|analytic_ilqr_rk4|analytic_ilqr_rk4_1thread) ;;
    *) echo "ERROR: unknown benchmark variant: $variant" >&2; exit 2 ;;
  esac
  if [[ -e "$OUTPUT_DIR/$variant.jsonl" || -e "$OUTPUT_DIR/$variant.profile.log" ]]; then
    echo "ERROR: refusing to overwrite $OUTPUT_DIR/$variant results; choose a new MPC_BENCH_OUTPUT." >&2
    exit 2
  fi
done
PACKAGE_PREFIX="$(ros2 pkg prefix tron2_ocs2)"
BENCHMARK_BIN="$PACKAGE_PREFIX/lib/tron2_ocs2/tron2_ocs2_benchmark"
if [[ ! -x "$BENCHMARK_BIN" ]]; then
  echo "ERROR: benchmark executable not found: $BENCHMARK_BIN. Rebuild and source the overlay." >&2
  exit 2
fi
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
printf 'running\n' > "$OUTPUT_DIR/run.status"
interrupt_run() {
  printf 'interrupted\n' > "$OUTPUT_DIR/run.status"
  echo "Benchmark interrupted; partial results retained in $OUTPUT_DIR. No further variants will run." >&2
  exit "$1"
}
trap 'interrupt_run 130' INT
trap 'interrupt_run 143' TERM
echo "Results: $OUTPUT_DIR"
echo "Executable: $BENCHMARK_BIN"
RESULTS=()
FAILED=0
for variant in "${VARIANTS[@]}"; do
  echo "Benchmark $variant: samples=$SAMPLES period=$PERIOD (solver only)"
  echo "Model initialization/code generation precedes the first result; solver progress will appear below."
  RESULTS+=("$OUTPUT_DIR/$variant.jsonl")
  # Execute the resolved binary directly: ros2run's Interrupt message must not
  # be mistaken for a JSON solve record. Keep both streams visible while logging.
  if "$BENCHMARK_BIN" \
      "$OUTPUT_DIR/$variant.info" \
      "$REPO_DIR/robot_description/tron2/SFYG_TRON2A/urdf/robot.urdf" \
      "$OUTPUT_DIR/codegen" "$SAMPLES" "$PERIOD" "${OBSERVATION_ARGS[@]}" \
      2> >(tee "$OUTPUT_DIR/$variant.profile.log" >&2) | tee "$OUTPUT_DIR/$variant.jsonl"; then
    printf 'complete\n' > "$OUTPUT_DIR/$variant.status"
  else
    RESULT_CODE=$?
    printf 'failed exit=%s\n' "$RESULT_CODE" > "$OUTPUT_DIR/$variant.status"
    if [[ "$RESULT_CODE" -eq 130 || "$RESULT_CODE" -eq 143 ]]; then
      interrupt_run "$RESULT_CODE"
    fi
    FAILED=1
    echo "FAILED $variant: inspect $OUTPUT_DIR/$variant.profile.log" >&2
  fi
done
python "$SCRIPT_DIR/summarize_mpc_benchmarks.py" "${RESULTS[@]}" \
  --expected-samples "$SAMPLES" --period "$PERIOD" | tee "$OUTPUT_DIR/summary.txt"
if [[ "$FAILED" -ne 0 ]]; then
  printf 'completed_with_errors\n' > "$OUTPUT_DIR/run.status"
  exit 1
fi
printf 'complete\n' > "$OUTPUT_DIR/run.status"
echo "Completed. Summary: $OUTPUT_DIR/summary.txt"
