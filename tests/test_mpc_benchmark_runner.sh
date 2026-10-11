#!/usr/bin/env bash
# Exercise the runner's process/log lifecycle without ROS, codegen or a solver.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MPC_RUNNER_FIXTURE="$REPO_DIR/.test-temp/mpc_runner_$$"
mkdir -p "$MPC_RUNNER_FIXTURE/package/lib/tron2_ocs2"
export MPC_RUNNER_EXECUTABLE="$MPC_RUNNER_FIXTURE/package/lib/tron2_ocs2/tron2_ocs2_benchmark"
cat > "$MPC_RUNNER_EXECUTABLE" <<'MOCK'
#!/usr/bin/env bash
printf '%s\n' "$1" >> "$MPC_RUNNER_FIXTURE/invocations"
printf '[MOCK_PROFILE] starting\n' >&2
if [[ "${MPC_FAKE_EXIT:-0}" -ne 0 ]]; then exit "$MPC_FAKE_EXIT"; fi
for sample in 0 1 2; do
  printf '{"sample":%s,"solve_duration_s":0.05}\n' "$sample"
done
MOCK
chmod +x "$MPC_RUNNER_EXECUTABLE"
ros2() { printf '%s\n' "$MPC_RUNNER_FIXTURE/package"; }
python() {
  local script="$1"
  shift
  case "$script" in
    */prepare_mpc_benchmarks.py)
      local output=""
      while [[ $# -gt 0 ]]; do
        if [[ "$1" == --output-dir ]]; then output="$2"; shift; fi
        shift
      done
      mkdir -p "$output"
      for variant in analytic_ilqr analytic_slq; do
        printf 'mock config\n' > "$output/$variant.info"
      done
      ;;
    */summarize_mpc_benchmarks.py) printf '{"mock_summary":true}\n' ;;
    *) echo "Unexpected Python invocation" >&2; return 2 ;;
  esac
}
export -f ros2 python
RUNNER="$REPO_DIR/scripts/ocs2/run_mpc_benchmarks.sh"
export MPC_BENCH_SAMPLES=3 MPC_BENCH_VARIANTS=analytic_ilqr
export MPC_BENCH_OUTPUT="$MPC_RUNNER_FIXTURE/complete"
bash "$RUNNER" > "$MPC_RUNNER_FIXTURE/complete.console" 2>&1
[[ "$(cat "$MPC_BENCH_OUTPUT/run.status")" == complete ]]
[[ "$(wc -l < "$MPC_BENCH_OUTPUT/analytic_ilqr.jsonl")" -eq 3 ]]
[[ "$(wc -l < "$MPC_RUNNER_FIXTURE/invocations")" -eq 1 ]]
if bash "$RUNNER" > "$MPC_RUNNER_FIXTURE/overwrite.console" 2>&1; then
  echo "FAIL: overwrote existing results" >&2; exit 1
else
  [[ $? -eq 2 ]]
fi
[[ "$(wc -l < "$MPC_BENCH_OUTPUT/analytic_ilqr.jsonl")" -eq 3 ]]
[[ "$(wc -l < "$MPC_RUNNER_FIXTURE/invocations")" -eq 1 ]]
export MPC_BENCH_OUTPUT="$MPC_RUNNER_FIXTURE/interrupted"
export MPC_BENCH_VARIANTS='analytic_ilqr analytic_slq' MPC_FAKE_EXIT=130
if bash "$RUNNER" > "$MPC_RUNNER_FIXTURE/interrupted.console" 2>&1; then
  echo "FAIL: interruption reported success" >&2; exit 1
else
  [[ $? -eq 130 ]]
fi
[[ "$(cat "$MPC_BENCH_OUTPUT/run.status")" == interrupted ]]
[[ ! -e "$MPC_BENCH_OUTPUT/analytic_slq.jsonl" ]]
[[ ! -e "$MPC_BENCH_OUTPUT/summary.txt" ]]
[[ "$(wc -l < "$MPC_RUNNER_FIXTURE/invocations")" -eq 2 ]]
[[ ! -s "$MPC_BENCH_OUTPUT/analytic_ilqr.jsonl" ]]
echo "PASS: selected variant, complete logs, overwrite refusal, and interruption stops subsequent variants."
