#!/usr/bin/env bash
# Same run root in terminals A/B. Stop A after control exits, then analyze.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
mode="${1:?Usage: test_task_space.sh sim|control|analyze CASE [RUN_ROOT]}"
case_name="${2:?Case: hold|walk_hold|arm_stand|arm_walk}"
run_root="${3:-$repo_root/logs/paint/task_space_v1}"
case "$case_name" in
  hold|arm_stand) command=(0 0 0) ;;
  walk_hold|arm_walk) command=(0.15 0 0) ;;
  *) echo "Unknown case: $case_name" >&2; exit 2 ;;
esac
motion=()
if [[ "$case_name" == arm_stand || "$case_name" == arm_walk ]]; then
  motion=(--arm-test-offsets 0 0.1 0 -0.05 0 0 --arm-test-start-delay 7
          --arm-test-move-duration 3 --arm-test-hold-duration 2)
fi
sim_root="${SIM_ROOT:-$HOME/workspace/limx_deploy/tron2_mujoco_sim}"
deploy_root="${DEPLOY_ROOT:-$HOME/workspace/limx_deploy/tron2_rl_deploy_python}"
model_root="${V3_MODEL_ROOT:-$deploy_root/controllers/model/v3_20000_20261016_181915}"
payload=(--payload-mass 0.5 --payload-dimensions 0.20 0.12 0.08 --payload-offset 0.15 0 0)
mkdir -p "$run_root"
case "$mode" in
  sim)
    python -u scripts/mujoco/run_wholebody_v3.py sim --sim-root "$sim_root" \
      --arm-pose configured "${payload[@]}" \
      --task-record-dir "$run_root/$case_name/sim" \
      2>&1 | tee "$run_root/${case_name}_sim.log"
    ;;
  control)
    test -f "$run_root/$case_name/sim/task_contract.json" || {
      echo "Start the matching simulator case in terminal A first." >&2; exit 2;
    }
    python -u scripts/mujoco/run_wholebody_v3.py control \
      --deploy-root "$deploy_root" --sim-root "$sim_root" --model-root "$model_root" \
      --arm-pose configured "${payload[@]}" --command "${command[@]}" --duration 30 \
      --low-wrench plan --preview-execution async --preview-max-age 0.15 \
      --publish-frequency 100 --arm-gravity-compensation \
      --arm-gravity-ramp-time 2 --arm-gravity-max-effort 30 "${motion[@]}" \
      --record-dir "$run_root/$case_name/control" \
      2>&1 | tee "$run_root/${case_name}_control.log"
    ;;
  analyze)
    test -f "$run_root/$case_name/sim/task_steps.jsonl" || {
      echo "Stop terminal A with Ctrl+C and wait for TASK RECORD saved before analysis." >&2; exit 2;
    }
    python scripts/mujoco/task_space_telemetry.py \
      --sim-record-dir "$run_root/$case_name/sim" \
      --arm-record-dir "$run_root/$case_name/control" \
      --output-dir "$run_root/$case_name/analysis"
    ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
