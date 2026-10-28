#!/usr/bin/env bash
# Run inside the existing limx environment. Does not install packages or stop jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOW="${LOW:-$PWD/logs/rsl_rl/sfyg_robust_5000_gpu1_20261027_131013/model_5000.pt}"
DEVICE="${DEVICE:-cuda:1}"
NUM_ENVS="${NUM_ENVS:-512}"
ITERATIONS="${ITERATIONS:-5000}"
RUN="${RUN:-logs/rsl_rl/sfyg_teacher_t1_$(date +%Y%m%d_%H%M%S)}"
[[ -f "$LOW" ]] || { echo "Missing frozen SFYG checkpoint: $LOW" >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo "Use a new RUN directory: $RUN" >&2; exit 1; }
mkdir -p "$RUN"
COMMON=(--low_checkpoint "$LOW" --delay_steps 2 --headless --device "$DEVICE"
        --kit_args="--/renderer/multiGpu/enabled=false")

echo "[T0/T1] RUN=$RUN DEVICE=$DEVICE LOW=$LOW"
echo "[T0] native force/reset/interface checks (not a walking acceptance)"
python -u scripts/rsl_rl/train_sfyg_teacher.py --mode smoke "${COMMON[@]}" \
  --num_envs 4 --steps 128 --output_dir "$RUN/interface" 2>&1 | tee "$RUN/interface.console.log"
python scripts/rsl_rl/check_sfyg_teacher_run.py "$RUN/interface" --mode smoke

echo "[T0] paired hold/reference at 0/2/4/6/8 N, and zero-force velocity probes"
python -u scripts/rsl_rl/train_sfyg_teacher.py --mode baseline "${COMMON[@]}" \
  --num_envs 4 --episodes 2 --seed 42 --output_dir "$RUN/baseline" 2>&1 | tee "$RUN/baseline.console.log"
python scripts/rsl_rl/check_sfyg_teacher_run.py "$RUN/baseline" --mode baseline

# A failed/incomplete T0 exits nonzero above. The train command also independently
# validates the report and its frozen checkpoint/settings binding.
echo "[T1] T0 passed; initialize a NEW teacher, keep the low-level frozen"
python -u scripts/rsl_rl/train_sfyg_teacher.py --mode train "${COMMON[@]}" \
  --baseline_report "$RUN/baseline/evaluation.json" \
  --num_envs "$NUM_ENVS" --max_iterations "$ITERATIONS" --save_interval 100 \
  --seed 42 --output_dir "$RUN/train" 2>&1 | tee "$RUN/train.console.log"
python scripts/rsl_rl/check_sfyg_teacher_run.py "$RUN/train" --mode train

echo "[DONE] teacher: $RUN/train/model_${ITERATIONS}.pt"
echo "[DONE] raw actor + execution manifest: $RUN/train/exported/model_${ITERATIONS}"
