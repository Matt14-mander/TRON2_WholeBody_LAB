#!/usr/bin/env bash
# Experimental current-T1 Student; completion is not locomotion acceptance.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
LOW="${LOW:-logs/rsl_rl/sfyg_robust_5000_gpu1_20261027_131013/model_5000.pt}"
TEACHER="${TEACHER:-logs/rsl_rl/sfyg_teacher_t1_20261028_130353/train/model_5000.pt}"
DEVICE="${DEVICE:-cuda:1}"
NUM_ENVS="${NUM_ENVS:-256}"
ITERATIONS="${ITERATIONS:-500}"
RUN="${RUN:-logs/rsl_rl/sfyg_student_s1_$(date +%Y%m%d_%H%M%S)}"
[[ -f "$LOW" && -f "$TEACHER" ]] || { echo 'Missing LOW or TEACHER checkpoint' >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo 'Use a new RUN directory' >&2; exit 1; }
mkdir -p "$RUN"
COMMON=(--low_checkpoint "$LOW" --teacher_checkpoint "$TEACHER" --headless
        --device "$DEVICE" --kit_args="--/renderer/multiGpu/enabled=false")
echo '[STUDENT] 2-update engineering check, then independent training and paired evaluation'
python -u scripts/rsl_rl/train_sfyg_student.py "${COMMON[@]}" \
  --mode train --development --num_envs 4 --max_iterations 2 --save_interval 1 \
  --output_dir "$RUN/interface" 2>&1 | tee "$RUN/interface.console.log"
python scripts/rsl_rl/check_sfyg_student_run.py "$RUN/interface" --mode train --iteration 2
python -u scripts/rsl_rl/train_sfyg_student.py "${COMMON[@]}" \
  --mode train --num_envs "$NUM_ENVS" --max_iterations "$ITERATIONS" --save_interval 100 \
  --output_dir "$RUN/train" 2>&1 | tee "$RUN/train.console.log"
python scripts/rsl_rl/check_sfyg_student_run.py "$RUN/train" --mode train --iteration "$ITERATIONS"
python -u scripts/rsl_rl/train_sfyg_student.py "${COMMON[@]}" \
  --mode eval --student_checkpoint "$RUN/train/model_${ITERATIONS}.pt" \
  --num_envs 4 --episodes 3 --forces 0 2 4 6 8 --controllers student teacher reference hold \
  --output_dir "$RUN/eval" 2>&1 | tee "$RUN/eval.console.log"
python scripts/rsl_rl/check_sfyg_student_run.py "$RUN/eval" --mode eval --iteration "$ITERATIONS"
echo "[DONE] $RUN/eval/evaluation.json (inspect task and estimator metrics; not blanket acceptance)"
