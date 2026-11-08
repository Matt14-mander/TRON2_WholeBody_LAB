#!/usr/bin/env bash
# Frozen S1 model_500 paired MuJoCo Fx intervention and Isaac/MuJoCo hold measurements.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
RUN="${RUN:-logs/mujoco/student500_arm_transfer_$(date +%Y%m%d_%H%M%S)}"
LOW=logs/rsl_rl/sfyg_robust_5000_gpu1_20261027_131013/model_5000.pt
TEACHER=logs/rsl_rl/sfyg_teacher_t1_20261028_130353/train/model_5000.pt
STUDENT=logs/rsl_rl/sfyg_student_s1_gpu0_500_20261107_141132/train/model_500.pt
mkdir -p "$(dirname "$RUN")"
mkdir "$RUN"
echo "[RUN] $RUN"
python -u scripts/mujoco/diagnose_sfyg_student.py --arm_transfer --seed 42 \
  --output_dir "$RUN/mujoco" 2>&1 | tee "$RUN/mujoco.log"
LIVESTREAM=0 python -u scripts/rsl_rl/train_sfyg_student.py \
  --mode eval --controllers hold --record_arm_hold --forces 0 4 8 \
  --student_checkpoint "$STUDENT" --low_checkpoint "$LOW" --teacher_checkpoint "$TEACHER" \
  --num_envs 1 --episodes 1 --seed 42 --headless --device "${DEVICE:-cuda:0}" \
  --kit_args="--/renderer/multiGpu/enabled=false" --output_dir "$RUN/isaac" 2>&1 | tee "$RUN/isaac.log"
python scripts/rsl_rl/check_sfyg_student_run.py "$RUN/isaac" --mode eval --iteration 500
python -u scripts/rsl_rl/compare_sfyg_arm_hold.py "$RUN" | tee "$RUN/comparison.log"
