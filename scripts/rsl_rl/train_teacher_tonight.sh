#!/usr/bin/env bash
# Run in the activated limx environment. Every stage uses a fresh output folder.
set -euo pipefail
cd "$(dirname "$0")/../.."
export LIVESTREAM=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
DEVICE="${TEACHER_DEVICE:-cuda:1}"
ENVS="${TEACHER_ENVS:-1024}"
ITERATIONS="${TEACHER_ITERATIONS:-3000}"
CHECKPOINT="${PAINT_CHECKPOINT:-$PWD/logs/rsl_rl/sfyg_tron_2a_wholebody_paper_v3_flat/2026-10-15_16-38-03/model_20000.pt}"
RUN="${TEACHER_RUN:-$PWD/logs/rsl_rl/teacher_quasistatic_$(date +%Y%m%d_%H%M%S)}"
test -f "$CHECKPOINT"
mkdir -p "$RUN"
printf '%s\n' "$RUN" > logs/rsl_rl/latest_teacher_run.txt
trap 'echo "[TEACHER PIPELINE] stopped; inspect stage logs in $RUN"' ERR
COMMON=(--checkpoint_path "$CHECKPOINT" --headless --device "$DEVICE"
  --payload_mass 0.5 --velocity_scale .15 .08 .25 --arm_scale .1)

check_stage() {
  python - "$RUN/$1" "$2" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
completion = json.loads((p / "completion.json").read_text())
assert completion["frozen_weights_verified"], completion
assert json.loads((p / "wrench_model_check.json").read_text())["valid"]
if sys.argv[2] != "train":
    r = json.loads((p / "evaluation.json").read_text())
    assert r["steps"] == 1500 and r["failures"] == 0 and r["timeouts"] >= 64, r
    assert r["peak_base_tilt_deg"] < 15 and r["peak_payload_tilt_from_reset_deg"] < 35, r
    # Preserve all-transient MSE for diagnosis, but gate tracking only after
    # the gravity feedforward ramp and settling interval of EACH episode.
    assert r["settled_responses"] > 0 and r["metric_settle_s"] == 4., r
    assert r["settled_tracking_mse"] < .01, r
    if sys.argv[2] == "force":
        assert r["force_active_samples"] > 0, r
print("[TEACHER GATE] PASS", p)
PY
}

python -u scripts/rsl_rl/train_paint_teacher.py "${COMMON[@]}" \
  --mode baseline --zero_force --num_envs 64 --steps 1500 \
  --output_dir "$RUN/hold_zero" 2>&1 | tee "$RUN/hold_zero.log"
check_stage hold_zero zero

python -u scripts/rsl_rl/train_paint_teacher.py "${COMMON[@]}" \
  --mode baseline --force_amplitude 1 --torque_amplitude .1 --num_envs 64 --steps 1500 \
  --output_dir "$RUN/hold_force" 2>&1 | tee "$RUN/hold_force.log"
check_stage hold_force force

python -u scripts/rsl_rl/train_paint_teacher.py "${COMMON[@]}" \
  --mode train --force_amplitude 3 --torque_amplitude .2 --num_envs 64 --max_iterations 2 \
  --output_dir "$RUN/ppo_smoke" 2>&1 | tee "$RUN/ppo_smoke.log"
check_stage ppo_smoke train

python -u scripts/rsl_rl/train_paint_teacher.py "${COMMON[@]}" \
  --mode train --force_amplitude 3 --torque_amplitude .2 \
  --initial_force_scale .2 --curriculum_steps 25000 \
  --num_envs "$ENVS" --max_iterations "$ITERATIONS" \
  --output_dir "$RUN/train" 2>&1 | tee "$RUN/train.log"
check_stage train train
for MODE in baseline play; do
  EXTRA=()
  if [[ "$MODE" == play ]]; then
    EXTRA=(--teacher_checkpoint "$RUN/train/model_${ITERATIONS}.pt")
  fi
  python -u scripts/rsl_rl/train_paint_teacher.py "${COMMON[@]}" \
    --mode "$MODE" "${EXTRA[@]}" --force_amplitude 3 --torque_amplitude .2 \
    --num_envs 64 --steps 1500 --output_dir "$RUN/eval_$MODE" \
    2>&1 | tee "$RUN/eval_$MODE.log"
done
python - "$RUN" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for mode in ("baseline", "play"):
    p = root / f"eval_{mode}"
    assert json.loads((p / "completion.json").read_text())["frozen_weights_verified"]
    print(mode, json.loads((p / "evaluation.json").read_text()))
PY
echo "[TEACHER PIPELINE] complete: $RUN/train"
