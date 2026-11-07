#!/usr/bin/env bash
# Actor-only warm start. All evaluation actions come from the exported architecture's native actor.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
LOW="${LOW:-logs/rsl_rl/sfyg_robust_5000_gpu1_20261027_131013/model_5000.pt}"
TEACHER="${TEACHER:-logs/rsl_rl/sfyg_teacher_t1_20261028_130353/train/model_5000.pt}"
STUDENT="${STUDENT:-logs/rsl_rl/sfyg_student_s1_gpu0_500_20261107_141132/train/model_500.pt}"
RUN="${RUN:-logs/rsl_rl/sfyg_actor4n_$(date +%Y%m%d_%H%M%S)}"
ITERATIONS="${ITERATIONS:-200}"
for ckpt in "$LOW" "$TEACHER" "$STUDENT"; do
  [[ -r "$ckpt" ]] || { echo "Missing checkpoint: $ckpt" >&2; exit 1; }
done
mkdir -p "$(dirname "$RUN")"
mkdir "$RUN"
COMMON=(--low_checkpoint "$LOW" --teacher_checkpoint "$TEACHER" --headless
        --device "${DEVICE:-cuda:0}" --kit_args="--/renderer/multiGpu/enabled=false")
python -u scripts/rsl_rl/train_sfyg_student.py "${COMMON[@]}" --mode train --finetune_actor_4n \
  --student_checkpoint "$STUDENT" --learning_rate 3e-5 --max_iterations "$ITERATIONS" \
  --num_envs "${NUM_ENVS:-256}" --save_interval 50 --output_dir "$RUN/train"
python scripts/rsl_rl/check_sfyg_student_run.py "$RUN/train" --mode train --iteration "$ITERATIONS"
python - "$RUN/train/training.jsonl" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    rows = [json.loads(line) for line in f]
print('4 N plateau / auxiliary frames across rollouts:',
      sum(r['four_plateau_frames'] for r in rows), '/', sum(r['raw_reference_frames'] for r in rows))
PY
for seed in 42 142; do
  OUT="$RUN/eval_seed${seed}"
  python -u scripts/rsl_rl/train_sfyg_student.py "${COMMON[@]}" --mode eval --controllers student \
    --student_checkpoint "$RUN/train/model_${ITERATIONS}.pt" --forces 0 2 4 6 8 \
    --num_envs 4 --episodes 3 --seed "$seed" --output_dir "$OUT"
  python scripts/rsl_rl/check_sfyg_student_run.py "$OUT" --mode eval --iteration "$ITERATIONS"
  python - "$OUT/evaluation.json" "$seed" <<'PY'
import json, math, sys
with open(sys.argv[1]) as f:
    reports = json.load(f)['reports']
for force in (0,2,4,6,8):
    group = [r for r in reports if r['force_n']==force]
    full = [r for r in group if r['completed'] and r['plateau_mean_velocity'] is not None]
    vx = sum(r['plateau_mean_velocity'][0] for r in full)/len(full) if full else float('nan')
    rmse = math.sqrt(sum(r['plateau_velocity_rmse'][0]**2 for r in full)/len(full)) if full else float('nan')
    stops = sum(r['low_speed_events'] for r in full) if force==4 and full else '-'
    print(f"[NATIVE] seed={sys.argv[2]} F={force}N passed={sum(r['gate']['passed'] for r in group)}/{len(group)} "
          f"release={sum(r['gate']['release'] for r in group)}/{len(group)} complete={len(full)}/{len(group)} "
          f"vx={vx:.4f} rmse={rmse:.4f} stops={stops}")
PY
done
