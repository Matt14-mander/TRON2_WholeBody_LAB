#!/usr/bin/env bash
# Same guide teacher environment and low checkpoint; no PPO or learned arm actions.
set -euo pipefail
cd "$(dirname "$0")/../.."
export LIVESTREAM=0
DEVICE="${TEACHER_DEVICE:-cuda:1}"
CHECKPOINT="${PAINT_CHECKPOINT:-$PWD/logs/rsl_rl/sfyg_tron_2a_wholebody_paper_v3_flat/2026-10-15_16-38-03/model_20000.pt}"
RUN="${TEACHER_PROBE_RUN:-$PWD/logs/rsl_rl/teacher_support_probe_$(date +%Y%m%d_%H%M%S)}"
test -f "$CHECKPOINT"
test ! -e "$RUN"
mkdir -p "$RUN"
for CASE in hold forward backward left right yaw_left yaw_right; do
  case "$CASE" in
    hold) CMD=(0 0 0) ;;
    forward) CMD=(.04 0 0) ;;
    backward) CMD=(-.04 0 0) ;;
    left) CMD=(0 .03 0) ;;
    right) CMD=(0 -.03 0) ;;
    yaw_left) CMD=(0 0 .04) ;;
    yaw_right) CMD=(0 0 -.04) ;;
  esac
  python -u scripts/rsl_rl/train_paint_teacher.py \
    --mode probe --teacher_contract guide --zero_force --probe_command "${CMD[@]}" \
    --checkpoint_path "$CHECKPOINT" --device "$DEVICE" --headless \
    --num_envs "${TEACHER_PROBE_ENVS:-64}" --steps 1500 --seed 42 \
    --payload_mass .5 --velocity_scale .15 .08 .25 --arm_scale .1 \
    --output_dir "$RUN/$CASE" 2>&1 | tee "$RUN/$CASE.log"
done
python - "$RUN" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
summary = {}
for case in ('hold','forward','backward','left','right','yaw_left','yaw_right'):
    p = root / case
    completion = json.loads((p/'completion.json').read_text())
    if not completion['frozen_weights_verified']:
        raise RuntimeError(f'{case}: frozen weights were modified')
    r = json.loads((p/'evaluation.json').read_text())
    summary[case] = {k:r[k] for k in ('probe_command','settled_mean_executed_base_command',
        'settled_mean_base_vx_vy_wz','settled_tracking_mse','settled_responses',
        'failures','timeouts','peak_base_tilt_deg')}
(root/'probe_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
print('Saved:',root/'probe_summary.json')
PY
