# MuJoCo arm precision baseline

Add `--record-dir NEW_DIRECTORY` to `run_wholebody_v3.py control`. Existing directories are rejected to prevent overwriting a trial. No policy weights, gains, targets or feedforward are changed by recording. Disk output happens after the loop; samples are buffered in memory, so use finite diagnostic runs.

Files:

- `contract.json`: arguments, joint order, units and telemetry definitions.
- `arm_steps.jsonl`: one sample per policy target publication (100 Hz with the command below); excludes pose preparation and shutdown damping.
- `arm_report.json`: run outcome, complete-run statistics, statistics after 2 seconds, last 5 seconds and each arm-test phase. Missing torque feedback is null, never substituted with estimated torque.

All six-element arrays follow `joints` in the contract. Position error is measured minus commanded (rad); velocity error has the same sign (rad/s). Torque units are Nm for this SFYG unit-gear motor model. `feedforward_effort` is the outgoing SDK effort. `estimated_pd_effort` is Kp*(target-q)+Kd*(velocity_target-dq); `estimated_total_effort` includes feedforward. `actuator_feedback` is received SDK state.tau, reordered using the SDK motor names. The checked SFYG simulator channel publishes MuJoCo actuator_force. It is neither joint reaction torque nor a real hardware measurement.

Feedback predates the outgoing command; estimates and feedback are not physics-synchronous. `state_age_s` measures local callback receipt age, not transport latency. `state_count` identifies repeated feedback samples. These logs cannot certify physics-rate torque peaks or saturation. Reports include per-joint signed mean, RMS and absolute peak for each error/torque field. `last_5s` helps inspect persistent loaded-pose offsets; phase reports separate warmup/outbound/hold/return/complete/aborted. A completed control loop alone does not imply the requested arm sequence finished: check the outcome flags.

## Server terminal A — restart before each trial

```bash
conda activate limx
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
python scripts/mujoco/run_wholebody_v3.py sim \
  --sim-root "$HOME/workspace/limx_deploy/tron2_mujoco_sim" \
  --arm-pose configured --payload-mass 0.5 \
  --payload-dimensions 0.20 0.12 0.08 --payload-offset 0.15 0 0
```

## Server terminal B — define once in this terminal

```bash
conda activate limx
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
set -o pipefail
export ARM_DIAG="$PWD/logs/paint/arm_precision_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARM_DIAG"
run_arm_record() {
  local case_name="$1"
  shift
  python -u scripts/mujoco/run_wholebody_v3.py control \
    --deploy-root "$HOME/workspace/limx_deploy/tron2_rl_deploy_python" \
    --sim-root "$HOME/workspace/limx_deploy/tron2_mujoco_sim" \
    --model-root "$HOME/workspace/limx_deploy/tron2_rl_deploy_python/controllers/model/v3_20000_20261016_181915" \
    --arm-pose configured --payload-mass 0.5 \
    --payload-dimensions 0.20 0.12 0.08 --payload-offset 0.15 0 0 \
    --duration 30 --low-wrench plan --preview-execution async \
    --publish-frequency 100 --record-dir "$ARM_DIAG/$case_name" \
    "$@" 2>&1 | tee "$ARM_DIAG/${case_name}.log"
}
run_arm_record hold --command 0 0 0
```

After hold finishes, restart terminal A's simulator, then run in the same B terminal:

```bash
run_arm_record arm_stand --command 0 0 0 \
  --arm-test-offsets 0 0.1 0 -0.05 0 0 \
  --arm-test-start-delay 7 --arm-test-move-duration 3 --arm-test-hold-duration 2
```

Restart A again, then run:

```bash
run_arm_record arm_walk --command 0.15 0 0 \
  --arm-test-offsets 0 0.1 0 -0.05 0 0 \
  --arm-test-start-delay 7 --arm-test-move-duration 3 --arm-test-hold-duration 2
```

Review `arm_report.json` for each case and `[V3 RESULT]` timing. Preserve approximately 50 Hz inference before comparing tracking. Share the three reports first; retain JSONL for time-series plots and signed arm2/arm3 load-error analysis.
