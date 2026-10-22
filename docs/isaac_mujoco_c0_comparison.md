# Isaac Lab / MuJoCo C0 fixed-arm comparison

Purpose: locate whether sustained zero-turn-command yaw drift also appears in Isaac Lab's full-arm support environment. This is a matched **task-condition** diagnostic, not a claim of identical dynamics. Keep v3 weights, gains and physical assets unchanged in C0.

Current differences that must not be hidden:

- Isaac support has no arm gravity effort; its action term sets arm effort to zero each physics tick. C0 therefore explicitly disables the optional MuJoCo arm gravity effort. Previously collected MuJoCo gravity-FF runs cannot substitute for C0.
- Isaac uses implicit PD at 200 Hz, MuJoCo MIT PD at 1000 Hz. Policy is 50 Hz in both. Isaac updates scripted high-level targets at 25 Hz; MuJoCo updates the continuous reference at 50 Hz. Compare steady 5–29 s, not transient curves sample-for-sample.
- Both use the continuous quintic walk reference: zero until 3 s, ramp 3–4 s. The new MuJoCo `--command-profile paint` is fixed-arm-only; it is incompatible with the fixed-command I/O audit and OCS2/arm-test replay.
- Same 0.5 kg box geometry/offset, but native arm inertias and COMs can differ. Earlier local native model checks gave added mass including payload 7.265 kg in MuJoCo versus approximately 7.661 kg in Isaac. Each backend's gravity preview models its own actual added bodies. These are retained differences to investigate after C0, not silently corrected parameters.

## Sync

Copy these files to their corresponding server paths:

- `scripts/rsl_rl/validate_paint_support.py` (now logs root quaternion and body angular velocity)
- `scripts/mujoco/run_wholebody_v3.py` (new paint velocity reference)
- `scripts/mujoco/test_task_space.sh` (optional gravity FF/profile/wrench environment switches)
- `scripts/mujoco/compare_support_backends.py` (offline C0 validation and pose comparison)

Keep the previously installed task-space and arm-recording modules. No remote control/deploy repository edits are required.

## 1. Isaac: hold and forward, one trial each

Stop other diagnostic simulator/control processes first to avoid unnecessary CPU scheduling interference. In the Isaac terminal:

```bash
conda activate limx
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
set -o pipefail

export C0_ROOT="$PWD/logs/paint/backend_c0_v1"
mkdir -p "$C0_ROOT"

LIVESTREAM=0 python -u scripts/rsl_rl/validate_paint_support.py \
  --checkpoint_path "$PWD/logs/rsl_rl/sfyg_tron_2a_wholebody_paper_v3_flat/2026-10-15_16-38-03/model_20000.pt" \
  --cases hold forward --episodes 1 --duration 30 \
  --arm_pose nominal --payload_mode attached \
  --payload_mass 0.5 --payload_dimensions 0.20 0.12 0.08 \
  --payload_offset 0.15 0 0 --low_wrench gravity --force 0 0 0 \
  --headless --device cuda:0 \
  --output_dir "$C0_ROOT/isaac" \
  2>&1 | tee "$C0_ROOT/isaac.log"
```

Inspect `report.json`, frozen-weights verification and `failure.json` if present. Do not trust Kit's process exit code alone. A fall/short episode is a result to diagnose, not a reason to relax termination. Do not run offline steady comparison for an insufficient-duration trial.

## 2. MuJoCo hold: terminal A and B

In both terminals first activate limx and cd to the repository.

Terminal A:

```bash
bash scripts/mujoco/test_task_space.sh sim hold \
  "$PWD/logs/paint/backend_c0_v1/mujoco"
```

Terminal B after A is ready:

```bash
ARM_GRAVITY_COMPENSATION=0 COMMAND_PROFILE=paint LOW_WRENCH=gravity \
  bash scripts/mujoco/test_task_space.sh control hold \
  "$PWD/logs/paint/backend_c0_v1/mujoco"
```

When B ends, Ctrl+C in A and wait for TASK RECORD saved.

## 3. MuJoCo forward: fresh simulator A

Terminal A:

```bash
bash scripts/mujoco/test_task_space.sh sim walk_hold \
  "$PWD/logs/paint/backend_c0_v1/mujoco"
```

Terminal B:

```bash
ARM_GRAVITY_COMPENSATION=0 COMMAND_PROFILE=paint LOW_WRENCH=gravity \
  bash scripts/mujoco/test_task_space.sh control walk_hold \
  "$PWD/logs/paint/backend_c0_v1/mujoco"
```

Again stop A gracefully after B ends. Do not set POLICY_IO_AUDIT=1 for this changing-command profile.

## 4. Offline comparisons, no new simulator needed

```bash
export C0_ROOT="$PWD/logs/paint/backend_c0_v1"

python scripts/mujoco/compare_support_backends.py \
  --isaac-dir "$C0_ROOT/isaac" \
  --mujoco-case "$C0_ROOT/mujoco/hold" \
  --case hold --output-dir "$C0_ROOT/comparison/hold"

python scripts/mujoco/compare_support_backends.py \
  --isaac-dir "$C0_ROOT/isaac" \
  --mujoco-case "$C0_ROOT/mujoco/walk_hold" \
  --case forward --output-dir "$C0_ROOT/comparison/forward"
```

The tool validates checkpoint hash, payload configuration, nominal arm targets, gravity preview, no external interaction, MuJoCo FF off and paint command profile. It checks the actual settled Isaac command and steady-window coverage. It compares yaw net change/slope, roll/pitch mean/std, displacement and per-joint arm error. Actual episode outcomes remain in the report. Native MuJoCo command selection uses wall time; motion rates use simulator time. Pose matches are statistical, not trajectory identities.

All output directories must be new. For reruns replace backend_c0_v1 with backend_c0_v2 consistently in every command. Final outputs are `comparison/hold/backend_comparison.json` and `comparison/forward/backend_comparison.json`.

## Interpretation and next layer

- Hold stable in both, forward drifts similarly in both: prioritize frozen policy behavior under the full articulated arm and the physical wrench distribution; verify the Isaac teacher wrapper history/phase contract before deciding to retrain.
- Hold stable in both, forward primarily drifts in MuJoCo: inspect native asset masses/COM/inertia, actuator limits/armature/PD implementation, contact/friction and command timing one category at a time. This does not by itself identify a simulator software bug.
- Hold already differs: resolve static carry posture, model gravity input and arm compliance first.
- Episodes fail or task-condition checks fail: address that mismatch before interpreting steady yaw statistics.

After C0, add equivalent model-based arm gravity effort to Isaac before the next FF-on/FF-on comparison. Do not compare Isaac FF-off directly with MuJoCo FF-on and call it a matched test. Later isolate native physics step/actuators/assets with declared changes; do not simultaneously change every parameter or adjust a fixed yaw bias to hide the discrepancy.
