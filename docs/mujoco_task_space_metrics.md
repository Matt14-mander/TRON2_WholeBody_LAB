# MuJoCo end-effector and payload metrics

This recorder is opt-in (`sim --task-record-dir NEW_DIRECTORY`). It uses the simulator's actual qpos/qvel with consistent FK in private MjData at 100 Hz. It never writes live state or control. Joint tracking logging in terminal B now includes monotonic timestamps and a same-host clock contract. SDK joint/IMU feedback alone does not provide reliable base world translation, so world metrics come from terminal A, not integrated IMU estimates.

Tracked points:

- End effector: `gripper_base_Link` body origin (no additional tool-center offset).
- Payload: `paint_payload_Link` center of mass; explicitly null if no payload.

Each point records measured and commanded position/quaternion in world/base coordinates, position error in both frames, SO(3) orientation error in radians, world linear velocity and finite-difference acceleration in m/s². Acceleration is the derivative of world velocity over simulator time; it does not contain a gravity offset and does not claim physics-rate impact peaks. The first sample after reset/pause or a >30 ms simulation-time gap has no acceleration.

The command reference uses the **currently received arm joint targets with the measured base pose and other measured joints**. Thus errors quantify arm kinematic tracking relative to its base. World-frame error is the same tracking error rotated into world axes. It is not error relative to an independent stationary-world or MPC task trajectory. Actual world velocities, acceleration and displacement include base motion and should be assessed separately. Quaternion arrays use wxyz; positions use metres; angles are explicitly rad or deg in field names.

`--payload-up-axis 0 0 1` selects the payload body-frame normal for world tilt (also used for the reported EE-frame tilt). The default local +Z is initially about 90° from world vertical with the current configured arm pose. This is reported honestly. `normal_change_from_first_control_deg` measures change from the first aligned control sample's normal, not absolute horizontality or a settled reference. `world_displacement_from_start_m` includes intentional walking and arm motion; it is not a tracking error. PAINT's desired payload-up axis/target orientation must be defined before interpreting absolute tilt as a task success criterion.

Terminal A buffers samples and saves `task_steps.jsonl` and `task_contract.json` on graceful shutdown. Stop with Ctrl+C or close the viewer normally; kill -9 loses buffered data. Terminal B saves its normal arm log. Analyze only after both have stopped. Offline alignment uses the same-host monotonic clock and the preceding controller publication, at most 30 ms old, for approximate phase labels; position targets come directly from the simulator's received command. Resets during the overlapping interval are rejected. Reports include alignment coverage and recording faults, arm outcome, all/after-2-second/last-5-second/phase summaries. Old controller logs without the new clock fields cannot be aligned.

## Server commands

Sync `run_wholebody_v3.py`, `arm_telemetry.py`, `task_space_telemetry.py`, and `test_task_space.sh` into the repository's `scripts/mujoco/` folder. Keep the existing `arm_gravity_compensation.py`. No sibling deploy/simulator sources need editing. The helper retains the diagnostic 150 ms preview age limit, 50 Hz policy, 100 Hz target publication, 0.5 kg load and gravity compensation.

Use these commands in both terminals first:

```bash
conda activate limx
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
```

Terminal A:

```bash
bash scripts/mujoco/test_task_space.sh sim walk_hold
```

Terminal B, after the simulator is ready:

```bash
bash scripts/mujoco/test_task_space.sh control walk_hold
```

When B finishes, stop A with Ctrl+C and wait for `[V3 TASK RECORD] saved`. Then in B:

```bash
bash scripts/mujoco/test_task_space.sh analyze walk_hold
```

Repeat with `arm_walk` in A/B/analyze to compare walking fixed arm versus walking moving arm. Each case needs its own fresh simulator. `hold` and `arm_stand` are also supported. Default root is `logs/paint/task_space_v1`; each output directory must be new. For reruns provide the **same new absolute run root** as the third argument in both terminals and the analysis command:

```bash
bash scripts/mujoco/test_task_space.sh sim walk_hold "$PWD/logs/paint/task_space_v2"
```

Outputs per case:

- `sim/task_steps.jsonl`: simulator samples including preparation/shutdown.
- `control/arm_steps.jsonl` and `arm_report.json`: controller records/outcome.
- `analysis/task_aligned.jsonl`: only samples overlapping the control interval.
- `analysis/task_report.json`: per-axis means/RMS/std/peaks, vector-norm RMS and phase summaries.

Review arm outcome and alignment coverage first. Compare after-2-second and matching phase windows rather than startup-inclusive peaks. These are diagnostic metrics, not a formal PAINT acceptance gate. Local tests cover world/base rotation, payload COM, common base translation invariance, net acceleration and reset handling, missing payload, phase alignment and rejection of old logs. The full local SFYG/0.5 kg model capture took about 0.11 ms/sample; actual server timing still requires verification.

## Reuse existing logs to inspect heading drift and inclination

Sync the additional `scripts/mujoco/analyze_task_heading.py` to the server. No simulator/controller rerun or change to existing records is needed. The script reads `analysis/task_aligned.jsonl` and its sibling `task_report.json` (for the selected normal axis). New output directories are required.

```bash
conda activate limx
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
set -o pipefail
export HEADING_ANALYSIS="$PWD/logs/paint/task_heading_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$HEADING_ANALYSIS"

for case_name in walk_hold arm_walk; do
  python -u scripts/mujoco/analyze_task_heading.py \
    --steps "$PWD/logs/paint/task_space_v1/$case_name/analysis/task_aligned.jsonl" \
    --output-dir "$HEADING_ANALYSIS/$case_name" \
    --settle-s 2 --plot \
    2>&1 | tee "$HEADING_ANALYSIS/${case_name}.log"
done
```

Each case produces `heading_report.json`, `heading_steps.jsonl` and `heading_payload.png`. Plotting uses matplotlib and can be omitted with no `--plot`.

- Heading is unwrapped ZYX yaw of the base forward axis, with net change, fitted slope and heading rate using simulator time. Rate is not the same quantity as body-frame gyroscope z at nonzero roll/pitch.
- Absolute chosen-normal inclination to world vertical is reported with mean, standard deviation and peak-to-peak variation. An RMS near 90° is the horizontal initial axis, not an oscillation amplitude.
- `normal_from_first_control_deg` preserves the old startup reference. `normal_from_settled_deg` and both frame-removed changes share the first sample after settling as their reference.
- `heading_removed_normal_from_settled_deg` removes only Rz(base yaw). Base roll/pitch, intended arm motion and arm tracking error remain.
- `base_relative_normal_from_settled_deg` removes the complete measured base orientation. Commanded arm motion still remains; this is not a tracking error.
- `normal_error_to_command_deg` measures the selected normal against received arm-target FK. `tilt_error_to_command_deg` is signed actual-minus-command inclination; a pure azimuth error may have zero inclination error but nonzero normal error.
- Angles do not decompose additively. Do not subtract these values or turn their ratios into percentages of error attributed to yaw.

The primary comparison is `settled` for both runs, then matching phase windows. Inspect straight-walk heading drift (commanded yaw is zero in these two runs), base roll/pitch and inclination variation before deciding whether to change locomotion or arm dynamic control. A phase summary of normal change retains the global settled reference; it is not re-zeroed at every phase. Intended arm motion and different phase lengths must be considered when comparing it.
