# Dynamic v3 deployment I/O audit

Static hashes, model dimensions and the gravity preview have passed. They do not show which tensors the running policy actually receives. This opt-in audit records the real SDK observation source, ONNX calls and SDK target publication without changing policy computation, targets or gains. Run the straight-walk fixed-arm case first to isolate deployment from trajectory generation.

Sync these four files into the server repository's `scripts/mujoco/`:

- `run_wholebody_v3.py`
- `task_space_telemetry.py`
- `policy_io_audit.py`
- `test_task_space.sh`

Keep previously installed `arm_telemetry.py` and `arm_gravity_compensation.py`. No sibling controller/simulator sources are modified.

## Collect a fresh 30-second walk_hold trial

Both terminals:

```bash
conda activate limx
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
```

Terminal A:

```bash
POLICY_IO_AUDIT=1 bash scripts/mujoco/test_task_space.sh sim walk_hold \
  "$PWD/logs/paint/deployment_io_v1"
```

Terminal B, after A is ready:

```bash
POLICY_IO_AUDIT=1 bash scripts/mujoco/test_task_space.sh control walk_hold \
  "$PWD/logs/paint/deployment_io_v1"
```

After B exits, stop A using Ctrl+C and wait for `[V3 TASK RECORD] saved`. In B:

```bash
set -o pipefail
bash scripts/mujoco/test_task_space.sh audit walk_hold \
  "$PWD/logs/paint/deployment_io_v1" \
  2>&1 | tee "$PWD/logs/paint/deployment_io_v1/audit.log"
```

Use a new run root such as deployment_io_v2 for a repeat and use that same root in all three commands. Preview age remains 150 ms for this diagnostic. Check recorded timing/outcome as well as audit results; an aborted rollout is not a full-duration acceptance result even if its recorded tensors are internally consistent.

## Evidence and limits

Control files are `policy_io_contract.json` and `policy_io_steps.jsonl` under `walk_hold/control`. The observer taps the actual ONNX session `.run()` and records copies of actual input and output tensors. Each session is executed once; computation is not repeated in the control loop. Records are buffered and saved before SDK teardown. Inputs include the actual q/dq, gyro, wxyz quaternion, previous action and command supplied to observation construction. Each inference is associated with its first successful target publication. Partial records are retained and marked incomplete.

Exact internal checks (tolerance 1e-5) independently recompute:

- The 42-dimensional proprioception, managed phase and specified v3 scales/default leg pose.
- Oldest-first 10-frame history, seeded with the first observation.
- The 78-dimensional policy input including normalized 5x6 wrench and actual command.
- Zero yaw command in this fixed-command trial, previous action, and default+0.25*clipped-action leg target mapping.

PyTorch replays each **recorded input**, using the strict frozen actor/encoder loader and the source checkpoint. Outputs are compared to the recorded ONNX results (tolerance 1e-4). If the checkpoint moved, use the Python CLI directly with `--checkpoint /real/path/model_20000.pt`. ONNX model hashes remain checked by the launch wrapper.

Simulator records include native q/dq, currently received targets, actual IMU sensor outputs, and independent base-body orientation/angular velocity calculated from the native state. Comparison by named joints avoids relying on declaration order. SDK samples are compared to the nearest native sample within 20 ms; target reception uses the best match within 30 ms after publication. These are sampling/transport diagnostics, not exact message IDs or automatic sensor PASS gates. Mean and component-wise RMS help locate a sign/axis issue. Native sensor and base-body poses may differ by a physics-step timestamp; do not interpret tiny residuals as frame errors. Source observation timestamps are local construction times, not hardware timestamps.

Read the report in this order:

1. Outcome/timing, complete vs incomplete frames.
2. `checks.command`, `proprio`, `history`, `managed_phase`, `previous_action`, `policy_input`, `published_leg_targets`, `v3_parameters`.
3. `pytorch_replay.encoder` and `.policy`.
4. `native_diagnostics`: q/dq, SDK/native gyro and quaternion, independent base-body gyro/quaternion, and actual received joint targets. Large errors require inspecting alignment as well as signs/order; they are not automatically assigned to the policy.

Report: `walk_hold/io_audit/io_audit_report.json`. Internal PASS establishes consistency with the specified v3 deployment contract. It does not certify Isaac/MuJoCo dynamics equivalence, physical wrench prediction accuracy, noise/latency equivalence, or PAINT task performance. If this audit is clean but yaw drift remains, proceed to the matched Isaac Lab/MuJoCo comparison rather than adding an unexplained fixed yaw bias.
