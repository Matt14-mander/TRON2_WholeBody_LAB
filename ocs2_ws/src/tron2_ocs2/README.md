# tron2_ocs2

OCS2 SLQ-MPC core for the sole-foot SFYG TRON2A whole-body controller. The
package includes a simulator-neutral localhost service; Isaac Lab and MuJoCo
specific adapters remain outside the native solver.

The optimized state is
`[p_WB(x,y,z), yaw,pitch,roll, q_arm(6), dq_arm(6)]`; the input is
`[v_Bx,v_By,yaw_rate,ddq_arm(6)]`. Planar velocity is body-frame so it can be
sent directly to the locomotion policy. The base height, pitch, and roll use
the paper's identified first-order response model, while each arm joint is a
double integrator.

The solver returns arm position/velocity/feed-forward effort, planar base
velocity, and five arm-on-base wrench samples at 0.0--0.8 s. Wrenches are
extracted from the `arm1_Joint` subtree during Pinocchio RNEA, negated to the
arm-on-base sign, and expressed at `base_Link` in the base frame. Feed-forward
arm effort is saturated by the six URDF effort limits before publication.

Runtime bridges should call `SolverCore::trySolve()`. On any solver or
validation failure it returns a zero, invalid command and resets MPC warm-start
state; `solve()` remains available for tests that need exceptions.

The repository root provides reproducible preparation and build scripts. They
pin the supported OCS2 revisions, apply the tracked compatibility patches,
require discoverable Pinocchio/hpp-fcl CMake packages, and work around the old
vendored GoogleTest build failure:

```bash
cd /path/to/TRON2_WholeBody_LAB
bash ocs2_ws/scripts/prepare_sources.sh
conda activate tron2_ocs2
bash ocs2_ws/scripts/build_tron2_ocs2.sh
source ocs2_ws/install/setup.bash
```

See [`../../README.md`](../../README.md) for dependency installation and
diagnostics. Do not continue to the commands below until
`ros2 pkg executables tron2_ocs2` lists all three executables.

Run a complete construction/solve/RNEA smoke pass:

```bash
(
  set -e
  CODEGEN_CC="$(command -v x86_64-conda-linux-gnu-gcc)"
  printf 'int main(void) { return 0; }\n' | "$CODEGEN_CC" -x c -fsyntax-only -
  export OCS2_CODEGEN_COMPILER="$CODEGEN_CC"
  ros2 run tron2_ocs2 tron2_ocs2_smoke \
    "$PWD/ocs2_ws/src/tron2_ocs2/config/task.info" \
    "$PWD/robot_description/tron2/SFYG_TRON2A/urdf/robot.urdf" \
    /tmp/tron2_ocs2_codegen
)
```

The first run should set `model.recompileLibraries=true`; after CppAD models
have been generated, set it back to `false` for fast startup. CppAD compiles
generated C code at runtime. The pinned OCS2 patch otherwise defaults to
`/usr/bin/gcc`, which may not have a working `cc1` in the Conda environment.
If the Conda compiler is absent, install `gcc_linux-64` from conda-forge in
the active `tron2_ocs2` environment before running the smoke test.

## Isaac Lab runtime bridge

The native solver and Isaac Lab run in separate Conda environments. A
localhost-only TCP service keeps ROS 2 and OCS2 libraries out of the Isaac Sim
Python process. Start the service from the OCS2 environment after building:

```bash
source ocs2_ws/install/setup.bash
export OCS2_CODEGEN_COMPILER="$(command -v x86_64-conda-linux-gnu-gcc)"
ros2 run tron2_ocs2 tron2_ocs2_bridge \
  "$PWD/ocs2_ws/src/tron2_ocs2/config/task.info" \
  "$PWD/robot_description/tron2/SFYG_TRON2A/urdf/robot.urdf" \
  /tmp/tron2_ocs2_codegen \
  5555
```

The server binds to `127.0.0.1` by default, accepts one persistent client, and
preserves the warm-started MPC instance between requests. Its line protocol
supports `SOLVE` and `RESET`; responses carry request identifiers so stale or
misordered solutions are rejected. Each request prints `elapsed_ms` on the
server so the achievable MPC rate can be measured on the deployment host.

In a second terminal, activate the Isaac Lab environment and run the WholeBody
PLAY task with `--ocs2`. The bridge currently supports one environment:

```bash
LIVESTREAM=2 python scripts/rsl_rl/play.py \
  --task Isaac-Limx-SFYG-TRON2A-WholeBody-Flat-Play-v0 \
  --num_envs 1 --headless --ocs2 \
  --ocs2_update_period 0.1 \
  --ocs2_max_solution_age 0.5 \
  --ocs2_target_position 0.35 0.0 0.85 \
  --ocs2_target_quaternion 1.0 0.0 0.0 0.0 \
  --checkpoint_path /absolute/path/to/model.pt
```

OCS2 owns the six arm position/velocity/feed-forward-effort targets and the
planar locomotion command. Its 5x6 base-wrench prediction replaces the zero
PLAY command observed by the policy. It is not applied as an external force:
the articulated arm already produces that reaction in simulation. Blocking
TCP and DDP work runs on a background thread, and observations submitted while
a solve is in flight are coalesced to the newest state. While no fresh result
is available, PLAY pauses physics but continues pumping the render/WebRTC event
loop. A timeout, invalid response, stale timestamp, or solver failure therefore
cannot advance the robot with mismatched inputs; the arm is held and both the
planar command and predicted wrench are zeroed.

## Offline trajectory export and replay

When the host cannot solve OCS2 in real time, export the complete primal
solution once. The default target is position `[0.35, 0.0, 0.85]`, identity
WXYZ orientation, and the default sample period is 0.02 s:

```bash
ros2 run tron2_ocs2 tron2_ocs2_trajectory_export \
  /tmp/tron2_ocs2_runtime.info \
  "$PWD/robot_description/tron2/SFYG_TRON2A/urdf/robot.urdf" \
  /tmp/tron2_ocs2_codegen \
  /tmp/tron2_reach.csv
```

An explicit target and sample period may be appended as:

```text
TARGET_X TARGET_Y TARGET_Z TARGET_QW TARGET_QX TARGET_QY TARGET_QZ SAMPLE_PERIOD ARRIVAL_TIME
```

`ARRIVAL_TIME` is measured from the initial observation and must lie within
the MPC horizon. A positive value creates a smooth position/quaternion
reference from the initial end-effector pose to the target. Omitting it keeps
the legacy immediate step reference used by earlier commands.

For a safer first dataset, prefer a small displacement from the *initial*
`gripper_base_Link` pose. This preserves its initial orientation instead of
forcing an identity wrist orientation:

```bash
ros2 run tron2_ocs2 tron2_ocs2_trajectory_export \
  /tmp/tron2_ocs2_runtime.info \
  "$PWD/robot_description/tron2/SFYG_TRON2A/urdf/robot.urdf" \
  /tmp/tron2_ocs2_codegen \
  /tmp/tron2_reach_safe.csv \
  --relative-target 0.02 0.0 0.0 0.02 1.0
```

Offline export checks every MPC knot and every 52-column output sample against
the URDF arm position limits with a 0.05 rad margin, arm velocity/effort
limits, and the walking policy's body-frame velocity-command limits. The MPC
uses soft constraint penalties, so a solve may complete but fail this strict
export check. Such a trajectory is rejected before the output CSV is opened;
do not clip its joint positions or use it for training or replay.

For reproducible batch generation over the initially validated compact
workspace, run:

```bash
python scripts/ocs2/generate_trajectory_dataset.py \
  --task-info /tmp/tron2_ocs2_runtime.info \
  --robot-urdf "$PWD/robot_description/tron2/SFYG_TRON2A/urdf/robot.urdf" \
  --library-dir /tmp/tron2_ocs2_codegen \
  --output-dir "$HOME/datasets/tron2_ocs2_safe_v2" \
  --count 200 \
  --seed 42 \
  --target-mode relative \
  --arrival-times 0.75 1.0
```

The relative-target defaults sample XYZ offsets within 0.03/0.02/0.02 m of
the home end-effector pose. This is a conservative *candidate generator*, not
a guarantee of feasibility: the strict checks decide what succeeds. The batch
tool writes one unchanged 52-column CSV per accepted solve, a `manifest.csv`
with targets/timing/numerical bounds, `failures.csv` for rejected solves, and
`contract.json` documenting the column order, arm limits, wrench frame/sign,
five prediction offsets, normalization, and 78-input/10-action walking actor
contract. `--target-mode absolute` retains the older absolute-world target
interface. Use a new output directory for this revised manifest; `--resume`
continues only an interrupted dataset with the same contract.

## Isaac Lab rollout collection

The OCS2 CSV files are expert plans, not measured executions. Collect aligned
policy inputs, simulated robot states, intent metadata, OCS2 labels, next
states, and success metrics in one Isaac Lab process with:

```bash
python scripts/rsl_rl/collect_ocs2_rollouts.py \
  --task Isaac-Limx-SFYG-TRON2A-WholeBody-Flat-Play-v0 \
  --trajectory_manifest "$HOME/datasets/tron2_ocs2_safe_v2/manifest.csv" \
  --output_dir "$HOME/datasets/tron2_ocs2_safe_v2_rollout" \
  --max_trajectories 10 \
  --start_delay 1.0 \
  --post_motion_duration 2.0 \
  --terminal_base_command -0.25 0.0 0.0 \
  --headless \
  --checkpoint_path /absolute/path/to/model.pt
```

Collection always uses deterministic base and joint reset offsets. Accepted
episodes are stored as compressed NPZ files under their train, validation, or
test split. Falls and excessive-tilt episodes are retained under `rejected/`
and recorded in `replay_failures.csv`. `rollout_manifest.csv` stores per-run
tracking metrics. Use `--resume_collection` to skip all trajectory ids already
present in either metadata file.

Replay requires only the Isaac Lab environment; do not start the TCP service:

```bash
LIVESTREAM=2 python scripts/rsl_rl/play.py \
  --task Isaac-Limx-SFYG-TRON2A-WholeBody-Flat-Play-v0 \
  --num_envs 1 --headless \
  --ocs2_trajectory /tmp/tron2_reach.csv \
  --checkpoint_path /absolute/path/to/model.pt
```

The CSV contains 52 columns: relative time, six arm positions, six arm
velocities, six feed-forward efforts, three body-frame planar commands, and
five row-major `[Fx,Fy,Fz,Tx,Ty,Tz]` base-frame wrench samples. Playback is
one-shot; after the final sample it holds terminal arm position/effort and
zeros arm velocity and base motion rather than looping the reach. The exporter
appends a 0.5 s transition to a static terminal sample whose feed-forward
effort and wrench are recomputed by RNEA with zero velocity and acceleration.
For diagnosis, `--ocs2_trajectory_zero_base_command` and
`--ocs2_trajectory_zero_wrench` independently disable those two policy inputs.
For a locomotion-focused test, append
`--ocs2_trajectory_terminal_base_command VX VY WZ`; PLAY smoothly blends to
that body-frame cruise command over the final 0.5 s instead of stopping.
`--ocs2_trajectory_start_delay SECONDS` holds the initial arm sample while the
locomotion/contact state settles before playback begins.
For repeatable controller validation, `--ocs2_deterministic_reset` zeros the
base pose/velocity and joint position/velocity reset offsets. This isolates
trajectory and policy stability from the randomized training reset state and
prevents a randomly reset arm from being pulled abruptly to the first sample.
