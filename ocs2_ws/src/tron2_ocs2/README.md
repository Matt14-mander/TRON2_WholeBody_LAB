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

Build from a ROS 2 workspace in which the OCS2 `ros2` branch and
`robot_description` are available:

```bash
cd ocs2_ws
colcon build --base-paths src ../robot_description --packages-up-to tron2_ocs2
source install/setup.bash
```

Run a complete construction/solve/RNEA smoke pass:

```bash
ros2 run tron2_ocs2 tron2_ocs2_smoke \
  "$PWD/src/tron2_ocs2/config/task.info" \
  "$PWD/../robot_description/tron2/SFYG_TRON2A/urdf/robot.urdf" \
  /tmp/tron2_ocs2_codegen
```

The first run should set `model.recompileLibraries=true`; after CppAD models
have been generated, set it back to `false` for fast startup.

## Isaac Lab runtime bridge

The native solver and Isaac Lab run in separate Conda environments. A
localhost-only TCP service keeps ROS 2 and OCS2 libraries out of the Isaac Sim
Python process. Start the service from the OCS2 environment after building:

```bash
source install_v3c/setup.bash
export OCS2_CODEGEN_COMPILER="$(command -v x86_64-conda-linux-gnu-gcc)"
ros2 run tron2_ocs2 tron2_ocs2_bridge \
  "$PWD/src/tron2_ocs2/config/task.info" \
  "$PWD/../robot_description/tron2/SFYG_TRON2A/urdf/robot.urdf" \
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
  "$PWD/../robot_description/tron2/SFYG_TRON2A/urdf/robot.urdf" \
  /tmp/tron2_ocs2_codegen \
  /tmp/tron2_reach.csv
```

An explicit target and sample period may be appended as:

```text
TARGET_X TARGET_Y TARGET_Z TARGET_QW TARGET_QX TARGET_QY TARGET_QZ SAMPLE_PERIOD
```

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
