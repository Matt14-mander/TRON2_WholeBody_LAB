# TRON2 Whole-Body + OCS2 Development Plan

## Control boundary

The sole-foot base RL policy retains its 10 leg-joint position outputs. OCS2
owns the six arm joints. The gripper remains an independent position-controlled
subsystem. Wheel-foot WholeBody control is outside the current development stage.

At deployment, OCS2 receives base state, arm state, and the desired end-effector
pose. It returns:

1. desired arm position and velocity;
2. feed-forward arm effort;
3. desired base planar velocity;
4. base-frame wrench samples at 0.0, 0.2, 0.4, 0.6, and 0.8 seconds.

The numerical interface is defined in
`bipedal_locomotion/controllers/ocs2_interface.py`.

## Development status

### Stage 1 - training interface (implemented)

- Separate whole-body SFYG asset with independently actuated arm and gripper.
- Smooth quadratic wrench-sequence generator at the existing 50 Hz policy rate.
- Noisy 30-D actor observation and clean privileged critic observations.
- Acceleration-dependent unobserved wrench disturbance.
- Dedicated Flat/Rough tasks and PPO experiment names.
- PLAY tasks preserve observation dimensions but disable synthetic wrench; they
  produce zero wrench until the OCS2 bridge is connected.

### Stage 2A - OCS2 solver core (implemented)

- Populate and validate the `robot_description` submodule.
- Create the ROS 2 `tron2_ocs2` package against the `ros2` branch of OCS2.
- Implement the paper-style 18-state/9-input floating-base arm model.
- Add end-effector tracking, nominal-arm, joint-limit, input, and self-collision
  costs/constraints.
- Add a configurable positive terminal-state regularizer so strict DDP
  numerical-stability checks remain enabled despite round-off in the terminal
  end-effector Hessian.
- Use Pinocchio RNEA arm-subtree forces to produce the base-frame wrench
  sequence without including the legs or trunk wrench.
- Add a full solver/RNEA smoke executable and dynamics unit test.

The implementation and build instructions are under `ocs2_ws/src/tron2_ocs2`.
The internal planar command is body-frame `[vx, vy, yaw_rate]`, matching the
locomotion policy command convention. The optimizer uses a 1.0 s horizon so all
five prediction offsets are always available.

### Stage 2B - runtime bridges (in progress)

- A localhost TCP service and non-blocking Isaac Lab PLAY client now connect
  the native C++ solver without mixing the OCS2 and Isaac Sim Conda environments.
- The PLAY adapter writes arm position/velocity/feed-forward effort targets,
  the planar base command, and the normalized 5x6 predicted-wrench observation.
- OCS2 runs on one background worker; requests are coalesced to the newest
  observation so a slow solver cannot block rendering or build an unbounded
  queue. Until a fresh solution is available, PLAY pumps the render/WebRTC
  event loop but pauses physics, preventing fallback falls and episode resets
  from invalidating a slow solve. Stale solutions, transport errors, and solver
  failures hold the arm and zero both the locomotion command and predicted
  wrench.
- Build and closed-loop simulation validation on the Linux Isaac Lab host is
  still required before this stage is marked implemented.
- Reuse the same transport-neutral contract for the MuJoCo deployment bridge.

### Stage 2C - offline trajectory replay (implemented, awaiting host validation)

- `tron2_ocs2_trajectory_export` performs one OCS2 solve and samples the full
  primal trajectory at 50 Hz into a fixed 52-column CSV contract.
- Each row contains arm position, velocity, feed-forward effort, planar base
  command, and the 5x6 predicted base wrench used by the WholeBody actor.
- The last optimized state is extended as a constant for wrench preview near
  the horizon boundary. A 0.5 s transition blends into an RNEA-recomputed
  static hold with zero arm velocity, acceleration, and planar base command.
- Isaac Lab accepts `--ocs2_trajectory PATH` as a mutually exclusive
  alternative to the live TCP bridge and interpolates the CSV against
  simulation time without any OCS2 process at PLAY time.
- A configurable terminal planar command supports continued walking after the
  one-shot arm motion; its final 0.5 s transition uses cubic smoothstep.
- An optional startup delay holds the first arm sample while locomotion,
  contacts, actuator targets, and policy observations settle.
- An optional deterministic-reset validation mode removes base and joint reset
  offsets so repeated falls can be separated from randomized startup states.

### Stage 3 - policy fidelity and hardware

- Add teacher/student distillation and a dedicated wrench RNN.
- Identify the base first-order response coefficients used by OCS2.
- Calibrate arm limits, inertias, friction, and actuator gains.
- Run no-wrench/current-wrench/predicted-wrench ablations before hardware tests.

## Safety invariants

- Never apply the synthetic training wrench while a simulated or physical arm
  already produces the same reaction wrench.
- Reject stale, non-finite, or dimensionally invalid OCS2 solutions.
- On solver failure, stop arm motion and send a zero planar base command.
- Confirm the wrench sign is "arm acting on base" in the current base frame.

## Model validation record

The `robot_description` submodule was initialized at commit
`8a11c12c7851a104dc4cafff1780276f624d2bc6`. The supplied SFYG model confirms
the expected `base_Link`, six `arm*_Joint` names, and two `gripper*_Joint`
names. Its URDF limits the arm joints to 100 Nm and 5 rad/s, and the gripper
joints to 10 N and 3 m/s. The WholeBody asset uses these limits. Its initial
arm2/arm3 positions are set near the middle of their asymmetric ranges rather
than at the zero-position joint limits.
