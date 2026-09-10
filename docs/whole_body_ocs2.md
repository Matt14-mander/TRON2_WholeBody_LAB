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
- Use Pinocchio RNEA arm-subtree forces to produce the base-frame wrench
  sequence without including the legs or trunk wrench.
- Add a full solver/RNEA smoke executable and dynamics unit test.

The implementation and build instructions are under `ocs2_ws/src/tron2_ocs2`.
The internal planar command is body-frame `[vx, vy, yaw_rate]`, matching the
locomotion policy command convention. The optimizer uses a 1.0 s horizon so all
five prediction offsets are always available.

### Stage 2B - runtime bridges (next)

- Connect the C++ solver output to Isaac Lab PLAY and add stale-solution and
  solver-failure fallbacks.
- Reuse the same transport-neutral contract for the MuJoCo deployment bridge.

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
