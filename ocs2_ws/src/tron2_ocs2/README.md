# tron2_ocs2

OCS2 SLQ-MPC core for the sole-foot SFYG TRON2A whole-body controller. This
package intentionally contains no Isaac Lab or MuJoCo transport code.

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
