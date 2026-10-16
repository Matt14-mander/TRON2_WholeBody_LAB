# Visualize v3 with a base-wrench preview in MuJoCo

The `sim` and `control` modes of `scripts/mujoco/run_wholebody_v3.py` accept
matching `--payload-mass`, `--payload-dimensions`, `--payload-offset` options.
`sim` forwards these to the existing SFYG simulator payload loader.

`control --low-wrench gravity|plan --sim-root ...` compiles a private copy of the
same SFYG XML, with the same optional payload. It uses native MuJoCo mass, COM,
principal inertia and FK; it does not depend on the Isaac URDF joint convention.
Extra bodies are classified relative to the SF reference URDF. `gravity` repeats
the current gravitational wrench; `plan` previews the active guarded joint-space
test with the current measured position offset held over the future trajectory.
Base orientation is held current and base acceleration is omitted, as in the
scripted Isaac diagnostic. Without an active arm test, plan reduces to gravity.

The prediction replaces only the controller's policy observation via a copy of
the immutable solution. Leg ONNX weights, arm command/PD and physical simulator
are unchanged. The existing compose_policy_input applies force 0.01 / torque
0.05 normalization. The reference point is base_Link origin and all samples are
in current base axes. OCS2 CSV and the new provider are mutually exclusive.

Both processes must use the same asset checkout and payload arguments. These
arguments do not handshake over SDK. Restart the simulator for each controller
trial, including between standing and arm tests. The initial SDK pose must match
the model configuration; configured is the original extended arm pose.

Observed local asset difference: this MuJoCo XML plus 0.5kg payload has 7.265kg
of added bodies relative to SF, while the supplied Isaac URDF has 7.661kg. The
native model provider therefore predicts a different wrench than the Isaac URDF
provider. This visual test does not establish simulator equivalence.

At upright configured pose, the local MuJoCo model predicts approximately
`[0,0,-71.27,-0.004,9.43,0]` N/Nm. The first-sample physical wrench is printed
as `wrench0` with every one-second control log. Arm test completion/abort flags
remain present: a disarmed arm test must not be counted as a successful swing.
The existing arm baseline-error guard is preserved.

Verified offline: compile the real payload XML, compare gravity to the static
five-point preview, and check analytical forces/torques with a small MuJoCo
fixture. The fixture also verifies that preview computation leaves a separate
live MjData untouched. Full SDK/viewer operation must be verified on the server.
