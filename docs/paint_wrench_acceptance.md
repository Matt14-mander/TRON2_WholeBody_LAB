# PAINT frozen locomotion: wrench-input ablation

`scripts/rsl_rl/validate_paint_support.py` now supports `--low_wrench zero|gravity|plan`.
Default is `zero`, preserving the original A/B/C tests. This does not enable a
prediction provider in teacher PPO; the validation action explicitly supplies it.

## Contract

- Arm/additional assembly acting **on** the base, not base acting on the arm.
- Moments about `base_Link` origin; all five samples in the **current** base axes.
- Sample times 0, 0.2, 0.4, 0.6, 0.8 seconds; each `[Fx,Fy,Fz,Tx,Ty,Tz]` in N/Nm.
- Flatten in sample-major order; multiply force by 0.01 and torque by 0.05;
  observation clip [-10,10] matches the training observation term.
- Input to actor: latent(3), proprioception(42), normalized wrench(30), command(3).
- Prediction changes observations only. Actual arm/payload remain physical;
  no duplicate synthetic reaction wrench is applied.

## Prediction sources

`gravity` sums weight and its moment for SFYG URDF bodies absent from the SF
reference, plus the attached payload. It uses measured joint angles and current
base orientation, repeating the static wrench at all five times. This includes
the adapter and fixed accessories, not just the moving arm links. These bodies
must be audited against the physical asset. A mass/COM check against the spawned
USD runs before using the first prediction. Missing bodies or differences above
5%/1g mass or 3cm COM abort, leaving `wrench_model_check.json` for inspection.
These tolerances are consistency checks, not evidence of accurate dynamics.

Use `robot_rl.urdf` for both models. The SFYG simulation configuration commands
negative gripper2 joint positions; its RL URDF reverses the gripper2 axis and
uses [-0.05,0] travel. Using the original `robot.urdf` with the same negative
coordinates produces a 9.75cm gripper2 COM mismatch at q=-0.04875m. The RL URDF
also contains a different arm6 inertia. Keep the runtime consistency check active.

`plan` evaluates the known quintic arm script over the horizon and sums
Newton-Euler force/moment balances including translational acceleration, angular
acceleration and gyroscopic terms. Present measured position offsets are held
constant over the future arm plan. Base pose is held constant; base acceleration
and future tracking errors are not predicted. Finite differences use 2ms.
This is a scripted model-based preview, **not an online MPC solution**. Arm PD
and zero arm feedforward remain the same for the input ablation. Gravity preview
should be checked first; dynamic preview cannot correct poor arm tracking itself.
Neither source currently supports the force/coupled cases, which are rejected.

The existing OCS2 bridge produces a compatible five-sample physical interface;
the setter can accept that source later. Its payload/inertia model, reference
point and convention must be checked before replacing the scripted source.

## Coverage diagnostic

Current repository generator defaults are Fz [-60,20]N. The supplied URDF's added
assembly plus 0.5kg payload totals 7.661kg, around -75.15N in upright stance.
Record this difference; do not clamp the physical force to -60N. Actual historical
checkpoint settings must be checked against its saved training configuration.
The script records raw and normalized sequences and the fraction of samples
outside current repository bounds. These bounds are not reconstructed from the
checkpoint, and do not include the training generator's unobserved disturbances.

## Server commands

Run in the repository after syncing the four runtime files: the validation script,
`paint_teacher_core.py`, `paint_wrench_preview.py`, and teacher `mdp.py`.

```bash
conda activate limx
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
set -o pipefail
CHECKPOINT="$PWD/logs/rsl_rl/sfyg_tron_2a_wholebody_paper_v3_flat/2026-10-15_16-38-03/model_20000.pt"
WRENCH_TEST="$PWD/logs/paint/wrench_input_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WRENCH_TEST"
COMMON=(--checkpoint_path "$CHECKPOINT" --payload_mode attached
  --arm_pose nominal --payload_mass 0.5 --payload_dimensions .20 .12 .08
  --payload_offset .15 0 0 --episodes 3 --duration 30 --headless --device cuda:1)

for source in zero gravity; do
  LIVESTREAM=0 python -u scripts/rsl_rl/validate_paint_support.py "${COMMON[@]}" \
    --cases hold --low_wrench "$source" --output_dir "$WRENCH_TEST/hold_$source" \
    2>&1 | tee "$WRENCH_TEST/hold_$source.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then break; fi
done
```

Inspect reports before proceeding. No-fall completion alone is insufficient:
compare steady zero-command velocity bias, payload tilt, acceleration and arm error.
If gravity mode helps and the hold is acceptable, compare arm previews:

```bash
for source in zero gravity plan; do
  LIVESTREAM=0 python -u scripts/rsl_rl/validate_paint_support.py "${COMMON[@]}" \
    --cases arm_hold arm_walk --arm_delta 0 .10 0 -.05 0 0 \
    --low_wrench "$source" --output_dir "$WRENCH_TEST/arm_$source" \
    2>&1 | tee "$WRENCH_TEST/arm_$source.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then break; fi
done
```

Use fresh output directories. The model is frozen throughout; no estimator or
teacher optimizer runs. Unit tests cover weight/moment signs, centripetal and
angular acceleration reactions, input slots/scales, reset isolation and physical
force independence. They do not validate PhysX or MuJoCo stability.
