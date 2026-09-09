# English | [中文](README_cn.md)

# TRON2_YG_LAB

Reinforcement learning training stack for the LimX **TRON2A** bipedal robot, built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) and using PPO to train locomotion policies. It supports SF/WF bases, legacy locked-arm SFYG/WFYG tasks, and the in-development **WFYG WholeBody architecture (OCS2 arm MPC + base RL)**.

## Repository Structure

```
.
├── exts/bipedal_locomotion/   # Isaac Lab extension: env/asset/MDP/robot cfg
├── rsl_rl/                    # Vendored rsl_rl fork (PPO + on-policy runner)
├── scripts/rsl_rl/            # Training/play entry points (train.py / play.py / cli_args.py)
├── robot_description/         # Git submodule — URDF/USD/STL robot description assets
└── docs/whole_body_ocs2.md    # WholeBody + OCS2 interface and development stages
```

## Requirements

- Isaac Sim **4.5.0** + Isaac Lab, with `isaaclab` / `isaaclab_tasks` / `isaaclab_rl` importable
- Python 3.10
- GPU (≥ 12 GB VRAM recommended for 4096-env training)

## Installation

```bash
# 1. Clone the repository with submodules
git clone --recurse-submodules https://github.com/limxdynamics/TRON2_YG_LAB.git
cd TRON2_YG_LAB
# If already cloned without submodules:
git submodule update --init --recursive

# 2. Editable install of the extension and vendored rsl_rl
pip install -e exts/bipedal_locomotion
pip install -e rsl_rl
```

The USD assets under the `robot_description` submodule are loaded at training/play startup and must be present; otherwise spawn will fail.

## WholeBody + OCS2 Development Status

Stage 1 is implemented: an independently actuated WFYG arm/gripper asset, a smooth 30-D future-wrench observation, acceleration-dependent unobserved disturbances, and dedicated WholeBody Flat/Rough tasks. The base policy retains its original ten leg/wheel actions; OCS2 owns the arm.

Training uses the lightweight wrench generator instead of running OCS2 in every parallel environment. WholeBody PLAY preserves the observation layout but supplies zero wrench until the OCS2 bridge is connected. See [docs/whole_body_ocs2.md](docs/whole_body_ocs2.md) for the interface contract, safety invariants, and remaining stages.

## Training

Task IDs are registered in [exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py](exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py).

Each morphology has two terrain variants: **Flat** (pure flat plane) and **Rough** (procedurally generated rough terrain with four sub-terrain types: flat / waves / boxes / random_rough, **no stairs**).

```bash
# === Flat ===
python scripts/rsl_rl/train.py --task Isaac-Limx-SF-TRON2A-Blind-Flat-v0   --num_envs 4096 --headless
python scripts/rsl_rl/train.py --task Isaac-Limx-WF-TRON2A-Blind-Flat-v0   --num_envs 4096 --headless --max_iterations 5000
python scripts/rsl_rl/train.py --task Isaac-Limx-SFYG-TRON2A-Blind-Flat-v0 --num_envs 4096 --headless
python scripts/rsl_rl/train.py --task Isaac-Limx-WFYG-TRON2A-Blind-Flat-v0 --num_envs 4096 --headless

# === Rough (procedural terrain, no stairs) ===
python scripts/rsl_rl/train.py --task Isaac-Limx-SF-TRON2A-Blind-Rough-v0   --num_envs 4096 --headless
python scripts/rsl_rl/train.py --task Isaac-Limx-WF-TRON2A-Blind-Rough-v0   --num_envs 4096 --headless
python scripts/rsl_rl/train.py --task Isaac-Limx-SFYG-TRON2A-Blind-Rough-v0 --num_envs 4096 --headless
python scripts/rsl_rl/train.py --task Isaac-Limx-WFYG-TRON2A-Blind-Rough-v0 --num_envs 4096 --headless

# === WholeBody: paper-aligned wrench-prediction training ===
python scripts/rsl_rl/train.py --task Isaac-Limx-WFYG-TRON2A-WholeBody-Flat-v0  --num_envs 4096 --headless
python scripts/rsl_rl/train.py --task Isaac-Limx-WFYG-TRON2A-WholeBody-Rough-v0 --num_envs 4096 --headless
```

Rough terrain configuration is defined in `BLIND_ROUGH_TERRAINS_CFG` in [cfg/SF_TRON2A/terrains_cfg.py](exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF_TRON2A/terrains_cfg.py) and [cfg/WF_TRON2A/terrains_cfg.py](exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/WF_TRON2A/terrains_cfg.py) (10×16 grid, curriculum on, difficulty 0~1). YG variants reuse the SF/WF rough terrain but exclude arm/gripper randomization and limit penalties as described in [YG Variant Design](#yg-variant-design).

Common options:

- `--checkpoint_path <path>` — resume from a specific `.pt` checkpoint (or set `resume=True` + `load_run`/`load_checkpoint` in cfg)
- `--video --video_interval 24000 --video_length 400` — enable video recording (auto-enables `--enable_cameras`)
- `--max_iterations N` — override the maximum iteration count in PPO cfg

Log path: `logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/`

## Resume Training

`agent_cfg.resume` defaults to `False`. You **must explicitly pass `--resume True`** to load a checkpoint ([scripts/rsl_rl/train.py:130-139](scripts/rsl_rl/train.py#L130-L139)). Two methods:

### Method A: Direct .pt path (recommended)

```bash
python scripts/rsl_rl/train.py \
    --task Isaac-Limx-SFYG-TRON2A-Blind-Rough-v0 \
    --num_envs 4096 --headless \
    --resume True \
    --checkpoint_path logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/model_<iter>.pt
```

### Method B: Look up by run name and checkpoint name

```bash
python scripts/rsl_rl/train.py \
    --task Isaac-Limx-SFYG-TRON2A-Blind-Rough-v0 \
    --num_envs 4096 --headless \
    --resume True \
    --load_run 2026-06-01_12-34-56_sfyg_rough \
    --checkpoint model_1500.pt
```

`--load_run` / `--checkpoint` support regex (e.g., `--load_run ".*"`, `--checkpoint "model_.*\.pt"`), matching the latest entry under `logs/rsl_rl/<experiment_name>/` in lexicographic order.

Notes:

1. **The task ID must match the original run**, otherwise obs/action dimension mismatch will cause loading failure. Changing only reward weights (which do not affect dimensions) is safe for resuming.
2. Resume creates a **new timestamped subdirectory** under `logs/rsl_rl/<experiment_name>/` for new logs, leaving the original run files untouched.
3. To train for N additional iterations: `--max_iterations` is a **cap, not an increment** — if you trained to 1500 and want 1000 more, pass `--max_iterations 2500`.

## Play / Deployment Preview

Use task IDs with the `-Play-v0` suffix. Play cfg uses fewer envs, disables domain randomization, and simplifies terrain.

```bash
# Flat
python scripts/rsl_rl/play.py \
    --task Isaac-Limx-SF-TRON2A-Blind-Flat-Play-v0 \
    --num_envs 32 \
    --checkpoint_path logs/rsl_rl/sf_tron_2a_flat/<run>/model_<iter>.pt

# Rough
python scripts/rsl_rl/play.py \
    --task Isaac-Limx-SF-TRON2A-Blind-Rough-Play-v0 \
    --num_envs 32 \
    --checkpoint_path logs/rsl_rl/sf_tron_2a_flat/<run>/model_<iter>.pt
```

Every training task has a corresponding `-Play-v0` variant. Four additional WFYG WholeBody Flat/Rough training/PLAY tasks are registered.

## Robot Morphologies

| Morphology | End-effector | Arms | Task ID Prefix |
|---|---|---|---|
| SF_TRON2A | sole foot (ankle pitch) | — | `Isaac-Limx-SF-TRON2A-...` |
| WF_TRON2A | wheel | — | `Isaac-Limx-WF-TRON2A-...` |
| SFYG_TRON2A | sole foot | 6-DoF arm + 2-finger prismatic gripper (locked) | `Isaac-Limx-SFYG-TRON2A-...` |
| WFYG_TRON2A | wheel | Same as above | `Isaac-Limx-WFYG-TRON2A-...` |
| WFYG WholeBody | wheel | 6-DoF arm reserved for OCS2; base observes future wrench | `Isaac-Limx-WFYG-TRON2A-WholeBody-...` |

### YG Variant Design

The arms remain locked in a fixed pose throughout (arm1~6 = 0 rad, gripper1/2 = 0.05/-0.05 m), held by a dedicated `arm_lock` `ImplicitActuator` group (stiffness 800, damping 40) in the asset config performing PD lock.

- Arm joints are **not** in `joint_order_name` → excluded from the RL action space and observation dimensions
- Domain randomization / reset / dof_limits reward explicitly exclude arm joints in YG env cfg, preventing disturbance to the lock PD or spurious penalties
- During training/inference, the arms serve as **payload only** and are invisible to the policy

For the full arm exclusion checklist, see [exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_solefoot_yg_tron2a_env_cfg.py](exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_solefoot_yg_tron2a_env_cfg.py) and [limx_wheelfoot_yg_tron2a_env_cfg.py](exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_wheelfoot_yg_tron2a_env_cfg.py).

WholeBody is an additive task and does not alter this legacy behavior. It uses separate `arm_mpc` and `gripper` actuator groups and trains the base policy with the wrench sequence expected from OCS2/Pinocchio at deployment. Arm joints are not added to the PPO action.

## Architecture Overview

See [CLAUDE.md](CLAUDE.md) for details. Three top-level packages:

1. **`exts/bipedal_locomotion/`** — Isaac Lab extension. All env/asset/MDP/robot configs live here.
2. **`rsl_rl/`** — Vendored fork. `scripts/rsl_rl/train.py` prepends this path to `sys.path`, overriding the system-installed version. Import uses `from rsl_rl.runner import OnPolicyRunner` (singular `runner`, not upstream's `runners`).
3. **`scripts/rsl_rl/`** — Entry-point scripts. **Not a package**; operates via `sys.path` manipulation. CLI parsing order is fixed: launcher args must be registered before `AppLauncher(args_cli)`.

### Task Wiring

Using `Isaac-Limx-SF-TRON2A-Blind-Flat-v0` as an example:

1. **`gym.register`**: `tasks/locomotion/robots/__init__.py` binds (env_cfg, ppo_cfg) to the task ID
2. **Env cfg**: `tasks/locomotion/robots/limx_solefoot_tron2a_env_cfg.py` inherits from `tasks/locomotion/cfg/SF_TRON2A/limx_base_env_cfg.py::SF_TRON2A_EnvCfg`, attaches assets + modifies MDP
3. **MDP terms**: `tasks/locomotion/mdp/{rewards,events,observations,curriculums,commands}.py`
4. **PPO cfg**: `tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py`, using the project's custom `RslRlPpoAlgorithmMlpCfg` type (not the upstream class)
5. **Asset cfg**: `assets/config/<robot>_cfg.py`, spawns USD (from `robot_description/`) + init joint pos + actuators

To add a new robot variant: increment one copy at each of the 5 layers above without modifying the existing TRON2 training stack.

## Submodule: robot_description

URL: [https://github.com/limx-tron2/robot-description](https://github.com/limx-tron2/robot-description)

Contains URDF / xacro / MuJoCo XML / mesh / USD assets for 6 TRON2 variants: `SF_TRON2A` / `WF_TRON2A` / `SFYG_TRON2A` / `WFYG_TRON2A` / `DA_TRON2A` / `DACH_TRON2A` (the last two are not used by this repository).

Update to the latest submodule commit:

```bash
cd robot_description && git pull origin main && cd ..
git add robot_description && git commit -m "chore: bump robot_description submodule"
```

## Testing / Linting

No test suite is configured. `pyproject.toml` contains `isort` + `pyright` configuration but no CI.

## License

[Apache 2.0](LICENCE).
