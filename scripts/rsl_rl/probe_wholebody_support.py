"""Evaluate the OLD WholeBody checkpoint with controlled arm/startup/phase changes.

Zero external force throughout. This probes execution sensitivity, not the causal
effect of training with wrench (which requires matched training ablations).
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rsl_rl"))
sys.path.insert(0, str(ROOT / "exts/bipedal_locomotion"))


def parser_for_probe():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--arm_mode", choices=("source", "hold", "stiff"), default="hold")
    p.add_argument("--phase", choices=("source", "frozen"), default="source")
    p.add_argument("--warmup", type=float, default=0., help="Seconds of leg PD at default pose before RL.")
    p.add_argument("--duration", type=float, default=20., help="RL seconds AFTER the PD preparation.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir")
    return p


def frozen_phase(env):
    import torch
    result = torch.zeros(env.num_envs, 2, device=env.device)
    result[:, 1] = 1
    return result


def main(args, app):
    import math
    import torch
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runner import OnPolicyRunner
    import bipedal_locomotion  # noqa: F401
    from bipedal_locomotion.controllers.ocs2_reset import configure_nominal_ocs2_reset
    from bipedal_locomotion.controllers.paint_teacher_core import ARM_JOINT_NAMES
    from bipedal_locomotion.controllers.standwalk_core import load_warm_start
    from bipedal_locomotion.tasks.standwalk.env import StandWalkRLEnv
    from bipedal_locomotion.tasks.standwalk.mdp import fall
    from isaaclab.managers import TerminationTermCfg

    if not math.isfinite(args.warmup) or args.warmup < 0 or not math.isfinite(args.duration) or args.duration <= 0:
        raise ValueError("Use finite nonnegative warmup and positive RL duration.")
    source = Path(args.checkpoint_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output = Path(args.output_dir or ROOT / "logs" / "paint" /
                  datetime.now().strftime("support_probe_%Y%m%d_%H%M%S_%f")).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    task = "Isaac-Limx-SFYG-TRON2A-WholeBody-Flat-Play-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    cfg.seed = args.seed
    cfg.episode_length_s = args.warmup + args.duration
    configure_nominal_ocs2_reset(cfg, SceneEntityCfg("robot"))
    cfg.observations.policy.enable_corruption = False
    cfg.observations.obsHistory.enable_corruption = False
    cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.
    cfg.commands.base_velocity.rel_standing_envs = 0.
    for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        setattr(cfg.commands.base_velocity.ranges, name, (0., 0.))
    if args.phase == "frozen":
        for group in (cfg.observations.policy, cfg.observations.obsHistory, cfg.observations.critic):
            group.gait_phase.func = frozen_phase
    for actuator in cfg.scene.robot.actuators.values():
        if hasattr(actuator, "min_delay"):
            actuator.min_delay = actuator.max_delay = 0
    if args.arm_mode == "stiff":
        cfg.scene.robot.actuators["arm_mpc"].stiffness = 800.
        cfg.scene.robot.actuators["arm_mpc"].damping = 40.
    # Preserve pose, mass, arm effort/velocity limits and solver settings across cases.
    cfg.terminations.support_fall = TerminationTermCfg(func=fall)
    torch.manual_seed(args.seed)
    env = RslRlVecEnvWrapper(StandWalkRLEnv(cfg))
    try:
        raw = env.unwrapped
        robot = raw.scene["robot"]
        arm_ids, names = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
        if tuple(names) != ARM_JOINT_NAMES:
            raise ValueError("Arm joint order mismatch.")
        agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=raw.device)
        report = load_warm_start(runner.alg.actor_critic, runner.alg.encoder,
                                torch.load(source, map_location="cpu", weights_only=False))
        policy = runner.get_inference_policy(raw.device)
        encoder = runner.get_inference_encoder(raw.device)
        obs = env.get_observations()
        nominal = robot.data.default_joint_pos[:, arm_ids].clone()
        zeros = torch.zeros_like(nominal)
        print(f"[PROBE] {vars(args)} load={report}", flush=True)
        (output / "contract.json").write_text(json.dumps({"args": vars(args), "load": report,
            "nominal_arm": nominal[0].tolist(), "env": cfg.to_dict()}, indent=2, default=str), encoding="utf-8")
        rl_steps = 0
        peak = 0.
        rl_peak = None
        result = {"completed": False, "failed_during": None}
        with torch.no_grad(), (output / "steps.jsonl").open("w", encoding="utf-8") as log:
            for step in range(raw.max_episode_length):
                if not app.is_running():
                    result["failed_during"] = "interrupted"
                    break
                preparing = step * raw.step_dt < args.warmup
                if args.arm_mode != "source":
                    robot.set_joint_position_target(nominal, joint_ids=arm_ids)
                    robot.set_joint_velocity_target(zeros, joint_ids=arm_ids)
                    robot.set_joint_effort_target(zeros, joint_ids=arm_ids)
                # Record real arm state/target BEFORE automatic reset can erase it.
                arm_record = {"arm_q": robot.data.joint_pos[0, arm_ids].tolist(),
                              "arm_dq": robot.data.joint_vel[0, arm_ids].tolist(),
                              "arm_target": robot.data.joint_pos_target[0, arm_ids].tolist(),
                              "arm_effort": robot.data.applied_torque[0, arm_ids].tolist()}
                action = (torch.zeros(1, 10, device=raw.device) if preparing else
                          policy(torch.cat((encoder(obs["obsHistory"].flatten(1)), obs["policy"], obs["commands"]), -1)))
                rl_steps += int(not preparing)
                obs, _, done, info = env.step(action)
                state = info["standwalk_state"]
                tilt = float(state["tilt_deg"][0])
                if not math.isfinite(tilt) or not torch.isfinite(action).all():
                    raise RuntimeError("Non-finite support probe state/action.")
                peak = max(peak, tilt)
                if not preparing:
                    rl_peak = max(rl_peak or 0., tilt)
                row = {"time_s": (step + 1) * raw.step_dt, "stage": "PD" if preparing else "RL",
                       "tilt_deg": tilt, "height_m": float(state["position"][0, 2]),
                       "velocity": state["linear_velocity"][0].tolist(), "yaw_rate": float(state["yaw_rate"][0]),
                       "base_wrench": raw.command_manager.get_term("arm_wrench").applied_wrench[0].tolist(),
                       "done": bool(done[0]), **arm_record}
                log.write(json.dumps(row) + "\n")
                if step % 50 == 0 or bool(done[0]):
                    print(f"[PROBE] t={row['time_s']:.2f}s stage={row['stage']} tilt={tilt:.2f} "
                          f"q={arm_record['arm_q']} target={arm_record['arm_target']}", flush=True)
                if bool(done[0]):
                    result["completed"] = bool(info["time_outs"][0]) and not bool(raw.reset_terminated[0])
                    if not result["completed"]:
                        result["failed_during"] = row["stage"]
                    break
        result.update(rl_seconds=rl_steps * raw.step_dt, peak_tilt_deg=peak, rl_peak_tilt_deg=rl_peak, output=str(output))
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("[PROBE RESULT] " + json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = parser_for_probe()
    if "--help" in sys.argv or "-h" in sys.argv:
        parser.parse_args()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        main(args, app)
    finally:
        app.close()
