"""Evaluate v3 without replacing its gait, rewards, actor, or physical asset.

training: the paper's arm-free SF model; wholebody: articulated SFYG transfer.
Fixed command schedules replace random velocity sampling for repeatable tests.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rsl_rl"))
sys.path.insert(0, str(ROOT / "exts/bipedal_locomotion"))
CASES = ("stand", "walk", "slow", "forward", "left", "right",
         "yaw_left", "yaw_right", "start_stop", "stop_start")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--robot", choices=("training", "wholebody"), default="training")
    parser.add_argument("--wrench", choices=("zero", "synthetic"), default="zero")
    parser.add_argument("--eval_cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transition_s", type=float, default=1.)
    parser.add_argument("--output_dir")
    return parser


def main(args, app):
    import torch
    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
    from isaaclab.utils import configclass
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    import bipedal_locomotion  # noqa: F401
    from bipedal_locomotion.controllers.ocs2_reset import configure_nominal_ocs2_reset
    from bipedal_locomotion.controllers.paint_teacher_core import FrozenLocomotion, ARM_JOINT_NAMES
    from bipedal_locomotion.controllers.standwalk_core import EpisodeMetrics, EVAL_CASES
    from bipedal_locomotion.tasks.standwalk.mdp import StandWalkVelocityCfg, fall
    from bipedal_locomotion.tasks.standwalk.env import StandWalkRLEnv

    if args.episodes < 1 or not math.isfinite(args.transition_s) or args.transition_s <= 0:
        raise ValueError("Positive episodes and finite positive transition_s are required")
    if len(set(args.eval_cases)) != len(args.eval_cases):
        raise ValueError("Duplicate evaluation cases")
    if args.robot == "wholebody" and args.wrench != "zero":
        raise ValueError("Synthetic arm wrench is only evaluated on the arm-free training asset")
    source = Path(args.checkpoint_path).expanduser().resolve()
    output = Path(args.output_dir or ROOT / "logs" / "paint" /
                  datetime.now().strftime("wholebody_eval_%Y%m%d_%H%M%S_%f")).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")

    # Load actor and encoder strictly; critic and optimizer are not used for evaluation.
    model = FrozenLocomotion(source, device=args.device)
    bound = getattr(model.actor[-1], "scale", None)
    if bound is None or not torch.isfinite(bound).all() or abs(float(bound) - 5.) > 1e-6:
        raise ValueError("This evaluator requires a v3 checkpoint with the 5*tanh(x/5) actor")
    task = "Isaac-Limx-SFYG-TRON2A-WholeBody-Flat"
    task += "-Play-v0" if args.robot == "wholebody" else "-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    cfg.seed = args.seed
    cfg.episode_length_s = 20.
    cfg.observations.policy.enable_corruption = False
    cfg.observations.obsHistory.enable_corruption = False
    configure_nominal_ocs2_reset(cfg, SceneEntityCfg("robot"))
    if args.wrench == "zero":
        cfg._disable_training_wrench_for_play()

    # Command order matters: the original gait reads the current standing mask.
    old = cfg.commands

    @configclass
    class EvaluationCommands:
        base_velocity = StandWalkVelocityCfg(evaluation_case=0, transition_s=args.transition_s)
        gait_command = deepcopy(old.gait_command)
        arm_wrench = deepcopy(old.arm_wrench)

    cfg.commands = EvaluationCommands()
    cfg.commands.gait_command.resampling_time_range = (1e9, 1e9)
    cfg.commands.gait_command.debug_vis = False
    cfg.terminations.evaluation_fall = TerminationTermCfg(func=fall)
    # Leave rewards, gait class, physical parameters and actuator delay unchanged.

    class ContactDiagnosticEnv(StandWalkRLEnv):
        def _state(self):
            state = super()._state()
            sensor = self.scene.sensors["contact_forces"]
            if not hasattr(self, "_evaluation_foot_ids"):
                ids, names = sensor.find_bodies(["ankle_pitch_L_Link", "ankle_pitch_R_Link"], preserve_order=True)
                if names != ["ankle_pitch_L_Link", "ankle_pitch_R_Link"]:
                    raise RuntimeError("Expected ordered left/right foot contact bodies")
                self._evaluation_foot_ids = ids
            state["foot_normal_force"] = sensor.data.net_forces_w[:, self._evaluation_foot_ids, 2].clone()
            state["gait_phase"] = self.command_manager.get_term("gait_command").gait_indices.clone()
            return state

    output.mkdir(parents=True)
    env = None
    results = []
    interrupted = False
    try:
        env = RslRlVecEnvWrapper(ContactDiagnosticEnv(cfg))
        raw = env.unwrapped
        robot = raw.scene["robot"]
        names = list(robot.joint_names)
        arm_ids = []
        if args.robot == "wholebody":
            arm_ids, matched = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
            if tuple(matched) != ARM_JOINT_NAMES:
                raise RuntimeError("SFYG arm joint order mismatch")
        elif any(name in names for name in ARM_JOINT_NAMES):
            raise RuntimeError("Training evaluation unexpectedly contains an articulated arm")
        command = raw.command_manager.get_term("base_velocity")
        wrench = raw.command_manager.get_term("arm_wrench")
        (output / "contract.json").write_text(json.dumps(dict(
            args=vars(args), checkpoint=str(source), task=task, joint_names=names,
            actor_bound=float(bound), delay="task defaults retained", env=cfg.to_dict()),
            indent=2, default=str), encoding="utf-8")
        print(f"[V3 EVAL] robot={args.robot} wrench={args.wrench} bound={float(bound)} "
              f"checkpoint={source} output={output}", flush=True)
        with (output / "evaluation_episodes.jsonl").open("w", encoding="utf-8") as episodes, \
                (output / "evaluation_steps.jsonl").open("w", encoding="utf-8") as steps_log:
            for case in args.eval_cases:
                command.cfg.evaluation_case, command.cfg.walk_velocity = EVAL_CASES[case]
                for trial in range(args.episodes):
                    if not app.is_running():
                        interrupted = True
                        break
                    raw.reset(seed=args.seed + trial)
                    # Explicitly hold all non-leg joints in SFYG transfer tests.
                    robot.set_joint_position_target(robot.data.default_joint_pos)
                    robot.set_joint_velocity_target(torch.zeros_like(robot.data.default_joint_pos))
                    obs = env.get_observations()
                    if (obs["policy"].shape != (1, 72) or obs["obsHistory"].shape != (1, 10, 42)
                            or obs["commands"].shape != (1, 3) or env.num_actions != 10):
                        raise RuntimeError("Expected original 78/420/10 observation/action contract")
                    metrics = EpisodeMetrics()
                    completed = False
                    action_peak = saturation = 0.
                    count = 0
                    termination = {}
                    for step in range(raw.max_episode_length):
                        if not app.is_running():
                            interrupted = True
                            break
                        with torch.no_grad():
                            latent = model.encoder(obs["obsHistory"].flatten(1))
                            action = model.actor(torch.cat((latent, obs["policy"], obs["commands"]), -1))
                        if not torch.isfinite(action).all() or float(action.abs().max()) > 5.00001:
                            raise RuntimeError("Non-finite or out-of-bound v3 deterministic action")
                        current_wrench = wrench.applied_wrench[0].tolist()
                        arm_state = dict(arm_q=robot.data.joint_pos[0, arm_ids].tolist(),
                                         arm_dq=robot.data.joint_vel[0, arm_ids].tolist()) if arm_ids else {}
                        action_peak = max(action_peak, float(action.abs().max()))
                        saturation += float((action.abs() >= 4.95).float().mean())
                        obs, _, done, info = env.step(action)
                        state = info["standwalk_state"]
                        if any(not torch.isfinite(value).all() for value in state.values()):
                            raise RuntimeError("Non-finite physical evaluation state")
                        count = step + 1
                        t = count * raw.step_dt
                        metrics.update(t, state["command"][0], state["linear_velocity"][0],
                                       state["yaw_rate"][0], state["position"][0, :2],
                                       state["tilt_deg"][0], raw.step_dt)
                        steps_log.write(json.dumps(dict(case=case, trial=trial, time_s=t,
                            command=state["command"][0].tolist(), velocity=state["linear_velocity"][0].tolist(),
                            yaw_rate=float(state["yaw_rate"][0]), tilt_deg=float(state["tilt_deg"][0]),
                            position=state["position"][0].tolist(),
                            gait_phase=float(state["gait_phase"][0]),
                            foot_normal_force=state["foot_normal_force"][0].tolist(),
                            action=action[0].tolist(),
                            base_wrench=current_wrench, action_abs_max=float(action.abs().max()), **arm_state)) + "\n")
                        if bool(done[0]):
                            completed = bool(info["time_outs"][0]) and not bool(raw.reset_terminated[0])
                            termination = {name: bool(raw.termination_manager.get_term(name)[0])
                                           for name in raw.termination_manager.active_terms}
                            break
                    row = dict(case=case, trial=trial, seed=args.seed + trial,
                        robot=args.robot, wrench=args.wrench,
                        **metrics.result(count * raw.step_dt, completed,
                                         require_standing=case in ("stand", "start_stop", "stop_start"),
                                         require_stop=case in ("start_stop", "stop_start")),
                        action_abs_max=action_peak, action_saturation_fraction=saturation / max(count, 1),
                        termination=termination)
                    results.append(row)
                    episodes.write(json.dumps(row) + "\n")
                    episodes.flush()
                    print(f"[EVAL] {case} trial={trial} completed={completed} passed={row['passed']} "
                          f"duration={row['duration_s']:.2f}s tilt={row['max_tilt_deg']:.2f}deg "
                          f"xy_rmse={row['linear_tracking_rmse']:.3f} "
                          f"yaw_rmse={row['yaw_tracking_rmse']:.3f} action_max={action_peak:.3f}", flush=True)
                    if interrupted:
                        break
                if interrupted:
                    break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        summary = dict(interrupted=interrupted, robot=args.robot, wrench=args.wrench,
            cases={case: dict(episodes=sum(r["case"] == case for r in results),
                             completed=sum(r["completed"] for r in results if r["case"] == case),
                             passed=sum(r["passed"] for r in results if r["case"] == case))
                   for case in args.eval_cases},
            all_passed=not interrupted and len(results) == len(args.eval_cases) * args.episodes
                       and all(r["passed"] for r in results))
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = make_parser()
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
