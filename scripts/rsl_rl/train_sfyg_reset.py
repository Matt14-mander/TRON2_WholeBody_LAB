"""Original SFYG, fixed-payload carry, and an isolated payload-robustness phase.

The initial PPO, runner, observations and rewards are verified against f441d67.
Only a launch adapter and the carry physical adapter are added. No v4 dependencies.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "rsl_rl"), str(ROOT / "exts/bipedal_locomotion")]


def parser_for_task():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("source", "carry", "robust"), default="source")
    p.add_argument("--mode", choices=("train", "eval", "probe"), default="train")
    p.add_argument("--num_envs", type=int)
    p.add_argument("--max_iterations", type=int, default=10000)
    p.add_argument("--save_interval", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir")
    p.add_argument("--checkpoint_path", help="Strict reset checkpoint; weights-only, fresh optimizers")
    p.add_argument("--payload_mass", type=float, default=.5)
    p.add_argument("--payload_mass_range", type=float, nargs=2, help="Robust only; default .25 .75 kg")
    p.add_argument("--friction_scale_range", type=float, nargs=2, help="Robust only; multiplier on authored friction, default .8 1.2")
    p.add_argument("--force_n", type=float, help="Robust only; horizontal force NORM bound, default 2 N")
    p.add_argument("--nominal_fraction", type=float, help="Robust only; default .4 train/probe, 0 eval")
    p.add_argument("--probe_steps", type=int, default=300, help="Robust physics diagnostic, not learning/acceptance")
    p.add_argument("--command", type=float, nargs=3, default=(.25, 0., 0.), help="Evaluation vx/vy/wz")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--video", action="store_true", help="Evaluation preview.mp4")
    return p


def main(args, app):
    import math
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    import bipedal_locomotion
    from bipedal_locomotion.controllers.sfyg_reset import SCHEMA, BASE, verify_source, load_weights
    from rsl_rl.runner import OnPolicyRunner
    if (min(args.max_iterations, args.save_interval, args.episodes) < 1
            or args.num_envs is not None and args.num_envs < 1
            or not math.isfinite(args.payload_mass) or args.payload_mass <= 0):
        raise ValueError("Positive finite mass and positive counts are required")
    if args.mode == "eval" and (not args.checkpoint_path or args.num_envs not in (None, 1)):
        raise ValueError("Evaluation needs a checkpoint and one environment")
    if args.video and args.mode != "eval":
        raise ValueError("Video is an evaluation option")
    if not all(math.isfinite(v) for v in args.command):
        raise ValueError("Evaluation command must be finite")
    robustness = None
    if args.phase == "robust":
        from dataclasses import asdict
        from bipedal_locomotion.controllers.payload_robustness import RobustSettings
        robustness = asdict(RobustSettings(
            mass_range=tuple(args.payload_mass_range or (.25, .75)),
            friction_scale_range=tuple(args.friction_scale_range or (.8, 1.2)),
            max_force_n=2. if args.force_n is None else args.force_n,
            nominal_mass_kg=args.payload_mass,
            nominal_fraction=(0. if args.mode == "eval" else .4) if args.nominal_fraction is None else args.nominal_fraction,
        ).validate())
        if args.mode == "train" and not args.checkpoint_path:
            raise ValueError("Robust training initializes from an existing carry/robust checkpoint")
    elif any(v is not None for v in (args.payload_mass_range, args.friction_scale_range, args.force_n, args.nominal_fraction)):
        raise ValueError("Randomization options require --phase robust")
    if args.mode == "probe" and (args.phase != "robust" or args.checkpoint_path or args.probe_steps < 1):
        raise ValueError("Probe needs robust phase, positive steps, and no learned checkpoint")
    core = verify_source(ROOT)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir or ROOT / "logs/rsl_rl" /
        ("sfyg_reset_" + args.phase + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    task = "Isaac-Limx-SFYG-TRON2A-Blind-Flat-v0"
    cfg = parse_env_cfg(task, device=args.device,
        num_envs=args.num_envs or (1 if args.mode == "eval" else 16 if args.mode == "probe" else 4096))
    cfg.seed = args.seed
    if args.phase in ("carry", "robust"):
        from bipedal_locomotion.tasks.locomotion.robots.limx_sfyg_carry_env_cfg import SFYGCarryFlatEnvCfg
        if robustness is not None:
            from bipedal_locomotion.tasks.locomotion.robots.limx_sfyg_robust_env_cfg import SFYGCarryRobustEnvCfg
            cfg = SFYGCarryRobustEnvCfg()
            cfg.robustness = robustness
        else:
            cfg = SFYGCarryFlatEnvCfg()
        cfg.scene.num_envs = args.num_envs or (1 if args.mode == "eval" else 16 if args.mode == "probe" else 4096)
        cfg.sim.device, cfg.seed = args.device, args.seed
        from bipedal_locomotion.controllers.paint_payload import create_payload_asset
        cfg.scene.robot.spawn.usd_path = create_payload_asset(cfg.scene.robot.spawn.usd_path,
            output / "robot_payload.usdc", args.payload_mass, (.20, .12, .08), (.15, 0., 0.))
    # Rendering/network-only adaptations; no friction, restitution, dynamics,
    # rewards, resets, observations or optimizer overrides in source training.
    cfg.commands.base_velocity.debug_vis = False
    cfg.scene.terrain.visual_material = None
    cfg.scene.light.spawn.texture_file = None
    if args.video:
        cfg.viewer.eye, cfg.viewer.lookat = (2.5, 2.5, 1.8), (0., 0., .65)
    if args.mode in ("eval", "probe"):
        c = cfg.commands.base_velocity
        c.heading_command, c.rel_heading_envs = False, 0.
        c.rel_standing_envs = float(not any(args.command))
        c.resampling_time_range = (1e9, 1e9)
        for key, value in zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), args.command):
            if abs(value) > (1. if key == "lin_vel_x" else .5 if key == "lin_vel_y" else 1.5):
                raise ValueError("Evaluation command exceeds the source envelope")
            setattr(c.ranges, key, (value, value))
        for group in (cfg.observations.policy, cfg.observations.obsHistory):
            group.enable_corruption = False
        for field in ("pose_range", "velocity_range"):
            cfg.events.reset_robot_base.params[field] = {
                name: (0., 0.) for name in cfg.events.reset_robot_base.params[field]}
        cfg.events.reset_robot_joints.params.update(position_range=(0., 0.), velocity_range=(0., 0.))
    agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent.seed, agent.device = args.seed, args.device
    agent.max_iterations, agent.save_interval = args.max_iterations, args.save_interval
    agent.experiment_name = "sfyg_reset_" + args.phase
    if cfg.events.push_robot is not None:
        raise ValueError("Reset route must have zero push disturbances")
    class RecordedEnv(ManagerBasedRLEnv):
        def state(self):
            robot = self.scene["robot"]
            data = robot.data
            feet = ["ankle_pitch_L_Link", "ankle_pitch_R_Link"]
            ids = self.scene.sensors["contact_forces"].find_bodies(feet, preserve_order=True)[0]
            arms = robot.find_joints(["arm[1-6]_Joint"], preserve_order=True)[0]
            result = {"velocity": data.root_lin_vel_b.clone(), "yaw_rate": data.root_ang_vel_b[:, 2].clone(),
                "tilt_deg": torch.rad2deg(torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1, 1))),
                "force_z": self.scene.sensors["contact_forces"].data.net_forces_w[:, ids, 2].clone(),
                "arm_error": (data.joint_pos[:, arms] - data.default_joint_pos[:, arms]).clone()}
            if robustness is not None:
                s = self._sfyg_robust
                result.update(payload_force_world=s.schedule.current.clone(),
                    payload_mass=s.masses[:,s.payload].clone(), friction_scale=s.frictions.clone(),
                    nominal_environment=s.nominal.clone())
            return result
        def _reset_idx(self, ids):
            active = ids[self.episode_length_buf[ids] > 0]
            if args.mode in ("eval", "probe") and len(active):
                s = self.state()
                self.terminal = (active.clone(), {k: v[active].clone() for k,v in s.items()})
            super()._reset_idx(ids)
        def step(self, action):
            self.terminal = None
            command = self.command_manager.get_command("base_velocity").clone()
            if robustness is not None:
                self._sfyg_robust.observe(command)
            result = super().step(action)
            if args.mode in ("eval", "probe"):
                s = self.state()
                if self.terminal is not None:
                    ids, values = self.terminal
                    for k in s:
                        s[k][ids] = values[k]
                s["command"] = command
                result[-1]["reset_state"] = s
            return result
    raw = RecordedEnv(cfg=cfg, render_mode="rgb_array" if args.video else None)
    env, runner = None, None
    try:
        env = RslRlVecEnvWrapper(raw)
        obs = env.get_observations()
        if obs["policy"].shape[1] != 42 or tuple(obs["obsHistory"].shape[1:]) != (10,42) or env.num_actions != 10:
            raise ValueError("Expected original SFYG policy42/history10x42/actions10")
        robot = raw.scene["robot"]
        masses = robot.root_physx_view.get_masses()
        payload_ids = robot.find_bodies("paint_payload_Link")[0] if args.phase != "source" else []
        if args.phase == "carry" and (len(payload_ids) != 1 or not torch.allclose(
                masses[:, payload_ids[0]], torch.full_like(masses[:, payload_ids[0]], args.payload_mass), atol=1e-4)):
            raise ValueError("Physical payload mass preflight failed")
        if robustness is not None:
            from bipedal_locomotion.tasks.locomotion.robots.limx_sfyg_robust_env_cfg import robust_state
            s = robust_state(raw)
            low, high = robustness["mass_range"]
            low, high = min(low,args.payload_mass), max(high,args.payload_mass)
            if (obs["critic"].shape[1] != 123 or len(payload_ids) != 1
                    or not bool(((masses[:,s.payload] >= low-1e-5) & (masses[:,s.payload] <= high+1e-5)).all())
                    or not torch.allclose(s.masses, masses.to(raw.device))):
                raise ValueError("Robust mass/critic preflight failed")
        manifest = {"schema": SCHEMA, "source_commit": BASE, "phase": args.phase,
            "core_files": core, "arguments": vars(args), "agent": agent.to_dict(),
            "environment": cfg.to_dict(),
            "observation_shapes": {k: list(v.shape[1:]) for k,v in obs.items()},
            "arm_lock": cfg.scene.robot.actuators["arm_lock"].to_dict(),
            "default_joint_pos": robot.data.default_joint_pos[0].tolist(),
            "total_mass_kg": masses.sum(-1).tolist(), "external_wrench": None if robustness else [0.,0.,0.,0.,0.,0.],
            "source_adaptations": ["disable velocity markers", "disable remote visual material/sky texture"]}
        if robustness is not None:
            manifest.update(robustness=robustness, nominal_environment_count=int(s.nominal.sum()),
                preflight_payload_masses=masses[:,s.payload].tolist(),
                preflight_friction_scales=s.frictions.tolist(),
                authored_friction_range=[float(s.materials[...,:2].min()),float(s.materials[...,:2].max())])
            print(f"[ROBUST] settings={robustness} nominal_envs={int(s.nominal.sum())}/{raw.num_envs} "
                  f"authored_friction={manifest['authored_friction_range']}",flush=True)
        (output / "contract.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        source_names = [*core, Path(__file__).relative_to(ROOT).as_posix(),
            "exts/bipedal_locomotion/bipedal_locomotion/controllers/sfyg_reset.py",
            "exts/bipedal_locomotion/bipedal_locomotion/controllers/paint_payload.py",
            "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_sfyg_carry_env_cfg.py"]
        if robustness is not None:
            source_names += ["exts/bipedal_locomotion/bipedal_locomotion/controllers/payload_robustness.py",
                "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_sfyg_robust_env_cfg.py"]
            if args.mode == "probe":
                source_names.append("scripts/rsl_rl/probe_sfyg_robust.py")
        manifest["adapter_sha256"] = {}
        for name in source_names:
            target = output / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
            if name not in core:
                manifest["adapter_sha256"][name] = hashlib.sha256(target.read_bytes()).hexdigest()
        (output / "contract.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        if args.mode == "probe":
            from probe_sfyg_robust import probe
            probe(env, args.probe_steps, output, app)
            return
        class SavedRunner(OnPolicyRunner):
            def log(self, locs, *a, **kw):
                super().log(locs, *a, **kw)
                if robustness is not None:
                    for name,value in raw._sfyg_robust.pop_metrics().items():
                        self.writer.add_scalar(name,value,locs["it"])
            def save(self, path, infos=None):
                super().save(path, infos)
                saved = torch.load(path, map_location="cpu", weights_only=False)
                saved["sfyg_reset_contract"] = manifest
                torch.save(saved, path)
        runner = SavedRunner(env, agent.to_dict(), log_dir=str(output), device=args.device)
        if args.checkpoint_path:
            checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
            load_weights(runner.alg, checkpoint, args.phase, evaluation=args.mode == "eval")
            args.checkpoint_phase = checkpoint["sfyg_reset_contract"]["phase"]
            manifest["checkpoint_phase"] = args.checkpoint_phase
            if args.mode == "eval" and args.phase == "carry" and checkpoint["sfyg_reset_contract"]["arguments"]["payload_mass"] != args.payload_mass:
                raise ValueError("Evaluate with the trained fixed payload mass")
            manifest["checkpoint_sha256"] = hashlib.sha256(Path(args.checkpoint_path).read_bytes()).hexdigest()
            (output / "contract.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        print(f"[RESET] phase={args.phase} source={BASE} mass={manifest['total_mass_kg'][0]:.4f} "
              f"shapes={manifest['observation_shapes']} LR={runner.alg.learning_rate} "
              f"early_stop={runner.alg.early_stop}", flush=True)
        if args.mode == "train":
            runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
        else:
            evaluate(env, runner, args, output, app)
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.close()
        raw.close()


def evaluate(env, runner, args, output, app):
    import torch
    policy = runner.get_inference_policy()
    encoder = runner.get_inference_encoder()
    raw = env.unwrapped
    writer = None
    if args.video:
        import imageio.v2 as imageio
        for _ in range(40):
            frame = raw.render()
        if frame is None or not bool(frame.any()):
            raise RuntimeError("Empty camera after render warmup")
        writer = imageio.get_writer(str(output / "preview.mp4"), fps=50, codec="libx264")
    reports = []
    try:
        with (output / "evaluation_steps.jsonl").open("w", encoding="utf-8") as stream:
            for episode in range(args.episodes):
                # PhysX lazy state caches can have been allocated by the prior
                # inference rollout. Reset must use the same inference scope.
                with torch.inference_mode():
                    raw.reset(seed=args.seed + episode)
                    obs = env.get_observations()
                rows, releases, previous = [], torch.zeros(2, dtype=torch.long), None
                interrupted = False
                with torch.inference_mode():
                    for step in range(raw.max_episode_length):
                        if not app.is_running():
                            interrupted = True
                            break
                        action = policy(torch.cat((encoder(obs["obsHistory"].flatten(1)), obs["policy"], obs["commands"]), -1))
                        obs, reward, done, info = env.step(action)
                        state = info["reset_state"]
                        if not all(torch.isfinite(v).all() for v in (*obs.values(), *state.values(), action, reward)):
                            raise FloatingPointError("Nonfinite evaluation tensors")
                        row = {k: v[0].tolist() for k,v in state.items()}
                        row.update(episode=episode, time_s=(step+1)*raw.step_dt,
                                   done=bool(done[0]), timeout=bool(info["time_outs"][0]))
                        stream.write(json.dumps(row) + "\n")
                        rows.append(row)
                        contact = (state["force_z"][0] > 5).cpu()
                        if previous is not None:
                            releases += previous & ~contact
                        previous = contact
                        if writer is not None:
                            writer.append_data(raw.render())
                        if step % 50 == 0 or bool(done[0]):
                            print(f"[EVAL] t={row['time_s']:.2f} cmd={row['command']} actual={row['velocity']} done={row['done']}", flush=True)
                        if bool(done[0]):
                            break
                kept = [r for r in rows if r["time_s"] >= 1.]
                velocity = torch.tensor([r["velocity"] for r in kept]) if kept else torch.empty(0,3)
                reports.append({"episode": episode, "interrupted": interrupted,
                    "completed": bool(rows and rows[-1]["timeout"] and rows[-1]["time_s"] >= 20.),
                    "duration_s": rows[-1]["time_s"] if rows else 0,
                    "mean_velocity_after_1s": velocity.mean(0).tolist() if kept else None,
                    "foot_releases": releases.tolist(),
                    "max_tilt_deg": max((r["tilt_deg"] for r in rows), default=0.),
                    "peak_arm_error_rad": max((max(abs(v) for v in r["arm_error"]) for r in rows), default=0.)})
                if interrupted:
                    break
    finally:
        if writer is not None:
            writer.close()
    result = {"kind": "source_reset_diagnostic", "is_acceptance": False,
        "command": args.command, "phase": args.phase,
        "checkpoint_phase": getattr(args,"checkpoint_phase",args.phase), "episodes": reports}
    (output / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[EVAL] {result}", flush=True)


if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    parser = parser_for_task()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.video:
        args.enable_cameras = True
    launcher = AppLauncher(args)
    try:
        main(args, launcher.app)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()
