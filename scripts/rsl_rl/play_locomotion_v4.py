"""Play v4 checkpoints with their physical payload and strict observation contract.

Run from a graphical desktop without --headless. --quick switches to one
continuous random-command rollout; --video records it with velocity overlays.
Without --quick, uses the deterministic v4 acceptance cases.
"""
import sys
import traceback
import json
import math
import time

from train_locomotion_v4 import main, parser_for_task


def preview(env, runner, args, output, app):
    """One continuous rollout with random commands; diagnostic, not acceptance."""
    import torch
    raw = env.unwrapped
    command = raw.command_manager.get_term("base_velocity")
    command.roles.fill_(1)  # Every episode is a moving role, not a fixed standing quota.
    command.cfg.resampling_time_range = (args.command_interval, args.command_interval)
    if args.base_command is not None:
        command.cfg.evaluation_case = 1
        command.cfg.evaluation_velocity = tuple(args.base_command)
    raw.reset(seed=args.seed)
    obs = env.get_observations()
    policy = runner.get_inference_policy(device=raw.device)
    encoder = runner.get_inference_encoder(device=raw.device)
    writer = None
    frames, steps, failures, timeouts = 0, 0, 0, 0
    requested_steps = max(1, round(args.seconds / raw.step_dt))
    video_path = output / "preview.mp4"
    try:
        if args.video:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw, ImageFont
            # Prime RTX without advancing physics; prevent silent black videos.
            frame = None
            for _ in range(40):
                frame = raw.render()
            if frame is None or not bool(frame.any()):
                raise RuntimeError("Camera still empty/black after warmup; check RTX rendering.")
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 20)
            except OSError:
                try:
                    font = ImageFont.load_default(size=20)
                except TypeError:
                    font = ImageFont.load_default()
            writer = imageio.get_writer(str(video_path), fps=round(1 / raw.step_dt), codec="libx264")
        with (output / "preview_steps.jsonl").open("w", encoding="utf-8") as stream:
            with torch.no_grad():
                for step in range(requested_steps):
                    if not app.is_running():
                        break
                    started = time.perf_counter()
                    action = policy(torch.cat((encoder(obs["obsHistory"].flatten(1)),
                                               obs["policy"], obs["commands"]), -1))
                    obs, reward, dones, infos = env.step(action)
                    state = infos["standwalk_state"]
                    if not all(torch.isfinite(v).all() for v in (*obs.values(), *state.values(), reward)):
                        raise FloatingPointError("Nonfinite preview observation/action state.")
                    steps += 1
                    done = bool(dones[0])
                    timeout = done and bool(infos["time_outs"][0])
                    timeouts += int(timeout)
                    failures += int(done and not timeout)
                    row = {"time_s": steps * raw.step_dt, "done": done, "timeout": timeout,
                           **{key: value[0].tolist() for key, value in state.items()}}
                    stream.write(json.dumps(row) + "\n")
                    cmd = row["command"]
                    vel = row["linear_velocity"]
                    line = (f"cmd vx/vy/wz={cmd[0]:+.3f}/{cmd[1]:+.3f}/{cmd[2]:+.3f}  "
                            f"actual={vel[0]:+.3f}/{vel[1]:+.3f}/{row['yaw_rate']:+.3f}")
                    if step % max(1, round(1 / raw.step_dt)) == 0 or done:
                        print(f"[PLAY] t={row['time_s']:.2f}s {line} reset={done}", flush=True)
                    if writer is not None:
                        frame = raw.render()
                        if frame is None:
                            raise RuntimeError("Renderer returned no frame while recording.")
                        canvas = Image.fromarray(frame).convert("RGB")
                        draw = ImageDraw.Draw(canvas)
                        draw.rectangle((0, 0, canvas.width, 90), fill="black")
                        draw.text((10, 5), f"t={row['time_s']:.2f}s  tilt={row['tilt_deg']:.1f} deg  "
                                  f"resets={failures + timeouts}  terminal={done}", font=font, fill="white")
                        draw.text((10, 35), line, font=font, fill="white")
                        import numpy as np
                        writer.append_data(np.asarray(canvas))
                        frames += 1
                    if args.real_time:
                        time.sleep(max(0., raw.step_dt - (time.perf_counter() - started)))
    finally:
        if writer is not None:
            writer.close()
        report = {"mode": "preview", "acceptance_evaluated": False,
                  "completed": steps == requested_steps, "steps": steps, "frames": frames,
                  "sim_time_s": steps * raw.step_dt, "failure_resets": failures,
                  "timeout_resets": timeouts, "video": str(video_path) if frames else None}
        (output / "preview.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[PLAY] {json.dumps(report)}", flush=True)


if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    parser = parser_for_task()
    parser.description = __doc__
    parser.set_defaults(mode="eval", num_envs=1, episodes=1, real_time=True)
    parser.add_argument("--fast", action="store_true", help="Disable real-time pacing")
    parser.add_argument("--quick", action="store_true", help="Continuous random-command preview instead of the evaluation suite")
    parser.add_argument("--seconds", type=float, default=30., help="Quick preview duration in simulation seconds")
    parser.add_argument("--command_interval", type=float, default=3., help="Seconds between random velocity commands")
    parser.add_argument("--base_command", type=float, nargs=3, help="Quick preview: fixed physical vx/vy/wz instead of random commands")
    parser.add_argument("--video", action="store_true", help="Quick preview: save preview.mp4 with velocity overlay")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.mode != "eval" or args.resume:
        parser.error("Play requires --mode eval without --resume.")
    if not args.checkpoint_path:
        parser.error("--checkpoint_path is required.")
    if args.num_envs != 1:
        parser.error("Play uses one environment for deterministic case evaluation.")
    if (not math.isfinite(args.seconds) or args.seconds <= 0
            or not math.isfinite(args.command_interval) or args.command_interval <= 0):
        parser.error("--seconds and --command_interval must be finite and positive.")
    if args.video or args.base_command is not None:
        args.quick = True
    if args.base_command is not None and any(not math.isfinite(v) or abs(v) > limit
                                            for v, limit in zip(args.base_command, args.velocity_limits)):
        parser.error("--base_command must be finite and inside --velocity_limits.")
    if args.video:
        args.enable_cameras = True
    args.real_time = not args.fast
    launcher = AppLauncher(args)
    try:
        main(args, launcher.app)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        launcher.app.close()
