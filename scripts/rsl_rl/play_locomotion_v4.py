"""Play v4 checkpoints with their physical payload and strict observation contract.

Run from a graphical desktop without --headless. Uses the same deterministic
cases, terminal diagnostics and result files as the v4 evaluation entry point.
"""
import sys
import traceback

from train_locomotion_v4 import main, parser_for_task


if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    parser = parser_for_task()
    parser.description = __doc__
    parser.set_defaults(mode="eval", num_envs=1, episodes=1, real_time=True)
    parser.add_argument("--fast", action="store_true", help="Disable real-time pacing")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.mode != "eval" or args.resume:
        parser.error("Play requires --mode eval without --resume.")
    if not args.checkpoint_path:
        parser.error("--checkpoint_path is required.")
    if args.num_envs != 1:
        parser.error("Play uses one environment for deterministic case evaluation.")
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
