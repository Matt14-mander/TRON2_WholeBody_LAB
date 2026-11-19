"""Standalone suspended leg preparation. Default read-only; never runs a policy."""
import argparse
import json
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import numpy as np
from shadow_sfyg_locomotion import Samples, Coordinates, subscribe
from sfyg_supported_prepare import SupportedPrepare, RECORDED, configuration
from sfyg_lifecycle import CommandRejected
from run_sfyg_lifecycle import MotorLease, make_sender
from sfyg_arm_guard import ArmGuard

CHECKS = ('joint_mapping_checked', 'imu_mapping_checked', 'low_level_dev_mode',
          'drive_enable_semantics_confirmed', 'exclusive_leg_control',
          'gantry_supports_full_weight', 'feet_clear_entire_path',
          'emergency_stop_and_robot_timeout_verified', 'arm_carry_verified')


def validate(profile, cfg, execute):
    if profile.get('quaternion_order') != 'wxyz':
        raise ValueError('This verified SDK path requires raw wxyz')
    if execute and (not str(profile.get('operator', '')).strip()
                    or profile.get('policy_sha256') != cfg['policy_sha256']
                    or any(profile.get(k) is not True for k in CHECKS)
                    or profile.get('payload_mass_kg') not in (0, .5)):
        raise ValueError('Review every suspended-preparation check; no ground-contact or automatic policy handoff')


def run(args):
    cfg = configuration(args.bundle_dir)
    profile = json.loads(args.profile.read_text())
    validate(profile, cfg, args.execute)
    coords = Coordinates(args.profile)
    ctl = SupportedPrepare(cfg)
    guard = ArmGuard(args.arm_status_file, cfg['policy_sha256'])
    if args.execute: guard.check()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'interface.json').write_text(json.dumps(dict(
        profile=profile, policy_inference=False, motor_output=args.execute,
        recorded_start=RECORDED.tolist(), default=ctl.default.tolist(),
        hardware_validated=False), indent=2))
    samples = Samples()
    robot = subscribe(args.robot_ip, samples)
    lease = MotorLease(make_sender(robot), samples, coords, ctl.kd, arm_guard=guard) if args.execute else None
    commands = queue.Queue()
    if lease:
        if robot.subscribeSensorJoy(lease.joystick) is False:
            raise RuntimeError('Remote subscription failed')
        def read():
            try:
                for line in sys.stdin: commands.put(line.strip())
            finally:
                lease.abort('terminal EOF')
                commands.put('abort')
        threading.Thread(target=read, daemon=True).start()
    def interrupt(*_):
        if lease: lease.abort('operator signal')
        commands.put('abort')
    old_handlers = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
    print('NO POLICY. WAIT sends nothing. Commands: prepare suspended | stop | release supported | abort | quit', flush=True)
    print('stop freezes the current reference; abort/Ctrl+C requests damping. Release requires support, may let legs sag.', flush=True)
    start = due = None
    waiting = time.monotonic()
    last_stamp = None
    previous = None
    frames = 0
    reason = 'completed'
    try:
        if lease: lease.start()
        with (args.output_dir/'steps.jsonl').open('w', buffering=1) as f:
            while True:
                now = time.monotonic()
                if lease and lease.fault: raise RuntimeError(lease.fault)
                sample = samples.snapshot(.04, .015)
                if sample is None:
                    if start is not None or now-waiting > 12: raise RuntimeError('Missing fresh joint/IMU')
                    time.sleep(.002)
                    continue
                if start is None: start = due = now
                if now-due > .03: raise RuntimeError('control deadline missed')
                _, joint, imu = sample
                stamp = joint[0], imu[0]
                if last_stamp and (stamp[0] == last_stamp[0] or stamp[1] == last_stamp[1]):
                    raise RuntimeError('duplicate joint/IMU sample')
                last_stamp = stamp
                q, dq, angular, gravity = coords.convert(joint, imu)
                event = commands.get_nowait() if not commands.empty() else None
                if event == 'quit':
                    if ctl.state == 'WAIT': break
                    print('Use stop then release supported after PAUSED/HOLD; no handoff while this process owns motors', flush=True)
                    event = None
                if not args.execute and event == 'abort': break
                if event == 'prepare suspended':
                    if lease.joy_at is None or now-lease.joy_at > 2: raise RuntimeError('No fresh remote heartbeat')
                    guard.check()
                try:
                    packet = ctl.tick(now-start, q, dq, gravity, event)
                except CommandRejected as exc:
                    print('[REJECTED]', exc, flush=True)
                    packet = ctl.tick(now-start, q, dq, gravity)
                if lease and ctl.state != 'WAIT': lease.renew(packet, now)
                if previous != ctl.state:
                    print('[STATE]', ctl.state, flush=True)
                    previous = ctl.state
                f.write(json.dumps(dict(time_s=now-start, state=ctl.state, q=q.tolist(), dq=dq.tolist(),
                    gravity=gravity.tolist(), packet=packet, motor_output=args.execute), allow_nan=False)+'\n')
                frames += 1
                if not args.execute and now-start >= 10:
                    preview = dict(start_error_rad=(q-RECORDED).tolist(),
                        start_within_envelope=bool(max(abs(q-RECORDED)) <= .08),
                        prepare_target=ctl.default.tolist(), policy_calls=0, motor_output=False)
                    print('[PREVIEW]', json.dumps(preview), flush=True)
                    (args.output_dir/'preview.json').write_text(json.dumps(preview, indent=2))
                    break
                if ctl.state == 'FAULT': raise RuntimeError(ctl.reason)
                if ctl.state == 'DONE': break
                due += .02
                time.sleep(max(0, due-time.monotonic()))
    except BaseException as exc:
        reason = str(exc)
        if lease: lease.abort(reason)
        raise
    finally:
        if lease: lease.close()
        for s, handler in old_handlers.items(): signal.signal(s, handler)
        (args.output_dir/'summary.json').write_text(json.dumps(dict(
            reason=reason, state=ctl.state, frames=frames, policy_calls=0,
            motor_output=args.execute, events=ctl.events, hardware_validated=False), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle_dir', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--arm_status_file', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--robot_ip', default='10.192.1.2')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    # One instance of this independent controller. No parallel lifecycle allowed.
    import fcntl
    lock_path = Path.home()/'sfyg_shadow/run/supported_prepare.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(a)
