"""Direct pyAgxArm J-mode carry commissioning, preview by default.

Operator confirms bounded inward recovery, enable, then carry separately.
No reset/calibration/gripper operation or leg HOLD interlock publication.
"""
import argparse
import math
import queue
import sys
import threading
import time


def recovery(q, limits):
    if len(q) != 6 or len(limits) != 6 or not all(math.isfinite(v) for v in q):
        raise ValueError("Expected six finite angles/limits")
    entry = []
    for v, (low, high) in zip(q, limits):
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError("Invalid controller limits")
        target = max(low, min(v, high))
        if abs(v-target) > math.radians(7):
            raise ValueError("Start exceeds bounded inward recovery (7 degrees); no automatic calibration")
        entry.append(target)
    return entry


def validate_target(q, limits):
    if len(q) != 6 or any(not math.isfinite(v) or not lo <= v <= hi
                         for v, (lo, hi) in zip(q, limits)):
        raise ValueError("Carry target outside controller limits")


def sample(robot):
    angle, status = robot.get_joint_angles(), robot.get_arm_status()
    now = time.time()
    if any(m is None or not 0 <= now-m.timestamp <= .3 for m in (angle, status)):
        raise RuntimeError("Missing/stale joint or controller status")
    if int(status.msg.arm_status) != 0:
        raise RuntimeError(f"Controller fault: {status.msg}")
    q = list(angle.msg)
    if len(q) != 6 or not all(math.isfinite(v) for v in q):
        raise RuntimeError("Invalid measured angles")
    return q, status.msg


def run(robot, execute, carry, commands, clock=time.monotonic, sleep=time.sleep):
    limits = []
    for i in range(1, 7):
        msg = robot.get_joint_angle_vel_limits(i)
        if msg is None:
            raise RuntimeError(f"Missing controller limits joint{i}")
        limits.append((float(msg.msg.min_angle_limit), float(msg.msg.max_angle_limit)))
    validate_target(carry, limits)
    q, _ = sample(robot)
    entry = recovery(q, limits)
    print("Current SDK deg:", [round(math.degrees(v), 3) for v in q], flush=True)
    print("Inward recovery SDK deg:", [round(math.degrees(v), 3) for v in entry], flush=True)
    print("Carry SDK deg:", [round(math.degrees(v), 3) for v in carry], flush=True)
    print("Target mapping and swept path require physical verification; 5% is a speed ratio, not rad/s.", flush=True)
    if not execute:
        print("PREVIEW ONLY: no enable, mode, speed or motion commands sent.", flush=True)
        return
    print("Supported robot/arm, exclusive CAN, clear recovery path: type ENABLE. Then CARRY at READY. STOP SUPPORTED exits.", flush=True)
    active, state, deadline, stable = False, "WAIT", None, None
    target = entry
    last_enable = -float("inf")
    ack_until = None
    last_mode_print = -float("inf")
    try:
        while True:
            measured, status = sample(robot)
            now = clock()
            command = commands.get_nowait() if not commands.empty() else ""
            if command == "ABORT":
                raise RuntimeError("Interrupted/terminal closed")
            if command == "STOP SUPPORTED":
                break
            if command == "ENABLE" and state == "WAIT":
                if max(abs(a-b) for a, b in zip(q, measured)) > .02:
                    raise RuntimeError("Start changed; rerun preview")
                # Preload the explicitly displayed legal entry, not an out-of-limit angle
                # which a different SDK version might silently clamp.
                active = True
                robot.set_auto_set_motion_mode_enabled(False)
                robot.set_speed_percent(5)
                robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
                robot.move_j(entry)
                state, deadline, stable = "ENABLING", now+10, None
                print("[ENABLING] preload legal recovery target; waiting for enable acknowledgement", flush=True)
            elif command == "CARRY" and state == "READY":
                target = carry
                robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
                robot.move_j(target)
                state, deadline, stable = "CARRYING", now+120, None
                ack_until = now+2
            elif command:
                print(f"Command refused in {state}", flush=True)
            if state == "ENABLING":
                if now-last_enable >= .2:
                    enabled = robot.enable()
                    last_enable = now
                    if enabled:
                        # Enabling and mode selection are separate handshakes. Some
                        # firmware may not retain the mode sent before enabling.
                        robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
                        robot.move_j(entry)
                        state = "RECOVERING"
                        ack_until = clock()+3
                        deadline = clock()+10
                        print("[RECOVERING] enable acknowledged; J mode and recovery target reissued", flush=True)
            if state in {"RECOVERING", "READY", "CARRYING", "HOLD"}:
                if int(status.ctrl_mode) != 1 or int(status.mode_feedback) != 1:
                    if now-last_mode_print >= .5:
                        print(f"[MODE] state={state} ctrl_mode={int(status.ctrl_mode):#04x} "
                              f"mode_feedback={int(status.mode_feedback):#04x} "
                              f"motion_status={int(status.motion_status):#04x}", flush=True)
                        last_mode_print = now
                    # Allow mode feedback a bounded acknowledgement interval.
                    if now > ack_until:
                        raise RuntimeError(f"CAN/J mode not acknowledged: ctrl_mode={int(status.ctrl_mode):#04x}, "
                                           f"mode_feedback={int(status.mode_feedback):#04x}; no carry sent unless CARRYING")
                error = max(abs(a-b) for a, b in zip(measured, target))
                settled = error < .02 and int(status.motion_status) == 0 and int(status.mode_feedback) == 1
                stable = (now if stable is None else stable) if settled else None
                if state in {"RECOVERING", "CARRYING"} and stable is not None and now-stable >= 1:
                    state = "READY" if state == "RECOVERING" else "HOLD"
                    deadline = None
                    print(f"[{state}] SDK deg={[round(math.degrees(v), 3) for v in measured]}", flush=True)
                if state in {"READY", "HOLD"} and (error > .05 or int(status.mode_feedback) != 1
                                                    or int(status.ctrl_mode) != 1):
                    raise RuntimeError("Position hold lost")
            if deadline is not None and now > deadline:
                raise RuntimeError(f"{state} timeout")
            sleep(.02)
    finally:
        if active:
            robot.electronic_emergency_stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--carry_sdk_deg", nargs=6, type=float, required=True,
                        help="Explicit operator-reviewed SDK angles, not inferred simulation mapping")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW
    cfg = create_agx_arm_config(robot=ArmModel.PIPER_X, firmeware_version=PiperFW.DEFAULT,
                               interface="socketcan", channel=args.can)
    robot = AgxArmFactory.create_arm(cfg)
    commands = queue.Queue()
    if args.execute:
        def read():
            for line in sys.stdin:
                commands.put(line.strip())
            commands.put("ABORT")
        threading.Thread(target=read, daemon=True).start()
    try:
        robot.connect()
        time.sleep(1)
        run(robot, args.execute, [math.radians(v) for v in args.carry_sdk_deg], commands)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
