"""Opt-in SFYG motor lifecycle; default is NO motor output. Linux/Jetson terminal."""
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
from sfyg_lifecycle import Lifecycle, CommandRejected, load_bundle, sdk_packet


CHECKS = ("joint_mapping_checked", "arm_in_carry_pose_and_payload_0p5kg",
          "low_level_dev_mode", "gantry_and_emergency_stop_ready")


class MotorLease:
    """Separate 200 Hz publisher. A stalled inference/logger cannot renew its lease.

    Best-effort host watchdog, NOT a replacement for robot-side command timeout
    or the physical emergency stop. SDK/network failure can prevent delivery.
    """
    def __init__(self, send, samples, coords, kd, clock=time.monotonic):
        self.send, self.samples, self.coords, self.clock = send, samples, coords, clock
        self.kd = list(kd)
        self.lock = threading.Lock()
        self.packet = None
        self.deadline = 0.
        self.fault = None
        self.armed = False
        self.joy_at = None
        self.closed = threading.Event()
        self.thread = None
        self.sent = 0

    def abort(self, reason):
        with self.lock:
            self.fault = self.fault or str(reason)

    def joystick(self, msg):
        try:
            buttons = list(msg.buttons)
            with self.lock:
                self.joy_at = self.clock()
            if len(buttons) < 6:
                self.abort("invalid joystick message")
            elif buttons[4] and (buttons[2] or buttons[0]):
                # Local app abort, never auto restart when the buttons are released.
                self.abort("remote L1+square/cross abort")
        except Exception as exc:
            self.abort(exc)

    def renew(self, packet, sample_time):
        with self.lock:
            if self.fault:
                return
            if self.clock()-sample_time > .04:
                self.fault = "inference exceeded fresh-sample deadline"
                return
            self.packet = {k: (list(v) if isinstance(v, list) else v) for k, v in packet.items()}
            self.deadline = sample_time+.06
            self.armed = True

    def pump(self):
        with self.lock:
            if not self.armed:
                return
            now = self.clock()
            if now > self.deadline:
                self.fault = self.fault or "motor lease expired"
            if self.joy_at is None or now-self.joy_at > 2.:
                self.fault = self.fault or "remote heartbeat missing"
            packet = self.packet
        try:
            fresh = self.samples.snapshot(.04, .015)
            if fresh is None:
                self.abort("joint/IMU stale or skewed")
        except Exception as exc:
            self.abort(exc)
        with self.lock:
            failed = self.fault is not None
        if failed:
            packet = dict(q=packet["q"], kp=[0.]*10, kd=self.kd, state="FAULT")
        self.send(sdk_packet(packet, self.coords))
        self.sent += 1

    def start(self):
        def loop():
            due = self.clock()
            while not self.closed.is_set():
                try:
                    self.pump()
                except Exception as exc:
                    self.abort(f"publisher: {exc}")
                due += .005
                if self.clock()-due > .02:
                    self.abort("publisher scheduling deadline")
                    due = self.clock()
                self.closed.wait(max(0., due-self.clock()))
        self.thread = threading.Thread(target=loop, name="sfyg_motor_watchdog", daemon=True)
        self.thread.start()

    def close(self):
        # Continue best-effort damping for one second after a fault/interrupt.
        if self.armed:
            self.abort("shutdown damping")
            time.sleep(1.)
        self.closed.set()
        if self.thread:
            self.thread.join(timeout=2.)


def make_sender(robot):
    from limxsdk import datatypes
    def send(packet):
        msg = datatypes.RobotCmd()
        msg.mode = [0.]*10
        msg.motor_names = [""]*10
        msg.q, msg.Kp, msg.Kd = packet["q"], packet["kp"], packet["kd"]
        msg.dq, msg.tau = [0.]*10, [0.]*10
        if robot.publishRobotCmd(msg) is False:
            raise RuntimeError("SDK rejected motor command")
    return send


def validate_preflight(args, cfg):
    if not args.execute:
        return None
    if not args.preflight:
        raise ValueError("--execute requires a reviewed --preflight JSON")
    profile = json.loads(args.preflight.read_text(encoding="utf-8"))
    if (profile.get("policy_sha256") != cfg["policy_sha256"]
            or any(profile.get(key) is not True for key in CHECKS)
            or not str(profile.get("operator", "")).strip()):
        raise ValueError("Preflight incomplete; bare robot must remain read-only with this loaded policy")
    Coordinates(args.preflight)  # Validate before SDK initialization or any motor output.
    return profile


def run(args):
    cfg, policy = load_bundle(args.bundle_dir)
    profile = validate_preflight(args, cfg)
    coords = Coordinates(args.preflight)
    ctl = Lifecycle(cfg, policy)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    (out/"interface.json").write_text(json.dumps(dict(
        policy_sha256=cfg["policy_sha256"], checkpoint=cfg["checkpoint_path"],
        motor_output=args.execute, preflight=profile, coordinates=coords.config,
        policy_period_s=.02, publisher_period_s=.005, host_lease_s=.06,
        previous_action="requested action after startup blend, before torque projection; raw actor in steady policy",
        prepare_target=ctl.default.tolist(), hardware_acceptance=False), indent=2), encoding="utf-8")
    samples = Samples()
    robot = subscribe(args.robot_ip, samples)
    lease = MotorLease(make_sender(robot), samples, coords, ctl.kd) if args.execute else None
    if lease:
        if robot.subscribeSensorJoy(lease.joystick) is False:
            raise RuntimeError("SDK joystick subscription failed")
    commands = queue.Queue()
    def read_commands():
        try:
            for line in sys.stdin:
                commands.put(line.strip())
        finally:
            if lease:
                lease.abort("terminal EOF")
            commands.put("abort")
    threading.Thread(target=read_commands, daemon=True).start()
    # SIGINT is an emergency damping request, not a normal stand-preserving stop.
    def interrupt(signum, frame):
        if lease:
            lease.abort(f"signal {signum}")
        commands.put("abort")
    old_handlers = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
    print("[SFYG] motor_output=", args.execute, "output=", out, flush=True)
    print("Commands (Enter): prepare supported | start grounded | vx 0.15 | stop | release supported | abort | quit", flush=True)
    print("WAIT sends no motor commands. Remote L1+square/cross aborts; terminal stop ramps to zero policy.", flush=True)
    start_wait = time.monotonic()
    start = last_stamp = due = None
    frames, reason, last_status = 0, "completed", None
    try:
        if lease:
            lease.start()
        with (out/"steps.jsonl").open("w", encoding="utf-8", buffering=1) as stream:
            while True:
                sample = samples.snapshot(.04, .015)
                now = time.monotonic()
                if lease and lease.fault:
                    raise RuntimeError(lease.fault)
                if sample is None:
                    if start is not None or now-start_wait > 12:
                        raise RuntimeError("Missing fresh joint/IMU data")
                    time.sleep(.002)
                    continue
                if start is None:
                    start = due = now
                if now-due > .03:
                    raise RuntimeError("50 Hz inference scheduling deadline missed")
                _, joint, imu = sample
                stamp = (joint[0], imu[0])
                if last_stamp and (stamp[0] == last_stamp[0] or stamp[1] == last_stamp[1]):
                    raise RuntimeError("Duplicate joint/IMU sample across policy ticks")
                last_stamp = stamp
                q, dq, angular, gravity = coords.convert(joint, imu)
                event = request = None
                input_line = commands.get_nowait() if not commands.empty() else ""
                if input_line == "quit":
                    if ctl.state == "WAIT":
                        break
                    print("Use stop, wait ZERO, attach support, release supported. abort is emergency damping.", flush=True)
                elif input_line:
                    mapping = {"prepare supported": "prepare_supported", "start grounded": "start_grounded",
                               "stop": "stop", "release supported": "release_supported", "abort": "abort"}
                    if input_line.startswith("vx "):
                        try:
                            request = [float(input_line.split()[1]), 0., 0.]
                        except (ValueError, IndexError):
                            print("Expected vx 0..0.20", flush=True)
                    else:
                        event = mapping.get(input_line)
                        if event is None:
                            print("Unknown command", flush=True)
                if lease and event == "prepare_supported" and (lease.joy_at is None or now-lease.joy_at > 2.):
                    raise RuntimeError("No live remote heartbeat; prepare refused")
                rejected = None
                try:
                    packet = ctl.tick(now-start, q, dq, angular, gravity, event, request)
                except CommandRejected as exc:
                    rejected = str(exc)
                    print("[REJECTED]", rejected, flush=True)
                    packet = ctl.tick(now-start, q, dq, angular, gravity)
                if lease and ctl.state != "WAIT":
                    lease.renew(packet, now)
                status = (ctl.state, ctl.ready)
                if status != last_status:
                    print(f"[STATE] {ctl.state} preparation_settled={ctl.ready}", flush=True)
                    last_status = status
                stream.write(json.dumps(dict(time_s=now-start, state=ctl.state, ready=ctl.ready,
                    event=input_line, rejected=rejected, joint_stamp=joint[0], imu_stamp=imu[0], q=q.tolist(), dq=dq.tolist(),
                    gravity=gravity.tolist(), angular=angular.tolist(), command=ctl.command.tolist(),
                    raw_action=ctl.raw.tolist(), applied_action=ctl.previous.tolist(), packet=packet,
                    observation=None if ctl.obs is None else ctl.obs.tolist(),
                    history=None if ctl.hist is None else ctl.hist.tolist(),
                    motor_output=args.execute), allow_nan=False)+"\n")
                frames += 1
                if ctl.state == "FAULT":
                    raise RuntimeError(ctl.reason)
                if ctl.state == "DONE":
                    break
                due += .02
                time.sleep(max(0., due-time.monotonic()))
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        ctl.fault(reason, 0. if start is None else time.monotonic()-start)
        if lease:
            lease.abort(reason)
        raise
    finally:
        if lease:
            lease.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        summary = dict(completed=reason == "completed", state=ctl.state, reason=reason, frames=frames,
                       motor_output=args.execute, events=ctl.events, hardware_acceptance=False,
                       publish_count=0 if lease is None else lease.sent,
                       watchdog_reason=None if lease is None or lease.fault == "shutdown damping" else lease.fault)
        (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("[SUMMARY]", summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--robot_ip", default="10.192.1.2")
    parser.add_argument("--preflight", type=Path, help="Coordinates and manually reviewed hardware checks")
    parser.add_argument("--execute", action="store_true", help="Enable motor publishing AFTER explicit prepare command")
    run(parser.parse_args())
