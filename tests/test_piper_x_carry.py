import math
from pathlib import Path
import queue
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/hardware"))
import piper_x_carry as direct

LIMITS = [(-2.6, 2.6), (0, math.pi), (-2.96, 0), (-1.55, 1.55), (-1.55, 1.55), (-2.09, 2.09)]


class Fake:
    def __init__(self):
        self.q = [0, -.10, .04, .59, 0, 0]
        self.calls = []
        self.OPTIONS = NS(MOTION_MODE=NS(J="j"))
    def get_joint_angle_vel_limits(self, i):
        a, b = LIMITS[i-1]
        return NS(msg=NS(min_angle_limit=a, max_angle_limit=b))
    def set_speed_percent(self, v): self.calls.append(("speed", v))
    def set_auto_set_motion_mode_enabled(self, value): self.calls.append(("auto_mode", value))
    def set_motion_mode(self, mode): self.calls.append(("mode", mode))
    def move_j(self, v):
        direct.validate_target(v, LIMITS)
        self.calls.append(("move", list(v)))
        self.q = list(v)
    def enable(self):
        self.calls.append(("enable",))
        return True
    def electronic_emergency_stop(self): self.calls.append(("stop",))


def sample(robot):
    return robot.q.copy(), NS(ctrl_mode=1, mode_feedback=1, motion_status=0)


class DirectTests(unittest.TestCase):
    def test_preview_no_write_calls(self):
        robot = Fake()
        with patch.object(direct, "sample", sample):
            direct.run(robot, False, [0, 1.57, -1.48, 0, 0, 0], queue.Queue())
        self.assertEqual(robot.calls, [])

    def test_recovery_never_expands_limits_or_moves_in_range_axes(self):
        q = [0, -.10, .04, .59, 0, 0]
        self.assertEqual(direct.recovery(q, LIMITS), [0, 0, 0, .59, 0, 0])
        q[1] = -.13
        with self.assertRaises(ValueError): direct.recovery(q, LIMITS)
        q[1] = float("nan")
        with self.assertRaises(ValueError): direct.recovery(q, LIMITS)

    def test_enable_then_explicit_carry_then_supported_stop(self):
        robot, commands = Fake(), queue.Queue()
        commands.put("ENABLE")
        now = [0.]
        def sleep(dt):
            now[0] += dt
            if 1.1 < now[0] < 1.12: commands.put("CARRY")
            if now[0] > 2.4: commands.put("STOP SUPPORTED")
        with patch.object(direct, "sample", sample):
            direct.run(robot, True, [0, 1.57, -1.48, 0, 0, 0], commands,
                       clock=lambda: now[0], sleep=sleep)
        names = [c[0] for c in robot.calls]
        self.assertLess(names.index("move"), names.index("enable"))
        self.assertEqual(names.count("move"), 3)
        enable_index = names.index("enable")
        self.assertEqual(names[enable_index+1:enable_index+3], ["mode", "move"])
        self.assertEqual(names[-1], "stop")

    def test_fault_after_enable_stops(self):
        robot, commands = Fake(), queue.Queue()
        commands.put("ENABLE")
        calls = [0]
        def read(r):
            calls[0] += 1
            if calls[0] > 2: raise RuntimeError("feedback lost")
            return sample(r)
        with patch.object(direct, "sample", read):
            with self.assertRaisesRegex(RuntimeError, "feedback lost"):
                direct.run(robot, True, [0, 1.57, -1.48, 0, 0, 0], commands, sleep=lambda _: None)
        self.assertEqual(robot.calls[-1], ("stop",))

    def test_slow_enable_gets_new_ack_window_and_unknown_mode_still_rejected(self):
        for acknowledge in (True, False):
            robot, commands = Fake(), queue.Queue()
            commands.put("ENABLE")
            now = [0.]
            original_enable = robot.enable
            def enable():
                now[0] += 2.5
                return original_enable()
            robot.enable = enable
            def read(r):
                mode = 1 if acknowledge and now[0] >= 4.5 else 255
                return r.q.copy(), NS(ctrl_mode=1, mode_feedback=mode, motion_status=0)
            def sleep(dt):
                now[0] += dt
                if acknowledge and now[0] > 6: commands.put("STOP SUPPORTED")
            with patch.object(direct, "sample", read):
                if acknowledge:
                    direct.run(robot, True, [0, 1.57, -1.48, 0, 0, 0], commands,
                               clock=lambda: now[0], sleep=sleep)
                else:
                    with self.assertRaisesRegex(RuntimeError, "mode_feedback=0xff"):
                        direct.run(robot, True, [0, 1.57, -1.48, 0, 0, 0], commands,
                                   clock=lambda: now[0], sleep=sleep)
            self.assertEqual(robot.calls[-1], ("stop",))
            self.assertFalse(any(c[0] == "move" and c[1][1] > 1 for c in robot.calls))


if __name__ == "__main__": unittest.main()
