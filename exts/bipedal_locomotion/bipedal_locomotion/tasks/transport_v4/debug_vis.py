"""Optional Isaac Sim force arrows and a live settings/observations/rewards HUD."""
import torch
from isaaclab.utils.math import quat_apply


class TransportDebugView:
    def __init__(self, env):
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.util.debug_draw")
        from isaacsim.util.debug_draw import _debug_draw
        import omni.ui as ui
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.env = env
        self.window = ui.Window("PAINT transport v4 | env 0", width=520, height=520)
        with self.window.frame:
            with ui.VStack():
                self.label = ui.Label("Starting transport...", word_wrap=True)

    def update(self, state, rewards, step):
        if step % 5:
            return
        env = self.env
        data = env.scene["robot"].data
        base = state["base_position"][0].detach()
        payload = state["payload_position"][0].detach()
        command = torch.cat((state["base_command"][0, :2], torch.zeros(1, device=env.device)))
        command = quat_apply(state["base_quaternion"][0:1], command[None])[0]
        starts, ends, colors, widths = [], [], [], []
        def arrow(origin, vector, color):
            if float(vector.norm()) < 1e-6:
                return
            endpoint = origin + vector
            side = torch.cross(vector, torch.tensor([0., 0., 1.], device=env.device), dim=0)
            if float(side.norm()) < 1e-6:
                side = torch.tensor([1., 0., 0.], device=env.device)
            side = side / side.norm() * min(.06, float(vector.norm()) * .15)
            back = vector * .2
            for a, b in ((origin, endpoint), (endpoint, endpoint - back + side), (endpoint, endpoint - back - side)):
                starts.append(a.tolist()); ends.append(b.tolist()); colors.append(color); widths.append(3.)
        arrow(base + torch.tensor([0., 0., .2], device=env.device), command, (0., 1., 0., 1.))
        arrow(payload, state["force_world"][0, 0] * .03, (1., .25, 0., 1.))
        arrow(payload, state["torque_world"][0, 0] * .12, (.25, .5, 1., 1.))
        self.draw.clear_lines()
        if starts:
            self.draw.draw_lines(starts, ends, colors, widths)
        values = lambda x: ", ".join(f"{float(v):+.3f}" for v in x)
        terms = "\n".join(f"{name}: {float(value[0]):+.6f}" for name, value in rewards.items())
        self.label.text = (
            f"Physics 200 Hz | control 50 Hz | episode {env.max_episode_length_s:.0f} s\n"
            f"Actor {env.cfg.actor_mode} | step {step}\n"
            "Green: base command | orange: payload force | blue: torque\n"
            f"command vx/vy/wz: {values(state['base_command'][0])}\n"
            f"actual  vx/vy/wz: {values(state['base_velocity'][0])}\n"
            f"reference c(t): {values(state['reference'][0])}\n"
            f"payload F_world: {values(state['force_world'][0, 0])}\n"
            f"arm q: {values(state['arm_position'][0])}\n"
            f"arm target: {values(state['arm_target'][0])}\n"
            f"base tilt: {float(state['base_tilt_deg'][0]):.2f} deg\n"
            f"Weighted rewards per control step:\n{terms}")

    def close(self):
        self.draw.clear_lines()
        self.window.visible = False
