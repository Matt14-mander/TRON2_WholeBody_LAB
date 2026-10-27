"""Simulator-free SFYG inference contract and deterministic actor+encoder export."""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
LEG_NAMES = [f"{name}_{side}_Joint" for side in "LR" for name in
             ("proximal_pitch", "proximal_roll", "proximal_yaw", "knee", "ankle_pitch")]
HOLD_NAMES = [f"arm{i}_Joint" for i in range(1, 7)] + ["gripper1_Joint", "gripper2_Joint"]
TERM_NAMES = ["base_ang_vel", "proj_gravity", "joint_pos", "joint_vel", "last_action", "gait_phase", "gait_command"]
WIDTHS = [3, 3, 10, 10, 10, 2, 4]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deployment_config(contract):
    if (contract.get("schema") != "sfyg_source_reset_v1" or contract.get("source_commit") != "f441d67"
            or contract.get("phase") not in ("source", "carry", "robust")):
        raise ValueError("Only strict SFYG reset source/carry/robust checkpoints are supported")
    env, agent = contract["environment"], contract["agent"]
    if agent.get("empirical_normalization", False) or agent.get("clip_actions") is not None:
        raise ValueError("Unexpected normalization or action clipping")
    shapes = contract["observation_shapes"]
    if shapes["policy"] != [42] or shapes["obsHistory"] != [10, 42] or shapes["commands"] != [3]:
        raise ValueError("Expected SFYG policy42/history10x42/commands3")
    action = env["actions"]["joint_pos"]
    if (action["joint_names"] != LEG_NAMES or action.get("clip") is not None
            or not action["use_default_offset"] or float(action["scale"]) != .25):
        raise ValueError("Unexpected action ordering, clipping or scale")
    groups = env["observations"]
    terms = []
    for name, width in zip(TERM_NAMES, WIDTHS):
        term = groups["policy"][name]
        hist = groups["obsHistory"][name]
        for key in ("func", "params", "scale", "clip"):
            if hist.get(key) != term.get(key):
                raise ValueError(f"Policy/history semantics differ: {name}/{key}")
        if term.get("modifiers"):
            raise ValueError("Unsupported observation modifier")
        terms.append(dict(name=name, width=width, scale=term.get("scale"), clip=term.get("clip")))
    for group_name in ("policy", "obsHistory"):
        # Dictionary order is the Isaac observation concatenation order.
        order = [k for k, v in groups[group_name].items() if isinstance(v, dict) and "func" in v]
        if order != TERM_NAMES:
            raise ValueError(f"Unsupported observation terms/order: {order}")
        for name in ("joint_pos", "joint_vel"):
            if groups[group_name][name]["params"]["asset_cfg"]["joint_names"] != LEG_NAMES:
                raise ValueError("Unexpected observation joint ordering")
    if groups["obsHistory"]["history_length"] != 10 or groups["obsHistory"]["flatten_history_dim"]:
        raise ValueError("Unexpected history layout")
    robot = env["scene"]["robot"]
    defaults = robot["init_state"]["joint_pos"]
    names = LEG_NAMES + HOLD_NAMES
    if set(defaults) != set(names):
        raise ValueError("SFYG defaults require all 18 named joints")
    actuators = []
    for name in names:
        matches = [a for a in robot["actuators"].values()
                   if any(re.fullmatch(pattern, name) for pattern in a["joint_names_expr"])]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous actuator mapping: {name}")
        a = matches[0]
        actuators.append(dict(kp=float(a["stiffness"]), kd=float(a["damping"]),
            effort=float(a.get("effort_limit_sim") or a["effort_limit"]),
            velocity=float(a.get("velocity_limit_sim") or a["velocity_limit"]),
            armature=float(a.get("armature") or 0), friction=float(a.get("friction") or 0),
            min_delay=int(a.get("min_delay", 0)), max_delay=int(a.get("max_delay", 0))))
    gait = []
    for key in ("frequencies", "offsets", "durations", "swing_height"):
        low, high = env["commands"]["gait_command"]["ranges"][key]
        if low != high:
            raise ValueError("This bridge requires the source fixed gait parameters")
        gait.append(float(low))
    return dict(schema="sfyg_mujoco_v1", phase=contract["phase"], names=names, leg_names=LEG_NAMES,
        defaults=[defaults[name] for name in names], actuators=actuators, terms=terms, gait=gait,
        physics_dt=float(env["sim"]["dt"]), decimation=int(env["decimation"]),
        root_pos=robot["init_state"]["pos"], root_quat=robot["init_state"]["rot"],
        self_collisions=bool(robot["spawn"]["articulation_props"]["enabled_self_collisions"]),
        payload_mass=0. if contract["phase"] == "source" else float(contract["arguments"]["payload_mass"]),
        command_limits=[1., .5, 1.5], observation_width=42, history_length=10,
        history_order="oldest_to_newest_time_major", history_reset="repeat_first_frame",
        gait_phase="episode_steps_dt_frequency_even_when_standing", action_clip=None,
        pd_implementation="MuJoCo general position+velocity bias, implicitfast, force limited",
        friction_coefficient=.5, contact_model="MuJoCo condim=3; PhysX static/dynamic and restitution are not identical",
        limitations=["No hard joint velocity cap in MuJoCo bridge; training limits are recorded",
                     "Contact solver and implicit PD integration differ from PhysX"])


class SFYGInference(nn.Module):
    def __init__(self, actor, encoder):
        super().__init__()
        self.actor, self.encoder = actor, encoder

    def forward(self, observation: torch.Tensor, history: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
        estimate = self.encoder(history.flatten(1))
        return self.actor(torch.cat((estimate, observation, commands), dim=-1))

    @torch.jit.export
    def estimate_velocity(self, history: torch.Tensor) -> torch.Tensor:
        return self.encoder(history.flatten(1))


def load_checkpoint(path):
    sys.path.insert(0, str(ROOT / "rsl_rl"))
    from rsl_rl.modules import ActorCritic, MLP_Encoder
    saved = torch.load(path, map_location="cpu", weights_only=False)
    contract = saved["sfyg_reset_contract"]
    cfg = deployment_config(contract)
    agent = contract["agent"]
    p, e = agent["policy"], agent["encoder"]
    actor = ActorCritic(48, saved["model_state_dict"]["critic.0.weight"].shape[1], 10,
        actor_hidden_dims=p["actor_hidden_dims"], critic_hidden_dims=p["critic_hidden_dims"], activation=p["activation"])
    encoder = MLP_Encoder(420, 3, hidden_dims=e["hidden_dims"], activation=e["activation"])
    for key in ("model_state_dict", "encoder_state_dict"):
        if not all(torch.isfinite(v).all() for v in saved[key].values()):
            raise ValueError("Nonfinite checkpoint weights")
    actor.load_state_dict(saved["model_state_dict"], strict=True)
    encoder.load_state_dict(saved["encoder_state_dict"], strict=True)
    return SFYGInference(actor.actor, encoder.encoder).eval(), cfg, contract


class ObservationHistory:
    """Isaac CircularBuffer startup padding and oldest->newest layout."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = None

    def push(self, observation):
        observation = np.asarray(observation, np.float32)
        if observation.shape != (42,) or not np.isfinite(observation).all():
            raise ValueError("Expected finite 42-D observation")
        if self.frames is None:
            self.frames = np.tile(observation, (10, 1))
        else:
            self.frames[:-1] = self.frames[1:].copy()
            self.frames[-1] = observation
        return self.frames.copy()


def make_observation(angular, gravity, joint_pos, joint_vel, previous_action, step, cfg):
    phase = np.float32(step) * np.float32(cfg["physics_dt"] * cfg["decimation"]) * np.float32(cfg["gait"][0])
    phase = np.remainder(phase, np.float32(1.)) * np.float32(2 * np.pi)
    values = [angular, gravity, np.asarray(joint_pos) - np.asarray(cfg["defaults"][:10]),
              joint_vel, previous_action, [np.sin(phase), np.cos(phase)], cfg["gait"]]
    parts = []
    for term, value in zip(cfg["terms"], values):
        value = np.asarray(value, np.float32)
        if value.shape != (term["width"],):
            raise ValueError(f"Wrong width for {term['name']}")
        if term["clip"] is not None:
            value = np.clip(value, *term["clip"])
        if term["scale"] is not None:
            value = value * np.float32(term["scale"])
        parts.append(value)
    result = np.concatenate(parts)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite observation")
    return result
