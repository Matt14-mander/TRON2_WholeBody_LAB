"""Headless S1 model_500 diagnostics: same-frame wrench substitution and direct SFYG commands."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from run_sfyg_teacher import load_bundle, step_frozen_low, termination, write_json
from sfyg_bridge import ROOT, SFYGBridge
from sfyg_student_policy import ROBUST_BUNDLE, S1_CHECKPOINT, StudentCheckpointPolicy
from sfyg_student_4n import shadow_actions
from sfyg_teacher_core import envelope, process_action


def shadow_commands(student, tick, force, executed):
    # command() has already advanced the history and saved its pre-action snapshot.
    history = torch.from_numpy(student.observation)[None]
    estimate = student.actor.estimate(history)
    true_label = torch.zeros_like(estimate)
    true_label[:, 0] = float(force[0]) / 40.
    pre = torch.cat((history, true_label), dim=1)
    full, fx_only = shadow_actions(student.actor.actor, pre, estimate)
    result = {'estimated': [float(student.raw[0]), float(executed[0])]}
    for name, raw in [('true_fx', fx_only), ('true_wrench', full)]:
        command = process_action(raw, student.time(tick), student.settings)
        result[name] = [float(raw[0, 0]), float(command[0, 0])]
    return result


def foot_contacts(bridge, feet):
    on_ground = set()
    for contact in bridge.data.contact:
        a, b = bridge.model.geom_bodyid[[contact.geom1, contact.geom2]]
        if a == 0:
            on_ground.add(b)
        elif b == 0:
            on_ground.add(a)
    return [body in on_ground for body in feet]


def summarize_case(rows, settings):
    begin = round((settings.settle + settings.ramp) / settings.dt)
    count = round(settings.hold / settings.dt)
    plateau = rows[begin:begin + count]
    result = dict(completed=rows[-1]['termination'] is None, termination=rows[-1]['termination'],
                  plateau_frames=len(plateau), expected_plateau_frames=count)
    if not plateau:
        return result
    cmd = np.array([r['cmd_vx'] for r in plateau])
    vx = np.array([r['vx'] for r in plateau])
    contact = np.array([r['foot_contact_lr'] for r in rows])
    releases = contact[:-1] & ~contact[1:]
    releases = releases[max(0, begin-1):begin+len(plateau)-1]
    result.update(cmd_vx=float(cmd.mean()), vx=float(vx.mean()),
                  rmse_vs_executed_cmd=float(np.sqrt(np.mean((vx-cmd)**2))),
                  dx_world_m=plateau[-1]['post_x_m']-plateau[0]['pre_x_m'],
                  foot_releases_lr=releases.sum(axis=0).tolist())
    if 'shadow_raw_cmd_vx' in plateau[0]:
        result['est_fx_n'] = float(np.mean([r['estimated_fx_n'] for r in plateau]))
        result['shadow'] = {}
        for mode in plateau[0]['shadow_raw_cmd_vx']:
            raw, command = np.array([r['shadow_raw_cmd_vx'][mode] for r in plateau]).T
            result['shadow'][mode] = dict(raw_vx=float(raw.mean()), cmd_vx=float(command.mean()),
                zero_fraction=float((command == 0).mean()), fraction_ge_010=float((command >= .1).mean()))
    return result


@torch.inference_mode()
def run_case(bridge, low, student, force_n, direct_vx, stream):
    bridge.reset()
    student.reset()
    settings = student.settings
    feet = [bridge.model.body(name).id for name in ('ankle_pitch_L_Link','ankle_pitch_R_Link')]
    rows = []
    for tick in range(round(settings.horizon / settings.dt)):
        force = student.force(tick, force_n)
        row = dict(tick=tick, action_time_s=tick*settings.dt, force_fx_n=float(force[0]),
                   pre_x_m=float(bridge.data.xpos[bridge.base, 0]))
        if direct_vx is None:
            arm = bridge.data.qpos[bridge.qpos[10:16]] - bridge.defaults[10:16]
            dq = bridge.data.qvel[bridge.qvel[10:16]]
            offset = bridge.target[10:16] - bridge.defaults[10:16]
            command = student.command(arm, dq, offset, tick)
            row['estimated_fx_n'] = student.estimated_wrench[0]
            if force_n == 8:
                row['shadow_raw_cmd_vx'] = shadow_commands(student, tick, force, command)
        else:
            amplitude = float(envelope(student.time(tick), settings)[0])
            command = np.array([direct_vx * amplitude, 0., 0.], np.float32)
        _, _, _, world_force = step_frozen_low(bridge, low, command, force)
        linear, _, gravity = bridge.base_state()
        tilt = float(np.rad2deg(np.arccos(np.clip(-gravity[2], -1, 1))))
        reason = termination(bridge, tilt, settings.dt)
        row.update(cmd_vx=float(command[0]), vx=float(linear[0]), time_s=float(bridge.data.time),
                   post_x_m=float(bridge.data.xpos[bridge.base, 0]), applied_world_force=world_force.tolist(),
                   foot_contact_lr=foot_contacts(bridge, feet), termination=reason)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        rows.append(row)
        if reason:
            break
    return summarize_case(rows, settings)


def run(output, bundle, seed):
    torch.set_num_threads(1)
    cfg, low = load_bundle(bundle)
    student = StudentCheckpointPolicy(S1_CHECKPOINT, cfg)
    settings = student.settings
    bridge = SFYGBridge(cfg, settings.payload_mass, cfg['friction_coefficient'], settings.delay_steps, seed)
    audit = bridge.audit()
    if max(audit[k] for k in ('authored_mass_error_kg','authored_com_error_m','authored_inertia_error')) > 1e-6:
        raise ValueError('Original robot mass/COM/inertia audit failed')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(student_checkpoint=str(S1_CHECKPOINT), student_sha256=student.sha,
                  low_checkpoint_sha256=cfg['checkpoint_sha256'], low_policy_sha256=cfg['policy_sha256'],
                  mujoco_version=mujoco.__version__, seed=seed, settings=vars(settings), bridge=audit,
                  iteration=500, is_acceptance=False, cases={})
    cases = [(8, None), (16, None)]
    cases += [(f, v) for f in (0, 4, 8) for v in (0., .10, .15, .20)]
    cases.append((16, 0.))
    for force, speed in cases:
        name = f'student_{force}N' if speed is None else f'low_{force}N_vx{speed:.2f}'
        with (output/f'{name}.jsonl').open('x', encoding='utf-8') as stream:
            case = run_case(bridge, low, student, force, speed, stream)
        case.update(force_n=force, direct_vx=speed)
        report['cases'][name] = case
        print(f'[CASE] {name} plateau={case["plateau_frames"]}/{case["expected_plateau_frames"]} '
              f'end={case["termination"] or "timeout"}', end=' ')
        if case['plateau_frames']:
            print(f'cmd={case["cmd_vx"]:.4f} vx={case["vx"]:.4f} dx={case["dx_world_m"]:.3f}m '
                  f'releases={case["foot_releases_lr"]}', end='')
        print(flush=True)
        for mode, values in case.get('shadow', {}).items():
            print(f'[8N SHADOW] {mode} nativeEstFx={case["est_fx_n"]:.2f}N raw={values["raw_vx"]:+.4f} '
                  f'cmd={values["cmd_vx"]:.4f} zero={values["zero_fraction"]:.1%} '
                  f'cmd>=0.10={values["fraction_ge_010"]:.1%}', flush=True)
    write_json(output/'summary.json', report)
    print(f'[REPORT] {output / "summary.json"}', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle_dir', type=Path, default=ROBUST_BUNDLE)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=Path, default=ROOT/'logs/mujoco'/
                        ('student500_controls_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    args = parser.parse_args()
    run(args.output_dir, args.bundle_dir, args.seed)
