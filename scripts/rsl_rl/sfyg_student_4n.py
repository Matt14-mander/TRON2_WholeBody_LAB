"""Paired evaluation of T1 student force estimates and plateau zero commands."""
import numpy as np
import torch


def shadow_actions(actor, pre, estimate):
    # Same history, same student actor. The Fx-only variant controls for errors
    # on the two unexcited estimator outputs as well.
    fx_only = estimate.clone()
    fx_only[:, 0] = pre[:, -3]
    return actor(pre), actor(torch.cat((pre[:, :144], fx_only), dim=-1))


def fill_zero(raw, command, reference, times, settings):
    begin = settings.settle + settings.ramp
    mask = ((times >= begin) & (times < begin + settings.hold)
            & ((reference[:, 0] - .1).abs() < 1e-6) & (command[:, 0] == 0))
    action = raw.clone()
    action[mask, 0] = .1
    return action, mask


def case_metrics(rows, settings):
    begin = settings.settle + settings.ramp
    end = begin + settings.hold
    plateau = [r for r in rows if begin <= round(r['action_time_s'], 4) < end]
    zero = [r for r in plateau if r['student_command_same_state'][0] == 0]
    active = [r for r in rows if settings.settle < round(r['time_s'], 4) <= end]
    vx = np.array([r['velocity'][0] for r in active])
    n = round(.5 / settings.dt)
    first_stop = None
    stops = 0
    moving = False
    i = 0
    while i + n <= len(active):
        window = vx[i:i+n]
        if moving and np.sqrt(np.mean(window**2)) <= .03:
            stops += 1
            if first_stop is None:
                first_stop = active[i]['time_s']
            moving = False
        elif not moving and window.mean() >= .05:
            moving = True
        else:
            i += 1
            continue
        i += n
    # Find the strongest raw dropout in the half second preceding the first
    # low-speed window. Compare all three raw outputs at that exact frame.
    pulse = [r for r in zero if first_stop is not None
             and first_stop-.5 <= r['action_time_s'] <= first_stop]
    detail = None
    if pulse:
        r = min(pulse, key=lambda x: x['student_raw_vx'])
        detail = {k:r[k] for k in ('action_time_s','student_raw_vx','true_fx_raw_vx','true_wrench_raw_vx')}
        detail['estimated_fx_n'] = r['pre_action_estimate'][0]*40
    return dict(plateau_frames=len(plateau), zero_frames=len(zero),
                filled_frames=sum(r['filled_zero'] for r in rows),
                fx_only_still_zero=sum(r['true_fx_raw_vx'] <= settings.deadband for r in zero),
                true_wrench_still_zero=sum(r['true_wrench_raw_vx'] <= settings.deadband for r in zero),
                negative_frames=sum(r['student_raw_vx'] < 0 for r in plateau),
                true_wrench_still_negative=sum(r['student_raw_vx'] < 0 and r['true_wrench_raw_vx'] < 0 for r in plateau),
                stops=stops, first_stop_time_s=first_stop, first_stop_pulse=detail)


def print_comparison(reports):
    base = {(r['episode'],r['environment']):r for r in reports if r['controller']=='student'}
    fill = {(r['episode'],r['environment']):r for r in reports if r['controller']=='student_fill_4n'}
    if not base or base.keys()!=fill.keys():
        raise ValueError('Missing paired 4 N cases')
    print('[SHADOW] Original zero frames on original student rollouts:')
    for key in ('zero_frames','fx_only_still_zero','true_wrench_still_zero','negative_frames','true_wrench_still_negative'):
        print(f'  {key}={sum(r["diagnostic_4n"][key] for r in base.values())}')
    print('[PULSE] episode/env | time_s | est_Fx_N | raw_est | raw_true_Fx | raw_true_wrench')
    for key,r in sorted(base.items()):
        p = r['diagnostic_4n']['first_stop_pulse']
        if p is not None:
            print(key, *(f'{p[k]:.4f}' for k in ('action_time_s','estimated_fx_n','student_raw_vx','true_fx_raw_vx','true_wrench_raw_vx')))
    pairs = [k for k in base if base[k]['completed'] and fill[k]['completed']]
    print(f'[PAIR] Both completed: {len(pairs)}/{len(base)}. Velocity/RMSE/stops below use these pairs.')
    for name,group in [('student',base),('fill_4n',fill)]:
        metrics = [group[k] for k in pairs]
        values = dict(passed=sum(r['gate']['passed'] for r in group.values()), total=len(group),
                      physical=sum(r['gate']['physical'] for r in group.values()),
                      release=sum(r['gate']['release'] for r in group.values()),
                      filled_frames=sum(r['diagnostic_4n']['filled_frames'] for r in group.values()))
        if metrics:
            values.update(vx=float(np.mean([r['plateau_mean_velocity'][0] for r in metrics])),
                          vx_rmse=float(np.sqrt(np.mean([r['plateau_velocity_rmse'][0]**2 for r in metrics]))),
                          stops=sum(r['diagnostic_4n']['stops'] for r in metrics))
        print(f'[PAIR] {name} {values}', flush=True)
