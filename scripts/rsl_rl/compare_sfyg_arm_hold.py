"""Compare pre-action arm measurements during matched zero-command force plateaus."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def statistics(rows, force, isaac, settings):
    dt = settings['dt']
    first = round((settings['settle'] + settings['ramp']) / dt)
    last = first + round(settings['hold'] / dt)
    window = [r for r in rows if first <= r['tick'] < last]
    if [r['tick'] for r in window] != list(range(first,last)):
        raise ValueError(f'Incomplete {force} N plateau: {len(window)}/{last-first} frames')
    names = [f'arm{i}_Joint' for i in range(1,7)]
    for r in window:
        command = r['command'] if isaac else [r['cmd_vx']]
        fx = r['force_base'][0] if isaac else r['force_fx_n']
        if (r['arm_names'] != names or np.any(np.asarray(command) != 0)
                or abs(r['action_time_s'] - r['tick']*dt) > 1e-5 or abs(fx-force) > 1e-4):
            raise ValueError(f'Unmatched arm order, command, time or force at tick {r["tick"]}')
    q = np.array([r['pre_arm_error_rad'] for r in window])
    dq = np.array([r['pre_arm_velocity_rad_s'] for r in window])
    estimate = np.array([r['pre_action_estimate'][0]*40 if isaac else r['estimated_fx_n'] for r in window])
    if q.shape != (len(window),6) or dq.shape != q.shape or not all(np.isfinite(x).all() for x in (q,dq,estimate)):
        raise ValueError('Invalid arm samples')
    return dict(q=q.mean(0), q_std=q.std(0), dq=dq.mean(0), dq_rms=np.sqrt((dq**2).mean(0)),
                est=float(estimate.mean()), frames=len(window))


def compare(root):
    isaac, mj = root/'isaac', root/'mujoco'
    run = load(isaac/'run.json')
    contract = load(isaac/'contract.json')['teacher_contract']
    manifest = load(mj/'summary.json')
    args = run['arguments']
    if (run['parent_student']['sha256'] != manifest['student_sha256']
            or run['parent_student']['iteration'] != 500 or manifest['iteration'] != 500
            or contract['low_sha256'] != manifest['low_checkpoint_sha256']
            or contract['settings'] != manifest['settings'] or args['seed'] != manifest['seed']
            or args['num_envs'] != 1 or args['episodes'] != 1):
        raise ValueError('Expected matched S1 model_500, low-level, settings, seed and one episode/environment')
    with (isaac/'steps.jsonl').open(encoding='utf-8') as f:
        records = [json.loads(line) for line in f]
    summary = []
    for sim in ('isaac','mujoco'):
        for force in (0,4,8):
            if sim == 'isaac':
                rows = [r for r in records if r['controller']=='hold' and r['force_n']==force
                        and r['episode']==0 and r['environment']==0]
            else:
                with (mj/f'low_{force}N_vx0.00.jsonl').open(encoding='utf-8') as f:
                    rows = [json.loads(line) for line in f]
            stats = statistics(rows,force,sim=='isaac',contract['settings'])
            if force == 0:
                zero_q = stats['q']
                zero_fx = stats['est']
            print(f'[HOLD] {sim} F={force} N estFx={stats["est"]:.3f} N '
                  f'deltaEstFx={stats["est"]-zero_fx:+.3f} N frames={stats["frames"]}')
            for j in range(6):
                summary.append(dict(sim=sim,force_n=force,joint=f'arm{j+1}_Joint',
                    q_error_mean_rad=stats['q'][j],delta_q_from_0N_rad=stats['q'][j]-zero_q[j],
                    q_error_std_rad=stats['q_std'][j],dq_mean_rad_s=stats['dq'][j],
                    dq_rms_rad_s=stats['dq_rms'][j],est_fx_n=stats['est'],frames=stats['frames']))
    with (root/'arm_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print('F(N) joint        mean error (mrad)     delta vs 0N (mrad)     dq RMS (rad/s)')
    print('                  Isaac      MuJoCo     Isaac      MuJoCo     Isaac      MuJoCo')
    for a,b in zip(summary[:18],summary[18:]):
        print(f'{a["force_n"]:>3}  {a["joint"]:<11}',end='')
        for key, scale in [('q_error_mean_rad',1000),('delta_q_from_0N_rad',1000),('dq_rms_rad_s',1)]:
            print(f'{scale*a[key]:10.4f}{scale*b[key]:10.4f}',end='')
        print()
    print(f'[REPORT] {root/"arm_comparison.csv"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    compare(parser.parse_args().run)
