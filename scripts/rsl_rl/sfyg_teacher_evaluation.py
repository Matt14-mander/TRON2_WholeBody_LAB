"""T0/T1 metrics on pre-reset states; weak-force immobility is not acceptance."""
import math
import numpy as np


def summarize(rows, settings, force_n, controller, probe_vx=0.):
    def array(items,key): return np.asarray([r[key] for r in items],dtype=float)
    plateau = [r for r in rows if settings.settle+settings.ramp+1 <= r['time_s'] <= settings.settle+settings.ramp+settings.hold]
    tail = [r for r in rows if r['time_s'] >= settings.horizon-1.5]
    post = [r for r in rows if r['time_s'] > settings.settle]
    completed = bool(rows and rows[-1]['timeout'] and not rows[-1]['failed']
                     and rows[-1]['time_s'] >= settings.horizon-settings.dt/2)
    expected = probe_vx if controller == 'probe' else force_n/40.
    if plateau:
        velocity = array(plateau,'velocity')
        target = array(plateau,'reference')
        if controller == 'probe': target[:,0] = probe_vx
        means = velocity.mean(0)
        rmse = np.sqrt(np.mean((velocity-target)**2,axis=0))
    else:
        means = rmse = np.full(3,float('nan'))
    release_rmse = np.sqrt(np.mean(array(tail,'velocity')**2,axis=0)) if tail else None
    # First sustained 0.5 s crossing half the moving reference, after force onset.
    onset = None
    window = max(1,round(.5/settings.dt))
    if expected > 0:
        for index in range(len(post)-window+1):
            block = post[index:index+window]
            if all(r['velocity'][0] >= expected*.5 for r in block):
                onset = block[0]['time_s']-settings.settle
                break
    reward_names = sorted(rows[0]['rewards']) if rows else []
    report = dict(controller=controller,force_n=force_n,probe_vx=probe_vx,
        completed=completed,duration_s=rows[-1]['time_s'] if rows else 0.,
        failed=any(r['failed'] for r in rows),plateau_samples=len(plateau),release_samples=len(tail),
        plateau_mean_velocity=means.tolist() if plateau else None,
        plateau_velocity_rmse=rmse.tolist() if plateau else None,
        release_velocity_rmse=release_rmse.tolist() if tail else None,
        onset_delay_s=onset,
        max_tilt_deg=max((r['tilt_deg'] for r in rows),default=0.),
        peak_arm_error_rad=max((max(abs(v) for v in r['arm_error']) for r in rows),default=0.),
        raw_vx_outside_limits_fraction=float(np.mean([(r['raw_action'][0]<0 or r['raw_action'][0]>settings.command_max) for r in post])) if post else None,
        episode_reward=sum(r['reward'] for r in rows),
        reward_terms={k:sum(r['rewards'][k] for r in rows) for k in reward_names})
    physical = completed and report['max_tilt_deg']<15 and report['peak_arm_error_rad']<.1
    if expected == 0:
        motion = bool(plateau and rmse[0]<.03 and rmse[1]<.05 and rmse[2]<.1)
    else:
        # RMS lateral motion includes gait oscillation; signed means alone are insufficient.
        motion = bool(plateau and rmse[0]<=max(.025,.5*expected)
            and .5*expected<=means[0]<=1.5*expected and rmse[1]<.2 and abs(means[1])<.05
            and rmse[2]<.15 and abs(means[2])<.05)
    release = bool(tail and release_rmse[0]<.03 and release_rmse[1]<.05 and release_rmse[2]<.1)
    report['gate'] = dict(physical=physical,tracking=motion,release=release,
                         passed=physical and motion and release)
    return report


def baseline_gate(reports):
    """All required cases must exist. Never accept an empty/partial report."""
    checks = {}
    for mode,forces in (('hold',(0.,)),('reference',(0.,4.,6.,8.))):
        for force in forces:
            cases = [r for r in reports if r['controller']==mode and r['force_n']==force]
            checks[f'{mode}_{force:g}N'] = bool(cases) and all(r['gate']['passed'] for r in cases)
    # Comparable full episodes, same seeds and environment indices, no invented
    # continuation after a reset. Include every reward term in the comparison.
    for force in (4.,6.,8.):
        hold = [r for r in reports if r['controller']=='hold' and r['force_n']==force]
        ref = [r for r in reports if r['controller']=='reference' and r['force_n']==force]
        pairs = {(r['episode'],r['environment']):r for r in hold}
        comparable = bool(ref) and len(hold)==len(ref)
        differences = []
        for r in ref:
            other = pairs.get((r['episode'],r['environment']))
            if other is None or not other['completed'] or not r['completed']:
                comparable = False
                continue
            differences.append(r['episode_reward']-other['episode_reward'])
        checks[f'reward_prefers_following_{force:g}N'] = bool(comparable and differences and np.mean(differences)>0.)
    return dict(passed=bool(checks) and all(checks.values()),checks=checks,
        scope='nominal fixed-arm forward 0/4/6/8N; 2N and zero-force command probes diagnostic only')
