"""Fixed-base, feet-clear fixture test of preparation, never policy inference."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'hardware'))
from sfyg_supported_prepare import SupportedPrepare, RECORDED, configuration
from sfyg_bridge import build_xml


def case(cfg, mass, delay, stop_at, out):
    xml, assets, _, _ = build_xml(cfg, mass, cfg['friction_coefficient'])
    base = xml.find(".//body[@name='base_Link']")
    base.set('pos', '0 0 1.6')
    base.set('quat', '1 0 0 0')
    ET.SubElement(xml.find('worldbody'), 'body', name='gantry_fixture', mocap='true', pos='0 0 1.6')
    eq = xml.find('equality')
    if eq is None: eq = ET.SubElement(xml, 'equality')
    ET.SubElement(eq, 'weld', body1='base_Link', body2='gantry_fixture', solref='.002 1')
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'), assets)
    data = mujoco.MjData(model)
    joints = np.array([model.joint(n).id for n in cfg['names']])
    qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    motors = np.array([model.actuator('pd/'+n).id for n in cfg['names']])
    data.qpos[qa] = cfg['defaults']
    data.qpos[qa[:10]] = RECORDED
    data.ctrl[motors] = data.qpos[qa]
    mujoco.mj_forward(model, data)
    ctl = SupportedPrepare(cfg)
    queue = deque(maxlen=delay+1)
    packet, done_at, holding_at = None, None, None
    peak_dq = peak_error = peak_tau = peak_active_error = peak_tilt = 0.
    base_id = model.body('base_Link').id
    last_target = RECORDED.copy()
    max_target_step = 0.
    contacts = 0
    dt = cfg['physics_dt']
    stopped = False
    with out.open('w') as f:
        for step in range(int(40/dt)):
            now = step*dt
            if step % cfg['decimation'] == 0:
                q, dq = data.qpos[qa[:10]].copy(), data.qvel[va[:10]].copy()
                event = 'prepare suspended' if step == 0 else None
                if stop_at and now >= stop_at and not stopped:
                    event, stopped = 'stop', True
                if ctl.state in ('HOLD', 'PAUSED'):
                    holding_at = now if holding_at is None else holding_at
                    if now-holding_at >= 3 and done_at is None:
                        event, done_at = 'release supported', now
                gravity = data.xmat[base_id].reshape(3,3).T @ np.array([0.,0.,-1.])
                peak_tilt = max(peak_tilt, float(np.degrees(np.arccos(np.clip(-gravity[2],-1,1)))))
                packet = ctl.tick(now, q, dq, gravity, event)
                error = float(max(abs(q-ctl.target)))
                peak_error = max(peak_error, error)
                if ctl.state not in ('RELEASE','DAMP','DONE'):
                    peak_active_error = max(peak_active_error, error)
                peak_dq = max(peak_dq, float(max(abs(dq))))
                max_target_step = max(max_target_step, float(max(abs(np.array(packet['q'])-last_target))))
                last_target = np.array(packet['q'])
                f.write(json.dumps(dict(time_s=now, state=ctl.state, q=q.tolist(), dq=dq.tolist(), packet=packet))+'\n')
                if ctl.state in ('FAULT','DONE'): break
            queue.append(packet)
            p = queue[0]
            model.actuator_gainprm[motors[:10],0] = p['kp']
            model.actuator_biasprm[motors[:10],1] = -np.array(p['kp'])
            model.actuator_biasprm[motors[:10],2] = -np.array(p['kd'])
            data.ctrl[motors[:10]] = p['q']
            data.ctrl[motors[10:]] = cfg['defaults'][10:]
            mujoco.mj_step(model, data)
            peak_tau = max(peak_tau, float(max(abs(data.actuator_force[motors[:10]]))))
            contacts += data.ncon
    passed = ctl.state == 'DONE' and holding_at is not None and contacts == 0
    return dict(passed=passed, mass=mass, delay_steps=delay, stop_at=stop_at,
                final_state=ctl.state, reason=ctl.reason, events=ctl.events,
                peak_speed=peak_dq, peak_tracking_error=peak_error, peak_torque=peak_tau,
                peak_active_tracking_error=peak_active_error, peak_fixture_tilt_deg=peak_tilt,
                max_target_step=max_target_step, contacts=contacts,
                fixture='fixed base at 1.6m, feet clear; NOT compliant gantry or ground stance',
                policy_calls=0, hardware_validated=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bundle_dir', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    a = p.parse_args()
    cfg = configuration(a.bundle_dir)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    reports = []
    for mass in (0., .5):
        for delay in (0, 4):
            for stop in (None, 7.):
                report = case(cfg, mass, delay, stop,
                              a.output_dir/f'mass{mass}_delay{delay}_stop{stop}.jsonl')
                reports.append(report)
                print(json.dumps(report), flush=True)
    (a.output_dir/'summary.json').write_text(json.dumps(reports, indent=2))
    if not all(r['passed'] for r in reports): raise SystemExit('Fixture validation failed')
