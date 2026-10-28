"""Native T0 interface audit. Artificial phase changes below are TEST ONLY."""
import torch

from probe_sfyg_robust import force_buffers
from sfyg_teacher_core import fingerprint


def audit(env):
    term = env.action_manager.get_term('transport')
    robot = term._asset
    if env.num_envs<2:
        raise ValueError('Interface smoke needs at least two environments for reset isolation')
    before = fingerprint(term.low)
    original_fixed = env.cfg.fixed_force
    report = dict(kind='sfyg_teacher_native_interface_audit',is_acceptance=False,
                  synthetic_phase_for_force_readback=True)
    with torch.inference_mode():
        try:
            env.cfg.fixed_force = 8.
            env.reset()
            # Exercise physical forces even with the local untrained LL that
            # cannot survive the real 3-second settle. This is never a walk gate.
            env.episode_length_buf[:] = 250
            obs = env.observation_manager.compute()
            torch.testing.assert_close(obs['policy'][:,-3:],torch.tensor([.2,0.,0.],device=env.device).expand(env.num_envs,-1))
            extreme = torch.full((env.num_envs,9),100.,device=env.device)
            calls = [0]
            original_apply = term.apply_actions
            def counted():
                calls[0] += 1
                original_apply()
            term.apply_actions = counted
            low_calls = term.low_calls
            try:
                env.step(extreme)
            finally:
                term.apply_actions = original_apply
            report['low_inferences_per_control'] = term.low_calls-low_calls
            report['force_writes_per_control'] = calls[0]
            if report['low_inferences_per_control']!=1 or report['force_writes_per_control']!=4:
                raise RuntimeError('Incorrect frozen inference / force reapplication rate')
            torch.testing.assert_close(term.command,term.command.new_tensor([.3,0.,0.]).expand_as(term.command))
            torch.testing.assert_close(robot.data.joint_pos_target[:,term.hold_ids],term.nominal)
            buffer,torque = force_buffers(robot)
            if buffer is None or torque is None: raise RuntimeError('Physical force readback unavailable')
            report['force_buffer_error_n'] = float((buffer[:,term.payload_id:term.payload_id+1]-term.local_force).abs().max())
            report['payload_force_norm_n'] = float(buffer[:,term.payload_id].norm(dim=-1).max())
            rest = torch.cat((buffer[:,:term.payload_id],buffer[:,term.payload_id+1:]),1)
            report['nonpayload_force_n'] = float(rest.abs().max())
            report['torque_nm'] = float(torque.abs().max())
            if abs(report['payload_force_norm_n']-8)>1e-4:
                raise RuntimeError('8 N force did not reach the physical payload')
            untouched = dict(low=term.low_history.data[1:].clone(),arm=term.arm_history.data[1:].clone(),
                force=buffer[1:].clone(),command=term.command[1:].clone(),last_leg=term.last_leg[1:].clone())
            env.reset(env_ids=torch.tensor([0],device=env.device))
            buffer,torque = force_buffers(robot)
            report['reset_force_residue_n'] = float(buffer[0].abs().max())
            for name,value in dict(low=term.low_history.data[1:],arm=term.arm_history.data[1:],
                force=buffer[1:],command=term.command[1:],last_leg=term.last_leg[1:]).items():
                torch.testing.assert_close(value,untouched[name],rtol=0,atol=0)
            if bool(term.low_history.data[0].any()) or bool(term.arm_history.data[0].any()):
                raise RuntimeError('Reset histories contain stale samples')
            report['partial_reset_isolated'] = True
            if any(report[k]>1e-5 for k in ('force_buffer_error_n','nonpayload_force_n','torque_nm','reset_force_residue_n')):
                raise RuntimeError('Physical force or reset audit failed')
            if fingerprint(term.low)!=before: raise RuntimeError('Frozen low weights changed')
            report['frozen_weights_verified'] = True
            report['passed'] = True
        finally:
            env.cfg.fixed_force = original_fixed
            env.reset()
    return report
