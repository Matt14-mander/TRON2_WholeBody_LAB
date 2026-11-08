"""Strict deployment of the existing privileged T1 actor; no retraining."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from sfyg_bridge import ROOT
from sfyg_policy import sha256

sys.path.insert(0,str(ROOT/'scripts/rsl_rl'))
from sfyg_teacher_core import Settings, History, execution_contract, envelope, process_action


class ArmHistoryPolicy:
    def __init__(self,contract,low_config):
        self.settings=Settings(**contract['settings']).validate()
        if contract.get('low_sha256')!=low_config.get('checkpoint_sha256'):
            raise ValueError('Policy is bound to a different low-level checkpoint')
        interface=contract['low_interface']
        if any(low_config.get(k)!=v for k,v in interface.items()):
            raise ValueError('Teacher/low-level observation, action or physical interface mismatch')
        expected=execution_contract(low_config['checkpoint_sha256'],interface,self.settings)
        if any(contract.get(k)!=v for k,v in expected.items()):
            raise ValueError('Unsupported teacher execution contract')
        self.history=History(1,8,18,'cpu')
        self.reset()

    def reset(self):
        self.history.reset(slice(None))
        self.observation=np.zeros(self.input_width,np.float32)
        self.raw=np.zeros(9,np.float32)

    def time(self,tick):
        if not isinstance(tick,(int,np.integer)) or tick<0: raise ValueError('Nonnegative integer control tick required')
        return torch.tensor([int(tick)],dtype=torch.long)*self.settings.dt

    def force(self,tick,force_n):
        if not np.isfinite(force_n) or not 0<=force_n<=self.settings.force_max:
            raise ValueError('T1 supports forward external force 0--8 N only')
        return np.array([float(envelope(self.time(tick),self.settings)[0])*force_n,0.,0.],np.float32)

    def arm_history(self,arm_error,arm_velocity,previous_target_offset,tick):
        parts=[np.asarray(v,np.float32) for v in (arm_error,arm_velocity,previous_target_offset)]
        if not all(v.shape==(6,) and np.isfinite(v).all() for v in parts):
            raise ValueError('Finite own arm6/dq6/target6 required')
        if np.any(parts[2]!=0): raise ValueError('T1 requires the original held arm target')
        frame=torch.from_numpy(np.concatenate(parts))[None]
        return self.history.append(frame,int(tick),valid=torch.tensor([tick>0])).flatten(1)

    def execute(self,observation,tick):
        raw=self.actor(observation)
        command=process_action(raw,self.time(tick),self.settings)[0]
        self.observation=observation[0].numpy().copy()
        self.raw=raw[0].numpy().copy()
        return command.numpy().copy()


class TeacherPolicy(ArmHistoryPolicy):
    input_width=147

    def __init__(self,directory,low_config):
        directory=Path(directory)
        self.manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        super().__init__(self.manifest,low_config)
        self.sha=sha256(directory/'policy.pt')
        if self.sha!=self.manifest.get('policy_sha256'):
            raise ValueError('Teacher actor SHA256 mismatch')
        with (directory/'policy.pt').open('rb') as stream:
            self.actor=torch.jit.load(stream,map_location='cpu').eval()
        for parameter in self.actor.parameters(): parameter.requires_grad_(False)
        with torch.inference_mode():
            sample=self.actor(torch.zeros(1,147))
        if sample.shape!=(1,9) or not torch.isfinite(sample).all():
            raise ValueError('Expected finite teacher actor147 -> raw9')

    @torch.inference_mode()
    def command(self,arm_error,arm_velocity,previous_target_offset,force_base,tick):
        force=np.asarray(force_base,np.float32)
        if (force.shape!=(3,) or not np.isfinite(force).all() or force[0]<0
                or force[0]>self.settings.force_max+1e-6 or np.any(force[1:]!=0)):
            raise ValueError('Teacher input must be known forward applied force, not grasp reaction')
        history=self.arm_history(arm_error,arm_velocity,previous_target_offset,tick)
        return self.execute(torch.cat((history,torch.from_numpy(force/40.)[None]),-1),tick)
