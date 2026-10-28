"""Simulator-free T0/T1 contracts; the SFYG training modules are never edited."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch

SCHEMA = "sfyg_transport_teacher_t1_v1"


@dataclass(frozen=True)
class Settings:
    payload_mass: float = .5
    force_max: float = 8.
    zero_probability: float = .2
    weak_probability: float = .1
    command_max: float = .3
    deadband: float = .02
    delay_steps: int = 2
    dt: float = .02
    settle: float = 3.
    ramp: float = 2.
    hold: float = 8.
    release_hold: float = 3.

    @property
    def horizon(self):
        return self.settle + 2*self.ramp + self.hold + self.release_hold

    def validate(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Finite teacher settings required")
        if (self.payload_mass != .5 or not 4 <= self.force_max <= 8
                or not 0 <= self.zero_probability < 1 or not 0 <= self.weak_probability < 1
                or self.zero_probability+self.weak_probability >= 1
                or self.command_max != .3 or self.deadband != .02
                or self.delay_steps not in range(5) or self.dt != .02
                or min(self.settle,self.ramp,self.hold,self.release_hold) <= 0):
            raise ValueError("T1 requires nominal 0.5 kg, forward force 4--8 N, declared timing/action limits")
        return self


def envelope(time, s):
    rise = (time-s.settle)/s.ramp
    fall = (s.settle+2*s.ramp+s.hold-time)/s.ramp
    return torch.minimum(rise,fall).clamp(0.,1.)


def sample_force(count, device, s):
    # Probabilities are fractions of ALL episodes, not conditional probabilities.
    category = torch.rand(count, device=device)
    force = torch.empty(count,device=device).uniform_(4.,s.force_max)
    weak = (category >= s.zero_probability) & (category < s.zero_probability+s.weak_probability)
    force[weak] = torch.empty(int(weak.sum()),device=device).uniform_(0.,2.)
    force[category < s.zero_probability] = 0.
    return force


class History:
    def __init__(self, count, length, width, device, repeat_first=False):
        self.data = torch.zeros(count,length,width,device=device)
        self.tick = torch.full((count,),-1,dtype=torch.long,device=device)
        self.repeat_first = repeat_first

    def reset(self, ids):
        self.data[ids] = 0
        self.tick[ids] = -1

    def append(self, frame, tick, valid=None):
        fresh = self.tick != tick
        if valid is not None:
            fresh &= valid
        first = fresh & (self.tick < 0)
        ids = fresh.nonzero().flatten()
        self.data[ids] = torch.cat((self.data[ids,1:],frame[ids,None]),1)
        if self.repeat_first:
            self.data[first] = frame[first,None]
        self.tick[fresh] = tick
        return self.data


def process_action(raw, time, settings):
    """Physical velocity units, forward only; unused outputs are still penalized."""
    if raw.ndim != 2 or raw.shape[1] != 9 or not torch.isfinite(raw).all():
        raise ValueError("Finite 9-D high-level raw actions required")
    command = torch.zeros(raw.shape[0],3,device=raw.device,dtype=raw.dtype)
    vx = raw[:,0].clamp(0.,settings.command_max)
    command[:,0] = torch.where((vx > settings.deadband) & (time >= settings.settle),vx,0.)
    return command


def low_observation(angular, gravity, q, dq, last_leg, steps, cfg):
    """Exact clean source42 terms, clipping BEFORE scaling; no true linear velocity."""
    phase = torch.remainder(steps * (cfg['physics_dt']*cfg['decimation']) * cfg['gait'][0],1.) * (2*math.pi)
    gait = q.new_tensor(cfg['gait']).expand(q.shape[0],-1)
    values = (angular,gravity,q-q.new_tensor(cfg['defaults'][:10]),dq,last_leg,
              torch.stack((phase.sin(),phase.cos()),-1),gait)
    parts = []
    for value,term in zip(values,cfg['terms']):
        if term['clip'] is not None:
            value = value.clamp(*term['clip'])
        if term['scale'] is not None:
            value = value*term['scale']
        parts.append(value)
    return torch.cat(parts,-1)


def reward_terms(actual, reference, command, raw, previous, gravity):
    error = actual-reference
    return {
        'linear_tracking': -2.*(1.-torch.exp(-error[:,:2].square().sum(-1)/.25)),
        'yaw_tracking': -(1.-torch.exp(-error[:,2].square()/.25)),
        'command_tracking': -.5*(command-reference).square().sum(-1),
        'base_tilt': -.5*gravity[:,:2].square().sum(-1),
        'raw_action': -.002*((raw-previous).square()+.5*raw.square()).sum(-1),
    }


def fingerprint(module):
    digest = hashlib.sha256()
    for key,value in module.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def frozen_low(path, device):
    # Reuse the strict, already tested SFYG checkpoint decoder. Caller puts the
    # existing scripts/mujoco directory on sys.path; no MuJoCo import is needed.
    from sfyg_policy import load_checkpoint, sha256
    low,cfg,saved = load_checkpoint(path)
    if cfg['phase'] not in ('carry','robust'):
        raise ValueError("Teacher requires a carry/robust SFYG checkpoint")
    if cfg['physics_dt'] != .005 or cfg['decimation'] != 4:
        raise ValueError("Expected source 200 Hz physics / 50 Hz control")
    low.requires_grad_(False).to(device).eval()
    return low,cfg,sha256(path),saved


def runner_config():
    return dict(num_steps_per_env=24,save_interval=100,empirical_normalization=False,
        logger='tensorboard',clip_actions=None,
        policy=dict(actor_hidden_dims=[128]*3,critic_hidden_dims=[128]*3,
                    activation='elu',init_noise_std=.1),
        encoder=dict(num_output_dim=0,hidden_dims=[1],activation='elu',output_detach=True),
        algorithm=dict(class_name='PPO',obs_history_len=1,value_loss_coef=1.,
            use_clipped_value_loss=True,clip_param=.2,entropy_coef=.001,
            num_learning_epochs=5,num_mini_batches=4,learning_rate=1e-4,
            schedule='adaptive',gamma=.99,lam=.95,desired_kl=.01,
            max_grad_norm=1.,early_stop=False,critic_take_latent=False))


def initialize_teacher(actor):
    # Source ActorCritic ignores init_noise_std. Correct this NEW instance only.
    # Small initial means keep the declared .1 exploration meaningful.
    with torch.no_grad():
        actor.logstd.fill_(math.log(.1))
        actor.actor[-1].weight.mul_(.01)
        actor.actor[-1].bias.zero_()


def execution_contract(low_sha, low_cfg, settings):
    contract = dict(schema=SCHEMA,low_sha256=low_sha,low_interface=deepcopy(low_cfg),
        settings=asdict(settings),actor_width=147,critic_width=177,action_width=9,
        actor_history='8x(q_arm-q_nominal,dq_arm,previous_executed_target-q_nominal); oldest first; zero reset',
        privilege='current external payload COM wrench in BASE axes / (40,40,10); NOT grasp reaction',
        freeze_arm=True,active_axes=['vx'],physics_dt=.005,control_dt=.02,
        action_semantics='raw physical vx clipped [0,.3], deadband .02; vy/wz/arm masked; no LL action clip',
        force_schedule='episode 3s settle + 2s rise + 8s hold + 2s fall + 3s release; force constant within control interval',
        reward_weights=dict(linear_tracking=-2.,yaw_tracking=-1.,command_tracking=-.5,
                            base_tilt=-.5,raw_action=-.002,failure_event=-50.),
        stage='T1_forward_fixed_arm',student_implemented=False)
    # Checkpoints contain tuples, while server gate/export manifests use JSON.
    # Canonicalize before binding either artifact to avoid false mismatches.
    return json.loads(json.dumps(contract))


def check_teacher(saved, contract):
    if json.loads(json.dumps(saved.get('sfyg_teacher_contract'))) != contract:
        raise ValueError("Teacher contract mismatch (low weights, physical settings, observations or action semantics)")
    if not all(torch.isfinite(v).all() for v in saved['model_state_dict'].values()):
        raise ValueError("Nonfinite teacher checkpoint")


def export_teacher(actor, directory, contract):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    model = deepcopy(actor.actor).cpu().eval()
    scripted = torch.jit.script(model)
    sample = torch.randn(7,147)
    with torch.no_grad():
        expected = model(sample)
    with (directory/'policy.pt').open('wb') as stream:
        torch.jit.save(scripted,stream)
    with (directory/'policy.pt').open('rb') as stream:
        restored = torch.jit.load(stream)
    torch.testing.assert_close(restored(sample),expected)
    manifest = dict(contract,policy_sha256=hashlib.sha256((directory/'policy.pt').read_bytes()).hexdigest(),
                    output='RAW 9-D actor output; execution contract MUST be applied')
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
