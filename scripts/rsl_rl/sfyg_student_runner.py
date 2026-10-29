"""Independent Student PPO + raw-Normal distillation; no source PPO edits."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch.distributions import Normal

from sfyg_student_core import StudentPolicy, estimator_step, latent_normal_kl, estimator_metrics
from sfyg_teacher_core import check_teacher, fingerprint


@dataclass(frozen=True)
class Config:
    steps: int = 24
    epochs: int = 5
    batches: int = 4
    learning_rate: float = 1e-4
    estimator_learning_rate: float = 1e-3
    distill_coef: float = .001
    entropy_coef: float = .001
    gamma: float = .99
    lam: float = .95
    clip: float = .2

    def validate(self):
        import math
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Finite Student settings required')
        if min(self.steps, self.epochs, self.batches, self.learning_rate, self.estimator_learning_rate) <= 0:
            raise ValueError('Positive Student counts/rates required')
        if min(self.distill_coef, self.entropy_coef) < 0 or not 0 < self.gamma <= 1 or not 0 <= self.lam <= 1 or not 0 < self.clip < 1:
            raise ValueError('Invalid Student coefficients')
        return self


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gae(rewards, values, next_values, failed, done, gamma, lam):
    """Timeouts bootstrap TRUE terminal values; recursion stops at any reset."""
    advantage = torch.zeros_like(rewards)
    tail = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (~failed[t]) * next_values[t] - values[t]
        tail = delta + gamma * lam * (~done[t]) * tail
        advantage[t] = tail
    return advantage, advantage + values


class StudentRunner:
    def __init__(self, teacher_path, contract, device='cpu', config=None):
        from rsl_rl.modules import ActorCritic
        self.device = device
        self.cfg = (config or Config()).validate()
        saved = torch.load(teacher_path, map_location=device, weights_only=False)
        check_teacher(saved, contract)
        self.teacher = ActorCritic(147, 177, 9, actor_hidden_dims=[128]*3,
                                  critic_hidden_dims=[128]*3, activation='elu').to(device)
        self.teacher.load_state_dict(saved['model_state_dict'], strict=True)
        self.teacher.requires_grad_(False).eval()
        self.teacher_hash = fingerprint(self.teacher)
        self.policy = StudentPolicy(self.teacher.actor).to(device)
        self.policy.actor.requires_grad_(True)
        self.critic = deepcopy(self.teacher.critic).requires_grad_(True)
        self.logstd = torch.nn.Parameter(self.teacher.logstd.detach().clone())
        self.parameters = list(self.policy.actor.parameters()) + list(self.critic.parameters()) + [self.logstd]
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.cfg.learning_rate)
        self.est_optimizer = torch.optim.Adam(self.policy.estimator.parameters(), lr=self.cfg.estimator_learning_rate)
        self.contract = dict(schema='sfyg_student_v1', teacher_sha256=sha256(teacher_path),
                             teacher_contract=deepcopy(contract), actor_input=144, raw_output=9,
                             estimator_target='BASE payload external [Fx/40,Fy/40,Mz/10]',
                             student_history='same clean T1 history; no added noise in S1',
                             execution='apply teacher contract once; raw Normal, no tanh',
                             scope='single robot forward fixed arm; experimental', is_acceptance=False)
        self.iteration = 0

    def distribution(self, history):
        mean = self.policy(history)
        std = self.logstd.exp().expand_as(mean)
        if not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
            raise FloatingPointError('Invalid Student standard deviation')
        return Normal(mean, std)

    def assert_teacher_frozen(self):
        if fingerprint(self.teacher) != self.teacher_hash or any(p.grad is not None for p in self.teacher.parameters()):
            raise RuntimeError('Frozen teacher changed')

    @torch.no_grad()
    def collect(self, env, obs, app=None):
        records = []
        for _ in range(self.cfg.steps):
            if app is not None and not app.is_running():
                raise RuntimeError('Simulator interrupted during collection')
            teacher = obs['policy'].detach().clone()
            critic = obs['critic'].detach().clone()
            if teacher.shape != (env.num_envs,147) or critic.shape != (env.num_envs,177):
                raise ValueError('Expected paired pre-action Teacher147 / critic177')
            if not bool(torch.isfinite(teacher).all() and torch.isfinite(critic).all()):
                raise FloatingPointError('Nonfinite observations')
            history = teacher[:,:144].clone()
            dist = self.distribution(history)
            action = dist.sample()
            value = self.critic(critic).squeeze(-1)
            # Snapshot everything before env.step can mutate observations.
            record = dict(history=history, teacher=teacher, critic=critic, action=action.clone(),
                          mean=dist.mean.clone(), std=dist.stddev.clone(),
                          logprob=dist.log_prob(action).sum(-1).clone(), value=value.clone())
            obs, reward, failed, timeout, info = env.step(action)
            failed, timeout = failed.bool(), timeout.bool() & ~failed.bool()
            next_value = self.critic(obs['critic']).squeeze(-1)
            if bool(timeout.any()):
                terminal = info.get('student_terminal_critic')
                if terminal is None:
                    raise RuntimeError('Timeout requires pre-reset terminal critic observations')
                next_value[timeout] = self.critic(terminal[timeout]).squeeze(-1)
            if not bool(torch.isfinite(reward).all() and torch.isfinite(next_value).all()):
                raise FloatingPointError('Nonfinite transition')
            record.update(reward=reward.clone(), failed=failed.clone(), done=(failed|timeout).clone(),
                          next_value=next_value.clone(), timeout=timeout.clone())
            records.append(record)
        data = {k:torch.stack([r[k] for r in records]) for k in records[0]}
        adv, returns = gae(data['reward'],data['value'],data['next_value'],data['failed'],data['done'],self.cfg.gamma,self.cfg.lam)
        data['returns'] = returns
        data['advantage'] = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        return {k:v.flatten(0,1) for k,v in data.items()}, obs

    def update(self, data):
        n = len(data['history'])
        if n < self.cfg.batches:
            raise ValueError('More minibatches than collected transitions')
        self.policy.train()
        sums = dict(policy=0., value=0., teacher_kl=0., entropy=0.)
        updates = 0
        # Hold estimator fixed throughout PPO. Update it in a separate phase,
        # then measure the induced policy drift; no custom KL early stopping.
        for _ in range(self.cfg.epochs):
            for ids in torch.randperm(n,device=data['history'].device).tensor_split(self.cfg.batches):
                dist = self.distribution(data['history'][ids])
                ratio = (dist.log_prob(data['action'][ids]).sum(-1)-data['logprob'][ids]).exp()
                a = data['advantage'][ids]
                policy_loss = -torch.minimum(ratio*a,ratio.clamp(1-self.cfg.clip,1+self.cfg.clip)*a).mean()
                value = self.critic(data['critic'][ids]).squeeze(-1)
                clipped = data['value'][ids]+(value-data['value'][ids]).clamp(-self.cfg.clip,self.cfg.clip)
                value_loss = torch.maximum((value-data['returns'][ids]).square(),(clipped-data['returns'][ids]).square()).mean()
                with torch.no_grad():
                    teacher_mean = self.teacher.actor(data['teacher'][ids])
                    teacher_std = self.teacher.logstd.exp().expand_as(teacher_mean)
                kl = latent_normal_kl(dist.mean,dist.stddev,teacher_mean,teacher_std).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss+value_loss+self.cfg.distill_coef*kl-self.cfg.entropy_coef*entropy
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite Student PPO loss')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters,1.,error_if_nonfinite=True)
                self.optimizer.step()
                for key,val in zip(sums,(policy_loss,value_loss,kl,entropy)): sums[key] += float(val.detach())
                updates += 1
        probe = data['history'][:min(n,1024)]
        with torch.no_grad():
            before = self.distribution(probe)
            before_mean,before_std = before.mean.clone(),before.stddev.clone()
        losses = []
        for _ in range(self.cfg.epochs):
            for ids in torch.randperm(n,device=probe.device).tensor_split(self.cfg.batches):
                losses.append(estimator_step(self.policy.estimator,self.est_optimizer,
                                            data['history'][ids],data['teacher'][ids,-3:]))
        with torch.no_grad():
            after = self.distribution(probe)
            drift = latent_normal_kl(before_mean,before_std,after.mean,after.stddev).mean()
            rollout_kl = latent_normal_kl(data['mean'][:len(probe)],data['std'][:len(probe)],after.mean,after.stddev).mean()
            estimate = estimator_metrics(self.policy.estimate(data['history']),data['teacher'][:,-3:])
        self.iteration += 1
        self.assert_teacher_frozen()
        return dict(iteration=self.iteration,**{k:v/updates for k,v in sums.items()},
                    estimator_loss=sum(losses)/len(losses),estimator_policy_kl=float(drift),
                    post_update_rollout_kl=float(rollout_kl),optimizer_steps=updates,estimator_steps=len(losses),
                    mean_reward=float(data['reward'].mean()),failures=int(data['failed'].sum()),
                    timeouts=int(data['timeout'].sum()),mean_noise_std=float(self.logstd.exp().mean()),
                    **estimate)

    def save(self, path):
        self.assert_teacher_frozen()
        torch.save(dict(student_contract=self.contract,config=asdict(self.cfg),iteration=self.iteration,
                        policy=self.policy.state_dict(),critic=self.critic.state_dict(),logstd=self.logstd.detach(),
                        optimizer=self.optimizer.state_dict(),estimator_optimizer=self.est_optimizer.state_dict()),path)

    def load(self, path, resume=False):
        saved = torch.load(path,map_location=self.device,weights_only=False)
        if saved['student_contract'] != self.contract:
            raise ValueError('Student teacher/low/interface binding mismatch')
        if resume and saved['config'] != asdict(self.cfg):
            raise ValueError('Resume requires identical training configuration')
        tensors = [*saved['policy'].values(),*saved['critic'].values(),saved['logstd']]
        if not all(bool(torch.isfinite(t).all()) for t in tensors):
            raise ValueError('Nonfinite Student checkpoint')
        self.policy.load_state_dict(saved['policy'],strict=True)
        self.critic.load_state_dict(saved['critic'],strict=True)
        with torch.no_grad(): self.logstd.copy_(saved['logstd'])
        if resume:
            self.optimizer.load_state_dict(saved['optimizer'])
            self.est_optimizer.load_state_dict(saved['estimator_optimizer'])
            self.iteration = saved['iteration']
        return saved['iteration']

    def export(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=False)
        model = deepcopy(self.policy).cpu().eval()
        with (directory/'policy.pt').open('wb') as stream: torch.jit.save(torch.jit.script(model),stream)
        with (directory/'policy.pt').open('rb') as stream: restored = torch.jit.load(stream)
        sample = torch.randn(7,144)
        with torch.no_grad(): torch.testing.assert_close(restored(sample),model(sample))
        manifest = dict(self.contract,policy_sha256=sha256(directory/'policy.pt'),
                        iteration=self.iteration,history_owned_by_caller=True)
        (directory/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
