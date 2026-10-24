"""Guide stage 2: student PPO + KL(student||teacher), separate online regression.

Both policies use the repository's common tanh-Normal transform. KL is evaluated
on latent Normals, where it equals KL of the transformed distributions.
"""
import math
import torch
from torch import nn
from ..storage.paint_storage import PaintStorage
from ..storage.rollout_storage import RolloutStorage


def normal_kl(mu, std, target_mu, target_std):
    return (torch.log(target_std / std)
            + (std.square() + (mu - target_mu).square()) / (2 * target_std.square()) - .5).sum(-1)


class PaintPPO:
    def __init__(self, actor_critic, teacher, estimator, num_envs, horizon, device="cpu",
                 distill_coef=1e-6, est_learning_rate=1e-3, learning_rate=1e-4,
                 num_learning_epochs=5, num_mini_batches=4, clip_param=.2,
                 gamma=.99, lam=.95, value_loss_coef=1., entropy_coef=.001,
                 max_grad_norm=1., desired_kl=.01, min_learning_rate=1e-5,
                 max_learning_rate=1e-4, schedule="adaptive", early_stop=True, **unused):
        if not actor_critic.bounded_distribution or not teacher.bounded_distribution:
            raise ValueError("PAINT v4 requires matching tanh-Normal policies")
        if (not all(math.isfinite(v) and v > 0 for v in
                    (est_learning_rate, learning_rate, min_learning_rate, max_learning_rate, max_grad_norm))
                or not math.isfinite(distill_coef) or distill_coef < 0
                or not min_learning_rate <= learning_rate <= max_learning_rate
                or num_learning_epochs < 1 or num_mini_batches < 1
                or desired_kl is not None and (not math.isfinite(desired_kl) or desired_kl <= 0)):
            raise ValueError("Invalid student optimizer settings")
        self.actor_critic, self.teacher, self.estimator = actor_critic, teacher, estimator
        self.teacher.requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate)
        self.estimator_optimizer = torch.optim.Adam(estimator.parameters(), lr=est_learning_rate)
        self.storage = PaintStorage(num_envs, horizon, device)
        self.device = device
        self.distill_coef, self.learning_rate = distill_coef, learning_rate
        self.epochs, self.batches = num_learning_epochs, num_mini_batches
        self.clip, self.gamma, self.lam = clip_param, gamma, lam
        self.value_coef, self.entropy_coef = value_loss_coef, entropy_coef
        self.use_clipped_value_loss = unused.get("use_clipped_value_loss", True)
        if (unused.get("normalize_advantage_per_mini_batch", False) or unused.get("rnd_cfg")
                or unused.get("symmetry_cfg") or unused.get("critic_take_latent", False)
                or unused.get("anneal_lr", False)):
            raise ValueError("Unsupported Student PPO extension; use transport v4 defaults")
        self.max_grad_norm, self.desired_kl = max_grad_norm, desired_kl
        self.min_lr, self.max_lr, self.schedule, self.early_stop = min_learning_rate, max_learning_rate, schedule, early_stop
        self.transition = RolloutStorage.Transition()
        self.last_update_stats = {}

    @torch.no_grad()
    def act(self, obs, teacher_obs, critic_obs):
        if obs.shape[1:] != (144,) or teacher_obs.shape[1:] != (147,) or critic_obs.shape[1:] != (177,):
            raise ValueError("Student requires history144 / teacher147 / critic177")
        if not all(torch.isfinite(x).all() for x in (obs, teacher_obs, critic_obs)):
            raise FloatingPointError("Nonfinite student observation")
        t = self.transition
        # Snapshot BEFORE env.step: Isaac observation buffers can be reused in place.
        t.observations, t.teacher_observations, t.critic_obs = obs.clone(), teacher_obs.clone(), critic_obs.clone()
        t.actions = self.actor_critic.act(torch.cat((obs, self.estimator(obs).detach()), -1))
        t.values = self.actor_critic.evaluate(critic_obs)
        t.actions_log_prob = self.actor_critic.get_actions_log_prob(t.actions)
        t.action_mean, t.action_sigma = self.actor_critic.action_mean, self.actor_critic.action_std
        t.observation_history = obs.new_zeros(len(obs), 1)
        t.commands = obs.new_zeros(len(obs), 0)
        return t.actions

    def process_env_step(self, rewards, dones, infos, next_obs):
        t = self.transition
        t.rewards, t.dones, t.next_observations = rewards.clone(), dones.clone(), next_obs
        if "time_outs" in infos:
            # Match repository PPO bootstrap; genuine failure overrides a simultaneous timeout.
            timeout = infos["time_outs"].bool().clone()
            if "transport_terminated" in infos:
                timeout &= ~infos["transport_terminated"].bool()
            t.rewards += self.gamma * t.values.squeeze(-1) * timeout
        if not torch.isfinite(t.rewards).all():
            raise FloatingPointError("Nonfinite student reward")
        self.storage.add_transitions(t)
        t.clear()

    @torch.no_grad()
    def compute_returns(self, critic_obs):
        self.storage.compute_returns(self.actor_critic.evaluate(critic_obs), self.gamma, self.lam)

    def update(self):
        stats = {name: 0. for name in ("value_loss", "surrogate_loss", "teacher_kl", "estimator_mse",
                                      "rollout_kl", "estimator_policy_kl")}
        updates, checks, peak_kl, stopped = 0, 0, 0., False
        with torch.no_grad():
            total = self.storage.num_envs * self.storage.num_transitions_per_env
            probe_ids = torch.linspace(0, total-1, min(total, 2048), device=self.device).long()
            probe_obs = self.storage.observations.flatten(0, 1)[probe_ids]
            probe_mu = self.storage.mu.flatten(0, 1)[probe_ids]
            probe_std = self.storage.sigma.flatten(0, 1)[probe_ids]
        for b in self.storage.batches(self.batches, self.epochs):
            obs = b["observations"]
            estimate = self.estimator(obs)
            self.actor_critic.update_distribution(torch.cat((obs, estimate.detach()), -1))
            mu, std = self.actor_critic.action_mean, self.actor_critic.action_std
            with torch.no_grad():
                self.teacher.update_distribution(b["teacher_observations"])
                teacher_mu, teacher_std = self.teacher.action_mean, self.teacher.action_std
                kl = float(normal_kl(b["mu"], b["sigma"], mu, std).mean())
            checks += 1
            peak_kl = max(peak_kl, kl)
            stats["rollout_kl"] += kl
            over = self.desired_kl is not None and kl > 1.5 * self.desired_kl
            if self.schedule == "adaptive" and self.desired_kl is not None:
                if over:
                    self.learning_rate = max(self.min_lr, self.learning_rate / 1.5)
                elif updates and kl < self.desired_kl / 2:
                    self.learning_rate = min(self.max_lr, self.learning_rate * 1.5)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate
            if over and self.early_stop:
                stopped = True
                break
            teacher_kl = normal_kl(mu, std, teacher_mu, teacher_std).mean()
            ratio = (self.actor_critic.get_actions_log_prob(b["actions"]) - b["actions_log_prob"].squeeze(-1)).exp()
            advantage = b["advantages"].squeeze(-1)
            surrogate = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(1-self.clip, 1+self.clip)).mean()
            value = self.actor_critic.evaluate(b["critic_obs"])
            clipped = b["values"] + (value - b["values"]).clamp(-self.clip, self.clip)
            value_loss = torch.maximum((value-b["returns"]).square(), (clipped-b["returns"]).square()).mean()
            if not self.use_clipped_value_loss:
                value_loss = (value-b["returns"]).square().mean()
            loss = surrogate + self.value_coef * value_loss - self.entropy_coef * self.actor_critic.entropy.mean() + self.distill_coef * teacher_kl
            regression = (estimate - b["teacher_observations"][:, -3:]).square().mean()
            if not torch.isfinite(loss + regression):
                raise FloatingPointError("Nonfinite student loss")
            self.optimizer.zero_grad(set_to_none=True)
            self.estimator_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # No actor/critic/KL gradient is permitted through estimate.detach().
            if any(p.grad is not None for p in self.estimator.parameters()):
                raise RuntimeError("Policy gradient leaked into intent estimator")
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            with torch.no_grad():
                self.actor_critic.update_distribution(torch.cat((obs, estimate.detach()), -1))
                before_mu = self.actor_critic.action_mean.clone()
                before_std = self.actor_critic.action_std.clone()
            regression.backward()
            nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm, error_if_nonfinite=True)
            self.estimator_optimizer.step()
            with torch.no_grad():
                self.actor_critic.update_distribution(torch.cat((obs, self.estimator(obs)), -1))
                drift = normal_kl(before_mu, before_std, self.actor_critic.action_mean, self.actor_critic.action_std).mean()
            for name, val in (("value_loss", value_loss), ("surrogate_loss", surrogate),
                              ("teacher_kl", teacher_kl), ("estimator_mse", regression), ("estimator_policy_kl", drift)):
                stats[name] += float(val.detach())
            updates += 1
        for name in stats:
            stats[name] /= max(1, checks if name == "rollout_kl" else updates)
        stats.update(optimizer_steps=updates, estimator_steps=updates, kl_checks=checks,
                     max_checked_kl=peak_kl, early_stopped=float(stopped), learning_rate=self.learning_rate)
        with torch.no_grad():
            self.actor_critic.update_distribution(torch.cat((probe_obs, self.estimator(probe_obs)), -1))
            stats["post_update_probe_kl"] = float(normal_kl(probe_mu, probe_std,
                self.actor_critic.action_mean, self.actor_critic.action_std).mean())
        self.last_update_stats = stats
        self.storage.clear()
        return stats
