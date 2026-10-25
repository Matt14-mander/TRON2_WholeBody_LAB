# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import math

from ..modules import ActorCritic, MLP_Encoder
from ..storage import RolloutStorage


class PPO:
    actor_critic: ActorCritic
    encoder: MLP_Encoder

    def __init__(
        self,
        num_group,
        encoder,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        min_learning_rate=1.0e-5,
        max_learning_rate=1.0e-2,
        vae_beta=1.0,
        est_learning_rate=1.0e-3,
        encoder_max_grad_norm=0.0,
        update_kl_guard=False,
        encoder_kl_fraction=.3,
        critic_take_latent=False,
        early_stop=False,
        anneal_lr=False,
        device="cpu",
        **kwargs,
    ):
        self.device = device
        self.num_group = num_group

        self.desired_kl = desired_kl
        self.early_stop = early_stop
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        if not 0.0 < self.min_learning_rate <= self.max_learning_rate:
            raise ValueError("learning-rate bounds must satisfy 0 < min <= max")
        self.learning_rate = min(
            max(self.learning_rate, self.min_learning_rate),
            self.max_learning_rate,
        )
        self.anneal_lr = anneal_lr
        self.vae_beta = vae_beta
        self.critic_take_latent = critic_take_latent
        self.encoder_max_grad_norm = float(encoder_max_grad_norm)
        self.update_kl_guard = bool(update_kl_guard)
        self.encoder_kl_fraction = float(encoder_kl_fraction)
        if self.update_kl_guard and (desired_kl is None or desired_kl <= 0
                                     or not 0 < self.encoder_kl_fraction <= 1):
            raise ValueError("KL guard requires positive desired_kl and encoder fraction in (0,1]")

        self.encoder = encoder

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(
            [{"params": self.actor_critic.parameters()}], lr=self.learning_rate
        )

        if self.encoder.num_output_dim != 0:
            self.extra_optimizer = optim.Adam(
                self.encoder.parameters(), lr=est_learning_rate
            )
        else:
            self.extra_optimizer = None
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.last_update_stats = {}

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        obs_history_shape,
        commands_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_history_shape,
            commands_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, obs_history, commands, critic_obs):
        critic_obs = torch.cat((critic_obs, commands), dim=-1)
        # act
        encoder_out = self.encoder.encode(obs_history)
        self.transition.actions = self.actor_critic.act(
            torch.cat((encoder_out, obs, commands), dim=-1)
        ).detach()

        # evaluate
        if self.critic_take_latent:
            critic_obs = torch.cat((critic_obs, encoder_out), dim=-1)
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()

        # storage
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_obs = critic_obs
        self.transition.observation_history = obs_history
        self.transition.commands = commands
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.transition.next_observations = next_obs
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        num_updates = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl = 0
        kl_checks = 0
        max_kl = 0.0
        early_stopped = False
        actor_rejected = encoder_rejected = actor_attempts = encoder_attempts = 0
        max_attempted_actor_kl = max_attempted_encoder_kl = 0.
        # This fixed, bounded probe is used for diagnostics and optional guards.
        # It is an empirical rollout probe, not a guarantee over all robot states.
        with torch.no_grad():
            total = self.storage.observations.shape[0] * self.storage.observations.shape[1]
            probe_ids = torch.linspace(0, total - 1, min(total, 2048), device=self.device).long()
            probe_obs = self.storage.observations.flatten(0, 1)[probe_ids]
            probe_history = self.storage.observation_history.flatten(0, 1)[probe_ids]
            probe_commands = self.storage.commands.flatten(0, 1)[probe_ids]
            old_probe_mu = self.storage.mu.flatten(0, 1)[probe_ids]
            old_probe_sigma = self.storage.sigma.flatten(0, 1)[probe_ids]

            def probe_distribution():
                self.actor_critic.update_distribution(torch.cat(
                    (self.encoder.encode(probe_history), probe_obs, probe_commands), -1))
                return (self.actor_critic.action_mean.detach().clone(),
                        self.actor_critic.action_std.detach().clone())

            def probe_kl(old_mu, old_sigma, mu, sigma):
                value = float((torch.log(sigma / old_sigma)
                    + (old_sigma.square() + (old_mu - mu).square()) / (2 * sigma.square())
                    - .5).sum(-1).mean())
                return max(0., value) if math.isfinite(value) else value

        generator = self.storage.mini_batch_generator(
            self.num_group,
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for (
            obs_batch,
            critic_obs_batch,
            obs_history_batch, _,
            group_commands_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            encoder_out_batch = self.encoder.encode(obs_history_batch)
            commands_batch = group_commands_batch
            self.actor_critic.act(
                torch.cat(
                    (encoder_out_batch, obs_batch, commands_batch),
                    dim=-1,
                )
            )

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )

            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            kl_mean = torch.tensor(0, device=self.device, requires_grad=False)
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

            # Include rejected batches: otherwise an update stopped on its
            # second batch reports only the near-zero KL of its first batch.
            kl_value = float(kl_mean)
            mean_kl += kl_value
            max_kl = max(max_kl, kl_value)
            kl_checks += 1
            should_stop = (self.desired_kl is not None and self.early_stop
                           and kl_value > self.desired_kl * 1.5)

            # KL
            if self.desired_kl != None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if should_stop or kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(
                            self.min_learning_rate, self.learning_rate / 1.5
                        )
                    elif num_updates > 0 and kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(
                            self.max_learning_rate, self.learning_rate * 1.5
                        )

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            if should_stop:
                early_stopped = True
                break

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            # print(ratio)
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            entropy_batch_mean = entropy_batch.mean()
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch_mean
            )

            if self.anneal_lr:
                frac = 1.0 - num_updates / (
                    self.num_learning_epochs * self.num_mini_batches
                )
                self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            if self.update_kl_guard:
                saved_model = deepcopy(self.actor_critic.state_dict())
                saved_optimizer = deepcopy(self.optimizer.state_dict())
            self.optimizer.step()
            actor_attempts += 1
            if self.update_kl_guard:
                with torch.no_grad():
                    mu, sigma = probe_distribution()
                    checked = probe_kl(old_probe_mu, old_probe_sigma, mu, sigma)
                    max_attempted_actor_kl = max(max_attempted_actor_kl, checked) if math.isfinite(checked) else float("inf")
                if not math.isfinite(checked) or checked > 1.5 * self.desired_kl:
                    self.actor_critic.load_state_dict(saved_model)
                    self.optimizer.load_state_dict(saved_optimizer)
                    self.learning_rate = max(self.min_learning_rate, self.learning_rate / 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate
                    actor_rejected += 1
                    early_stopped = True
                    break

            num_updates += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        # The supervised encoder also changes the effective policy. Optional
        # guards cover BOTH its cumulative drift and total old-to-new drift.
        with torch.no_grad():
            before_encoder_mu, before_encoder_sigma = probe_distribution()
            post_actor_kl = probe_kl(old_probe_mu, old_probe_sigma, before_encoder_mu, before_encoder_sigma)

        num_updates_extra = 0
        mean_extra_loss = 0
        if self.extra_optimizer is not None:
            generator = self.storage.encoder_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
            for (
                next_obs_batch,
                critic_obs_batch,
                obs_history_batch,
            ) in generator:
                if self.encoder.is_mlp_encoder:
                    self.encoder.encode(obs_history_batch)
                    encode_batch = self.encoder.get_encoder_out()

                if self.encoder.is_mlp_encoder:
                    extra_loss = (
                        (encode_batch[:, 0:3] - critic_obs_batch[:, 0:3]).pow(2).mean()
                    )
                else:
                    extra_loss = torch.zeros_like(value_loss)

                self.extra_optimizer.zero_grad()
                extra_loss.backward()
                if self.encoder_max_grad_norm > 0.0:
                    nn.utils.clip_grad_norm_(
                        self.encoder.parameters(), self.encoder_max_grad_norm
                    )
                if self.update_kl_guard:
                    saved_encoder = deepcopy(self.encoder.state_dict())
                    saved_extra_optimizer = deepcopy(self.extra_optimizer.state_dict())
                self.extra_optimizer.step()
                encoder_attempts += 1
                if self.update_kl_guard:
                    with torch.no_grad():
                        mu, sigma = probe_distribution()
                        drift = probe_kl(before_encoder_mu, before_encoder_sigma, mu, sigma)
                        total_drift = probe_kl(old_probe_mu, old_probe_sigma, mu, sigma)
                        max_attempted_encoder_kl = max(max_attempted_encoder_kl, drift) if math.isfinite(drift) else float("inf")
                    if (not math.isfinite(drift) or not math.isfinite(total_drift)
                            or drift > self.desired_kl * self.encoder_kl_fraction
                            or total_drift > self.desired_kl * 1.5):
                        self.encoder.load_state_dict(saved_encoder)
                        self.extra_optimizer.load_state_dict(saved_extra_optimizer)
                        for group in self.extra_optimizer.param_groups:
                            group["lr"] = max(1e-6, group["lr"] / 1.5)
                        encoder_rejected += 1
                        break

                num_updates_extra += 1
                mean_extra_loss += extra_loss.item()

        mean_value_loss /= max(1, num_updates)
        if num_updates_extra > 0:
            mean_extra_loss /= num_updates_extra
        mean_surrogate_loss /= max(1, num_updates)
        mean_kl /= max(1, kl_checks)
        with torch.no_grad():
            after_encoder_mu, after_encoder_sigma = probe_distribution()
            encoder_policy_kl = probe_kl(before_encoder_mu, before_encoder_sigma,
                                         after_encoder_mu, after_encoder_sigma)
            post_update_kl = probe_kl(old_probe_mu, old_probe_sigma, after_encoder_mu, after_encoder_sigma)
        self.last_update_stats = {
            "optimizer_steps": num_updates,
            "planned_optimizer_steps": self.num_learning_epochs * self.num_mini_batches,
            "kl_checks": kl_checks,
            "max_checked_kl": max_kl,
            "early_stopped": float(early_stopped),
            "encoder_steps": num_updates_extra,
            "actor_attempts": actor_attempts,
            "actor_rejected_updates": actor_rejected,
            "encoder_attempts": encoder_attempts,
            "encoder_rejected_updates": encoder_rejected,
            "max_attempted_actor_probe_kl": max_attempted_actor_kl,
            "max_attempted_encoder_probe_kl": max_attempted_encoder_kl,
            "encoder_learning_rate": self.extra_optimizer.param_groups[0]["lr"] if self.extra_optimizer else 0.,
            "post_actor_probe_kl": post_actor_kl,
            "encoder_policy_probe_kl": encoder_policy_kl,
            "post_update_probe_kl": post_update_kl,
        }
        self.storage.clear()

        return (mean_value_loss, mean_extra_loss, mean_surrogate_loss, mean_kl)
