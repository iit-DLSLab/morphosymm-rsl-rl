# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.storage import RolloutStorage

from morphosymm_rsl_rl.algorithms.ppo import PPO


class PPOSymmDataAugmented(PPO):
    """PPO whose rollout storage holds every symmetry-group replica of each collected transition.

    Unlike RSL-RL's built-in ``symmetry_cfg`` (which only tiles a mini-batch during ``update()``, repeating the
    same returns/values for every replica), this class augments the rollout *as it is collected*: every
    observation, action, and distribution parameter is transformed by each non-identity group element and stored
    as its own transition, so ``compute_returns`` bootstraps proper GAE targets for every replica. The actor's
    escnn group ``G`` (shared by construction with the critic, see :class:`~morphosymm_rsl_rl.modules.SymmModel`)
    drives every transform; only the storage sizing, transition augmentation, and return bootstrap need overriding,
    the loss computation in ``update()`` is inherited unchanged from :class:`~morphosymm_rsl_rl.algorithms.ppo.PPO`.
    """

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPOSymmDataAugmented:
        """Construct PPO, then resize its storage to hold every symmetry replica of each transition."""
        alg: PPOSymmDataAugmented = PPO.construct_algorithm(obs, env, cfg, device)  # type: ignore[assignment]

        num_replica = alg._raw_actor.num_replica  # type: ignore[attr-defined]
        tiled_obs = TensorDict(
            {key: PPOSymmDataAugmented._tile(value, num_replica) for key, value in obs.items()},
            batch_size=[obs.batch_size[0] * num_replica],
            device=obs.device,
        )
        alg.storage = RolloutStorage(
            "rl", env.num_envs * num_replica, cfg["num_steps_per_env"], tiled_obs, [env.num_actions], device
        )
        alg.transition = RolloutStorage.Transition()
        return alg

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step, replicated across the symmetry group, and update the normalizers."""
        # Update the normalizers
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        # Replicate the transition across the symmetry group before it is stored
        self._augment_transition()

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute returns and advantages over the symmetry-augmented storage."""
        st = self.storage
        # The critic is invariant under the symmetry group by construction, so evaluating it once on the original
        # observations and tiling the result is equivalent to (and cheaper than) re-evaluating every replica.
        last_values = self._tile(self.critic(obs).detach(), self._raw_actor.num_replica)  # type: ignore[attr-defined]
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def _augment_transition(self) -> None:
        """Replicate every field of the current transition across the non-identity symmetry group elements."""
        t = self.transition
        raw_actor = self._raw_actor  # type: ignore[attr-defined]
        elements = raw_actor.G.elements[1:]  # The identity replica is the original, already-collected sample.

        t.observations = self._augment_observations(t.observations)
        t.actions = torch.cat(
            [t.actions] + [raw_actor.actor_out_type.transform_fibers(t.actions, g) for g in elements], dim=0
        )
        mean, std = t.distribution_params
        augmented_mean = torch.cat(
            [mean] + [raw_actor.actor_out_type.transform_fibers(mean, g) for g in elements], dim=0
        )
        # Standard deviations are variance-like: a reflection must not flip their sign.
        augmented_std = torch.abs(
            torch.cat([std] + [raw_actor.actor_out_type.transform_fibers(std, g) for g in elements], dim=0)
        )
        t.distribution_params = (augmented_mean, augmented_std)
        t.actions_log_prob = self._tile(t.actions_log_prob, raw_actor.num_replica)
        t.values = self._tile(t.values, raw_actor.num_replica)
        t.rewards = self._tile(t.rewards, raw_actor.num_replica)
        t.dones = self._tile(t.dones, raw_actor.num_replica)

    def _augment_observations(self, obs: TensorDict) -> TensorDict:
        """Replicate every observation group used by the actor or critic across the symmetry group.

        Groups used by neither model (e.g. an RND-only observation group) are simply tiled without a transform,
        since this class has no representation to transform them by.
        """
        raw_actor, raw_critic = self._raw_actor, self._raw_critic  # type: ignore[attr-defined]
        augmented: dict[str, torch.Tensor] = {}
        for group_name, value in self._transform_obs_group(obs, raw_actor.obs_groups, raw_actor.actor_in_type).items():
            augmented[group_name] = value
        for group_name, value in self._transform_obs_group(
            obs, raw_critic.obs_groups, raw_critic.critic_in_type
        ).items():
            # An observation group shared by both models must already carry a consistent representation; keep the
            # actor's transform in that case rather than overwriting it with an equivalent one from the critic.
            augmented.setdefault(group_name, value)
        for key in obs.keys():
            if key not in augmented:
                augmented[key] = self._tile(obs[key], raw_actor.num_replica)

        batch_size = next(iter(augmented.values())).shape[0]
        return TensorDict(augmented, batch_size=[batch_size], device=obs.device)

    def _transform_obs_group(self, obs: TensorDict, obs_groups: list[str], in_type) -> dict[str, torch.Tensor]:
        """Transform the concatenated observation groups feeding one model, then split them back apart."""
        raw_actor = self._raw_actor  # type: ignore[attr-defined]
        widths = [obs[group_name].shape[-1] for group_name in obs_groups]
        flat = torch.cat([obs[group_name] for group_name in obs_groups], dim=-1)
        replicas = [in_type.transform_fibers(flat, g) for g in raw_actor.G.elements[1:]]

        per_group = {group_name: [obs[group_name]] for group_name in obs_groups}
        for replica in replicas:
            for group_name, chunk in zip(obs_groups, torch.split(replica, widths, dim=-1)):
                per_group[group_name].append(chunk)
        return {group_name: torch.cat(chunks, dim=0) for group_name, chunks in per_group.items()}

    @staticmethod
    def _tile(value: torch.Tensor, num_replica: int) -> torch.Tensor:
        """Repeat a batch-first tensor `num_replica` times along the batch dimension."""
        return value.repeat(num_replica, *([1] * (value.dim() - 1)))
