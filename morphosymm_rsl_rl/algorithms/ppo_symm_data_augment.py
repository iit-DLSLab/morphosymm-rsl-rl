# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import escnn
import torch
from escnn.group import Group
from escnn.nn import FieldType
from tensordict import TensorDict

from rsl_rl.algorithms import PPO as RslRlPPO
from rsl_rl.env import VecEnv
from rsl_rl.storage import RolloutStorage

from morphosymm_rsl_rl.algorithms.ppo import PPO
from morphosymm_rsl_rl.symm_utils import configure_observation_space_representations


class _AugmentedRolloutStorage(RolloutStorage):
    """Rollout storage over symmetry-augmented transitions.

    Mirrors :meth:`RolloutStorage.mini_batch_generator`, but also records -- on ``last_batch_is_identity``, right
    before each mini-batch is yielded -- which of its rows are genuinely collected samples (the identity replica)
    rather than the analytically symmetrized replicas added by :meth:`PPOSymmDataAugmented._augment_transition`.
    ``PPOSymmDataAugmented`` uses this to restrict the adaptive learning-rate schedule's KL statistic to real
    policy drift; see :func:`_identity_only_kl_divergence` for why that matters.
    """

    def __init__(self, *args, num_original_envs: int, **kwargs) -> None:
        """Initialize the storage, additionally recording how many environments each replica block spans."""
        super().__init__(*args, **kwargs)
        self.num_original_envs = num_original_envs
        self.last_batch_is_identity: torch.Tensor | None = None

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        """Yield the same mini-batches as the base implementation, tagging each one's identity-replica rows."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Each stored timestep replicates `num_original_envs` genuine samples into `self.num_envs` rows, with the
        # identity replica occupying the first `num_original_envs` of every such block (see `_augment_transition`).
        is_identity_per_step = torch.arange(self.num_envs, device=self.device) < self.num_original_envs
        is_identity_flat = is_identity_per_step.repeat(self.num_transitions_per_env)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)  # type: ignore

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]
                self.last_batch_is_identity = is_identity_flat[batch_idx]

                yield RolloutStorage.Batch(
                    observations=observations[batch_idx],  # type: ignore
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                )


def _identity_only_kl_divergence(actor: object, storage: _AugmentedRolloutStorage) -> None:
    """Patch ``actor.get_kl_divergence`` so the adaptive learning-rate schedule reflects only real policy drift.

    ``old_distribution_params`` for the analytically symmetrized replicas (see ``_augment_transition``) is a
    transform of the identity replica's own targets, not something the plain (non-equivariant) actor network has
    necessarily learned to reproduce yet. Left untreated, that equivariance gap -- not the actual policy step size
    -- dominates ``rsl_rl.algorithms.PPO.update``'s KL average and collapses the adaptive learning rate toward its
    floor from the very first update, regardless of how the policy is actually changing. Restricting the statistic
    to the identity-replica rows recovers the schedule's intended meaning without touching ``update()`` itself.
    """
    original_get_kl_divergence = actor.get_kl_divergence  # type: ignore[attr-defined]

    def get_kl_divergence(old_distribution_params, new_distribution_params):
        kl = original_get_kl_divergence(old_distribution_params, new_distribution_params)
        mask = storage.last_batch_is_identity
        if mask is None or not torch.any(mask):
            return kl
        return kl.new_full(kl.shape, kl[mask].mean())

    actor.get_kl_divergence = get_kl_divergence  # type: ignore[attr-defined]


class PPOSymmDataAugmented(PPO):
    """PPO whose rollout storage holds every symmetry-group replica of each collected transition.

    Unlike RSL-RL's built-in ``symmetry_cfg`` (which only tiles a mini-batch during ``update()``, repeating the
    same returns/values for every replica), this class augments the rollout *as it is collected*: every
    observation, action, and distribution parameter is transformed by each non-identity group element and stored
    as its own transition, so ``compute_returns`` bootstraps proper GAE targets for every replica.

    This is the data-augmentation counterpart to the equivariant/invariant architecture built by
    :class:`~morphosymm_rsl_rl.algorithms.ppo.PPO` (via :class:`~morphosymm_rsl_rl.modules.SymmModel`): the two
    mechanisms are alternatives, so this class builds a plain, non-equivariant actor/critic -- exactly like
    RSL-RL's own PPO would -- and instead derives the escnn group and observation/action ``FieldType``\\ s it needs
    directly from ``morphologycal_symmetries_cfg``, independently of the actor/critic networks. Only the storage
    sizing, transition augmentation, and return bootstrap need overriding; the loss computation in ``update()`` is
    inherited unchanged from :class:`~morphosymm_rsl_rl.algorithms.ppo.PPO`.
    """

    G: Group
    num_replica: int
    actor_in_type: FieldType
    critic_in_type: FieldType
    actor_out_type: FieldType

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPOSymmDataAugmented:
        """Build a plain RSL-RL PPO, then resize its storage to hold every symmetry replica of each transition."""
        symm_cfg = cfg["morphologycal_symmetries_cfg"]

        # Resolve the escnn group and the FieldTypes describing the actor/critic observations and the actions.
        # This is independent of the actor/critic networks, which are plain (non-equivariant) here.
        all_space_names = list(
            dict.fromkeys(
                [
                    *symm_cfg["obs_space_names_actor"],
                    *symm_cfg["obs_space_names_critic"],
                    *symm_cfg["action_space_names"],
                ]
            )
        )
        G, representations = configure_observation_space_representations(
            symm_cfg["robot_name"], all_space_names, symm_cfg["joints_order"]
        )
        gspace = escnn.gspaces.no_base_space(G)
        actor_in_type = FieldType(gspace, [representations[name] for name in symm_cfg["obs_space_names_actor"]])
        critic_in_type = FieldType(gspace, [representations[name] for name in symm_cfg["obs_space_names_critic"]])
        actor_out_type = FieldType(gspace, [representations[name] for name in symm_cfg["action_space_names"]])

        # Build the actor, critic, storage, and algorithm exactly like RSL-RL's own plain PPO would.
        alg: PPOSymmDataAugmented = RslRlPPO.construct_algorithm(obs, env, cfg, device)  # type: ignore[assignment]
        alg.G = G
        alg.num_replica = len(G.elements)
        alg.actor_in_type = actor_in_type
        alg.critic_in_type = critic_in_type
        alg.actor_out_type = actor_out_type

        tiled_obs = TensorDict(
            {key: PPOSymmDataAugmented._tile(value, alg.num_replica) for key, value in obs.items()},
            batch_size=[obs.batch_size[0] * alg.num_replica],
            device=obs.device,
        )
        alg.storage = _AugmentedRolloutStorage(
            "rl",
            env.num_envs * alg.num_replica,
            cfg["num_steps_per_env"],
            tiled_obs,
            [env.num_actions],
            device,
            num_original_envs=env.num_envs,
        )
        alg.transition = RolloutStorage.Transition()
        _identity_only_kl_divergence(alg.actor, alg.storage)
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
        # The critic is a plain (non-equivariant) network here, so unlike a value bootstrapped from an invariant
        # critic it cannot be evaluated once and tiled -- each replica's transformed observations are evaluated
        # separately, exactly as `_augment_transition` does for every other stored step.
        last_values = self.critic(self._augment_observations(obs)).detach()
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
        elements = self.G.elements[1:]  # The identity replica is the original, already-collected sample.

        t.observations = self._augment_observations(t.observations)
        t.actions = torch.cat(
            [t.actions] + [self.actor_out_type.transform_fibers(t.actions, g) for g in elements], dim=0
        )
        mean, std = t.distribution_params
        augmented_mean = torch.cat(
            [mean] + [self.actor_out_type.transform_fibers(mean, g) for g in elements], dim=0
        )
        # Standard deviations are variance-like: a reflection must not flip their sign.
        augmented_std = torch.abs(
            torch.cat([std] + [self.actor_out_type.transform_fibers(std, g) for g in elements], dim=0)
        )
        t.distribution_params = (augmented_mean, augmented_std)
        # Under a signed-permutation group action, transforming a diagonal Gaussian's sample, mean, and standard
        # deviation together leaves its log-density unchanged, so the identity replica's log-probability is exact
        # for every replica regardless of whether the actor network itself is equivariant.
        t.actions_log_prob = self._tile(t.actions_log_prob, self.num_replica)
        # Unlike `actions_log_prob`, the critic's value has no such invariance here -- it is a plain (non-invariant)
        # network -- so every replica's transformed observations must be evaluated separately.
        t.values = self.critic(t.observations).detach()
        t.rewards = self._tile(t.rewards, self.num_replica)
        t.dones = self._tile(t.dones, self.num_replica)

    def _augment_observations(self, obs: TensorDict) -> TensorDict:
        """Replicate every observation group used by the actor or critic across the symmetry group.

        Groups used by neither model (e.g. an RND-only observation group) are simply tiled without a transform,
        since this class has no representation to transform them by.
        """
        raw_actor, raw_critic = self._raw_actor, self._raw_critic  # type: ignore[attr-defined]
        augmented: dict[str, torch.Tensor] = {}
        for group_name, value in self._transform_obs_group(obs, raw_actor.obs_groups, self.actor_in_type).items():
            augmented[group_name] = value
        for group_name, value in self._transform_obs_group(
            obs, raw_critic.obs_groups, self.critic_in_type
        ).items():
            # An observation group shared by both models must already carry a consistent representation; keep the
            # actor's transform in that case rather than overwriting it with an equivalent one from the critic.
            augmented.setdefault(group_name, value)
        for key in obs.keys():
            if key not in augmented:
                augmented[key] = self._tile(obs[key], self.num_replica)

        batch_size = next(iter(augmented.values())).shape[0]
        return TensorDict(augmented, batch_size=[batch_size], device=obs.device)

    def _transform_obs_group(self, obs: TensorDict, obs_groups: list[str], in_type: FieldType) -> dict[str, torch.Tensor]:
        """Transform the concatenated observation groups feeding one model, then split them back apart."""
        widths = [obs[group_name].shape[-1] for group_name in obs_groups]
        flat = torch.cat([obs[group_name] for group_name in obs_groups], dim=-1)
        replicas = [in_type.transform_fibers(flat, g) for g in self.G.elements[1:]]

        per_group = {group_name: [obs[group_name]] for group_name in obs_groups}
        for replica in replicas:
            for group_name, chunk in zip(obs_groups, torch.split(replica, widths, dim=-1)):
                per_group[group_name].append(chunk)
        return {group_name: torch.cat(chunks, dim=0) for group_name, chunks in per_group.items()}

    @staticmethod
    def _tile(value: torch.Tensor, num_replica: int) -> torch.Tensor:
        """Repeat a batch-first tensor `num_replica` times along the batch dimension."""
        return value.repeat(num_replica, *([1] * (value.dim() - 1)))
