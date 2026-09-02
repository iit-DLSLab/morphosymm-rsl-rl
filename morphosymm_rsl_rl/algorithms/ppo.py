# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from tensordict import TensorDict

from rsl_rl.algorithms import PPO as RslRlPPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from morphosymm_rsl_rl.modules import SymmModel


class PPO(RslRlPPO):
    """RSL-RL v5.4.2 PPO with morphologically-symmetric (equivariant actor / invariant critic) construction.

    Everything else -- the update step, RND, and RSL-RL's built-in mini-batch symmetry data augmentation / mirror
    loss (``symmetry_cfg``) -- is inherited unchanged from :class:`rsl_rl.algorithms.PPO`.
    """

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct PPO using SymmModel actor and critic networks."""
        # Resolve class callables
        algorithm_name = cfg["algorithm"].pop("class_name")
        alg_class: type[PPO] = PPO if algorithm_name == "PPO" else resolve_callable(algorithm_name)  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # The morphological symmetry configuration (robot, joint order, observation/action representations) is
        # shared by the actor and the critic; each SymmModel instance re-derives its own escnn group from it.
        symm_cfg = cfg["morphologycal_symmetries_cfg"].copy()
        symm_cfg.pop("class_name", None)
        symm_cfg.pop("use_data_augmentation", None)  # consumed by SymmOnPolicyRunner, not a SymmModel kwarg

        # Initialize the actor
        cfg["actor"].pop("class_name", None)
        actor: SymmModel = SymmModel(
            obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"], **symm_cfg
        ).to(device)
        print(f"Actor Model: {actor}")

        # Initialize the critic
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            raise ValueError("`share_cnn_encoders` is not supported by SymmModel.")
        cfg["critic"].pop("class_name", None)
        critic: SymmModel = SymmModel(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"], **symm_cfg).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize the algorithm
        alg: PPO = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg
