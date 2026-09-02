# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from morphosymm_rsl_rl.algorithms.ppo import PPO as SymmPPO
from morphosymm_rsl_rl.algorithms.ppo_symm_data_augment import PPOSymmDataAugmented


class SymmOnPolicyRunner(OnPolicyRunner):
    """RSL-RL on-policy runner selecting a morphologically-symmetric PPO implementation.

    The choice between the plain equivariant PPO and its transition-augmented variant is read from
    ``morphologycal_symmetries_cfg.use_data_augmentation`` -- ``algorithm.class_name`` is overwritten
    unconditionally and does not need to be set by the caller.
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Select the symmetry-aware PPO variant, then delegate all runner behavior to RSL-RL."""
        symm_cfg = train_cfg.get("morphologycal_symmetries_cfg") or {}
        use_data_augmentation = symm_cfg.get("use_data_augmentation", False)
        train_cfg["algorithm"]["class_name"] = PPOSymmDataAugmented if use_data_augmentation else SymmPPO

        # Delegate construction, learning, logging, checkpoints, and exports to the upstream runner
        super().__init__(env, train_cfg, log_dir, device)
