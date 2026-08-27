# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from morphosymm_rl.algorithms.ppo import PPO as SymmPPO
from morphosymm_rl.algorithms.ppo_symm_data_augment import PPOSymmDataAugmented


class SymmOnPolicyRunner(OnPolicyRunner):
    """RSL-RL on-policy runner selecting a morphologically-symmetric PPO implementation."""

    _ALGORITHM_CLASSES = {
        "PPO": SymmPPO,
        "PPOSymmDataAugmented": PPOSymmDataAugmented,
    }

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Use the local PPO variants for symmetric configs, then delegate all runner behavior to RSL-RL."""
        # Select the symmetry-aware algorithm while keeping the standard RSL-RL configuration shape
        class_name = train_cfg["algorithm"]["class_name"]
        train_cfg["algorithm"]["class_name"] = self._ALGORITHM_CLASSES.get(class_name, class_name)

        # Delegate construction, learning, logging, checkpoints, and exports to the upstream runner
        super().__init__(env, train_cfg, log_dir, device)
