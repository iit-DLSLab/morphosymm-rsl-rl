from __future__ import annotations

import copy
import warnings
from typing import Any

import escnn
import numpy as np
import torch
import torch.nn as nn
from escnn.nn import FieldType
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.distribution import Distribution, GaussianDistribution
from rsl_rl.utils import resolve_callable
from symm_learning.models import EMLP, IMLP
from symm_learning.nn import EquivMultivariateNormal
from tensordict import TensorDict
from torch.distributions import Normal

from morphosymm_rsl_rl.symm_utils import configure_observation_space_representations


class _EquivariantEmpiricalNormalization(EmpiricalNormalization):
    """Empirical normalization with statistics symmetrized over group orbits."""

    def __init__(self, field_type: FieldType) -> None:
        super().__init__(field_type.size)
        group_matrices = torch.stack(
            [
                torch.as_tensor(
                    field_type.representation(group_element),
                    dtype=torch.get_default_dtype(),
                )
                for group_element in field_type.fibergroup.elements
            ]
        )
        self.register_buffer("_group_matrices", group_matrices)
        self._validate_signed_permutation_action()

    def _validate_signed_permutation_action(self) -> None:
        for matrix in self._group_matrices:
            rounded_matrix = matrix.round()
            is_signed_permutation = (
                torch.allclose(matrix, rounded_matrix)
                and torch.all((rounded_matrix == -1) | (rounded_matrix == 0) | (rounded_matrix == 1))
                and torch.all(rounded_matrix.abs().sum(dim=0) == 1)
                and torch.all(rounded_matrix.abs().sum(dim=1) == 1)
            )
            if not is_signed_permutation:
                raise ValueError(
                    "Empirical observation normalization is only guaranteed equivariant for signed-permutation "
                    "representations. Disable observation normalization for this symmetry representation."
                )

    @torch.jit.unused
    def update(self, observations: torch.Tensor) -> None:
        """Update statistics using the complete symmetry orbit of each sample."""
        orbit_observations = torch.einsum("gij,bj->gbi", self._group_matrices, observations)
        orbit_observations = orbit_observations.reshape(-1, observations.shape[-1])
        super().update(orbit_observations)


# ---------------------------------------------------------------------------------------------------------------
# Symmetric weight initialization.
#
# EMLP/IMLP do not hard-constrain their weights to be equivariant; instead, the network is initialized by
# projecting its (randomly initialized) linear layers onto the subspace of weights that commute with the group
# action on their input and output representations. Training can then drift away from this subspace, but starting
# there gives the policy a strong equivariant bias. This block operates directly on the underlying `nn.Linear`
# layers found inside the EMLP/IMLP networks built by `SymmModel`.
# ---------------------------------------------------------------------------------------------------------------


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    if isinstance(module, nn.Linear):
        return [module]
    return [layer for layer in module.children() if isinstance(layer, nn.Linear)]


def _field_representation_matrices(field_type: FieldType, elements, device: torch.device, dtype: torch.dtype):
    return [field_type.fiber_representation(g).to(device=device, dtype=dtype) for g in elements]


def _hidden_representation_matrices(G, size: int, elements, device: torch.device, dtype: torch.dtype):
    regular_size = G.regular_representation.size
    regular_blocks = size // regular_size
    trivial_size = size % regular_size
    matrices = []

    for g in elements:
        blocks = [
            torch.as_tensor(G.regular_representation(g), device=device, dtype=dtype) for _ in range(regular_blocks)
        ]
        if trivial_size:
            blocks.append(torch.eye(trivial_size, device=device, dtype=dtype))
        matrices.append(torch.block_diag(*blocks) if blocks else torch.empty((0, 0), device=device, dtype=dtype))

    return matrices


def _project_linear_layer(layer: nn.Linear, in_mats, out_mats) -> None:
    projected_weight = torch.zeros_like(layer.weight)
    original_weight_norm = torch.linalg.vector_norm(layer.weight)

    for in_mat, out_mat in zip(in_mats, out_mats):
        projected_weight += out_mat.transpose(0, 1) @ layer.weight @ in_mat
    projected_weight /= len(in_mats)

    projected_weight_norm = torch.linalg.vector_norm(projected_weight)
    if original_weight_norm > 0 and projected_weight_norm > 0:
        projected_weight *= original_weight_norm / projected_weight_norm

    layer.weight.copy_(projected_weight)

    if layer.bias is not None:
        projected_bias = torch.zeros_like(layer.bias)
        for out_mat in out_mats:
            projected_bias += out_mat.transpose(0, 1) @ layer.bias
        layer.bias.copy_(projected_bias / len(out_mats))


def _project_mlp_initialization(
    mlp: nn.Module,
    in_mats,
    out_mats,
    hidden_mats_factory,
    name: str,
) -> bool:
    linear_layers = _linear_layers(mlp)
    if not linear_layers:
        warnings.warn(f"Could not symmetrize {name}: no linear layers found.", stacklevel=2)
        return False

    expected_in_features = in_mats[0].shape[0]
    if linear_layers[0].in_features != expected_in_features:
        warnings.warn(
            f"Could not symmetrize {name}: first layer expects {linear_layers[0].in_features} inputs, "
            f"but the symmetry representation has size {expected_in_features}.",
            stacklevel=2,
        )
        return False

    previous_mats = in_mats
    for layer_idx, layer in enumerate(linear_layers):
        is_last_layer = layer_idx == len(linear_layers) - 1
        next_mats = out_mats if is_last_layer else hidden_mats_factory(layer.out_features)

        if layer.in_features != previous_mats[0].shape[0] or layer.out_features != next_mats[0].shape[0]:
            warnings.warn(
                f"Could not symmetrize {name}: layer {layer_idx} has shape "
                f"{layer.out_features}x{layer.in_features}, expected "
                f"{next_mats[0].shape[0]}x{previous_mats[0].shape[0]}.",
                stacklevel=2,
            )
            return False

        _project_linear_layer(layer, previous_mats, next_mats)
        previous_mats = next_mats

    return True


class _EquivariantActorNet(nn.Module):
    """Equivariant expert network wrapped to accept and return plain tensors, like a standard MLP head."""

    def __init__(
        self,
        in_type: FieldType,
        out_type: FieldType,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        small_init_output: bool,
    ) -> None:
        super().__init__()
        self.in_type = in_type
        self.net = EMLP(in_type=in_type, out_type=out_type, bias=True, hidden_units=list(hidden_dims), activation=activation)
        if small_init_output:
            with torch.no_grad():
                head = self.net.net[-1]
                head.weights.mul_(0.2)
                if head.bias is not None:
                    head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.in_type(x)).tensor

    def export(self) -> nn.Module:
        """Return a plain PyTorch module for deployment, stripping escnn-specific wrapping."""
        return self.net.export()


class _InvariantCriticNet(nn.Module):
    """Invariant value network wrapped to accept and return plain tensors, like a standard MLP head."""

    def __init__(
        self,
        in_type: FieldType,
        out_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        small_init_output: bool,
    ) -> None:
        super().__init__()
        self.in_type = in_type
        self.net = IMLP(in_type=in_type, out_dim=out_dim, bias=True, hidden_units=list(hidden_dims), activation=activation)
        if small_init_output:
            with torch.no_grad():
                self.net.head.weight.zero_()
                if self.net.head.bias is not None:
                    self.net.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.in_type(x)).tensor

    def export(self) -> nn.Module:
        """Return a plain PyTorch module for deployment, stripping escnn-specific wrapping."""
        return self.net.export()


class EquivGaussianDistribution(Distribution):
    """Gaussian action distribution whose covariance respects the actor's output symmetry representation.

    Parameter names (``init_std``, ``std_range``, ``std_type``, ``learn_std``) follow RSL-RL's own
    :class:`~rsl_rl.modules.distribution.GaussianDistribution` so this distribution plugs into the same
    ``distribution_cfg`` configuration shape.

    With a state-independent covariance (the default), one variance parameter is learned per irreducible action
    subspace instead of per raw action dimension, so the noise itself stays symmetric. With a state-dependent
    covariance, the actor network emits the covariance parameters directly alongside the mean.
    """

    def __init__(
        self,
        output_dim: int,
        actor_out_type: FieldType,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1e-6, 1e6),
        std_type: str = "scalar",
        learn_std: bool = True,
        state_dependent_std: bool = False,
    ) -> None:
        """Initialize the equivariant Gaussian distribution."""
        super().__init__(output_dim)
        if std_type not in {"scalar", "log"}:
            raise ValueError(f"Unknown standard deviation type: {std_type}. Should be 'scalar' or 'log'.")

        self.actor_out_type = actor_out_type
        self.state_dependent_std = state_dependent_std
        self.std_type = std_type
        self.action_gaussian = EquivMultivariateNormal(y_type=actor_out_type)

        # A state-independent covariance is the symmetric counterpart of RSL-RL's default learned action noise.
        # One parameter is maintained per irreducible action subspace, and `EquivMultivariateNormal` maps these to
        # component variances without breaking equivariance.
        if not state_dependent_std:
            self.std_range = (max(float(std_range[0]), 1e-6), float(std_range[1]))
            self.log_std_range = (
                float(np.log(self.std_range[0])),
                float(np.log(self.std_range[1])),
            )
            noise_shape = (self.action_gaussian.n_cov_params,)
            if std_type == "scalar":
                self.std_param = nn.Parameter(init_std * torch.ones(noise_shape), requires_grad=learn_std)
            else:
                self.log_std_param = nn.Parameter(
                    torch.log(init_std * torch.ones(noise_shape)), requires_grad=learn_std
                )

        self._distribution: torch.distributions.Distribution | None = None
        Normal.set_default_validate_args(False)

    @property
    def input_type(self) -> FieldType:
        """Return the FieldType the actor network must output into."""
        return self.action_gaussian.in_type if self.state_dependent_std else self.actor_out_type

    @property
    def input_dim(self) -> int:
        """Return the number of raw values the actor network must output."""
        return self.input_type.size

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the action distribution from the latest actor forward pass."""
        if self.state_dependent_std:
            distribution_parameters = self.input_type(mlp_output)
        else:
            if self.std_type == "scalar":
                std = self.std_param.clamp(self.std_range[0], self.std_range[1])
                log_variances = 2.0 * torch.log(std)
            else:
                log_std = self.log_std_param.clamp(self.log_std_range[0], self.log_std_range[1])
                log_variances = 2.0 * log_std
            expanded_log_variances = log_variances.expand(*mlp_output.shape[:-1], -1)
            distribution_parameters = self.action_gaussian.in_type(
                torch.cat((mlp_output, expanded_log_variances), dim=-1)
            )
        self._distribution = self.action_gaussian.get_distribution(distribution_parameters)

    def sample(self) -> torch.Tensor:
        """Sample actions from the current distribution."""
        return self._distribution.sample()  # type: ignore[union-attr]

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Return the action mean, which always occupies the leading `output_dim` entries."""
        return mlp_output[..., : self.output_dim]

    def as_deterministic_output_module(self) -> nn.Module:
        """Return the export-friendly module that extracts the mean from the raw actor output."""
        return _ActionMean(self.output_dim)

    @property
    def mean(self) -> torch.Tensor:
        """Return the current action mean."""
        return self._distribution.mean  # type: ignore[union-attr]

    @property
    def std(self) -> torch.Tensor:
        """Return the current per-dimension action standard deviation."""
        return self._distribution.stddev  # type: ignore[union-attr]

    @property
    def entropy(self) -> torch.Tensor:
        """Return the joint entropy of the multivariate action distribution."""
        return self._distribution.entropy()  # type: ignore[union-attr]

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return mean and standard deviation for rollout storage."""
        return (self.mean, self.std)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Return the joint action log probability."""
        return self._distribution.log_prob(outputs)  # type: ignore[union-attr]

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute an approximate KL divergence for the adaptive learning-rate schedule.

        The joint distribution's covariance need not be diagonal, but PPO's adaptive schedule only needs an
        approximate KL. The same diagonal-Gaussian formula as the standard `GaussianDistribution` is applied to the
        marginal mean/std, matching this policy's previous (RSL-RL v3) behavior.
        """
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        return torch.distributions.kl_divergence(Normal(old_mean, old_std), Normal(new_mean, new_std)).sum(dim=-1)


class _ActionMean(nn.Module):
    """Exportable module that extracts the action mean from the actor's raw output."""

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.num_actions = num_actions

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output[..., : self.num_actions]


class _TorchSymmModel(nn.Module):
    """Exportable symmetric model for JIT."""

    def __init__(self, model: SymmModel) -> None:
        """Create a TorchScript-friendly copy of a SymmModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        # escnn networks are not directly deep-copy/JIT compatible; `export()` strips them to plain layers.
        self.mlp = model.mlp.export()
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on pre-concatenated observations."""
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op; the symmetric policy is feed-forward)."""
        pass


class _OnnxSymmModel(nn.Module):
    """Exportable symmetric model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: SymmModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a SymmModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = model.mlp.export()
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]


class SymmModel(MLPModel):
    """RSL-RL v5 MLP-model interface backed by a morphological-symmetry-respecting network.

    The same class is used for both the actor (equivariant mean, optionally state-dependent covariance) and the
    critic (invariant value estimate), selected through ``obs_set``. Both the actor and critic FieldTypes are
    derived once from the robot's symmetry group so they stay consistent across the two model instances.
    """

    is_recurrent = False
    _ESCNN_CACHE_NAMES = frozenset({"matrix", "expanded_bias"})

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        *,
        obs_space_names_actor: list[str] | tuple[str, ...] | None = None,
        obs_space_names_critic: list[str] | tuple[str, ...] | None = None,
        action_space_names: list[str] | tuple[str, ...] | None = None,
        joints_order: list[str] | None = None,
        robot_name: str | None = None,
        state_dependent_std: bool = False,
        small_init_output: bool = True,
        symmetric_initialization: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize a morphologically-symmetric actor or critic model.

        The observation-space name lists must describe the flattened tensors formed by the configured ``actor``
        and ``critic`` observation groups, in exactly the same order.
        """
        # Skip MLPModel.__init__: the network here is escnn-based (EMLP/IMLP), not the generic plain MLP it builds.
        nn.Module.__init__(self)
        if kwargs:
            print(f"SymmModel.__init__ got unexpected arguments, which will be ignored: {list(kwargs)}")

        if obs_space_names_actor is None or obs_space_names_critic is None or action_space_names is None:
            raise ValueError(
                "obs_space_names_actor, obs_space_names_critic, and action_space_names must all be configured."
            )
        if joints_order is None or robot_name is None:
            raise ValueError("joints_order and robot_name must be configured.")
        if obs_set not in ("actor", "critic"):
            raise ValueError(f"SymmModel only supports obs_set 'actor' or 'critic', got {obs_set!r}.")

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.obs_normalization = obs_normalization

        # Load all representations together. Apart from avoiding duplicate robot loads, this guarantees that the
        # actor, critic, and action FieldTypes share the exact same escnn group instance, whichever of the two
        # models (actor or critic) is being built right now.
        all_space_names = list(dict.fromkeys([*obs_space_names_actor, *obs_space_names_critic, *action_space_names]))
        self.G, representations = configure_observation_space_representations(
            robot_name, all_space_names, joints_order
        )
        gspace = escnn.gspaces.no_base_space(self.G)
        self.num_replica = len(self.G.elements)
        self.actor_in_type = FieldType(gspace, [representations[name] for name in obs_space_names_actor])
        self.critic_in_type = FieldType(gspace, [representations[name] for name in obs_space_names_critic])
        self.actor_out_type = FieldType(gspace, [representations[name] for name in action_space_names])

        own_in_type = self.actor_in_type if obs_set == "actor" else self.critic_in_type
        if self.obs_dim != own_in_type.size:
            raise ValueError(
                f"The configured representations describe {own_in_type.size} {obs_set} observations, but the "
                f"environment provides {self.obs_dim}. Check the order and contents of the morphological symmetry "
                "configuration."
            )

        self.obs_normalizer = (
            _EquivariantEmpiricalNormalization(own_in_type) if obs_normalization else nn.Identity()
        )

        if obs_set == "actor":
            if output_dim != self.actor_out_type.size:
                raise ValueError(
                    f"The configured representations describe {self.actor_out_type.size} actions, but the "
                    f"environment expects {output_dim} actions."
                )
            if distribution_cfg is None:
                raise ValueError("SymmModel requires a `distribution_cfg` for the actor.")
            distribution_cfg = distribution_cfg.copy()
            # The `class_name` is validated against RSL-RL's own `GaussianDistribution` (rather than requiring a
            # custom dotted path in the config) so a standard `distribution_cfg` works unmodified; the model that is
            # actually built is always the equivariant counterpart below.
            dist_class = resolve_callable(distribution_cfg.pop("class_name"))
            if dist_class is not GaussianDistribution:
                raise ValueError("SymmModel actors currently support only `GaussianDistribution`.")
            self.distribution = EquivGaussianDistribution(
                output_dim, self.actor_out_type, state_dependent_std=state_dependent_std, **distribution_cfg
            )
            self.mlp = _EquivariantActorNet(
                self.actor_in_type, self.distribution.input_type, hidden_dims, activation, small_init_output
            )
        else:
            if distribution_cfg is not None:
                raise ValueError("SymmModel critics do not accept a `distribution_cfg`.")
            self.distribution = None
            self.mlp = _InvariantCriticNet(self.critic_in_type, output_dim, hidden_dims, activation, small_init_output)

        if symmetric_initialization:
            self._apply_symmetric_initialization(obs_set, own_in_type)

        model_params = sum(p.numel() for p in self.mlp.parameters() if p.requires_grad)
        print(f"{obs_set.capitalize()} [{model_params / 1e6:.2f}M params]:\n{self.mlp.net}")

    def _apply_symmetric_initialization(self, obs_set: str, in_type: FieldType) -> None:
        """Project the freshly initialized weights onto the symmetry-equivariant subspace."""
        with torch.no_grad():
            dtype = next(self.mlp.parameters()).dtype
            device = next(self.mlp.parameters()).device
            elements = self.G.elements
            hidden_mats_cache: dict[int, list[torch.Tensor]] = {}

            def hidden_mats_factory(size: int) -> list[torch.Tensor]:
                if size not in hidden_mats_cache:
                    hidden_mats_cache[size] = _hidden_representation_matrices(self.G, size, elements, device, dtype)
                return hidden_mats_cache[size]

            in_mats = _field_representation_matrices(in_type, elements, device, dtype)

            if obs_set == "actor":
                actor_out_mats = _field_representation_matrices(self.actor_out_type, elements, device, dtype)
                actor_layers = _linear_layers(self.mlp.net)
                if actor_layers and actor_layers[-1].out_features == 2 * self.actor_out_type.size:
                    # The actor also emits state-dependent covariance parameters, which mirror the mean under the
                    # unsigned action representation.
                    out_mats = [torch.block_diag(action_mat, action_mat.abs()) for action_mat in actor_out_mats]
                else:
                    out_mats = actor_out_mats
                _project_mlp_initialization(self.mlp.net, in_mats, out_mats, hidden_mats_factory, "actor")
                self._project_action_std_initialization(actor_out_mats)
            else:
                critic_out_mats = [torch.ones((1, 1), device=device, dtype=dtype) for _ in elements]
                _project_mlp_initialization(self.mlp.net, in_mats, critic_out_mats, hidden_mats_factory, "critic")

    def _project_action_std_initialization(self, actor_out_mats: list[torch.Tensor]) -> None:
        """Symmetrize the state-independent noise parameter, if one was created."""
        if self.distribution is None or self.distribution.state_dependent_std:
            return
        std_param = getattr(self.distribution, "std_param", None)
        if std_param is None:
            std_param = getattr(self.distribution, "log_std_param", None)
        if std_param is None or std_param.ndim != 1 or std_param.shape[0] != self.actor_out_type.size:
            return
        projected_std = torch.zeros_like(std_param)
        for action_mat in actor_out_mats:
            projected_std += action_mat.abs().transpose(0, 1) @ std_param
        std_param.copy_(projected_std / len(actor_out_mats))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load learned state while ignoring escnn's mode-dependent linear caches.

        ``escnn.nn.Linear`` adds ``matrix`` and ``expanded_bias`` buffers in evaluation mode and removes them in
        training mode. Checkpoints therefore contain different keys depending on the mode in which they were saved,
        even though these buffers are derived from the learned parameters.
        """
        was_training = self.training
        self.train()
        learned_state = {
            key: value
            for key, value in state_dict.items()
            if key.rsplit(".", maxsplit=1)[-1] not in self._ESCNN_CACHE_NAMES
        }
        try:
            super().load_state_dict(learned_state, strict=strict)
        finally:
            self.train(was_training)
        return True

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchSymmModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxSymmModel(self, verbose)
