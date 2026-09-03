import dataclasses
import math
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
from openpi.shared.tactile_type import TactileType
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Tactile / torque configuration (JAX-only; PyTorch side handles its own config).
    # By default, models ignore tactile.
    tactile_type: TactileType = TactileType.NO
    # Per-timestep tactile dimension (e.g., 14 for 7 joints * 2 arms), before history concatenation.
    tactile_dim: int = 14
    # Effective input dim for the tactile MLP projector. Usually tactile_dim * len(tactile_history).
    # When not set explicitly (e.g., from training config + data.tactile_history), we fall back to tactile_dim.
    tactile_dim_in: int | None = None
    # Tactile/force stream configuration and loss behavior.
    # Reference-frame and history-window handling.
    # Implementation note.
    tactile_history: int | None = None
    # Action/force dimensions and loss handling.
    # Tactile/force stream configuration and loss behavior.
    # Action/force dimensions and loss handling.
    effective_action_dim: int | None = None
    # Optional action-only objective; keeps checkpoint architecture and tactile conditioning unchanged.
    supervised_action_dim: int | None = None
    # Tactile/force stream configuration and loss behavior.
    # Tactile/force stream configuration and loss behavior.
    tactile_loss_weight: float = 0.1
    # Action/force dimensions and loss handling.
    # Padding dimensions and their loss weighting.
    # Padding dimensions and their loss weighting.
    # Tactile/force stream configuration and loss behavior.
    padding_loss_weight: float = 1.0
    # Loss component computation and logging behavior.
    # Action/force dimensions and loss handling.
    # Tactile/force stream configuration and loss behavior.
    # Action/force dimensions and loss handling.
    # Implementation note.
    # Action/force dimensions and loss handling.
    # Tactile/force stream configuration and loss behavior.
    expert_his_c_fut_loss_mode: str = "weighted_full"
    # Tactile/force stream configuration and loss behavior.
    # Implementation note.
    # Tactile/force stream configuration and loss behavior.
    tactile_encoder_type: str = "mlp"
    # Encoder configuration and sequence handling.
    # Reference-frame and history-window handling.
    # Reference-frame and history-window handling.
    tactile_use_reference_frame: bool = False
    # Tactile/force stream configuration and loss behavior.
    # Implementation note.
    # Implementation note.
    tactile_diff_from_reference: bool = True

    # Tactile/force stream configuration and loss behavior.
    tactile_prefix_dim_in: int | None = None
    tactile_prefix_history: int | None = None
    tactile_prefix_encoder_type: str | None = None
    tactile_prefix_use_reference_frame: bool | None = None
    tactile_prefix_diff_from_reference: bool | None = None
    # Opt-in low-rank adaptation of the prefix TCN; zero preserves legacy architecture.
    tactile_prefix_lora_rank: int = 0
    tactile_prefix_lora_alpha: float = 16.0

    # Tactile/force stream configuration and loss behavior.
    # Tactile/force stream configuration and loss behavior.
    # Tactile/force stream configuration and loss behavior.
    # Tactile/force stream configuration and loss behavior.
    # Tactile/force stream configuration and loss behavior.
    tactile_streams: tuple[str, ...] = ()

    # Tactile/force stream configuration and loss behavior.
    # Tactile/force stream configuration and loss behavior.
    # Tactile/force stream configuration and loss behavior.
    #
    # Tactile/force stream configuration and loss behavior.
    # Action/force dimensions and loss handling.
    tactile_suffix_placement: str = "suffix"

    def __post_init__(self):
        if self.tactile_prefix_lora_rank < 0:
            raise ValueError("tactile_prefix_lora_rank must be nonnegative")
        if self.tactile_prefix_lora_rank:
            if not math.isfinite(self.tactile_prefix_lora_alpha) or self.tactile_prefix_lora_alpha <= 0:
                raise ValueError("tactile_prefix_lora_alpha must be finite and positive")
            if (
                self.tactile_type is not TactileType.EXPERT_HIS_C_FUT
                or "tactile_prefix" not in self.tactile_streams
                or self.tactile_prefix_encoder_type != "tcn"
                or not self.tactile_prefix_dim_in
                or self.tactile_prefix_dim_in <= 0
            ):
                raise ValueError("Prefix LoRA requires an enabled EXPERT_HIS_C_FUT tactile_prefix TCN")
        if self.supervised_action_dim is not None and not 1 <= self.supervised_action_dim <= self.action_dim:
            raise ValueError("supervised_action_dim must be between 1 and action_dim")
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.tactile_dim_in is None:
            object.__setattr__(self, "tactile_dim_in", self.tactile_dim)
        if self.effective_action_dim is None:
            object.__setattr__(self, "effective_action_dim", self.action_dim)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
