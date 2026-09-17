import dataclasses
from typing import Literal, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


def global_norm(updates: at.PyTree, *, reduction_dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """Compute a tree norm with FP32 accumulation by default.

    BF16 gradients remain BF16 tensors; only the numerically sensitive sum of
    squares is promoted. This avoids allocating a second full gradient tree.
    """
    dtype = jnp.dtype(reduction_dtype)
    squared_norms = [jnp.sum(jnp.square(jnp.asarray(x, dtype=dtype)), dtype=dtype) for x in jax.tree.leaves(updates)]
    return jnp.sqrt(sum(squared_norms, start=jnp.zeros((), dtype=dtype)))


def clip_by_global_norm(max_norm: float, *, reduction_dtype: jnp.dtype = jnp.float32) -> optax.GradientTransformation:
    """Clip a gradient tree using a norm accumulated in ``reduction_dtype``."""

    def init_fn(_):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        norm = global_norm(updates, reduction_dtype=reduction_dtype)
        scale = jnp.minimum(jnp.asarray(1.0, dtype=norm.dtype), jnp.asarray(max_norm, dtype=norm.dtype) / norm)
        # Keep the gradient tree in its original storage dtype. Only the norm
        # and scalar clipping factor use FP32, avoiding a second FP32 update tree.
        clipped = jax.tree.map(lambda x: x * scale.astype(x.dtype), updates)
        return clipped, state

    return optax.GradientTransformation(init_fn, update_fn)


def optimizer_state_dtype(tx: optax.GradientTransformation, dtype: jnp.dtype) -> optax.GradientTransformation:
    """Store every floating optimizer-state leaf in an explicit dtype."""
    dtype = jnp.dtype(dtype)

    def cast_floating(tree):
        return jax.tree.map(
            lambda x: x.astype(dtype)
            if hasattr(x, "dtype") and jnp.issubdtype(jnp.dtype(x.dtype), jnp.floating)
            else x,
            tree,
        )

    def init_fn(params):
        return cast_floating(tx.init(params))

    def update_fn(updates, state, params=None):
        transformed, new_state = tx.update(updates, state, params)
        return transformed, cast_floating(new_state)

    return optax.GradientTransformation(init_fn, update_fn)


@runtime_checkable
class LRScheduleConfig(Protocol):
    def create(self) -> optax.Schedule: ...


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@runtime_checkable
class OptimizerConfig(Protocol):
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    # Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value.
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0
    moment_dtype: Literal["match_parameter", "float32"] = "match_parameter"

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        fp32_moments = self.moment_dtype == "float32"
        tx = optax.adamw(
            lr,
            b1=self.b1,
            b2=self.b2,
            eps=self.eps,
            mu_dtype=jnp.float32 if fp32_moments else None,
            weight_decay=self.weight_decay,
            mask=weight_decay_mask,
        )
        if fp32_moments:
            # Optax exposes an override only for the first moment. Casting the
            # initialized state also makes the second moment FP32; subsequent
            # updates preserve it because FP32 state dominates BF16 inputs.
            tx = optimizer_state_dtype(tx, jnp.float32)

        return optax.chain(clip_by_global_norm(self.clip_gradient_norm), tx)


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


@dataclasses.dataclass(frozen=True)
class StatelessSGD(OptimizerConfig):
    """Scheduled SGD without momentum tensors, for memory-bound full fine-tuning."""

    clip_gradient_norm: float = 1.0

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for stateless SGD"
        return optax.chain(clip_by_global_norm(self.clip_gradient_norm), optax.sgd(lr, momentum=None))


def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    lr = lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)
