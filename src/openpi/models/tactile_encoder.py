import logging
from typing import Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


logger = logging.getLogger("openpi")


class MLPTactileEncoder(nnx.Module):
    """Tactile/force stream configuration and loss behavior.

    Implementation note.
        Tactile/force stream configuration and loss behavior.
        Implementation note.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        emb_dim: int,
        rngs: Optional[nnx.Rngs] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.proj_in = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.proj_out = nnx.Linear(hidden_dim, emb_dim, rngs=rngs)

    def __call__(self, tactile: jax.Array) -> at.Float[at.Array, "b emb"]:
        if tactile.ndim < 2:
            raise ValueError(f"MLPTactileEncoder expects input with rank >= 2, got shape {tactile.shape}.")
        batch_size = tactile.shape[0]
        tactile_flat = tactile.reshape(batch_size, -1)
        if tactile_flat.shape[-1] != self.in_dim:
            raise ValueError(
                f"MLPTactileEncoder: expected flattened dim={self.in_dim}, "
                f"got {tactile_flat.shape[-1]} (input shape={tactile.shape})."
            )
        hidden = self.proj_in(tactile_flat)
        hidden = nnx.swish(hidden)
        return self.proj_out(hidden)


class TactileTCNBlock(nnx.Module):
    """Implementation note.

    Implementation note.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        rngs: Optional[nnx.Rngs] = None,
    ):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1 for TactileTCNBlock.")
        self.kernel_size = kernel_size
        kernels: dict[str, nnx.Linear] = {}
        for k in range(kernel_size):
            kernels[f"kernel_{k}"] = nnx.Linear(in_dim, out_dim, rngs=rngs)
        self.kernels = nnx.Dict(**kernels)
        self.residual_proj = None
        if in_dim != out_dim:
            self.residual_proj = nnx.Linear(in_dim, out_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        if x.ndim != 3:
            raise ValueError(f"TactileTCNBlock expects input of rank 3, got shape {x.shape}.")
        b, n, d = x.shape
        k = self.kernel_size

        # Padding dimensions and their loss weighting.
        pad = jnp.zeros((b, k - 1, d), dtype=x.dtype)
        x_pad = jnp.concatenate([pad, x], axis=1)  # [b, n + k - 1, d]

        # Implementation note.
        y = 0.0
        for idx, linear in self.kernels.items():
            # Implementation note.
            offset = int(idx.split("_")[1])
            start = (k - 1 - offset)
            end = start + n
            x_slice = x_pad[:, start:end, :]  # [b, n, d]
            y = y + linear(x_slice)

        residual = x if self.residual_proj is None else self.residual_proj(x)
        return nnx.swish(y + residual)


class TactileTCNEncoder(nnx.Module):
    """Tactile/force stream configuration and loss behavior.

    Implementation note.
    - has_reference_frame=True：
        Reference-frame and history-window handling.
        Implementation note.
        Implementation note.
    - has_reference_frame=False：
        Reference-frame and history-window handling.
    """

    def __init__(
        self,
        in_dim: int,
        history_len: int,
        hidden_dim: int,
        emb_dim: int,
        has_reference_frame: bool,
        diff_from_reference: bool,
        num_layers: int = 2,
        kernel_size: int = 3,
        rngs: Optional[nnx.Rngs] = None,
    ):
        super().__init__()
        if history_len <= 0:
            raise ValueError("TactileTCNEncoder.history_len must be > 0.")
        if num_layers < 1:
            raise ValueError("TactileTCNEncoder.num_layers must be >= 1.")
        self.history_len = history_len  # Implementation note.
        self.has_reference_frame = has_reference_frame
        self.diff_from_reference = diff_from_reference

        blocks: dict[str, TactileTCNBlock] = {}
        for i in range(num_layers):
            block_in = in_dim if i == 0 else hidden_dim
            blocks[f"block_{i}"] = TactileTCNBlock(
                in_dim=block_in,
                out_dim=hidden_dim,
                kernel_size=kernel_size,
                rngs=rngs,
            )
        self.blocks = nnx.Dict(**blocks)
        # Implementation note.
        self.out_proj = nnx.Linear(hidden_dim, emb_dim, rngs=rngs)

    def __call__(self, tactile: jax.Array) -> at.Float[at.Array, "b emb"]:
        # Implementation note.
        if tactile.ndim == 2:
            tactile_seq = tactile[:, None, :]
        elif tactile.ndim == 3:
            tactile_seq = tactile
        else:
            raise ValueError(
                f"TactileTCNEncoder expects input of rank 2 or 3, got shape {tactile.shape}."
            )

        b, n, d = tactile_seq.shape
        H = self.history_len

        if self.has_reference_frame:
            # Reference-frame and history-window handling.
            if n < 2:
                raise ValueError(
                    "TactileTCNEncoder with has_reference_frame=True expects at least 2 frames, "
                    f"got {n} (shape={tactile_seq.shape})."
                )
            if self.diff_from_reference:
                # Reference-frame and history-window handling.
                max_hist = min(H, n - 1)
                baseline = tactile_seq[:, 0:1, :]           # [B, 1, d]
                history = tactile_seq[:, 1 : 1 + max_hist]  # [B, max_hist, d]
                history = history - baseline                 # [B, max_hist, d]
                if max_hist != H:
                    logger.warning(
                        "TactileTCNEncoder(diff): expected history_len=%d (excluding baseline), "
                        "but only %d history frames available; using %d frames.",
                        H,
                        n - 1,
                        max_hist,
                    )
                seq_for_tcn = history
            else:
                # Reference-frame and history-window handling.
                max_steps = min(H + 1, n)
                seq_for_tcn = tactile_seq[:, :max_steps, :]  # Tactile/force stream configuration and loss behavior.
                if max_steps != H + 1:
                    logger.warning(
                        "TactileTCNEncoder(full-seq): expected 1+history_len=%d frames, "
                        "but only %d available; using %d frames.",
                        H + 1,
                        n,
                        max_steps,
                    )
        else:
            # Reference-frame and history-window handling.
            if n >= H:
                seq_for_tcn = tactile_seq[:, -H:, :]
            else:
                seq_for_tcn = tactile_seq
                logger.warning(
                    "TactileTCNEncoder: expected history_len=%d, but only %d frames available; "
                    "using all frames.",
                    H,
                    n,
                )

        h = seq_for_tcn
        for block in self.blocks.values():
            h = block(h)
        # Tactile/force stream configuration and loss behavior.
        h_last = h[:, -1, :]  # [b, hidden_dim]
        return self.out_proj(h_last)  # [b, emb_dim]


def create_tactile_encoder(
    *,
    encoder_type: str,
    tactile_dim_in: int,
    tactile_history: Optional[int],
    has_reference_frame: bool,
    diff_from_reference: bool,
    expert_width: int,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Tactile/force stream configuration and loss behavior.

    Encoder configuration and sequence handling.
    Encoder configuration and sequence handling.
    """
    if tactile_dim_in <= 0:
        raise ValueError("create_tactile_encoder: tactile_dim_in must be > 0 when encoder is enabled.")

    if encoder_type == "tcn":
        if tactile_history is None:
            raise ValueError(
                "Pi0Config.tactile_history must be set when tactile_encoder_type='tcn'. "
                "For example, both tacforce and tacfield can use 8."
            )
        # Reference-frame and history-window handling.
        # Reference-frame and history-window handling.
        # Implementation note.
        steps_for_dim = tactile_history + 1 if has_reference_frame else tactile_history
        if steps_for_dim <= 0:
            raise ValueError("tactile_history must be > 0 for TCN encoder.")
        if tactile_dim_in % steps_for_dim != 0:
            raise ValueError(
                "Pi0Config.tactile_dim_in ({tactile_dim_in}) must be divisible by "
                f"effective_steps={steps_for_dim} when using TCN encoder; "
                f"got tactile_dim_in={tactile_dim_in}, history={tactile_history}, "
                f"has_reference_frame={has_reference_frame}."
            )
        per_step_dim = tactile_dim_in // steps_for_dim
        return TactileTCNEncoder(
            in_dim=per_step_dim,
            history_len=tactile_history,
            hidden_dim=2 * expert_width,
            emb_dim=expert_width,
            has_reference_frame=has_reference_frame,
            diff_from_reference=diff_from_reference,
            rngs=rngs,
        )

    # Encoder configuration and sequence handling.
    return MLPTactileEncoder(
        in_dim=tactile_dim_in,
        hidden_dim=2 * expert_width,
        emb_dim=expert_width,
        rngs=rngs,
    )


