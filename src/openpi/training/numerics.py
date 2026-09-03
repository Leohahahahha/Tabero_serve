"""Finite-value checks for rejecting, never silently skipping, invalid updates."""

import jax
import jax.numpy as jnp


def leaf_finite_flags(tree):
    leaves = jax.tree.leaves(tree)
    return jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]) if leaves else jnp.ones((0,), dtype=bool)


def all_finite(tree):
    return jnp.all(leaf_finite_flags(tree))


def first_nonfinite_leaf(tree):
    flags = leaf_finite_flags(tree)
    if flags.size == 0:
        return jnp.array(-1, dtype=jnp.int32)
    return jnp.where(jnp.all(flags), -1, jnp.argmax(~flags))


def accept_finite_update(previous, candidate, valid):
    """Return the old parameters AND optimizer state if any update check fails."""
    return jax.lax.cond(valid, lambda: candidate, lambda: previous)
