import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.training import numerics


@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf, -jnp.inf])
def test_finite_checks_find_invalid_leaf(bad):
    tree = {"a": jnp.ones(2), "b": jnp.array([bad])}
    assert not numerics.all_finite(tree)
    assert int(numerics.first_nonfinite_leaf(tree)) == 1
    assert numerics.all_finite({})
    assert int(numerics.first_nonfinite_leaf({})) == -1


@pytest.mark.parametrize("valid", [True, False])
def test_compiled_update_retains_old_state_when_invalid(valid):
    previous = {"params": jnp.array([1.0, 2.0]), "optimizer": jnp.array([0.2]), "step": jnp.array(3)}
    candidate = {
        "params": jnp.array([2.0, 3.0]) if valid else jnp.full(2, jnp.nan),
        "optimizer": jnp.array([0.4]) if valid else jnp.array([jnp.nan]),
        "step": jnp.array(4),
    }
    result = jax.jit(numerics.accept_finite_update)(previous, candidate, jnp.array(valid))
    expected = candidate if valid else previous
    for key in result:
        np.testing.assert_array_equal(result[key], expected[key])
    assert numerics.all_finite(result)


def test_real_train_step_rejects_invalid_candidate():
    from openpi.models.model import BaseModel
    from openpi.models.model import Observation
    from openpi.training.config import TrainConfig
    from openpi.training.utils import TrainState
    from scripts.train import train_step

    class TinyModel(BaseModel):
        def __init__(self):
            self.kernel = nnx.Param(jnp.ones((1, 1)))

        def compute_loss(self, rng, observation, actions, train):
            return jnp.square(actions * self.kernel.value)

        def sample_actions(self, rng, observation, **kwargs):
            raise NotImplementedError("This fixture only exercises the training update")

    model = TinyModel()
    params = nnx.state(model)
    tx = optax.adam(1e-3)
    state = TrainState(
        step=0, params=params, model_def=nnx.graphdef(model), opt_state=tx.init(params), tx=tx, ema_decay=None
    )
    config = TrainConfig(name="finite-unit-test")
    obs = Observation(images={}, image_masks={}, state=jnp.ones((1, 1)))
    step = jax.jit(lambda s, a: train_step(config, jax.random.key(0), s, (obs, a)))
    new_state, info = step(state, jnp.full((1, 1, 1), jnp.nan))
    assert not info["update_applied"]
    assert new_state.step == 0
    for old, new in zip(jax.tree.leaves(state.params), jax.tree.leaves(new_state.params), strict=True):
        np.testing.assert_array_equal(old, new)
    assert numerics.all_finite(new_state.opt_state)
    valid_state, info = step(state, jnp.ones((1, 1, 1)))
    assert info["update_applied"]
    assert valid_state.step == 1
