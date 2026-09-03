"""Small CPU-only modules; no pretrained model, optimizer step or training loop."""

import dataclasses

import flax.nnx as nnx
from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models.pi0 import action_only_flow_loss
from openpi.models.tactile_encoder import TactileLoRALinear
from openpi.models.tactile_encoder import create_tactile_encoder
from openpi.training import config
from openpi.training import weight_loaders

BASELINE = "pi0_lora_tacfield_local_smoke"
ADAPTED = "pi0_lora_tacfield_local_tactile_lora_smoke"


def tiny_encoder(rank=2, seed=0):
    return create_tactile_encoder(
        encoder_type="tcn",
        tactile_dim_in=9 * 6,
        tactile_history=8,
        has_reference_frame=True,
        diff_from_reference=False,
        expert_width=8,
        rngs=nnx.Rngs(seed),
        lora_rank=rank,
        lora_alpha=2.0,
    )


def flat_params(model):
    return flatten_dict(nnx.state(model, nnx.Param).to_pure_dict(), sep="/")


def fake_restore(monkeypatch, loaded):
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda p: p)
    monkeypatch.setattr("openpi.models.model.restore_params", lambda *a, **k: loaded)


def test_tactile_lora_configuration_preserves_baseline():
    base, adapted = config.get_config(BASELINE), config.get_config(ADAPTED)
    assert base.model.tactile_prefix_lora_rank == 0
    assert base.weight_loader.strict_allow_missing_regex is None
    assert config.get_config("pi0_lora_tacfield_tabero").model.tactile_prefix_lora_rank == 0
    assert adapted.model.tactile_prefix_lora_rank == 16
    assert adapted.model.tactile_prefix_lora_alpha == 16
    assert adapted.model.supervised_action_dim == 7
    assert adapted.model.tactile_loss_weight == adapted.model.padding_loss_weight == 0
    assert adapted.data == base.data
    assert adapted.lr_schedule == base.lr_schedule
    assert adapted.weight_loader.params_path == base.weight_loader.params_path
    assert adapted.weight_loader.strict
    param = nnx.Param(jnp.ones(1))
    for path in flat_params(tiny_encoder()):
        assert adapted.trainable_filter(("tactile_prefix_encoder", *path.split("/")), param) == ("lora_" in path)
    assert adapted.trainable_filter(("PaliGemma", "llm", "lora_a"), param)
    assert not adapted.trainable_filter(("PaliGemma", "img", "kernel"), param)
    assert not adapted.trainable_filter(("action_out_proj", "kernel"), param)


def test_rank_zero_keeps_original_linear_paths():
    encoder = tiny_encoder(rank=0)
    assert type(encoder.out_proj) is nnx.Linear
    assert type(encoder.blocks["block_0"].kernels["kernel_0"]) is nnx.Linear
    params = flat_params(encoder)
    assert len(params) == 16
    assert all(path.endswith(("/kernel", "/bias")) for path in params)
    assert params["blocks/block_0/kernels/kernel_0/kernel"].shape == (6, 16)
    assert params["out_proj/kernel"].shape == (16, 8)


def test_configured_encoder_parameter_counts_without_allocating_weights():
    cfg = config.get_config(ADAPTED).model

    def parameter_shapes():
        encoder = create_tactile_encoder(
            encoder_type=cfg.tactile_prefix_encoder_type,
            tactile_dim_in=cfg.tactile_prefix_dim_in,
            tactile_history=cfg.tactile_prefix_history,
            has_reference_frame=cfg.tactile_prefix_use_reference_frame,
            diff_from_reference=cfg.tactile_prefix_diff_from_reference,
            expert_width=gemma.get_config(cfg.paligemma_variant).width,
            rngs=nnx.Rngs(0),
            lora_rank=cfg.tactile_prefix_lora_rank,
            lora_alpha=cfg.tactile_prefix_lora_alpha,
        )
        return nnx.state(encoder, nnx.Param).to_pure_dict()

    shapes = flatten_dict(jax.eval_shape(parameter_shapes), sep="/")
    assert sum(np.prod(v.shape) for k, v in shapes.items() if "/lora_" in k) == 779008
    assert sum(np.prod(v.shape) for k, v in shapes.items() if "/lora_" not in k) == 65239040


def test_restore_base_then_zero_adapters_preserves_output(monkeypatch):
    base, adapted = tiny_encoder(rank=0), tiny_encoder(seed=7)
    original_base = nnx.state(base, nnx.Param).to_pure_dict()
    fake_restore(monkeypatch, {"tactile_prefix_encoder": original_base})
    reference = {"tactile_prefix_encoder": nnx.state(adapted, nnx.Param).to_pure_dict()}
    restored = config.get_config(ADAPTED).weight_loader.load(reference)
    state = nnx.state(adapted)
    state.replace_by_pure_dict(restored["tactile_prefix_encoder"])
    nnx.update(adapted, state)
    inputs = jax.random.normal(jax.random.key(2), (2, 9, 6))
    np.testing.assert_array_equal(base(inputs), adapted(inputs))
    params = flat_params(adapted)
    assert len(params) == 32
    for path, value in params.items():
        if path.endswith("lora_b"):
            np.testing.assert_array_equal(value, 0)
        elif path.endswith("lora_a"):
            assert np.any(value != 0)
        else:
            np.testing.assert_array_equal(value, flat_params(base)[path])


def test_action_only_backward_reaches_only_selected_tactile_adapters():
    encoder = tiny_encoder()
    before = {k: np.array(v) for k, v in flat_params(encoder).items()}
    inputs = jax.random.normal(jax.random.key(3), (2, 9, 6))

    def loss(module):
        # Tiny stand-in action projection, with ignored slots as in the real objective.
        pred = jnp.pad(module(inputs)[:, None, :], ((0, 0), (0, 0), (0, 24)))
        return action_only_flow_loss(pred, jnp.zeros_like(pred), 7).mean()

    diff = nnx.DiffState(0, config.get_config(ADAPTED).trainable_filter)
    grads = flatten_dict(nnx.grad(loss, argnums=diff)(encoder).to_pure_dict(), sep="/")
    assert len(grads) == 16
    assert all("lora_" in path and np.isfinite(value).all() for path, value in grads.items())
    assert all(np.any(value != 0) for path, value in grads.items() if path.endswith("lora_b"))
    assert all(np.all(value == 0) for path, value in grads.items() if path.endswith("lora_a"))
    # Backward alone must not modify any parameter (no optimizer is called).
    for path, value in flat_params(encoder).items():
        np.testing.assert_array_equal(value, before[path])

    # A synthetic nonzero B tests that A can receive gradients after initialization.
    encoder.out_proj.lora_b.value = jnp.full_like(encoder.out_proj.lora_b.value, 0.02)
    grads = flatten_dict(nnx.grad(loss, argnums=diff)(encoder).to_pure_dict(), sep="/")
    assert np.any(grads["out_proj/lora_a"] != 0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_linear_lora_scaling_dtype_and_jit(dtype):
    layer = TactileLoRALinear(3, 4, rank=2, alpha=6.0, rngs=nnx.Rngs(0))
    layer.kernel.value = layer.kernel.value.astype(dtype)
    layer.bias.value = layer.bias.value.astype(dtype)
    inputs = jnp.ones((2, 5, 3), dtype=dtype)
    layer.lora_a.value = jnp.full((3, 2), 0.25)
    layer.lora_b.value = jnp.full((2, 4), 0.5)
    base = inputs @ layer.kernel.value + layer.bias.value
    expected = base + 3.0 * ((inputs @ layer.lora_a.value.astype(dtype)) @ layer.lora_b.value.astype(dtype))
    result = nnx.jit(lambda module, x: module(x))(layer, inputs)
    assert result.dtype == dtype
    np.testing.assert_allclose(np.asarray(result, dtype=np.float32), np.asarray(expected, dtype=np.float32))


def checkpoint_fixture():
    base = {
        "tactile_prefix_encoder": nnx.state(tiny_encoder(rank=0), nnx.Param).to_pure_dict(),
        "PaliGemma": {"llm": {"lora_a": np.full(2, 7.0, np.float32)}},
    }
    adapted = {
        "tactile_prefix_encoder": nnx.state(tiny_encoder(), nnx.Param).to_pure_dict(),
        "PaliGemma": {"llm": {"lora_a": np.zeros(2, np.float32)}},
    }
    return base, adapted


def test_allow_new_adapters_preserves_training_shape_placeholders(monkeypatch):
    loaded, reference = checkpoint_fixture()
    shapes = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), reference)
    fake_restore(monkeypatch, loaded)
    restored = config.get_config(ADAPTED).weight_loader.load(shapes)
    flat = flatten_dict(restored, sep="/")
    new = [v for k, v in flat.items() if k.startswith("tactile_prefix_encoder/") and "/lora_" in k]
    assert len(new) == 16
    assert all(isinstance(value, jax.ShapeDtypeStruct) for value in new)
    np.testing.assert_array_equal(restored["PaliGemma"]["llm"]["lora_a"], 7)
    for key, value in flatten_dict(loaded, sep="/").items():
        np.testing.assert_array_equal(flat[key], value)


@pytest.mark.parametrize("fault", ["base_missing", "backbone_lora_missing", "shape", "extra", "partial_adapter"])
def test_new_adapter_exception_does_not_weaken_pretrained_checks(monkeypatch, fault):
    loaded, reference = checkpoint_fixture()
    flat = flatten_dict(loaded, sep="/")
    base_path = "tactile_prefix_encoder/out_proj/kernel"
    if fault == "base_missing":
        del flat[base_path]
    elif fault == "backbone_lora_missing":
        del flat["PaliGemma/llm/lora_a"]
    elif fault == "shape":
        flat[base_path] = np.ones(3)
    elif fault == "extra":
        flat["tactile_prefix_encoder/out_proj/unknown"] = np.ones(1)
    else:
        flat["tactile_prefix_encoder/out_proj/lora_a"] = np.ones((16, 2))
    fake_restore(monkeypatch, unflatten_dict(flat, sep="/"))
    with pytest.raises(ValueError, match="Strict checkpoint mismatch"):
        config.get_config(ADAPTED).weight_loader.load(reference)


def test_existing_tactile_adapters_restore_without_reset(monkeypatch):
    _, reference = checkpoint_fixture()
    flat = flatten_dict(reference, sep="/")
    loaded = unflatten_dict({k: np.full_like(v, 0.3) for k, v in flat.items()}, sep="/")
    fake_restore(monkeypatch, loaded)
    restored = config.get_config(ADAPTED).weight_loader.load(reference)
    for value in jax.tree.leaves(restored):
        np.testing.assert_array_equal(value, np.full_like(value, 0.3))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tactile_prefix_lora_rank": -1},
        {"tactile_prefix_lora_alpha": 0},
        {"tactile_prefix_lora_alpha": float("nan")},
        {"tactile_prefix_encoder_type": "mlp"},
        {"tactile_streams": ()},
    ],
)
def test_reject_invalid_tactile_lora_config(kwargs):
    with pytest.raises(ValueError, match="LoRA|lora"):
        dataclasses.replace(config.get_config(ADAPTED).model, **kwargs)


def test_reject_allowlist_without_strict_mode():
    with pytest.raises(ValueError, match="strict=True"):
        weight_loaders.CheckpointWeightLoader("unused", strict_allow_missing_regex=".*")
