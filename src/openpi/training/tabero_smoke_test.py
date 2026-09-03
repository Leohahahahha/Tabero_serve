import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from openpi import transforms
from openpi.models.pi0 import action_only_flow_loss
from openpi.policies.libero_policy import TaberoActionOnlyInputs
from openpi.policies.libero_policy import TaberoActionOnlyOutputs
from openpi.training import config
from openpi.training import weight_loaders
from openpi.training.data_loader import EpisodeSafeLeRobotDataset


def test_smoke_config_and_split():
    cfg = config.get_config("pi0_lora_tacfield_local_smoke")
    data = cfg.data.base_config
    assert len(data.episodes) == 26
    assert len(data.validation_episodes) == 3
    assert set(data.episodes).isdisjoint(data.validation_episodes)
    assert set(data.episodes) | set(data.validation_episodes) == set(range(29))
    assert cfg.model.action_dim == 32
    assert cfg.model.supervised_action_dim == 7
    assert cfg.weight_loader.strict
    assert "pi0_base" not in cfg.weight_loader.params_path
    assert cfg.model.tactile_streams == ("tactile_prefix",)
    assert cfg.model.tactile_loss_weight == cfg.model.padding_loss_weight == 0
    assert not cfg.freeze_filter(("PaliGemma", "llm", "lora_a"), nnx.Param(jnp.ones(1)))
    assert cfg.freeze_filter(("tactile_prefix_encoder", "kernel"), nnx.Param(jnp.ones(1)))
    assert cfg.freeze_filter(("PaliGemma", "img", "kernel"), nnx.Param(jnp.ones(1)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"episodes": (1, 1)},
        {"episodes": ()},
        {"episodes": (-1,)},
        {"episodes": (1,), "validation_episodes": (1,)},
        {"validation_episodes": (2,)},
    ],
)
def test_invalid_split(kwargs):
    with pytest.raises(ValueError, match="Episode selectors|selector cannot|Validation requires"):
        config.DataConfig(**kwargs)


def test_action_only_loss_ignores_force_and_padding():
    pred = jnp.ones((2, 3, 32))
    target = jnp.zeros_like(pred).at[..., 7:].set(12345)
    np.testing.assert_allclose(action_only_flow_loss(pred, target, 7), 1)
    grads = jax.grad(lambda p: action_only_flow_loss(p, target, 7).mean())(pred)
    np.testing.assert_array_equal(grads[..., 7:], 0)
    assert np.all(np.asarray(grads[..., :7]) != 0)


def test_tactile_and_absolute_action_round_trip():
    cfg = config.get_config("pi0_lora_tacfield_local_smoke")
    state = np.arange(7, dtype=np.float32) / 100
    state[6] = 0.02
    actions = np.ones((50, 7), dtype=np.float32)
    motion = np.arange(9 * 198 * 2, dtype=np.float32).reshape(9, 198, 2)
    data = {
        "image": np.zeros((32, 32, 3), np.uint8),
        "wrist_image": np.zeros((32, 32, 3), np.uint8),
        "state": state.copy(),
        "actions": actions.copy(),
        "tactile_marker_motion": motion,
        "wrist_wrench": np.full(6, np.nan),
        "tactile_depth": np.full((2, 2, 2), np.nan),
    }
    data = TaberoActionOnlyInputs(cfg.model.model_type)(data)
    np.testing.assert_array_equal(data["tactile_prefix"], motion.reshape(9, 396))
    assert "wrist_wrench" not in data
    assert "tactile_depth" not in data
    assert "tactile_suffix" not in data
    mask = transforms.make_bool_mask(6, -1)
    data = transforms.DeltaActions(mask)(data)
    np.testing.assert_array_equal(data["actions"][:, 6], actions[:, 6])
    data = transforms.AbsoluteActions(mask)(data)
    np.testing.assert_allclose(data["actions"], actions)
    assert TaberoActionOnlyOutputs()({"actions": np.zeros((50, 32))})["actions"].shape == (50, 7)
    with pytest.raises(ValueError, match="non-finite"):
        TaberoActionOnlyOutputs()({"actions": np.full((50, 32), np.nan)})


def test_strict_checkpoint_rejects_missing_lora(monkeypatch):
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda p: p)
    monkeypatch.setattr("openpi.models.model.restore_params", lambda *a, **k: {"base": np.ones(2)})
    loader = weight_loaders.CheckpointWeightLoader("unused", strict=True)
    with pytest.raises(ValueError, match="Strict checkpoint mismatch"):
        loader.load({"base": np.ones(2), "lora_a": np.ones(2)})


@pytest.mark.parametrize("loaded", [{"base": np.ones(3)}, {"base": np.ones(2), "extra": np.ones(2)}])
def test_strict_checkpoint_rejects_extra_and_wrong_shapes(monkeypatch, loaded):
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda p: p)
    monkeypatch.setattr("openpi.models.model.restore_params", lambda *a, **k: loaded)
    with pytest.raises(ValueError, match="Strict checkpoint mismatch"):
        weight_loaders.CheckpointWeightLoader("unused", strict=True).load({"base": np.ones(2)})


def test_strict_checkpoint_preserves_existing_lora(monkeypatch):
    loaded = {"base": np.full(2, 3.0), "lora_a": np.full(2, 7.0)}
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda p: p)
    monkeypatch.setattr("openpi.models.model.restore_params", lambda *a, **k: loaded)
    restored = weight_loaders.CheckpointWeightLoader("unused", strict=True).load(
        {"base": np.zeros(2, np.float32), "lora_a": np.zeros(2, np.float32)}
    )
    np.testing.assert_array_equal(restored["lora_a"], np.full(2, 7.0))
    np.testing.assert_array_equal(restored["base"], np.full(2, 3.0))
    assert restored["lora_a"].dtype == np.float32


def test_existing_config_unchanged():
    cfg = config.get_config("pi0_lora_tacfield_tabero")
    assert cfg.model.supervised_action_dim is None
    assert cfg.data.action_only is False
    assert cfg.eval_interval == 0
    assert not cfg.weight_loader.strict


def test_noncontiguous_episode_chunks_do_not_cross_boundaries():
    dataset = EpisodeSafeLeRobotDataset.__new__(EpisodeSafeLeRobotDataset)
    dataset.episodes = [4, 14, 24]
    dataset.episode_data_index = {"from": torch.tensor([0, 3, 6]), "to": torch.tensor([3, 6, 9])}
    dataset.delta_indices = {"actions": [0, 1, 2]}
    for row, ep, expected in [(2, 4, [2, 2, 2]), (3, 14, [3, 4, 5]), (8, 24, [8, 8, 8])]:
        indices, _ = dataset._get_query_indices(row, ep)  # noqa: SLF001 - regression for upstream query hook
        assert indices["actions"] == expected
