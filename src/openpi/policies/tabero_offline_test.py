import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.policies import tabero_offline as offline


def make_sample(episode=4, frame=0, length=2, horizon=3):
    state = np.array([0.4, 0.1, 0.3, 0.0, 0.0, 0.0, 0.02], dtype=np.float32)
    return {
        "image": np.zeros((3, 8, 8), np.float32),
        "wrist_image": np.zeros((3, 8, 8), np.float32),
        "state": state,
        "tactile_marker_motion": np.zeros((9, 198, 2), np.float32),
        "actions": np.tile(state, (horizon, 1)),
        "episode_index": episode,
        "frame_index": frame,
        "actions_is_pad": frame + np.arange(horizon) >= length,
        "task": "move recorded object",
        "wrist_wrench": np.ones(6),
        "tactile_depth": np.ones(1),
    }


class HoldPolicy:
    def infer(self, observation, *, noise):
        assert set(observation) == {"image", "wrist_image", "state", "tactile_marker_motion", "prompt"}
        return {"actions": np.tile(observation["state"], (len(noise), 1))}


def test_metrics_have_physical_units():
    target = np.zeros((2, 7))
    prediction = target.copy()
    prediction[:, :2] = [0.003, 0.004]
    prediction[:, 5] = np.pi / 2
    prediction[:, 6] = 0.002
    errors = offline.action_errors(prediction, target)
    np.testing.assert_allclose(errors["position_mm"], 5)
    np.testing.assert_allclose(errors["rotation_deg"], 90)
    np.testing.assert_allclose(errors["gripper_mm"], 2)


def test_rotation_branch_equivalence():
    errors = offline.rotation_error_deg([[np.pi, 0, 0], [0, 0, 0]], [[-np.pi, 0, 0], [0, 0, 2 * np.pi]])
    np.testing.assert_allclose(errors, 0, atol=1e-10)
    error = offline.rotation_error_deg([0, 0, np.pi - 0.01], [0, 0, -np.pi + 0.01])
    assert float(error) == pytest.approx(np.rad2deg(0.02))


def test_mask_and_noncontiguous_episode_selection():
    assert offline.select_anchors({4: 3, 14: 2}, stride=2) == [(0, 4, 0), (2, 4, 2), (3, 14, 0)]
    assert offline.select_anchors({4: 5}, max_frames_per_episode=2) == [(0, 4, 0), (4, 4, 4)]
    np.testing.assert_array_equal(offline.valid_horizon_mask(2, 3, 3, [False, True, True]), [True, False, False])
    with pytest.raises(ValueError, match="boundary"):
        offline.valid_horizon_mask(2, 3, 3, [False, False, False])


def test_observation_does_not_leak_or_alias_labels():
    sample = make_sample()
    observation = offline.policy_observation(sample)
    assert "actions" not in observation
    assert "wrist_wrench" not in observation
    assert "tactile_depth" not in observation
    observation["state"][:] = 99
    observation["image"][:] = 1
    observation["tactile_marker_motion"][:] = 1
    assert sample["state"][0] != 99
    assert not sample["image"].any()
    assert not sample["tactile_marker_motion"].any()


def test_noise_is_paired_by_anchor_not_evaluation_order():
    a = offline.sampling_noise(42, 14, 12, 50, 32)
    offline.sampling_noise(42, 4, 0, 50, 32)
    np.testing.assert_array_equal(a, offline.sampling_noise(42, 14, 12, 50, 32))
    assert not np.array_equal(a, offline.sampling_noise(43, 14, 12, 50, 32))


def test_evaluator_excludes_padded_targets_and_exports(tmp_path):
    samples = [make_sample(frame=0), make_sample(frame=1)]
    for sample in samples:
        sample["actions"][sample["actions_is_pad"], 0] = 999
    summary = offline.evaluate_dataset(HoldPolicy(), samples, {4: 2}, tmp_path, horizon=3, make_plots=False)
    assert summary["status"] == "complete"
    result = summary["overall"]
    assert result["anchors"] == 2
    assert result["valid_chunk_targets"] == 3
    assert result["excluded_padded_targets"] == 3
    assert result["valid_chunk"]["position_mm"]["max"] == 0
    assert result["hold_current_state_baseline"]["first_action"]["position_mm"]["mean"] == 0
    assert result["by_horizon"][2]["position_mm"]["count"] == 0
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "complete"
    with (tmp_path / "predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert sum(row["valid"] == "1" for row in rows) == 3
    assert all(row["position_mm"] == "" and row["target_frame"] == "" for row in rows if row["valid"] == "0")
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as arrays:
        assert arrays["prediction"].shape == (2, 3, 7)
        assert arrays["valid"].sum() == 3


def test_headless_plot(tmp_path):
    samples = [make_sample(frame=0), make_sample(frame=1)]
    offline.evaluate_dataset(HoldPolicy(), samples, {4: 2}, tmp_path, horizon=3, make_plots=True)
    assert (tmp_path / "episode_000004.png").read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_invalid_output_stops_and_preserves_failure_report(tmp_path, value):
    class InvalidPolicy:
        def infer(self, observation, *, noise):
            return {"actions": np.full((3, 7), value)}

    with pytest.raises(ValueError, match="non-finite"):
        offline.evaluate_dataset(InvalidPolicy(), [make_sample(length=1)], {4: 1}, tmp_path, horizon=3)
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["frame"] == 0
    assert failure["completed_anchors"] == 0
    assert not (tmp_path / "summary.json").exists()


def test_gripper_violations_reported_without_clipping():
    sample = make_sample()
    record = offline.infer_anchor(HoldPolicy(), sample, episode=4, frame=0, length=2, horizon=3, action_dim=32, seed=42)
    record["prediction"][0, 6] = -0.01
    record["prediction"][1, 6] = 0.1
    report = offline.summarize_records([record])
    assert report["diagnostics"]["gripper_outside_0_to_0_0425_m_count"] == 2
    assert record["prediction"][0, 6] == -0.01


@pytest.mark.parametrize(("field", "value"), [("episode_index", 14), ("frame_index", 1)])
def test_index_mismatch_rejected(field, value):
    sample = make_sample()
    sample[field] = value
    with pytest.raises(ValueError, match="ordering"):
        offline.infer_anchor(HoldPolicy(), sample, episode=4, frame=0, length=2, horizon=3, action_dim=32, seed=42)


def test_normalized_images_rejected():
    sample = make_sample()
    sample["image"][:] = -1
    with pytest.raises(ValueError, match="normalized"):
        offline.policy_observation(sample)


def test_cli_help_and_invalid_parameters_do_not_load_model():
    from scripts.eval_tabero_offline import parse_args

    with pytest.raises(SystemExit) as result:
        parse_args(["--help"])
    assert result.value.code == 0
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint=/tmp/checkpoint", "--output-dir=/tmp/output", "--stride=0"])


def test_incomplete_checkpoint_rejected(tmp_path):
    from scripts.eval_tabero_offline import require_local_checkpoint

    for name in ("params", "assets", "train_state"):
        (tmp_path / name).mkdir()
    (tmp_path / "_CHECKPOINT_METADATA").write_text("{}")
    with pytest.raises(ValueError, match="finalized"):
        require_local_checkpoint(tmp_path)


def test_validation_dataset_selector_preserves_training_config():
    from openpi.training.config import DataConfig
    from scripts.eval_tabero_offline import validation_dataset_config

    original = DataConfig(episodes=(0, 1), validation_episodes=(4, 14, 24))
    selected = validation_dataset_config(original, (4, 14))
    assert selected.episodes == (4, 14)
    assert selected.validation_episodes is None
    assert original.episodes == (0, 1)
    with pytest.raises(ValueError, match="validation"):
        validation_dataset_config(original, (0,))


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("use_tactile", [False, True])
def test_factory_uses_checkpoint_stats_strict_restore_and_absolute_outputs(monkeypatch, tmp_path, strict, use_tactile):
    from openpi import transforms
    from openpi.models import model
    from openpi.policies import libero_policy
    from openpi.policies import policy
    from openpi.policies import policy_config
    from openpi.training import checkpoints

    mask = transforms.make_bool_mask(6, -1)
    data_config = SimpleNamespace(
        asset_id="local/test",
        use_quantile_norm=False,
        model_transforms=transforms.Group(),
        data_transforms=transforms.Group(
            inputs=[
                (libero_policy.TaberoActionOnlyInputs if use_tactile else libero_policy.TaberoNoTactActionOnlyInputs)(
                    model_type=model.ModelType.PI0
                ),
                transforms.DeltaActions(mask),
            ],
            outputs=[transforms.AbsoluteActions(mask), libero_policy.TaberoActionOnlyOutputs()],
        ),
    )
    calls = {}

    def load(params, *, remove_extra_params):
        calls["remove_extra_params"] = remove_extra_params
        return "fake_model"

    config = SimpleNamespace(
        model=SimpleNamespace(load=load),
        assets_dirs=tmp_path,
        data=SimpleNamespace(create=lambda *_: data_config),
        policy_metadata={},
    )
    stats = {
        "state": transforms.NormStats(mean=np.ones(7) * 0.1, std=np.ones(7) * 0.5),
        "actions": transforms.NormStats(mean=np.ones(7) * 0.01, std=np.ones(7) * 0.2),
        "tactile_prefix": transforms.NormStats(mean=np.zeros(396), std=np.ones(396)),
    }

    def load_stats(directory, asset_id):
        assert directory == tmp_path / "assets"
        assert asset_id == "local/test"
        return stats

    monkeypatch.setattr(policy_config.download, "maybe_download", lambda _: tmp_path)
    monkeypatch.setattr(model, "restore_params", lambda *_, **__: {})
    monkeypatch.setattr(checkpoints, "load_norm_stats", load_stats)
    monkeypatch.setattr(policy, "Policy", lambda *_, **kwargs: kwargs)
    constructed = policy_config.create_trained_policy(config, tmp_path, strict_params=strict)
    assert calls["remove_extra_params"] is not strict
    sample = make_sample()
    inputs = transforms.compose(constructed["transforms"])(offline.policy_observation(sample, use_tactile=use_tactile))
    assert ("tactile_prefix" in inputs) is use_tactile
    target = sample["actions"].copy()
    target[:, :3] += 0.01
    delta = target.copy()
    delta[:, :6] -= sample["state"][:6]
    normalized = transforms.Normalize({"actions": stats["actions"]})({"actions": delta})["actions"]
    outputs = transforms.compose(constructed["output_transforms"])({"state": inputs["state"], "actions": normalized})
    np.testing.assert_allclose(outputs["actions"], target, atol=1e-6)
