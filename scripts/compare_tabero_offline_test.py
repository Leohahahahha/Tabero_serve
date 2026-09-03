import hashlib
import json

import numpy as np
import pytest

from scripts.compare_tabero_offline import compare_runs


def make_runs(tmp_path, fault=None):
    dirs = []
    for use_tactile in (True, False):
        directory = tmp_path / ("touch" if use_tactile else "no_touch")
        directory.mkdir()
        dirs.append(directory)
        stats = {key: {"mean": [0.0], "std": [1.0]} for key in ("state", "actions")}
        if use_tactile:
            stats["tactile_prefix"] = {"mean": [42.0], "std": [1.0]}
        elif fault == "stats":
            stats["actions"]["mean"] = [1.0]
        stats_path = directory / "norm_stats.json"
        stats_path.write_text(json.dumps({"norm_stats": stats}))
        manifest = {
            "status": "complete",
            "dataset_root": "/recorded",
            "dataset_info_sha256": "same",
            "episode_lengths": {"4": 2},
            "selected_anchors": 2,
            "stride": 1,
            "max_frames_per_episode": 0,
            "seed": 42,
            "num_denoise_steps": 10,
            "action_horizon": 2,
            "output_action_dim": 7,
            "internal_action_dim": 32,
            "norm_stats_path": str(stats_path),
            "norm_stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        }
        # Old touch run has no modality field; new RGB+state baseline must have it.
        if not use_tactile:
            manifest["input_modality"] = "rgb_state"
            if fault == "seed":
                manifest["seed"] = 1
            if fault == "incomplete":
                manifest["status"] = "failed"
        (directory / "manifest.json").write_text(json.dumps(manifest))
        target = np.zeros((2, 2, 7))
        pred = target.copy()
        pred[..., 0] = 0.001 if use_tactile else 0.003
        pred[1, 1, 0] = 99  # Must be masked out of chunk metrics.
        if not use_tactile and fault == "target":
            target[0, 0, 0] = 1
        np.savez(
            directory / "predictions.npz",
            episode=[4, 4],
            frame=[0, 1],
            prediction=pred,
            target=target,
            state=np.zeros((2, 7)),
            valid=np.array([[True, True], [True, False]]),
        )
    return dirs


def test_paired_comparison_sign_units_and_padding(tmp_path):
    result = compare_runs(*make_runs(tmp_path))
    assert result["anchors"] == 2
    assert result["valid_chunk_targets"] == 3
    assert result["matched_training_verified"] is False
    for scope in ("first_action", "valid_chunk"):
        metric = result["overall"][scope]["position_mm"]
        assert metric["touch"]["mean"] == pytest.approx(1)
        assert metric["no_touch"]["mean"] == pytest.approx(3)
        assert metric["no_touch_minus_touch"]["mean"] == pytest.approx(2)
        assert metric["prediction_difference"]["mean"] == pytest.approx(2)
    assert "4" in result["episodes"]


@pytest.mark.parametrize("fault", ["seed", "target", "stats", "incomplete"])
def test_unpaired_runs_rejected(tmp_path, fault):
    with pytest.raises(ValueError, match="Unpaired|normalization differs|Incomplete"):
        compare_runs(*make_runs(tmp_path, fault))


def test_mutated_norm_assets_rejected(tmp_path):
    dirs = make_runs(tmp_path)
    (dirs[1] / "norm_stats.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        compare_runs(*dirs)
