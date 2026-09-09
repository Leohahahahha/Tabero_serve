"""Server portability tests with real configs/stats; model construction is mocked."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts import serve_tabero


@pytest.mark.parametrize(
    ("name", "use_tactile", "asset_id"),
    [
        ("pi0_lora_tacfield_local_tactile_lora_smoke", True, "local/tabero_lerobot_compact_v1"),
        ("pi0_lora_tabero_v3_touch_20k", True, "local/tabero_lerobot_compact_v3"),
        ("pi0_lora_tabero_rgb_state", False, "local/tabero_lerobot_compact_v1"),
    ],
)
def test_server_uses_moved_checkpoint_assets_and_strict_restore(monkeypatch, tmp_path, name, use_tactile, asset_id):
    from openpi import transforms
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.training import config as configs

    checkpoint = tmp_path / "moved" / "2999"
    (checkpoint / "params").mkdir(parents=True)
    stats = {
        "state": transforms.NormStats(mean=np.zeros(7), std=np.ones(7)),
        "actions": transforms.NormStats(mean=np.zeros(7), std=np.ones(7)),
        "tactile_prefix": transforms.NormStats(mean=np.zeros(396), std=np.ones(396)),
    }
    normalize.save(checkpoint / "assets" / asset_id, stats)
    conversion = Path(__file__).parents[1] / "examples/fr3_deploy/tabero_conversion.json"
    # Avoid tokenizer downloads and model/GPU initialization; actual data config and norm loading run.
    monkeypatch.setattr(configs.ModelTransformFactory, "__call__", lambda *_: transforms.Group())
    calls = {}

    def create(config, directory, **kwargs):
        calls.update(config=config, directory=directory, kwargs=kwargs)
        return "mock_policy"

    monkeypatch.setattr(policy_config, "create_trained_policy", create)
    result, metadata = serve_tabero.load_policy(name, checkpoint, conversion, 10)
    assert result == "mock_policy"
    assert calls["directory"] == checkpoint
    assert calls["config"].data.assets.assets_dir == str(checkpoint / "assets")
    assert calls["kwargs"] == {"sample_kwargs": {"num_steps": 10}, "strict_params": True}
    assert metadata["use_tactile"] is use_tactile
    assert metadata["asset_id"] == asset_id
    assert metadata["action_horizon"] == 50
    assert metadata["conversion_sha256"] == hashlib.sha256(conversion.read_bytes()).hexdigest()
    if use_tactile:
        assert metadata["tactile_marker_shape"] == [9, 198, 2]
        assert metadata["tactile_marker_dtype"] == "float32"
        assert metadata["tactile_marker_layout"] == "reference_then_8_history_frames_left_then_right"


def test_wrong_simulation_config_rejected_before_loading(monkeypatch, tmp_path):
    (tmp_path / "params").mkdir()
    with pytest.raises(ValueError, match="real-FR3"):
        serve_tabero.load_policy("pi0_lora_tacfield_tabero", tmp_path, tmp_path / "not-read.json", 10)


def test_server_reports_config_checkpoint_asset_mismatch(monkeypatch, tmp_path):
    from openpi import transforms
    from openpi.shared import normalize
    from openpi.training import config as configs

    checkpoint = tmp_path / "20000"
    (checkpoint / "params").mkdir(parents=True)
    stats = {
        "state": transforms.NormStats(mean=np.zeros(7), std=np.ones(7)),
        "actions": transforms.NormStats(mean=np.zeros(7), std=np.ones(7)),
        "tactile_prefix": transforms.NormStats(mean=np.zeros(396), std=np.ones(396)),
    }
    normalize.save(checkpoint / "assets/local/tabero_lerobot_compact_v3", stats)
    monkeypatch.setattr(configs.ModelTransformFactory, "__call__", lambda *_: transforms.Group())

    with pytest.raises(ValueError, match="compact_v1.*compact_v3.*exact training config"):
        serve_tabero.load_policy(
            "pi0_lora_tacfield_local_tactile_lora_smoke",
            checkpoint,
            Path(__file__).parents[1] / "examples/fr3_deploy/tabero_conversion.json",
            10,
        )
