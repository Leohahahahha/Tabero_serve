#!/usr/bin/env python3
"""Serve an absolute-action FR3 policy using checkpoint-local normalization assets."""

import argparse
import dataclasses
import hashlib
import json
import logging
from pathlib import Path


def validate_tactile_contract(policy_metadata, conversion, use_tactile):
    if not use_tactile:
        return
    expected_tactile = {
        "tactile_input": "rolling_9x198x2_marker_coordinates_left_then_right",
        "tactile_marker_shape": [9, 198, 2],
        "tactile_marker_dtype": "float32",
        "tactile_marker_layout": "reference_then_8_history_frames_left_then_right",
    }
    for key, value in expected_tactile.items():
        if policy_metadata.get(key) != value:
            raise ValueError(f"Tactile policy metadata mismatch: {key}")
    marker = conversion.get("marker_field", {})
    if (
        marker.get("shape") != [9, 198, 2]
        or marker.get("history_length") != 8
        or marker.get("side_order") != ["left", "right"]
    ):
        raise ValueError("Conversion does not provide the required left-then-right [9,198,2] markers")


def load_policy(config_name, checkpoint, conversion_path, denoise_steps):
    from openpi.policies import policy_config
    from openpi.training import config as configs

    checkpoint = checkpoint.resolve(strict=True)
    if not (checkpoint / "params").is_dir():
        raise ValueError("Pass the complete JAX step directory containing params/ and assets/, not params/ itself")
    config = configs.get_config(config_name)
    if (
        getattr(config.model, "supervised_action_dim", None) != 7
        or not getattr(config.data, "action_only", False)
        or config.policy_metadata.get("action_representation") != "absolute_xyz_axis_angle_single_finger_m"
    ):
        raise ValueError("Select a real-FR3 action-only training config matching this checkpoint")
    use_tactile = "tactile_prefix" in config.model.tactile_streams
    # Avoid original-host paths even during DataConfig.create(). No dataset is opened.
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data, assets=dataclasses.replace(config.data.assets, assets_dir=str(checkpoint / "assets"))
        ),
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    required_stats = {"state", "actions"} | ({"tactile_prefix"} if use_tactile else set())
    if data_config.norm_stats is None or not required_stats.issubset(data_config.norm_stats):
        raise ValueError(f"Checkpoint assets must contain {sorted(required_stats)} normalization statistics")
    stats_path = checkpoint / "assets" / data_config.asset_id / "norm_stats.json"
    conversion = json.loads(conversion_path.read_bytes())
    if conversion["output_contract"] != "tabero_action_only_lerobot_v2.1":
        raise ValueError("Unexpected training conversion metadata")
    validate_tactile_contract(config.policy_metadata, conversion, use_tactile)
    metadata = {
        **config.policy_metadata,
        "deployment_protocol": "tabero_fr3_absolute_v1",
        "use_tactile": use_tactile,
        "action_horizon": config.model.action_horizon,
        "config": config_name,
        "checkpoint": str(checkpoint),
        "norm_stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "conversion_sha256": hashlib.sha256(conversion_path.read_bytes()).hexdigest(),
    }
    policy = policy_config.create_trained_policy(
        config, checkpoint, sample_kwargs={"num_steps": denoise_steps}, strict_params=True
    )
    return policy, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi0_lora_tacfield_local_tactile_lora_smoke")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--conversion", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-denoise-steps", type=int, default=10)
    args = parser.parse_args()
    if args.num_denoise_steps <= 0:
        parser.error("--num-denoise-steps must be positive")
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    policy, metadata = load_policy(args.config, args.checkpoint, args.conversion, args.num_denoise_steps)
    logging.info("Deployment metadata: %s", json.dumps(metadata, ensure_ascii=False))
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
