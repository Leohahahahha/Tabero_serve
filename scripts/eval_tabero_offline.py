"""Offline FR3 action evaluation on held-out recordings; never connects to a robot."""

import argparse
import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi0_lora_tacfield_local_tactile_lora_smoke")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Full checkpoint directory, e.g. .../2999")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory; existing directories are refused"
    )
    parser.add_argument("--episodes", type=int, nargs="+", help="Default: all validation episodes; train IDs refused")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--max-frames-per-episode", type=int, default=0, help="0 = all; otherwise evenly sampled anchors"
    )
    parser.add_argument("--num-denoise-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--check-only", action="store_true", help="CPU metadata and boundary-sample checks, no model load"
    )
    args = parser.parse_args(argv)
    if args.stride < 1 or args.max_frames_per_episode < 0 or args.num_denoise_steps < 1 or args.seed < 0:
        parser.error("stride/denoise steps must be positive; seed/frame cap must be nonnegative")
    if args.episodes is not None and (not args.episodes or len(args.episodes) != len(set(args.episodes))):
        parser.error("Episode IDs must be unique")
    return args


def require_local_checkpoint(path):
    path = path.resolve()
    for name in ("params", "assets", "train_state"):
        if not (path / name).is_dir():
            raise ValueError(f"Missing {path / name}; pass the checkpoint root, not its params directory")
    metadata = json.loads((path / "_CHECKPOINT_METADATA").read_text())
    if not metadata.get("commit_timestamp_nsecs"):
        raise ValueError("Checkpoint is not finalized")
    if (path / "model.safetensors").exists():
        raise ValueError("This evaluator expects the trained JAX checkpoint")
    return path


def validation_dataset_config(data_config, episodes):
    """The dataset factory's episodes selector must not retain the training split validator."""
    if not episodes or not set(episodes).issubset(data_config.validation_episodes or ()):
        raise ValueError("Select only configured validation episodes")
    if set(episodes) & set(data_config.episodes or ()):
        raise ValueError("Training/validation overlap")
    return dataclasses.replace(data_config, episodes=tuple(episodes), validation_episodes=None)


def validate_model_contract(config):
    """Accept only the explicit real-FR3 tactile or separately trained RGB+state contract."""
    from openpi.shared.tactile_type import TactileType

    model = config.model
    if (
        not config.policy_metadata
        or config.policy_metadata.get("action_representation") != "absolute_xyz_axis_angle_single_finger_m"
        or model.supervised_action_dim != 7
        or model.action_dim != 32
        or model.action_horizon != 50
    ):
        raise ValueError("Use the real-FR3 absolute 7D action-only config")
    if model.tactile_type is TactileType.NO:
        if (
            model.tactile_streams
            or model.tactile_prefix_lora_rank
            or config.policy_metadata.get("tactile_input") != "none"
        ):
            raise ValueError("Inconsistent no-tactile config")
        return False
    if model.tactile_streams != ("tactile_prefix",) or model.tactile_prefix_lora_rank != 16:
        raise ValueError("Expected action-only tactile LoRA rank-16 architecture")
    return True


def main(argv=None):
    args = parse_args(argv)
    # All actual resources are local. Do not download datasets or model weights implicitly.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    if args.check_only:
        os.environ["JAX_PLATFORMS"] = "cpu"
    checkpoint = require_local_checkpoint(args.checkpoint)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")

    from openpi.policies import tabero_offline as offline
    from openpi.training import config as configs
    from openpi.training import data_loader

    config = configs.get_config(args.config)
    use_tactile = validate_model_contract(config)
    # Even config creation must use checkpoint-local stats, not mutable training assets.
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data, assets=dataclasses.replace(config.data.assets, assets_dir=str(checkpoint / "assets"))
        ),
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    episodes = tuple(sorted(args.episodes if args.episodes is not None else data_config.validation_episodes or ()))
    if not episodes or not set(episodes).issubset(data_config.validation_episodes or ()):
        raise ValueError("Select only the configured held-out validation episodes")
    if set(episodes) & set(data_config.episodes or ()):
        raise ValueError("Training/validation overlap")
    root = Path(data_config.root)
    if output_dir.is_relative_to(root.resolve()) or output_dir.is_relative_to(checkpoint):
        raise ValueError("Output must be outside the dataset and checkpoint directories")
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    if info["codebase_version"] != "v2.1" or info["fps"] != 10:
        raise ValueError("Expected LeRobot v2.1, 10 Hz compact dataset")
    all_lengths = {
        r["episode_index"]: r["length"]
        for r in (json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines())
    }
    lengths = {episode: all_lengths[episode] for episode in episodes}
    stats_path = checkpoint / "assets" / data_config.asset_id / "norm_stats.json"
    required_stats = {"state", "actions"} | ({"tactile_prefix"} if use_tactile else set())
    if data_config.norm_stats is None or not required_stats.issubset(data_config.norm_stats):
        raise ValueError(f"Checkpoint lacks required normalization statistics: {sorted(required_stats)}")
    horizon, action_dim = config.model.action_horizon, config.model.action_dim
    anchors = offline.select_anchors(lengths, args.stride, args.max_frames_per_episode)
    manifest = {
        "status": "initializing",
        "config": args.config,
        "input_modality": "rgb_state_touch" if use_tactile else "rgb_state",
        "config_repr": repr(config),
        "checkpoint": str(checkpoint),
        "dataset_root": str(root),
        "episode_lengths": lengths,
        "selected_anchors": len(anchors),
        "stride": args.stride,
        "max_frames_per_episode": args.max_frames_per_episode,
        "seed": args.seed,
        "num_denoise_steps": args.num_denoise_steps,
        "action_horizon": horizon,
        "internal_action_dim": action_dim,
        "output_action_dim": 7,
        "check_only": args.check_only,
        "norm_stats_path": str(stats_path),
        "norm_stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "dataset_info_sha256": hashlib.sha256(info_path.read_bytes()).hexdigest(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.version,
        "robot_connection": False,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), Path(offline.__file__).resolve(), Path(configs.__file__).resolve())
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "eval.log")],
        force=True,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    try:
        dataset = data_loader.create_torch_dataset(
            validation_dataset_config(data_config, episodes), horizon, config.model
        )
        if args.check_only:
            import numpy as np

            offset = 0
            for episode, length in lengths.items():
                for frame in sorted({0, length - 1}):
                    sample = dataset[offset + frame]
                    if int(sample["episode_index"]) != episode or int(sample["frame_index"]) != frame:
                        raise ValueError("Dataset index mismatch")
                    offline.policy_observation(sample, use_tactile=use_tactile)
                    offline.valid_horizon_mask(frame, length, horizon, sample["actions_is_pad"])
                    target = np.asarray(sample["actions"])
                    if target.shape != (horizon, 7) or not np.isfinite(target).all():
                        raise ValueError("Invalid raw action targets")
                    logging.info(
                        "Checked episode=%d frame=%d, valid targets=%d", episode, frame, min(horizon, length - frame)
                    )
                offset += length
            manifest["status"] = "checks_passed_no_inference"
        else:
            import jax

            from openpi.policies import policy_config

            devices = jax.devices()
            if len(devices) != 1 or devices[0].platform != "gpu":
                raise ValueError("Select exactly one available GPU with CUDA_VISIBLE_DEVICES and JAX_PLATFORMS=cuda")
            manifest["jax_version"] = jax.__version__
            manifest["devices"] = [str(device) for device in devices]
            logging.info(
                "Loading %s; %d anchors, episodes=%s. No robot connection.", checkpoint, len(anchors), episodes
            )
            policy = policy_config.create_trained_policy(
                config, checkpoint, sample_kwargs={"num_steps": args.num_denoise_steps}, strict_params=True
            )
            summary = offline.evaluate_dataset(
                policy,
                dataset,
                lengths,
                output_dir,
                horizon=horizon,
                action_dim=action_dim,
                seed=args.seed,
                stride=args.stride,
                max_frames_per_episode=args.max_frames_per_episode,
                make_plots=not args.no_plots,
                use_tactile=use_tactile,
            )
            manifest["status"] = "complete"
            logging.info("First-action errors: %s", summary["overall"]["first_action"])
            logging.info("Hold-state baseline: %s", summary["overall"]["hold_current_state_baseline"]["first_action"])
        logging.info("Finished: %s; output=%s", manifest["status"], output_dir)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        logging.exception("Evaluation stopped; retain logs and partial outputs")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
