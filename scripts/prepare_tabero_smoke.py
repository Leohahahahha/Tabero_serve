"""Read-only dataset audit and train-only normalization for a local Tabero config.

Does not decode unused depth/wrench columns or modify original metadata/labels.
Run with JAX_PLATFORMS=cpu. Reports are written only to --output-dir.
"""

import argparse
import json
import pathlib
import subprocess

import av
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from openpi import transforms as _transforms
from openpi.shared import normalize
from openpi.training import config as configs


def action_chunks(actions, horizon):
    """Replicate the last target at episode end; never cross episode boundaries."""
    indices = np.minimum(np.arange(len(actions))[:, None] + np.arange(horizon), len(actions) - 1)
    return actions[indices].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi0_lora_tacfield_local_smoke")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    config = configs.get_config(args.config)
    data = config.data.base_config
    root = pathlib.Path(data.root)
    info = json.loads((root / "meta/info.json").read_text())
    conversion = json.loads((root / "meta/tabero_conversion.json").read_text())
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    task_rows = [json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines()]
    task_mapping = {int(row["task_index"]): row["task"] for row in task_rows}
    if len(task_mapping) != len(task_rows):
        raise ValueError("Duplicate task_index values in tasks.jsonl")
    expected_prompt = (config.policy_metadata or {}).get("task_prompt")
    if expected_prompt is not None and set(task_mapping.values()) != {expected_prompt}:
        raise ValueError(f"Task prompt does not match config policy metadata: {task_mapping}")
    train_ids, val_ids = set(data.episodes), set(data.validation_episodes)
    if train_ids & val_ids or train_ids | val_ids != {e["episode_index"] for e in episodes}:
        raise ValueError("Split must be disjoint and cover every episode")
    stats = {key: normalize.RunningStats() for key in ("state", "actions", "tactile_prefix")}
    reports = []
    for episode in episodes:
        index, length = episode["episode_index"], episode["length"]
        chunk = index // info["chunks_size"]
        path = root / info["data_path"].format(episode_chunk=chunk, episode_index=index)
        table = pq.read_table(
            path,
            columns=[
                "state",
                "actions",
                "tactile_marker_motion",
                "frame_index",
                "episode_index",
                "task_index",
            ],
        )
        arrays = {key: np.asarray(table[key].to_pylist()) for key in table.column_names}
        state, actions, tactile = (arrays[key] for key in ("state", "actions", "tactile_marker_motion"))
        if state.shape != (length, 7) or actions.shape != (length, 7) or tactile.shape != (length, 9, 198, 2):
            raise ValueError(f"Episode {index}: invalid tensor dimensions")
        if not all(np.isfinite(x).all() for x in (state, actions, tactile)):
            raise ValueError(f"Episode {index}: non-finite data")
        if not np.array_equal(arrays["frame_index"].reshape(-1), np.arange(length)):
            raise ValueError(f"Episode {index}: frame index mismatch")
        if not np.all(arrays["episode_index"] == index):
            raise ValueError(f"Episode {index}: episode index mismatch")
        episode_task_indices = {int(x) for x in arrays["task_index"].reshape(-1)}
        if not episode_task_indices or not episode_task_indices <= set(task_mapping):
            raise ValueError(f"Episode {index}: task indices not found in tasks.jsonl: {episode_task_indices}")
        if (
            min(state[:, 6].min(), actions[:, 6].min()) < -1e-6
            or max(state[:, 6].max(), actions[:, 6].max()) > 0.042501
        ):
            raise ValueError(f"Episode {index}: gripper outside single-finger meter range")
        video_info = {}
        for key in ("image", "wrist_image"):
            video = root / info["video_path"].format(episode_chunk=chunk, episode_index=index, video_key=key)
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                # Decode every frame to check corruption/count, not only container headers.
                decoded = sum(1 for _ in container.decode(video=0))
                if decoded != length:
                    raise ValueError(f"Episode {index}: {key} has {decoded}, expected {length} frames")
                video_info[key] = {"frames": decoded, "height": stream.height, "width": stream.width}
        jumps = np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=-1)
        rot_jump = np.linalg.norm(np.diff(actions[:, 3:6], axis=0), axis=-1)
        rotation = Rotation.from_rotvec(actions[:, 3:6])
        physical_jump = (rotation[:-1].inv() * rotation[1:]).magnitude()
        next_state_error = float(np.max(np.abs(actions[:-1] - state[1:]))) if length > 1 else 0.0
        relative_rotation = (
            Rotation.from_rotvec(state[:, 3:6]).inv() * Rotation.from_rotvec(actions[:, 3:6])
        ).magnitude()
        relative_translation = np.linalg.norm(actions[:, :3] - state[:, :3], axis=-1)
        if (config.policy_metadata or {}).get(
            "action_label_source"
        ) == "next_observation_state_within_episode" and next_state_error > 1e-7:
            raise ValueError(f"Episode {index}: action is not the next observation state ({next_state_error=})")
        # Index 0 is a fixed reference grid. Indices 1..8 are the rolling
        # coordinate history, so only those history slots shift in time.
        reference_error = float(np.max(np.abs(tactile[:, 0] - tactile[0, 0])))
        history_error = float(np.max(np.abs(tactile[1:, 1:-1] - tactile[:-1, 2:]))) if length > 1 else 0.0
        report = {
            "episode": index,
            "length": length,
            "split": "train" if index in train_ids else "validation",
            "action_position_jump_rows_gt_5cm": (np.flatnonzero(jumps > 0.05) + 1).tolist(),
            "max_action_position_jump_m": float(jumps.max(initial=0)),
            "axis_angle_jumps_gt_1rad": int(np.sum(rot_jump > 1)),
            "branch_like_jumps": int(np.sum((rot_jump > 1) & (physical_jump < 0.1))),
            "next_state_action_max_abs_error": next_state_error,
            "so3_relative_rotation_max_deg": float(np.rad2deg(relative_rotation.max(initial=0))),
            "translation_steps_gt_2mm": int(np.sum(relative_translation > 0.002)),
            "translation_step_count": len(relative_translation),
            "compacted_timing": index in set(conversion.get("compacted_source_episode_indices", [])),
            "task_indices": sorted(episode_task_indices),
            "reference_grid_static_error": reference_error,
            "max_history_shift_error": history_error,
            "initial_history_repeat_error": float(np.max(np.abs(tactile[0, 1:] - tactile[0, -1:]))),
            "videos": video_info,
        }
        reports.append(report)
        if index in train_ids:
            targets = action_chunks(actions, config.model.action_horizon)
            if config.data.extra_delta_transform:
                if getattr(config.data, "use_so3_relative_actions", False):
                    targets = _transforms.RelativePoseActions()({"state": state, "actions": targets})["actions"]
                else:
                    targets[..., :6] -= state[:, None, :6]
            # Float64 moments avoid cancellation from large marker coordinates with small motion.
            stats["state"].update(state.astype(np.float64))
            stats["actions"].update(targets.astype(np.float64))
            stats["tactile_prefix"].update(tactile.astype(np.float64).reshape(length, 9, -1))
        print(json.dumps({k: v for k, v in report.items() if k != "videos"}), flush=True)
    norm_stats = {key: accumulator.get_statistics() for key, accumulator in stats.items()}
    asset_id = config.data.assets.asset_id or config.data.repo_id
    asset_dir = config.assets_dirs / asset_id
    normalize.save(asset_dir, norm_stats)
    summary = {
        "config": args.config,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dataset_root": str(root),
        "train_episodes": sorted(train_ids),
        "validation_episodes": sorted(val_ids),
        "train_frames": sum(r["length"] for r in reports if r["split"] == "train"),
        "validation_frames": sum(r["length"] for r in reports if r["split"] == "validation"),
        "action_horizon": config.model.action_horizon,
        "extra_delta_transform": config.data.extra_delta_transform,
        "use_so3_relative_actions": getattr(config.data, "use_so3_relative_actions", False),
        "timing_policy": conversion.get("timing_policy"),
        "compacted_episode_indices": conversion.get("compacted_source_episode_indices", []),
        "total_missing_candidate_steps": conversion.get("total_missing_candidate_steps", 0),
        "translation_steps_gt_2mm": sum(r["translation_steps_gt_2mm"] for r in reports),
        "translation_step_count": sum(r["translation_step_count"] for r in reports),
        "authoritative_task_mapping": task_mapping,
        "episode_task_metadata_mismatches": [
            episode["episode_index"]
            for episode in episodes
            if set(episode.get("tasks", [])) != {task_mapping[index] for index in task_mapping}
        ],
        "norm_stats_path": str(asset_dir / "norm_stats.json"),
        "episodes": reports,
        "norm_summary": {
            k: {"dimensions": len(v.mean), "min_std": float(v.std.min()), "max_std": float(v.std.max())}
            for k, v in norm_stats.items()
        },
        "limitations": [
            "Labels were not cropped or corrected",
            "Compacted timing cannot reconstruct the missing sensor samples or their exact transition locations",
            "The real-robot 0.02 m/s guard is intentionally not used to clip expert training labels",
            "Rolling nine-frame tactile history; no reference subtraction",
            "Training prompts come from tasks.jsonl via task_index; episodes.jsonl task strings are audit metadata",
            "Not robot success evaluation",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_audit.json").write_text(json.dumps(summary, indent=2))
    (asset_dir / "split_provenance.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2)
    )
    print(f"Wrote audit to {args.output_dir}, train-only statistics to {asset_dir}", flush=True)


if __name__ == "__main__":
    main()
