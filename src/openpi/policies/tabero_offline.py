"""Recorded-observation action evaluation. No robot, simulator, or model loading here."""

import csv
import json
import logging
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_error_deg(prediction, target):
    """SO(3) shortest angle, invariant to axis-angle branch changes."""
    prediction, target = np.broadcast_arrays(
        np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    )
    shape = prediction.shape[:-1]
    # Some SciPy versions require writable buffers; broadcast_to returns read-only views.
    relative = Rotation.from_rotvec(prediction.reshape(-1, 3).copy()).inv() * Rotation.from_rotvec(
        target.reshape(-1, 3).copy()
    )
    return np.rad2deg(relative.magnitude()).reshape(shape)


def action_errors(prediction, target):
    prediction, target = np.broadcast_arrays(
        np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    )
    if prediction.shape[-1] != 7 or not np.isfinite((prediction, target)).all():
        raise ValueError("Metrics require finite absolute 7D actions")
    return {
        "position_mm": 1000 * np.linalg.norm(prediction[..., :3] - target[..., :3], axis=-1),
        "rotation_deg": rotation_error_deg(prediction[..., 3:6], target[..., 3:6]),
        "gripper_mm": 1000 * np.abs(prediction[..., 6] - target[..., 6]),
    }


def distribution(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    if not values.size:
        return {"count": 0}
    if not np.isfinite(values).all():
        raise ValueError("Non-finite metric values")
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "p50": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def select_anchors(episode_lengths, stride=1, max_frames_per_episode=0):
    if stride < 1 or max_frames_per_episode < 0:
        raise ValueError("stride must be positive and max_frames_per_episode nonnegative")
    offset = 0
    anchors = []
    for episode, length in episode_lengths.items():
        if length < 1:
            raise ValueError(f"Empty episode: {episode}")
        frames = np.arange(0, length, stride, dtype=int)
        if max_frames_per_episode and len(frames) > max_frames_per_episode:
            frames = frames[np.linspace(0, len(frames) - 1, max_frames_per_episode, dtype=int)]
        anchors.extend((offset + int(frame), episode, int(frame)) for frame in frames)
        offset += length
    return anchors


def valid_horizon_mask(frame, length, horizon, padding):
    if not 0 <= frame < length or horizon < 1:
        raise ValueError("Invalid episode frame/horizon")
    valid = frame + np.arange(horizon) < length
    padding = np.asarray(padding)
    if padding.dtype != np.bool_ or padding.shape != (horizon,) or not np.array_equal(~padding, valid):
        raise ValueError("Action padding disagrees with episode boundary; refusing misaligned evaluation")
    return valid


def policy_observation(sample, *, use_tactile=True):
    """Allowlist inputs and copy arrays: labels/metadata cannot leak into the policy."""
    keys = ("image", "wrist_image", "state") + (("tactile_marker_motion",) if use_tactile else ())
    observation = {key: np.asarray(sample[key]).copy() for key in keys}
    if observation["state"].shape != (7,):
        raise ValueError("Expected raw state [7]")
    if use_tactile and observation["tactile_marker_motion"].shape != (9, 198, 2):
        raise ValueError("Expected tactile_marker_motion [9,198,2]")
    if not all(np.isfinite(value).all() for value in observation.values()):
        raise ValueError("Non-finite recorded observation")
    for key in ("image", "wrist_image"):
        image = observation[key]
        if image.ndim != 3 or (image.shape[0] != 3 and image.shape[-1] != 3):
            raise ValueError(f"Expected RGB image, got {key}: {image.shape}")
        if np.issubdtype(image.dtype, np.floating) and (image.min() < 0 or image.max() > 1):
            raise ValueError("Raw floating images must be in [0,1]; do not pass normalized training batches")
    prompt = sample.get("prompt", sample.get("task"))
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Recorded task prompt is required")
    observation["prompt"] = prompt
    return observation


def sampling_noise(seed, episode, frame, horizon, action_dim):
    """Fixed per-anchor noise allows paired comparisons regardless of stride."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, episode, frame]))
    return rng.standard_normal((horizon, action_dim)).astype(np.float32)


def infer_anchor(policy, sample, *, episode, frame, length, horizon, action_dim, seed, use_tactile=True):
    if int(np.asarray(sample["episode_index"]).item()) != episode:
        raise ValueError("Dataset episode ordering mismatch")
    if int(np.asarray(sample["frame_index"]).item()) != frame:
        raise ValueError("Dataset frame ordering mismatch")
    target = np.asarray(sample["actions"], dtype=np.float64).copy()
    if target.shape != (horizon, 7) or not np.isfinite(target).all():
        raise ValueError("Expected finite raw absolute targets [horizon,7]")
    valid = valid_horizon_mask(frame, length, horizon, sample["actions_is_pad"])
    observation = policy_observation(sample, use_tactile=use_tactile)
    state = observation["state"].astype(np.float64).copy()
    noise = sampling_noise(seed, episode, frame, horizon, action_dim)
    start = time.perf_counter()
    prediction = np.asarray(policy.infer(observation, noise=noise)["actions"], dtype=np.float64)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if prediction.shape != (horizon, 7) or not np.isfinite(prediction).all():
        raise ValueError("Policy returned wrong-shaped or non-finite actions; no clipping/replacement applied")
    return {
        "episode": episode,
        "frame": frame,
        "prediction": prediction.copy(),
        "target": target,
        "valid": valid,
        "state": state,
        "policy_call_wall_ms": elapsed_ms,
    }


def summarize_records(records):
    if not records:
        raise ValueError("No evaluated anchors")
    pred = np.stack([r["prediction"] for r in records])
    target = np.stack([r["target"] for r in records])
    valid = np.stack([r["valid"] for r in records])
    state = np.stack([r["state"] for r in records])
    metrics = action_errors(pred, target)
    baseline = action_errors(np.broadcast_to(state[:, None], pred.shape), target)
    result = {
        "anchors": len(records),
        "valid_chunk_targets": int(valid.sum()),
        "excluded_padded_targets": int((~valid).sum()),
        "first_action": {key: distribution(value[:, 0]) for key, value in metrics.items()},
        "valid_chunk": {key: distribution(value[valid]) for key, value in metrics.items()},
        "hold_current_state_baseline": {
            "first_action": {key: distribution(value[:, 0]) for key, value in baseline.items()},
            "valid_chunk": {key: distribution(value[valid]) for key, value in baseline.items()},
        },
        "by_horizon": [
            {
                "offset": offset,
                **{key: distribution(value[:, offset][valid[:, offset]]) for key, value in metrics.items()},
            }
            for offset in range(pred.shape[1])
        ],
    }
    # Diagnostic thresholds, NOT a robot safety certificate. Do not clip predictions.
    gripper = pred[..., 6][valid]
    adjacency = valid[:, 1:] & valid[:, :-1]
    pred_jumps = 1000 * np.linalg.norm(np.diff(pred[..., :3], axis=1), axis=-1)[adjacency]
    target_jumps = 1000 * np.linalg.norm(np.diff(target[..., :3], axis=1), axis=-1)[adjacency]
    first_move = 1000 * np.linalg.norm(pred[:, 0, :3] - state[:, :3], axis=-1)
    result["diagnostics"] = {
        "nonfinite_output_values": 0,
        "gripper_outside_0_to_0_0425_m_count": int(((gripper < 0) | (gripper > 0.0425)).sum()),
        "gripper_min_m": float(gripper.min()),
        "gripper_max_m": float(gripper.max()),
        "prediction_xyz_min_m": pred[..., :3][valid].min(axis=0).tolist(),
        "prediction_xyz_max_m": pred[..., :3][valid].max(axis=0).tolist(),
        "first_target_distance_from_state_mm": distribution(first_move),
        "first_target_distance_over_50mm_count": int((first_move > 50).sum()),
        "predicted_within_chunk_position_step_mm": distribution(pred_jumps),
        "predicted_within_chunk_step_over_50mm_count": int((pred_jumps > 50).sum()),
        "recorded_within_chunk_position_step_mm": distribution(target_jumps),
        "recorded_within_chunk_step_over_50mm_count": int((target_jumps > 50).sum()),
    }
    return result


def save_episode_plot(records, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    frames = np.array([r["frame"] for r in records])
    pred = np.stack([r["prediction"][0] for r in records])
    target = np.stack([r["target"][0] for r in records])
    state = np.stack([r["state"] for r in records])
    errors = action_errors(pred, target)
    figure = Figure(figsize=(13, 10), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2).ravel()
    for ax, dim, name in zip(axes[:4], (0, 1, 2, 6), ("X", "Y", "Z", "Single finger"), strict=True):
        ax.plot(frames, target[:, dim] * 1000, label="Recorded target")
        ax.plot(frames, pred[:, dim] * 1000, label="Predicted first action", alpha=0.8)
        ax.plot(frames, state[:, dim] * 1000, label="Current state", linestyle=":", alpha=0.6)
        ax.set_ylabel(f"{name} (mm)")
    axes[0].legend(fontsize=8)
    axes[4].plot(frames, errors["rotation_deg"])
    axes[4].set_ylabel("SO(3) rotation error (deg)")
    axes[5].plot(frames, errors["position_mm"])
    axes[5].set_ylabel("Position L2 error (mm)")
    for ax in axes:
        ax.set_xlabel("Recorded episode frame (10 Hz compact timeline)")
        ax.grid(alpha=0.2)
    figure.suptitle(f"Episode {records[0]['episode']}: open-loop first-action comparison (not a rollout)")
    figure.savefig(path, dpi=140)


def evaluate_dataset(
    policy,
    dataset,
    episode_lengths,
    output_dir,
    *,
    horizon=50,
    action_dim=32,
    seed=42,
    stride=1,
    max_frames_per_episode=0,
    make_plots=True,
    use_tactile=True,
):
    """Evaluate every selected anchor once; caller creates a fresh output directory."""
    output_dir = Path(output_dir)
    anchors = select_anchors(episode_lengths, stride, max_frames_per_episode)
    if len(dataset) != sum(episode_lengths.values()):
        raise ValueError("Dataset length disagrees with episode metadata")
    records = []
    columns = ["episode", "frame", "offset", "target_frame", "valid", "position_mm", "rotation_deg", "gripper_mm"]
    columns += [
        f"{source}_{name}"
        for source in ("prediction", "target")
        for name in ("x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad", "finger_m")
    ]
    with (output_dir / "predictions.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index, (dataset_index, episode, frame) in enumerate(anchors):
            try:
                record = infer_anchor(
                    policy,
                    dataset[dataset_index],
                    episode=episode,
                    frame=frame,
                    length=episode_lengths[episode],
                    horizon=horizon,
                    action_dim=action_dim,
                    seed=seed,
                    use_tactile=use_tactile,
                )
            except Exception as exc:
                failure = {
                    "status": "failed",
                    "episode": episode,
                    "frame": frame,
                    "completed_anchors": len(records),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                (output_dir / "failure.json").write_text(json.dumps(failure, indent=2, allow_nan=False))
                raise
            records.append(record)
            errors = action_errors(record["prediction"], record["target"])
            for offset, valid in enumerate(record["valid"]):
                writer.writerow(
                    [
                        episode,
                        frame,
                        offset,
                        frame + offset if valid else "",
                        int(valid),
                        *[float(errors[key][offset]) if valid else "" for key in errors],
                        *record["prediction"][offset],
                        *record["target"][offset],
                    ]
                )
            handle.flush()
            if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(anchors):
                logging.info(
                    "Evaluated %d/%d anchors; episode=%d frame=%d infer_wall=%.1f ms",
                    index + 1,
                    len(anchors),
                    episode,
                    frame,
                    record["policy_call_wall_ms"],
                )
    np.savez_compressed(
        output_dir / "predictions.npz",
        episode=np.array([r["episode"] for r in records]),
        frame=np.array([r["frame"] for r in records]),
        prediction=np.stack([r["prediction"] for r in records]),
        target=np.stack([r["target"] for r in records]),
        valid=np.stack([r["valid"] for r in records]),
        state=np.stack([r["state"] for r in records]),
        policy_call_wall_ms=np.array([r["policy_call_wall_ms"] for r in records]),
    )
    summary = {"status": "complete", "overall": summarize_records(records), "episodes": {}}
    summary["input_modality"] = "rgb_state_touch" if use_tactile else "rgb_state"
    summary["latency"] = {
        "first_call_including_compilation_ms": records[0]["policy_call_wall_ms"],
        "subsequent_policy_call_wall_ms": distribution([r["policy_call_wall_ms"] for r in records[1:]]),
        "note": "Includes policy transforms and synchronization; excludes video loading/robot/network latency.",
    }
    summary["limitations"] = [
        "Recorded-observation open-loop evaluation, not a robot/simulator rollout or task success rate.",
        "Overlapping chunk targets are correlated; valid_chunk weights anchor/offset pairs, not unique frames.",
        "Single fixed noise draw per anchor; compare checkpoints with identical seed/selection/sampler settings.",
        "No clipping, cropping or relabeling; startup target jumps and rotation-label issues remain in the data.",
        "50 mm is a diagnostic threshold, not a robot workspace/speed/collision safety limit.",
        "Baseline holds current state at every horizon offset; it is not another trained policy.",
    ]
    for episode in episode_lengths:
        subset = [r for r in records if r["episode"] == episode]
        summary["episodes"][str(episode)] = summarize_records(subset)
        if make_plots:
            save_episode_plot(subset, output_dir / f"episode_{episode:06d}.png")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary
