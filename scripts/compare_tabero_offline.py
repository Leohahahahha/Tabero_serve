"""Compare two completed offline runs on CPU; never load a model or contact a robot."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from openpi.policies.tabero_offline import action_errors
from openpi.policies.tabero_offline import distribution


def load_run(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete offline run: {directory}")
    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("episode", "frame", "prediction", "target", "valid", "state")}
    stats_bytes = Path(manifest["norm_stats_path"]).read_bytes()
    if hashlib.sha256(stats_bytes).hexdigest() != manifest["norm_stats_sha256"]:
        raise ValueError("Normalization assets changed since evaluation")
    stats = json.loads(stats_bytes)["norm_stats"]
    return manifest, arrays, {key: stats[key] for key in ("state", "actions")}


def compare_runs(touch_dir, no_touch_dir):
    touch_manifest, touch, touch_stats = load_run(touch_dir)
    base_manifest, base, base_stats = load_run(no_touch_dir)
    if base_manifest.get("input_modality") != "rgb_state":
        raise ValueError("--no-touch must be an RGB+state-only evaluation")
    if touch_manifest.get("input_modality", "rgb_state_touch") != "rgb_state_touch":
        raise ValueError("--touch must be a tactile-conditioned evaluation")
    paired_fields = (
        "dataset_root",
        "dataset_info_sha256",
        "episode_lengths",
        "selected_anchors",
        "stride",
        "max_frames_per_episode",
        "seed",
        "num_denoise_steps",
        "action_horizon",
        "output_action_dim",
        "internal_action_dim",
    )
    for field in paired_fields:
        if field not in touch_manifest or field not in base_manifest or touch_manifest[field] != base_manifest[field]:
            raise ValueError(f"Unpaired evaluation setting: {field}")
    if touch_stats != base_stats:
        raise ValueError("Shared state/action normalization differs")
    for field in ("episode", "frame", "target", "state", "valid"):
        if not np.array_equal(touch[field], base[field]):
            raise ValueError(f"Unpaired recorded arrays: {field}")
    n, horizon = len(touch["episode"]), touch_manifest["action_horizon"]
    if n != touch_manifest["selected_anchors"] or n == 0:
        raise ValueError("Wrong anchor count")
    if touch["frame"].shape != (n,) or touch["episode"].shape != (n,):
        raise ValueError("Invalid anchor IDs")
    if len(set(zip(touch["episode"].tolist(), touch["frame"].tolist(), strict=True))) != n:
        raise ValueError("Duplicate anchor IDs")
    if touch["valid"].shape != (n, horizon) or touch["valid"].dtype != np.bool_ or not touch["valid"][:, 0].all():
        raise ValueError("Invalid valid-target mask")
    if touch["state"].shape != (n, 7) or not np.isfinite(touch["state"]).all():
        raise ValueError("Invalid state array")
    for run in (touch, base):
        if run["prediction"].shape != (n, horizon, 7) or run["target"].shape != (n, horizon, 7):
            raise ValueError("Invalid action shape")
    touch_errors = action_errors(touch["prediction"], touch["target"])
    base_errors = action_errors(base["prediction"], base["target"])
    prediction_change = action_errors(touch["prediction"], base["prediction"])

    def summarize(indices):
        valid = touch["valid"][indices]
        result = {}
        for scope in ("first_action", "valid_chunk"):
            select = (lambda x: x[indices, 0]) if scope == "first_action" else (lambda x: x[indices][valid])
            result[scope] = {}
            for metric in touch_errors:
                a, b = select(touch_errors[metric]), select(base_errors[metric])
                result[scope][metric] = {
                    "touch": distribution(a),
                    "no_touch": distribution(b),
                    "no_touch_minus_touch": distribution(b - a),
                    "fraction_touch_lower_error": float(np.mean(a < b)),
                    "prediction_difference": distribution(select(prediction_change[metric])),
                }
        return result

    return {
        "status": "complete",
        "touch_run": str(Path(touch_dir).resolve()),
        "no_touch_run": str(Path(no_touch_dir).resolve()),
        "paired_eval_settings_verified": True,
        "matched_training_verified": False,
        "anchors": n,
        "valid_chunk_targets": int(touch["valid"].sum()),
        "overall": summarize(np.ones(n, dtype=bool)),
        "episodes": {str(ep): summarize(touch["episode"] == ep) for ep in np.unique(touch["episode"])},
        "limitations": [
            "Positive mean(no_touch_minus_touch) favors tactile; negative favors no tactile.",
            "Checks evaluation alignment, NOT equal training initialization/budget: inspect training logs.",
            "Old tactile recovery warm-started local smoke99; published-initialized baseline comparison is exploratory.",
            "Three episodes and correlated frames/chunks do not establish task success or statistical significance.",
            "Raw startup and rotation-label issues remain; no cropping, relabeling or safety certification.",
            "Recorded targets/state agree; original RGB/task file contents are not hashed by the legacy evaluator.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--touch", type=Path, required=True)
    parser.add_argument("--no-touch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or any(output.is_relative_to(p.resolve()) for p in (args.touch, args.no_touch)):
        raise ValueError("Choose a fresh output directory outside both evaluation runs")
    result = compare_runs(args.touch, args.no_touch)
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ["Scope / metric | touch | no touch | no touch - touch (positive favors touch)"]
    for scope, metrics in result["overall"].items():
        for name, values in metrics.items():
            lines.append(
                f"{scope} / {name} | {values['touch']['mean']:.4f} | "
                f"{values['no_touch']['mean']:.4f} | {values['no_touch_minus_touch']['mean']:+.4f}"
            )
    lines.extend(["", *result["limitations"]])
    (output / "comparison.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Full per-episode comparison: {output / 'comparison.json'}")


if __name__ == "__main__":
    main()
