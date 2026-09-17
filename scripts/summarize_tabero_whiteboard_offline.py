"""Build a compact comparison table from completed whiteboard offline evaluations."""

import argparse
import csv
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing one completed evaluation per child directory")
    return parser.parse_args(argv)


def metric(summary, scope, name, statistic):
    return summary["overall"][scope][name][statistic]


def collect(root):
    rows = []
    for summary_path in sorted(root.glob("*/summary.json")):
        run_dir = summary_path.parent
        manifest = json.loads((run_dir / "manifest.json").read_text())
        summary = json.loads(summary_path.read_text())
        if manifest.get("status") != "complete" or summary.get("status") != "complete":
            raise ValueError(f"Incomplete evaluation: {run_dir}")
        row = {
            "target": run_dir.name,
            "config": manifest["config"],
            "action_source_mode": manifest["action_source_mode"],
            "optimization_method": manifest["optimization_method"],
            "tactile_lora_rank": manifest["tactile_lora_rank"],
            "anchors": summary["overall"]["anchors"],
            "first_position_mean_mm": metric(summary, "first_action", "position_mm", "mean"),
            "first_position_p95_mm": metric(summary, "first_action", "position_mm", "p95"),
            "first_rotation_mean_deg": metric(summary, "first_action", "rotation_deg", "mean"),
            "first_rotation_p95_deg": metric(summary, "first_action", "rotation_deg", "p95"),
            "first_gripper_mean_mm": metric(summary, "first_action", "gripper_mm", "mean"),
            "chunk_position_mean_mm": metric(summary, "valid_chunk", "position_mm", "mean"),
            "chunk_rotation_mean_deg": metric(summary, "valid_chunk", "rotation_deg", "mean"),
            "chunk_gripper_mean_mm": metric(summary, "valid_chunk", "gripper_mm", "mean"),
            "first_force_l2_mean_n": summary["overall"]["wrist_wrench"]["first_action"]["force_l2_n"]["mean"],
            "first_torque_l2_mean_nm": summary["overall"]["wrist_wrench"]["first_action"]["torque_l2_nm"]["mean"],
            "chunk_force_l2_mean_n": summary["overall"]["wrist_wrench"]["valid_chunk"]["force_l2_n"]["mean"],
            "chunk_torque_l2_mean_nm": summary["overall"]["wrist_wrench"]["valid_chunk"]["torque_l2_nm"]["mean"],
            "subsequent_policy_call_mean_ms": summary["latency"]["subsequent_policy_call_wall_ms"].get("mean"),
        }
        rows.append(row)
    if not rows:
        raise ValueError(f"No completed child evaluations found under {root}")
    return rows


def main(argv=None):
    args = parse_args(argv)
    root = args.root.resolve()
    rows = collect(root)
    fieldnames = list(rows[0])
    with (root / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (root / "comparison.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    print(f"Wrote {len(rows)} rows to {root / 'comparison.csv'}")


if __name__ == "__main__":
    main()
