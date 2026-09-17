"""Compare action/state continuity in Tabero compact dataset v1 and v3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

ROOTS = {
    "v1": Path("/data/yanghaojun/datasets/tabero_lerobot_compact_v1"),
    "v3": Path("/data/yanghaojun/datasets/tabero_lerobot_compact_v3"),
}
OUT = Path(__file__).with_name("v1_v3_comparison.json")


def summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def physical_rotation_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    relative = Rotation.from_rotvec(left[..., 3:6]).inv() * Rotation.from_rotvec(
        right[..., 3:6]
    )
    return np.rad2deg(relative.magnitude())


def load_episode(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["state", "actions"]).to_pydict()
    return (
        np.asarray(table["state"], dtype=np.float32),
        np.asarray(table["actions"], dtype=np.float32),
    )


def analyze(root: Path) -> tuple[dict[str, object], list[tuple[np.ndarray, np.ndarray]]]:
    episodes: list[tuple[np.ndarray, np.ndarray]] = []
    rows: list[dict[str, object]] = []
    previous_last_action: np.ndarray | None = None
    for path in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
        states, actions = load_episode(path)
        episodes.append((states, actions))
        action_state_pos = np.linalg.norm(actions[:, :3] - states[:, :3], axis=1) * 1000
        action_state_component_rot = np.linalg.norm(
            actions[:, 3:6] - states[:, 3:6], axis=1
        )
        action_state_physical_rot = physical_rotation_deg(states, actions)
        action_steps_pos = np.linalg.norm(actions[1:, :3] - actions[:-1, :3], axis=1) * 1000
        action_steps_component_rot = np.linalg.norm(
            actions[1:, 3:6] - actions[:-1, 3:6], axis=1
        )
        action_steps_physical_rot = physical_rotation_deg(actions[:-1], actions[1:])
        rows.append(
            {
                "episode": int(path.stem.rsplit("_", 1)[1]),
                "frames": len(states),
                "first_action_state_position_mm": float(action_state_pos[0]),
                "max_action_state_position_mm": float(action_state_pos.max()),
                "max_adjacent_action_position_mm": float(action_steps_pos.max()),
                "max_adjacent_action_physical_rotation_deg": float(
                    action_steps_physical_rot.max()
                ),
                "first_action_equals_previous_last_action": bool(
                    previous_last_action is not None
                    and np.array_equal(actions[0], previous_last_action)
                ),
                "action_state_rotvec_branch_frames": int(
                    np.sum(
                        (action_state_component_rot > 3.0)
                        & (action_state_physical_rot < 10.0)
                    )
                ),
                "adjacent_action_rotvec_branch_steps": int(
                    np.sum(
                        (action_steps_component_rot > 3.0)
                        & (action_steps_physical_rot < 10.0)
                    )
                ),
            }
        )
        previous_last_action = actions[-1]

    states = np.concatenate([item[0] for item in episodes])
    actions = np.concatenate([item[1] for item in episodes])
    step_left = np.concatenate([item[1][:-1] for item in episodes])
    step_right = np.concatenate([item[1][1:] for item in episodes])
    action_state_pos = np.linalg.norm(actions[:, :3] - states[:, :3], axis=1) * 1000
    action_state_component_rot = np.linalg.norm(actions[:, 3:6] - states[:, 3:6], axis=1)
    action_state_physical_rot = physical_rotation_deg(states, actions)
    action_state_gripper = np.abs(actions[:, 6] - states[:, 6]) * 1000
    action_steps_pos = np.linalg.norm(step_right[:, :3] - step_left[:, :3], axis=1) * 1000
    action_steps_component_rot = np.linalg.norm(step_right[:, 3:6] - step_left[:, 3:6], axis=1)
    action_steps_physical_rot = physical_rotation_deg(step_left, step_right)
    action_steps_gripper = np.abs(step_right[:, 6] - step_left[:, 6]) * 1000
    result: dict[str, object] = {
        "episodes": len(episodes),
        "frames": len(states),
        "action_state": {
            "position_mm": summary(action_state_pos),
            "physical_rotation_deg": summary(action_state_physical_rot),
            "rotvec_component_norm_rad": summary(action_state_component_rot),
            "gripper_mm": summary(action_state_gripper),
            "position_over_50mm": int(np.sum(action_state_pos > 50)),
            "physical_rotation_over_20deg": int(np.sum(action_state_physical_rot > 20)),
            "rotvec_branch_frames": int(
                np.sum((action_state_component_rot > 3.0) & (action_state_physical_rot < 10.0))
            ),
        },
        "adjacent_action": {
            "position_mm": summary(action_steps_pos),
            "physical_rotation_deg": summary(action_steps_physical_rot),
            "rotvec_component_norm_rad": summary(action_steps_component_rot),
            "gripper_mm": summary(action_steps_gripper),
            "position_over_50mm": int(np.sum(action_steps_pos > 50)),
            "physical_rotation_over_20deg": int(np.sum(action_steps_physical_rot > 20)),
            "rotvec_branch_steps": int(
                np.sum((action_steps_component_rot > 3.0) & (action_steps_physical_rot < 10.0))
            ),
        },
        "first_action_state_over_50mm_episodes": int(
            sum(row["first_action_state_position_mm"] > 50 for row in rows)
        ),
        "first_action_equals_previous_last_action_episodes": int(
            sum(row["first_action_equals_previous_last_action"] for row in rows)
        ),
        "per_episode": rows,
    }
    return result, episodes


def main() -> None:
    results: dict[str, object] = {}
    loaded: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for name, root in ROOTS.items():
        results[name], loaded[name] = analyze(root)

    alignment_rows = []
    for episode, ((v1_state, _), (v3_state, v3_action)) in enumerate(
        zip(loaded["v1"], loaded["v3"], strict=True)
    ):
        alignment_rows.append(
            {
                "episode": episode,
                "v3_frames_equal_v1_minus_one": len(v3_state) == len(v1_state) - 1,
                "v3_state_equals_v1_state_without_last": bool(
                    np.array_equal(v3_state, v1_state[:-1])
                ),
                "v3_action_equals_v1_next_state": bool(
                    np.array_equal(v3_action, v1_state[1:])
                ),
            }
        )
    results["v3_contract"] = {
        "all_episodes_exact": all(
            row["v3_frames_equal_v1_minus_one"]
            and row["v3_state_equals_v1_state_without_last"]
            and row["v3_action_equals_v1_next_state"]
            for row in alignment_rows
        ),
        "episodes": alignment_rows,
    }
    OUT.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    compact = {
        name: {
            key: value
            for key, value in result.items()
            if key != "per_episode"
        }
        for name, result in results.items()
        if name in ROOTS
    }
    compact["v3_contract"] = {
        "all_episodes_exact": results["v3_contract"]["all_episodes_exact"]
    }
    print(json.dumps(compact, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
