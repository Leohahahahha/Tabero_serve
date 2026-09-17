"""Summarize same-frame action/state offsets in the converted Tabero dataset."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

DATA = Path("/data/yanghaojun/datasets/tabero_lerobot_compact_v1/data/chunk-000")


def describe(values: np.ndarray) -> str:
    q = np.percentile(values, [50, 95, 99])
    return (
        f"n={len(values)} p50={q[0]:.3f} p95={q[1]:.3f} "
        f"p99={q[2]:.3f} max={values.max():.3f}"
    )


def rotation_deg(actions: np.ndarray, states: np.ndarray) -> np.ndarray:
    relative = Rotation.from_rotvec(states[:, 3:6]).inv() * Rotation.from_rotvec(
        actions[:, 3:6]
    )
    return np.rad2deg(relative.magnitude())


def main() -> None:
    groups: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "all": [],
        "after_constant_prefix": [],
        "after_20_frames": [],
    }
    position_steps: list[np.ndarray] = []
    prefix_lengths: list[int] = []

    for path in sorted(DATA.glob("episode_*.parquet")):
        columns = pq.read_table(path, columns=["state", "actions"]).to_pydict()
        states = np.asarray(columns["state"], dtype=np.float64)
        actions = np.asarray(columns["actions"], dtype=np.float64)
        changed = np.flatnonzero(np.any(actions != actions[0], axis=1))
        prefix = int(changed[0]) if changed.size else len(actions)
        prefix_lengths.append(prefix)
        groups["all"].append((actions, states))
        groups["after_constant_prefix"].append((actions[prefix:], states[prefix:]))
        groups["after_20_frames"].append((actions[20:], states[20:]))
        if prefix + 1 < len(actions):
            position_steps.append(
                np.linalg.norm(
                    actions[prefix + 1 :, :3] - actions[prefix:-1, :3], axis=1
                )
                * 1000
            )

    print(
        "constant_prefix_frames:",
        describe(np.asarray(prefix_lengths, dtype=np.float64)),
    )
    for name, arrays in groups.items():
        actions = np.concatenate([pair[0] for pair in arrays])
        states = np.concatenate([pair[1] for pair in arrays])
        position = np.linalg.norm(actions[:, :3] - states[:, :3], axis=1) * 1000
        rotation = rotation_deg(actions, states)
        gripper = np.abs(actions[:, 6] - states[:, 6]) * 1000
        print(f"\n{name}")
        print("position_mm:", describe(position), f">50={np.mean(position > 50):.3%}")
        print("rotation_deg:", describe(rotation), f">20={np.mean(rotation > 20):.3%}")
        print("gripper_mm:", describe(gripper))

    steps = np.concatenate(position_steps)
    print("\npost-prefix adjacent action position step mm:", describe(steps))
    print(">50mm:", f"{np.mean(steps > 50):.3%}")


if __name__ == "__main__":
    main()
