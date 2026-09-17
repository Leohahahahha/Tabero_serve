"""Read-only audit of recorded FR3 state/action boundaries; no model or robot use.

Run with /data/yanghaojun/envs/tabero-smoke/bin/python and this file's absolute path.
Outputs are written beside this script; all source data remain unchanged.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


OUT = Path(__file__).resolve().parent
RAW = Path('/home/yanghaojun/.codex/attachments/7a3207e0-6628-4ea2-b07c-332c8eb1a530/robot_state_events.jsonl')
DATA = Path('/data/yanghaojun/datasets/tabero_lerobot_compact_v1')
EVAL = Path('/data/yanghaojun/outputs/offline_eval/offline_2999_20260903_125340')


def pos_error(a, b):
    return np.linalg.norm(np.asarray(a)[..., :3] - np.asarray(b)[..., :3], axis=-1) * 1000


def rot_error(a, b):
    return np.rad2deg((Rotation.from_rotvec(a).inv() * Rotation.from_rotvec(b)).magnitude())


def main():
    raw = [json.loads(line) for line in RAW.read_text().splitlines()]
    assert all(row['type'] == 'robot_state' and row['ok'] for row in raw)
    pose = np.asarray([row['state']['pose'] for row in raw])
    query_time = np.asarray([row['t_query_mid'] for row in raw])
    finger = np.asarray([row['state']['gripper_width'] / 2 for row in raw])
    tree = cKDTree(pose[:, :3])
    saved = np.load(EVAL / 'predictions.npz')
    conversion = json.loads((DATA / 'meta/tabero_conversion.json').read_text())
    episodes = []
    previous_action = None
    ep4 = None
    for path in sorted((DATA / 'data/chunk-000').glob('*.parquet')):
        episode = int(path.stem.rsplit('_', 1)[1])
        table = pq.read_table(path, columns=['state', 'actions']).to_pydict()
        state, action = np.asarray(table['state']), np.asarray(table['actions'])
        changes = np.flatnonzero(np.any(action != action[0], axis=1))
        prefix = int(changes[0]) if changes.size else len(action)
        raw_distance, raw_indices = tree.query(state[:, :3])
        assert raw_distance.max() < 1e-7
        gap = pos_error(action, state)
        component_gap = np.linalg.norm(action[:, 3:6] - state[:, 3:6], axis=1)
        physical_gap = rot_error(action[:, 3:6], state[:, 3:6])
        item = {
            'episode': episode,
            'frames': len(state),
            'constant_action_prefix_frames': prefix,
            'first_action_state_mm': float(gap[0]),
            'first_action_next_state_mm': float(pos_error(action[0], state[1])),
            'last_action_state_mm': float(gap[-1]),
            'first_action_equals_previous_last_all7': bool(previous_action is not None and np.array_equal(action[0], previous_action)),
            'max_action_position_step_mm': float(pos_error(action[1:], action[:-1]).max()),
            'nearest_raw_xyz_max_error_mm': float(raw_distance.max() * 1000),
            'raw_rotvec_delta_over3rad_but_physical_under10deg_frames': int(((component_gap > 3) & (physical_gap < 10)).sum()),
        }
        select = saved['episode'] == episode
        if select.any():
            assert np.array_equal(saved['state'][select], state)
            assert np.array_equal(saved['target'][select, 0], action)
            # Check every saved chunk against its own episode, including end padding.
            indices = np.minimum(np.arange(len(state))[:, None] + np.arange(50), len(state) - 1)
            assert np.array_equal(saved['target'][select], action[indices])
            assert np.array_equal(saved['valid'][select], np.arange(len(state))[:, None] + np.arange(50) < len(state))
            prediction = saved['prediction'][select, 0]
            error = pos_error(prediction, action)
            item['prediction_target_mean_position_mm'] = float(error.mean())
            item['prefix_prediction_target_mean_position_mm'] = float(error[:prefix].mean())
            item['after_prefix_prediction_target_mean_position_mm'] = float(error[prefix:].mean())
            item['last_prediction_target_position_mm'] = float(error[-1])
            item['last_prediction_target_rotation_deg'] = float(rot_error(prediction[-1:, 3:6], action[-1:, 3:6])[0])
            # Descriptive lag sweep; not a causal latency measurement or relabeling rule.
            start, stop = 25, len(state) - 10
            item['interior_action_t_vs_state_t_plus_k_mean_mm'] = {
                str(k): float(pos_error(action[start:stop], state[start + k:stop + k]).mean())
                for k in range(8)
            }
            if episode == 4:
                assert np.array_equal(raw_indices, np.arange(1228, 1450))
                raw_rotation = Rotation.from_quat(pose[raw_indices, 3:])
                converted_rotation = Rotation.from_rotvec(state[:, 3:6])
                item['raw_matching_jsonl_lines_inclusive'] = [1229, 1450]
                item['raw_matching_max_rotation_error_deg'] = float(np.rad2deg((raw_rotation.inv() * converted_rotation).magnitude()).max())
                item['raw_matching_max_finger_error_mm'] = float(np.max(abs(finger[raw_indices] - state[:, 6])) * 1000)
                item['gap_before_first_raw_state_sec'] = float(query_time[1228] - query_time[1227])
                item['first_11_states_max_displacement_from_start_mm'] = float(pos_error(state[:11], state[0]).max())
                item['frame0_prediction_state_mm'] = float(pos_error(prediction[0], state[0]))
                item['frame0_prediction_target_mm'] = float(error[0])
                ep4 = (state, action, prediction, raw_indices)
        episodes.append(item)
        previous_action = action[-1]
    summary = {
        'sources': {'raw': str(RAW), 'dataset': str(DATA), 'evaluation': str(EVAL)},
        'raw_rows': len(raw),
        'raw_contains_action_events': False,
        'episode4_conversion_timing': next(x for x in conversion['timing_reports'] if x['output_episode_index'] == 4),
        'episodes': episodes,
        'first_action_state_over50mm_episode_count': sum(x['first_action_state_mm'] > 50 for x in episodes),
        'first_action_equals_previous_last_all7_episode_count': sum(x['first_action_equals_previous_last_all7'] for x in episodes),
        'unresolved': 'Raw action events and collection/conversion source are unavailable; stale acquisition cache versus cross-episode converter carry-forward is not localized.',
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    assert ep4 is not None
    state, action, prediction, raw_indices = ep4
    with (OUT / 'episode_000004_frames.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['frame', 'raw_jsonl_line', 'raw_query_time', 'action_state_mm', 'action_next_state_mm', 'prediction_target_mm', 'prediction_state_mm', 'prediction_rotation_error_deg'] + [f'{kind}_{dim}' for kind in ('state', 'action', 'prediction') for dim in ('x_m', 'y_m', 'z_m', 'rx_rad', 'ry_rad', 'rz_rad', 'finger_m')])
        for i in range(len(state)):
            writer.writerow([i, raw_indices[i] + 1, query_time[raw_indices[i]], pos_error(action[i], state[i]), pos_error(action[i], state[i + 1]) if i + 1 < len(state) else '', pos_error(prediction[i], action[i]), pos_error(prediction[i], state[i]), rot_error(prediction[i:i + 1, 3:6], action[i:i + 1, 3:6])[0], *state[i], *action[i], *prediction[i]])
    make_plot(*ep4)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('episodes', 'sources', 'episode4_conversion_timing')}, indent=2))
    print('Episode 4:', json.dumps(episodes[4], indent=2))


def make_plot(state, action, prediction, raw_indices):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(14, 7), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    for row, frames in enumerate((np.arange(25), np.arange(len(state) - 25, len(state)))):
        for col, dim in enumerate((0, 2)):
            ax = axes[row, col]
            ax.plot(frames, action[frames, dim] * 1000, color='#1764b0', label='Recorded action', linewidth=2)
            ax.plot(frames, prediction[frames, dim] * 1000, color='#e77619', label='Predicted first action', alpha=.8)
            ax.plot(frames, state[frames, dim] * 1000, color='#23854d', label='Measured state (raw matched)', linestyle='--')
            ax.set_ylabel(('X' if dim == 0 else 'Z') + ' (mm)')
            if row == 0:
                ax.axvspan(0, 10.5, color='#ee7777', alpha=.12)
        axes[row, 2].plot(frames, pos_error(action[frames], state[frames]), color='#1764b0', label='Recorded action vs state')
        axes[row, 2].plot(frames, pos_error(prediction[frames], action[frames]), color='#e77619', label='Prediction vs recorded action')
        axes[row, 2].set_ylabel('Position difference (mm)')
        for ax in axes[row]:
            ax.grid(alpha=.2)
            ax.set_xlabel('Episode frame (10 Hz)')
    axes[0, 0].set_title('Start: frames 0-10 reuse previous episode final action')
    axes[0, 1].set_title('Measured state remains at reset pose')
    axes[0, 2].set_title('Startup label mismatch: 423.36 mm')
    axes[1, 0].set_title('End: motion followed by convergence')
    axes[1, 1].set_title('Final recorded target vs state: 3.63 mm')
    axes[1, 2].set_title('Final prediction vs target: 32.64 mm')
    axes[0, 1].legend(fontsize=8)
    axes[0, 2].legend(fontsize=8)
    figure.suptitle(f'Episode 4 boundary audit | raw feedback JSONL lines {raw_indices[0] + 1}-{raw_indices[-1] + 1}')
    figure.savefig(OUT / 'episode_000004_boundaries.png', dpi=140)


if __name__ == '__main__':
    main()
