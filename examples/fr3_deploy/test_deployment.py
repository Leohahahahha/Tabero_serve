"""CPU contract and controller fault tests; no ROS, GPU, or hardware required."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import core
import dmtac_w_ipc as ipc
import numpy as np
import pytest
import run
from scipy.spatial.transform import Rotation
import transport

from scripts import serve_tabero

ROOT = Path(__file__).parent


@pytest.fixture
def config():
    return run.load_config(ROOT / "config.json")


def state():
    return np.array([0.45, 0, 0.25, np.pi, 0, 0, 0.02])


def test_http_units_quaternion_and_gripper_roundtrip():
    action = np.array([0.4, -0.1, 0.3, 0, 0, np.pi / 2, 0.03])
    pose, width = core.action_to_http(action)
    np.testing.assert_allclose(pose, [0.4, -0.1, 0.3, 0, 0, np.sqrt(0.5), np.sqrt(0.5)])
    assert width == pytest.approx(0.06)
    np.testing.assert_allclose(core.pose_to_state(pose, width), action, atol=1e-7)
    with pytest.raises(KeyError, match="gripper_width"):
        core.state_from_http({"pose": pose, "gripper_pos": 0.7})


@pytest.mark.parametrize("invalid", [np.nan, -0.001, 0.043])
def test_bad_gripper_rejected(invalid):
    action = state()
    action[6] = invalid
    with pytest.raises(ValueError, match="finite|single-finger"):
        core.action_to_http(action)


def test_invalid_quaternion_is_not_replaced_with_identity():
    with pytest.raises(ValueError, match="quaternion"):
        core.pose_to_state([0.4, 0, 0.3, 0, 0, 0, 0], 0.04)


def test_rotation_branch_and_rate_limits(config):
    measured = state()
    measured[3] = np.pi - 0.005
    target = measured.copy()
    target[3] = -np.pi + 0.005
    target[0] += 0.04
    target[6] += 0.01
    guard = core.TargetGuard(config["limits"], measured)
    command = guard.prepare(target, measured, 10)  # dt is capped at 100 ms.
    assert command[0] - measured[0] == pytest.approx(0.002)
    assert command[6] - measured[6] == pytest.approx(0.001)
    relative = Rotation.from_rotvec(measured[3:6]).inv() * Rotation.from_rotvec(command[3:6])
    assert relative.magnitude() <= 0.0100001
    np.testing.assert_array_equal(guard.last, measured)  # Must not commit before HTTP success.


@pytest.mark.parametrize("kind", ["workspace", "translation", "rotation", "tracking"])
def test_reject_unsafe_targets(config, kind):
    measured, target = state(), state()
    guard = core.TargetGuard(config["limits"], measured)
    if kind == "workspace":
        target[2] = 0.01
    elif kind == "translation":
        target[0] += 0.06
    elif kind == "rotation":
        target[3] += 1
    else:
        measured[0] -= 0.04
    with pytest.raises(ValueError, match="workspace|jump|tracking"):
        guard.prepare(target, measured)


def test_marker_grid_history_and_left_right_order():
    history = core.MarkerHistory(scale=2)
    left = np.zeros((240, 320, 2), np.float32)
    right = np.zeros_like(left)
    left[..., 0] = 1
    right[..., 1] = 3
    first = history.append(left, right)
    assert first.shape == (9, 198, 2)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first[0, [0, 98, 99, 197]], [[0, 0], [319, 239], [0, 0], [319, 239]])
    np.testing.assert_array_equal(first[8, [0, 98, 99, 197]], [[2, 0], [321, 239], [0, 6], [319, 245]])
    np.testing.assert_array_equal(first[1:], np.repeat(first[1:2], 8, axis=0))
    left[..., 0] = 5
    second = history.append(left, right)
    np.testing.assert_array_equal(second[1:8], first[2:9])
    assert second[-1, 0, 0] == 10
    np.testing.assert_array_equal(second[0], first[0])


def test_marker_contract_rejects_wrong_shape_dtype_and_nonfinite():
    valid = core.MarkerHistory().append(
        np.zeros((240, 320, 2), np.float32),
        np.zeros((240, 320, 2), np.float32),
    )
    assert core.validate_tactile_marker_motion(valid).dtype == np.float32
    for invalid, reason in (
        (np.zeros((8, 198, 2), np.float32), "shape"),
        (valid.astype(np.float64), "float32"),
        (np.full((9, 198, 2), np.nan, np.float32), "non-finite"),
        (np.zeros((9, 198, 2), np.float32), "reference grid"),
    ):
        with pytest.raises(ValueError, match=reason):
            core.validate_tactile_marker_motion(invalid)


def test_marker_summary_separates_left_and_right_current_motion():
    marker = core.MarkerHistory().append(
        np.zeros((240, 320, 2), np.float32),
        np.zeros((240, 320, 2), np.float32),
    )
    marker[-1, :99, 0] += 3
    marker[-1, 99:, 1] += 4
    summary = core.tactile_marker_summary(marker)
    assert summary["shape"] == [9, 198, 2]
    assert summary["dtype"] == "float32"
    assert summary["left_motion_mean"] == pytest.approx(3)
    assert summary["right_motion_mean"] == pytest.approx(4)


def test_remote_tactile_policy_rejects_bad_marker_before_send():
    policy = object.__new__(transport.RemotePolicy)
    policy.metadata = {"use_tactile": True, "action_horizon": 50}
    sent = []
    policy.ws = SimpleNamespace(send=sent.append)
    policy.packer = SimpleNamespace(pack=lambda value: value)
    marker = core.MarkerHistory().append(
        np.zeros((240, 320, 2), np.float32),
        np.zeros((240, 320, 2), np.float32),
    )
    sample = core.Sample({"tactile_marker_motion": marker.astype(np.float64)}, 1.0, 1.0)
    with pytest.raises(ValueError, match="float32"):
        policy.infer(sample)
    assert sent == []


@pytest.mark.parametrize("mode", ["full", "shear_depth"])
def test_real_ipc_pack_decode_parity(mode):
    arrays = {
        spec.name: np.full(spec.shape, index + 1, dtype=spec.dtype)
        for index, spec in enumerate(ipc.get_modality_specs(mode))
    }
    arrays["shear"][149, 191] = [7, -8]
    payload = ipc.pack_modalities(arrays, output_mode=mode)
    layout = ipc.packed_layout_metadata(mode)
    msg = SimpleNamespace(
        height=1, width=len(payload), step=len(payload), encoding="8UC1", data=payload, is_bigendian=False
    )
    decoded = core.decode_packed_shear(msg, layout, ipc.PACKED_IMAGE_ENCODING)
    np.testing.assert_array_equal(decoded, arrays["shear"])
    msg.step -= 1
    with pytest.raises(ValueError, match="packed size"):
        core.decode_packed_shear(msg, layout, ipc.PACKED_IMAGE_ENCODING)


def test_rgb_row_padding_bgr_and_big_endian_shear():
    msg = SimpleNamespace(
        height=1, width=2, step=8, encoding="bgr8", is_bigendian=False, data=bytes([1, 2, 3, 4, 5, 6, 0, 0])
    )
    np.testing.assert_array_equal(core.decode_image(msg), [[[3, 2, 1], [6, 5, 4]]])
    msg = SimpleNamespace(
        height=1,
        width=1,
        step=12,
        encoding="32FC2",
        is_bigendian=True,
        data=np.array([1.5, -3.5, 0], dtype=">f4").tobytes(),
    )
    np.testing.assert_array_equal(core.decode_image(msg), [[[1.5, -3.5]]])


def test_training_crop_and_conversion():
    conversion = json.loads((ROOT / "tabero_conversion.json").read_text())
    core.validate_conversion(conversion)
    rgb = np.zeros((540, 960, 3), np.uint8)
    rgb[0, 350] = [1, 2, 3]
    result = core.crop_front(rgb, conversion)
    assert result.shape == (520, 390, 3)
    np.testing.assert_array_equal(result[0, 0], [1, 2, 3])
    with pytest.raises(ValueError, match="resolution"):
        core.crop_front(rgb[:520], conversion)


def test_chunk_discards_elapsed_actions_and_expires():
    chunk = core.Chunk(np.repeat(state()[None], 50, axis=0), 10.0, state())
    _, index = chunk.select(10.245, 5)
    assert index == 2
    with pytest.raises(ValueError, match="expired"):
        chunk.select(10.51, 5)


def test_action_distance_metrics_uses_so3_shortest_angle():
    left = state()
    right = left.copy()
    left[3] = np.pi - 0.01
    right[3] = -np.pi + 0.01
    right[0] += 0.03
    right[6] += 0.004
    result = core.action_distance_metrics(left, right)
    assert result["position_m"] == pytest.approx(0.03)
    assert result["rotation_rad"] == pytest.approx(0.02)
    assert result["single_finger_m"] == pytest.approx(0.004)


def test_server_contract_rejects_wrong_modalities_or_conversion():
    meta = {
        "deployment_protocol": core.PROTOCOL,
        "action_representation": "absolute_xyz_axis_angle_single_finger_m",
        "action_dim": 7,
        "dataset_fps": 10,
        "use_tactile": True,
        "conversion_sha256": "abc",
        "predicts_wrench": False,
        "tactile_input": core.TACTILE_INPUT,
        "tactile_marker_shape": list(core.TACTILE_MARKER_SHAPE),
        "tactile_marker_dtype": "float32",
        "tactile_marker_layout": core.TACTILE_MARKER_LAYOUT,
    }
    core.validate_metadata(meta, use_tactile=True, conversion_sha256="abc")
    with pytest.raises(ValueError, match="use_tactile"):
        core.validate_metadata(meta, use_tactile=False, conversion_sha256="abc")
    with pytest.raises(ValueError, match="conversion_sha256"):
        core.validate_metadata(meta, use_tactile=True, conversion_sha256="def")
    wrong = {**meta, "tactile_marker_shape": [8, 198, 2]}
    with pytest.raises(ValueError, match="tactile_marker_shape"):
        core.validate_metadata(wrong, use_tactile=True, conversion_sha256="abc")


def test_live_config_uses_stream_specific_qos_and_url_overrides():
    config = run.load_config(ROOT / "config.json")
    assert config["front_topic"].endswith("/compressed")
    assert config["front_compressed"] is True
    assert config["qos"]["front"] == {"reliability": "reliable", "depth": 1}
    assert config["qos"]["wrist"] == {"reliability": "reliable", "depth": 1}
    assert config["qos"]["tactile"] == {"reliability": "best_effort", "depth": 1}
    changed = run.load_config(
        ROOT / "config.json",
        robot_url="http://172.31.179.19:5000",
        policy_url="ws://192.168.1.20:8000",
    )
    assert changed["robot_url"] == "http://172.31.179.19:5000"
    assert changed["policy_url"] == "ws://192.168.1.20:8000"


def test_server_rejects_wrong_tactile_conversion_before_model_inference():
    metadata = {
        "tactile_input": core.TACTILE_INPUT,
        "tactile_marker_shape": list(core.TACTILE_MARKER_SHAPE),
        "tactile_marker_dtype": "float32",
        "tactile_marker_layout": core.TACTILE_MARKER_LAYOUT,
    }
    conversion = {"marker_field": {"shape": [9, 198, 2], "history_length": 8, "side_order": ["left", "right"]}}
    serve_tabero.validate_tactile_contract(metadata, conversion, use_tactile=True)
    with pytest.raises(ValueError, match="left-then-right"):
        serve_tabero.validate_tactile_contract(
            metadata,
            {"marker_field": {"shape": [9, 198, 2], "history_length": 8, "side_order": ["right", "left"]}},
            use_tactile=True,
        )


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def time(self):
        return self.now

    def is_set(self):
        return False

    def wait(self, seconds):
        self.now += max(seconds, 0.00001)
        return False


def controller_fixture(
    monkeypatch, config, *, latency=0.145, enabled_until=float("inf"), invalid=False, fail_gripper=False
):
    clock = Clock()
    measured = state()
    records = []

    class Source:
        def arm(self):
            pass

        def snapshot(self):
            return core.Sample({"state": measured.copy()}, clock.now, clock.now)

        def measured_state(self):
            return measured.copy()

        def enable_is_fresh(self, after):
            return clock.now < enabled_until

    class Policy:
        def infer(self, sample, *, warmup=False):
            actions = np.repeat(sample.data["state"][None], 50, axis=0)
            actions[:, 0] += 0.0001 * np.arange(50)
            actions[:, 6] += 0.001
            if invalid:
                actions[:, 0] += 0.1
            return core.Chunk(actions, sample.monotonic, sample.data["state"].copy())

        def close(self):
            pass

    class Robot:
        def command_pose(self, pose):
            records.append(("pose", np.asarray(pose).copy()))
            measured[:6] = core.pose_to_state(pose, 2 * measured[6])[:6]

        def command_width(self, width):
            records.append(("width", width))
            if fail_gripper:
                raise TimeoutError("uncertain gripper command")
            measured[6] = width / 2

        def read_state(self, max_age):
            return measured.copy(), clock.now

    class Pool:
        def __init__(self, **kwargs):
            pass

        def submit(self, fn, sample):
            output = fn(sample)
            ready = clock.now + latency
            return SimpleNamespace(done=lambda: clock.now >= ready, result=lambda: output)

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(run, "time", clock)
    monkeypatch.setattr(run, "ThreadPoolExecutor", Pool)
    log = io.StringIO()

    def execute(*, live):
        run.control_loop(Source(), Policy(), Robot(), config, execute=live, duration=0.65, stopped=clock, log=log)

    return execute, records, log


def test_shadow_makes_no_robot_writes_and_uses_delayed_index(monkeypatch, config):
    execute, records, log = controller_fixture(monkeypatch, config)
    execute(live=False)
    assert records == []
    rows = [json.loads(line) for line in log.getvalue().splitlines()]
    chunks = [row for row in rows if row["event"] == "inference_chunk"]
    ticks = [row for row in rows if row["event"] == "control_tick"]
    assert all(len(row["actions"]) == 50 for row in chunks)
    assert len({row["chunk_id"] for row in chunks}) == len(chunks)
    assert ticks[0]["chunk_index"] == 2
    assert ticks[0]["observation_age_sec"] == pytest.approx(0.2)
    expected_action0 = state()
    expected_action0[6] += 0.001
    assert ticks[0]["action0"] == pytest.approx(expected_action0.tolist())
    assert ticks[0]["distances"]["action0_vs_observation"]["position_m"] == pytest.approx(0)
    assert ticks[0]["distances"]["selected_vs_current"]["position_m"] == pytest.approx(0.0002)
    assert ticks[0]["distances"]["selected_vs_action0"]["position_m"] == pytest.approx(0.0002)
    assert ticks[0]["distances"]["current_vs_observation"]["position_m"] == pytest.approx(0)
    assert ticks[0]["distances"]["selected_vs_previous_tick"] is None
    assert ticks[0]["chunk_switched_since_previous_tick"] is False


def test_shadow_records_rejected_predictions_without_writes(monkeypatch, config):
    execute, records, log = controller_fixture(monkeypatch, config, invalid=True)
    execute(live=False)
    assert records == []
    rows = [json.loads(line) for line in log.getvalue().splitlines()]
    ticks = [row for row in rows if row["event"] == "control_tick"]
    assert len(ticks) >= 2
    assert all(not row["ok"] for row in ticks)


def test_live_splits_pose_and_width_then_holds_on_exit(monkeypatch, config):
    execute, records, log = controller_fixture(monkeypatch, config)
    execute(live=True)
    assert records[0][0] == "pose"
    assert records[0][1].shape == (7,)
    assert records[1][0] == "width"
    assert records[1][1] == pytest.approx(0.042)
    assert records[-1][0] == "pose"  # final measured hold
    rows = [json.loads(row) for row in log.getvalue().splitlines()]
    assert all(row["ok"] for row in rows if row["event"] == "control_tick")


def test_enable_loss_stops_and_holds(monkeypatch, config):
    execute, records, _ = controller_fixture(monkeypatch, config, enabled_until=100.25)
    with pytest.raises(RuntimeError, match="heartbeat"):
        execute(live=True)
    assert [kind for kind, _ in records] == ["pose", "width", "pose"]


def test_late_inference_never_actuates(monkeypatch, config):
    execute, records, _ = controller_fixture(monkeypatch, config, latency=0.4)
    with pytest.raises(RuntimeError, match="too old"):
        execute(live=True)
    assert records == []


def test_invalid_first_target_never_actuates(monkeypatch, config):
    execute, records, log = controller_fixture(monkeypatch, config, invalid=True)
    with pytest.raises(ValueError, match="jump"):
        execute(live=True)
    assert records == []
    rows = [json.loads(row) for row in log.getvalue().splitlines()]
    assert not next(row for row in rows if row["event"] == "control_tick")["ok"]


def test_partial_http_failure_is_not_retried(monkeypatch, config):
    execute, records, log = controller_fixture(monkeypatch, config, fail_gripper=True)
    with pytest.raises(TimeoutError, match="uncertain"):
        execute(live=True)
    assert [kind for kind, _ in records] == ["pose", "width", "pose"]
    rows = [json.loads(row) for row in log.getvalue().splitlines()]
    assert not next(row for row in rows if row["event"] == "control_tick")["ok"]


def test_http_rejects_stale_measured_stamp(monkeypatch):
    robot = transport.RobotHttp("http://unused.invalid")
    pose, width = core.action_to_http(state())
    monkeypatch.setattr(
        robot, "post", lambda *_: {"pose": pose, "gripper_width": width, "stamp": {"to_sec": transport.time.time() - 5}}
    )
    with pytest.raises(ValueError, match="stale"):
        robot.read_state(0.25)
    robot.close()


def test_wire_paths_and_uncertain_requests_not_retried(monkeypatch):
    robot = transport.RobotHttp("http://unused.invalid", timeout=0.12)
    sent = []

    def post(url, **kwargs):
        sent.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None, headers={}, text="ok")

    monkeypatch.setattr(robot.session, "post", post)
    robot.command_pose([0.4, 0, 0.3, 0, 0, 0, 1])
    robot.command_width(0.04)
    assert sent[0] == ("http://unused.invalid/pose", {"json": {"arr": [0.4, 0, 0.3, 0, 0, 0, 1]}, "timeout": 0.12})
    assert sent[1] == ("http://unused.invalid/move_gripper", {"json": {"gripper_width": 0.04}, "timeout": 0.12})
    robot.close()
