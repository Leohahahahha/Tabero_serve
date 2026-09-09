"""CPU-only FR3 deployment contracts. No ROS, model loading, or network on import."""

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

PROTOCOL = "tabero_fr3_absolute_v1"
TACTILE_INPUT = "rolling_9x198x2_marker_coordinates_left_then_right"
TACTILE_MARKER_SHAPE = (9, 198, 2)
TACTILE_MARKER_DTYPE = np.dtype(np.float32)
TACTILE_MARKER_LAYOUT = "reference_then_8_history_frames_left_then_right"
SINGLE_FINGER_MIN_M = 0.0
SINGLE_FINGER_MAX_M = 0.0425


def finite(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected finite {shape}, got {array.shape}")
    return array


def pose_to_state(pose, width):
    pose = finite(pose, (7,), "XYZ + XYZW pose")
    norm = np.linalg.norm(pose[3:])
    if not 0.99 <= norm <= 1.01:
        raise ValueError("Invalid measured quaternion norm")
    if not np.isfinite(width) or not 0 <= width <= 0.085001:
        raise ValueError("Measured gripper_width must be total width in meters [0,0.085]")
    return np.concatenate([pose[:3], Rotation.from_quat(pose[3:]).as_rotvec(), [width / 2]]).astype(np.float32)


def action_to_http(action):
    action = finite(action, (7,), "absolute model action")
    if not SINGLE_FINGER_MIN_M <= action[6] <= SINGLE_FINGER_MAX_M + 1e-7:
        raise ValueError("Predicted single-finger position outside [0,0.0425] m")
    return np.concatenate([action[:3], Rotation.from_rotvec(action[3:6]).as_quat()]), float(2 * action[6])


def state_from_http(payload):
    # Deliberately no fallback to gripper_pos: supplied server reports that in [0,1].
    return pose_to_state(payload["pose"], float(payload["gripper_width"]))


def marker_reference_grid():
    y = np.rint(np.linspace(0, 239, 9)).astype(int)
    x = np.rint(np.linspace(0, 319, 11)).astype(int)
    gx, gy = np.meshgrid(x.astype(np.float32), y.astype(np.float32))
    side = np.stack([gx, gy], axis=-1).reshape(99, 2)
    return np.concatenate([side, side]).astype(np.float32)


def marker_positions(left, right, scale=1.0):
    left = finite(left, (240, 320, 2), "left shear")
    right = finite(right, (240, 320, 2), "right shear")
    y = np.rint(np.linspace(0, 239, 9)).astype(int)
    x = np.rint(np.linspace(0, 319, 11)).astype(int)
    reference = marker_reference_grid()
    shear = np.concatenate([left[np.ix_(y, x)].reshape(99, 2), right[np.ix_(y, x)].reshape(99, 2)])
    current = reference + np.float32(scale) * shear.astype(np.float32)
    if not np.isfinite(current).all():
        raise ValueError("Non-finite marker coordinates")
    return reference, current.astype(np.float32)


class MarkerHistory:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.frames = deque(maxlen=8)

    def append(self, left, right):
        reference, current = marker_positions(left, right, self.scale)
        if not self.frames:
            self.frames.extend(current.copy() for _ in range(7))
        self.frames.append(current)
        return validate_tactile_marker_motion(np.stack([reference, *self.frames]).astype(np.float32))


def validate_tactile_marker_motion(value):
    """Reject any observation that is not the exact tactile training contract."""
    array = np.asarray(value)
    if array.shape != TACTILE_MARKER_SHAPE:
        raise ValueError(f"tactile_marker_motion must have shape {TACTILE_MARKER_SHAPE}, got {array.shape}")
    if array.dtype != TACTILE_MARKER_DTYPE:
        raise ValueError(f"tactile_marker_motion must have dtype float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("tactile_marker_motion contains non-finite values")
    if not np.array_equal(array[0], marker_reference_grid()):
        raise ValueError("tactile_marker_motion[0] is not the fixed left-then-right reference grid")
    return np.ascontiguousarray(array)


def tactile_marker_summary(value):
    marker = validate_tactile_marker_motion(value)
    displacement = marker[-1] - marker[0]
    left_norm = np.linalg.norm(displacement[:99], axis=-1)
    right_norm = np.linalg.norm(displacement[99:], axis=-1)
    return {
        "shape": list(marker.shape),
        "dtype": marker.dtype.name,
        "left_motion_mean": float(left_norm.mean()),
        "left_motion_max": float(left_norm.max()),
        "right_motion_mean": float(right_norm.mean()),
        "right_motion_max": float(right_norm.max()),
    }


def decode_image(msg):
    """Decode ROS Image including row padding and endianness; output RGB or shear."""
    formats = {
        "rgb8": ("u1", 3),
        "bgr8": ("u1", 3),
        "rgba8": ("u1", 4),
        "bgra8": ("u1", 4),
        "32FC2": (">f4" if msg.is_bigendian else "<f4", 2),
    }
    if msg.encoding not in formats:
        raise ValueError(f"Unsupported image encoding {msg.encoding}")
    dtype, channels = formats[msg.encoding]
    dtype = np.dtype(dtype)
    row_bytes = msg.width * channels * dtype.itemsize
    if msg.width <= 0 or msg.height <= 0 or msg.step < row_bytes or len(msg.data) != msg.height * msg.step:
        raise ValueError("Invalid ROS image size/step/data")
    array = np.ndarray(
        (msg.height, msg.width, channels),
        dtype=dtype,
        buffer=bytes(msg.data),
        strides=(msg.step, channels * dtype.itemsize, dtype.itemsize),
    ).copy()
    if msg.encoding == "32FC2":
        return array.astype(np.float32)
    array = array[..., :3]
    if msg.encoding in ("bgr8", "bgra8"):
        array = array[..., ::-1]
    return np.ascontiguousarray(array)


def crop_front(image, conversion):
    spec = conversion["front_image_crop"]
    if list(image.shape) != spec["source_shape"] or image.dtype != np.uint8:
        raise ValueError(f"Front camera resolution/dtype differs from training: {image.shape}/{image.dtype}")
    if spec["enabled"]:
        x0, y0, x1, y1 = spec["roi_xyxy"]
        if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
            raise ValueError("Invalid training front ROI")
        image = image[y0:y1, x0:x1]
    if list(image.shape) != spec["output_shape"]:
        raise ValueError("Front crop does not match training metadata")
    return np.ascontiguousarray(image)


def decode_packed_shear(msg, layout, encoding):
    """Use the collector's dmtac_w_ipc metadata, never inferred binary offsets."""
    size = int(layout["packed_frame_bytes"])
    if (msg.height, msg.width, msg.step, msg.encoding, len(msg.data)) != (1, size, size, encoding, size):
        raise ValueError("DM-Tac packed size/encoding differs from configured IPC schema")
    if layout["byte_order_little_endian"] != 1 or msg.is_bigendian:
        raise ValueError("DM-Tac packed payload must be little endian")
    if [layout[f"shear_{k}"] for k in ("height", "width", "channels", "itemsize")] != [240, 320, 2, 4]:
        raise ValueError("Unsupported IPC shear shape/type")
    start, length = int(layout["shear_start"]), int(layout["shear_len"])
    if length != 240 * 320 * 2 * 4 or start < 0 or start + length > size:
        raise ValueError("Invalid IPC shear byte range")
    shear = np.frombuffer(bytes(msg.data), dtype="<f4", count=240 * 320 * 2, offset=start)
    return shear.reshape(240, 320, 2).copy()


def validate_conversion(conversion):
    if conversion["output_contract"] != "tabero_action_only_lerobot_v2.1":
        raise ValueError("Unsupported conversion contract")
    field = conversion["marker_field"]
    if field["shape"] != [9, 198, 2] or field["history_length"] != 8 or field["side_order"] != ["left", "right"]:
        raise ValueError("Unsupported tactile marker shape/order")
    for key, size, count in (("grid_y_indices", 240, 9), ("grid_x_indices", 320, 11)):
        if field[key] != np.rint(np.linspace(0, size - 1, count)).astype(int).tolist():
            raise ValueError("Marker grid differs from the supported converter")
    if not np.isfinite(field["shear_scale"]):
        raise ValueError("Invalid shear scale")
    grip = conversion["gripper"]
    if grip["output_coordinate"] != "single_finger_absolute_position" or grip["open_width_m"] != 0.085:
        raise ValueError("Unsupported gripper convention")


def validate_metadata(metadata, *, use_tactile, conversion_sha256):
    expected = {
        "deployment_protocol": PROTOCOL,
        "action_representation": "absolute_xyz_axis_angle_single_finger_m",
        "action_dim": 7,
        "dataset_fps": 10,
        "use_tactile": use_tactile,
        "conversion_sha256": conversion_sha256,
        "predicts_wrench": False,
    }
    if use_tactile:
        expected.update(
            {
                "tactile_input": TACTILE_INPUT,
                "tactile_marker_shape": list(TACTILE_MARKER_SHAPE),
                "tactile_marker_dtype": TACTILE_MARKER_DTYPE.name,
                "tactile_marker_layout": TACTILE_MARKER_LAYOUT,
            }
        )
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Server metadata mismatch: {key}={metadata.get(key)!r}, expected {value!r}")


@dataclass(frozen=True)
class Sample:
    data: dict
    monotonic: float
    oldest_capture: float


@dataclass(frozen=True)
class Chunk:
    actions: np.ndarray
    observation_time: float
    observation_state: np.ndarray


def action_distance_metrics(left, right):
    """Physical differences between two absolute FR3 action/state vectors."""
    left = finite(left, (7,), "left action/state")
    right = finite(right, (7,), "right action/state")
    rotation = Rotation.from_rotvec(left[3:6]).inv() * Rotation.from_rotvec(right[3:6])
    return {
        "position_m": float(np.linalg.norm(left[:3] - right[:3])),
        "rotation_rad": float(rotation.magnitude()),
        "single_finger_m": float(abs(left[6] - right[6])),
    }


def validate_sensor_timing(stamps, now, max_age, max_skew):
    """Validate multimodal wall-clock stamps and report the stream causing skew."""
    if not stamps:
        raise ValueError("No sensor timestamps")
    nonfinite = [name for name, stamp in stamps.items() if not np.isfinite(stamp)]
    if nonfinite:
        raise ValueError(f"Non-finite sensor timestamps: {nonfinite}")
    oldest = min(stamps, key=stamps.get)
    newest = max(stamps, key=stamps.get)
    ages = {name: round(now - stamp, 6) for name, stamp in stamps.items()}
    oldest_age = now - stamps[oldest]
    skew = stamps[newest] - stamps[oldest]
    if oldest_age > max_age:
        raise ValueError(f"Stale sensor stream: oldest={oldest}, age={oldest_age:.3f}s, ages_sec={ages}")
    if skew > max_skew:
        raise ValueError(
            "Sensor capture timestamps are not synchronized: "
            f"skew={skew:.3f}s, oldest={oldest}, newest={newest}, ages_sec={ages}"
        )


def sensor_stamps(frames, keys):
    """Extract required stream stamps with an actionable startup error."""
    missing = [key for key in keys if key not in frames]
    if missing:
        raise ValueError(f"Waiting for sensor streams: missing={missing}")
    return {key: frames[key][1] for key in keys}


class TargetGuard:
    """Saturate finite predictions while rejecting faults in measured robot state."""

    def __init__(self, limits, initial_state):
        self.limits = limits
        self.low = finite(limits["workspace_min"], (3,), "workspace_min")
        self.high = finite(limits["workspace_max"], (3,), "workspace_max")
        if np.any(self.low >= self.high):
            raise ValueError("workspace_min must be less than workspace_max")
        for key in (
            "max_target_distance_m",
            "max_target_rotation_rad",
            "max_translation_m_s",
            "max_rotation_rad_s",
            "max_gripper_width_m_s",
            "max_tracking_distance_m",
            "max_tracking_rotation_rad",
        ):
            if not np.isfinite(limits[key]) or limits[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        self.last = finite(initial_state, (7,), "initial state").copy()
        self.check_workspace(self.last)
        action_to_http(self.last)
        self.last_bounded_target = self.last.copy()
        self.last_limits_applied = ()

    def check_workspace(self, action):
        if np.any(action[:3] < self.low) or np.any(action[:3] > self.high):
            raise ValueError(f"XYZ outside configured workspace: {action[:3].tolist()}")

    def prepare(self, target, measured, dt=0.1):
        target = finite(target, (7,), "target")
        measured = finite(measured, (7,), "measured state")
        self.check_workspace(measured)
        action_to_http(measured)
        limits = self.limits
        r_measured = Rotation.from_rotvec(measured[3:6])
        r_last = Rotation.from_rotvec(self.last[3:6])
        r_target = Rotation.from_rotvec(target[3:6])
        if np.linalg.norm(self.last[:3] - measured[:3]) > limits["max_tracking_distance_m"]:
            raise ValueError("Robot is not tracking the commanded position")
        if (r_measured.inv() * r_last).magnitude() > limits["max_tracking_rotation_rad"]:
            raise ValueError("Robot is not tracking the commanded orientation")

        applied = []
        bounded = target.copy()

        workspace_xyz = np.clip(bounded[:3], self.low, self.high)
        if not np.array_equal(workspace_xyz, bounded[:3]):
            applied.append("workspace")
            bounded[:3] = workspace_xyz

        target_delta = bounded[:3] - measured[:3]
        target_distance = np.linalg.norm(target_delta)
        if target_distance > limits["max_target_distance_m"]:
            applied.append("max_target_distance_m")
            bounded[:3] = measured[:3] + target_delta * limits["max_target_distance_m"] / target_distance

        target_rotation_delta = (r_measured.inv() * r_target).as_rotvec()
        target_rotation_distance = np.linalg.norm(target_rotation_delta)
        if target_rotation_distance > limits["max_target_rotation_rad"]:
            applied.append("max_target_rotation_rad")
            target_rotation_delta *= limits["max_target_rotation_rad"] / target_rotation_distance
        bounded[3:6] = (r_measured * Rotation.from_rotvec(target_rotation_delta)).as_rotvec()

        bounded_finger = np.clip(target[6], SINGLE_FINGER_MIN_M, SINGLE_FINGER_MAX_M)
        if bounded_finger != target[6]:
            applied.append("gripper_position_m")
        bounded[6] = bounded_finger

        dt = min(max(dt, 0.0), 0.1)  # A delayed tick cannot create a large catch-up step.
        result = bounded.copy()
        dp = bounded[:3] - self.last[:3]
        translation_step = limits["max_translation_m_s"] * dt
        if np.linalg.norm(dp) > translation_step:
            applied.append("max_translation_m_s")
        dp *= min(1.0, translation_step / max(np.linalg.norm(dp), 1e-12))
        result[:3] = self.last[:3] + dp
        r_bounded = Rotation.from_rotvec(bounded[3:6])
        dr = (r_last.inv() * r_bounded).as_rotvec()
        rotation_step = limits["max_rotation_rad_s"] * dt
        if np.linalg.norm(dr) > rotation_step:
            applied.append("max_rotation_rad_s")
        dr *= min(1.0, rotation_step / max(np.linalg.norm(dr), 1e-12))
        result[3:6] = (r_last * Rotation.from_rotvec(dr)).as_rotvec()
        finger_step = 0.5 * limits["max_gripper_width_m_s"] * dt
        if abs(bounded[6] - self.last[6]) > finger_step:
            applied.append("max_gripper_width_m_s")
        result[6] = np.clip(bounded[6], self.last[6] - finger_step, self.last[6] + finger_step)
        self.check_workspace(result)
        action_to_http(result)
        self.last_bounded_target = bounded
        self.last_limits_applied = tuple(applied)
        return result

    def commit(self, command):
        self.last = command.copy()
