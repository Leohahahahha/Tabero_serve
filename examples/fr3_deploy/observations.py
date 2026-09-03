"""ROS2 cameras/tactile plus direct measured HTTP state; samples history at 10 Hz."""

import importlib
import threading
import time

from core import MarkerHistory
from core import Sample
from core import crop_front
from core import decode_image
from core import decode_packed_shear
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from transport import RobotHttp


class LiveObservations:
    def __init__(self, config, conversion, *, use_tactile):
        self.config = config
        self.conversion = conversion
        self.use_tactile = use_tactile
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.frames = {}
        self.sample = None
        self.problem = "Waiting for sensors"
        self.fatal = None
        self.enabled = False
        self.enable_time = 0.0
        self.armed = False
        self.enable_lost = False
        self.history = MarkerHistory(conversion["marker_field"]["shear_scale"])
        self.reader = RobotHttp(config["robot_url"], config["http_timeout_sec"])
        self.node = Node("tabero_fr3_observations")
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subscriptions = []
        if use_tactile and config["tactile_format"] == "packed":
            ipc = importlib.import_module(config["dmtac_ipc_module"])
            self.layout = ipc.packed_layout_metadata(config["tactile_output_mode"])
            self.packed_encoding = ipc.PACKED_IMAGE_ENCODING
            if self.layout["schema_version"] not in conversion["source_tactile_schema_versions"]:
                raise ValueError("Configured DM-Tac IPC schema differs from training source schema")
        for key in ("front", "wrist", *(("left", "right") if use_tactile else ())):
            compressed = key in ("front", "wrist") and config[f"{key}_compressed"]
            self.subscriptions.append(
                self.node.create_subscription(
                    CompressedImage if compressed else Image,
                    config[f"{key}_topic"],
                    lambda msg, key=key, compressed=compressed: self.on_image(key, msg, compressed=compressed),
                    qos,
                )
            )
        self.subscriptions.append(self.node.create_subscription(Bool, config["enable_topic"], self.on_enable, qos))
        self.threads = [
            threading.Thread(target=target, daemon=True) for target in (self.spin, self.poll_state, self.sample_loop)
        ]
        for thread in self.threads:
            thread.start()

    def on_enable(self, msg):
        with self.lock:
            now = time.monotonic()
            if self.armed and (not msg.data or now - self.enable_time >= self.config["enable_timeout_sec"]):
                self.enable_lost = True
            self.enabled = bool(msg.data)
            self.enable_time = now

    def arm(self):
        with self.lock:
            self.armed = True

    def enable_is_fresh(self, after):
        with self.lock:
            return (
                self.enabled
                and not self.enable_lost
                and self.enable_time > after
                and time.monotonic() - self.enable_time < self.config["enable_timeout_sec"]
            )

    def on_image(self, key, msg, *, compressed):
        try:
            stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
            if not np.isfinite(stamp) or stamp <= 0:
                raise ValueError(f"{key}: missing acquisition timestamp")
            age = time.time() - stamp
            if age < -0.05 or age > self.config["max_sensor_age_sec"]:
                raise ValueError(f"{key}: stale/future camera timestamp, age={age:.3f}s; check host clocks")
            with self.lock:
                if key in self.frames and stamp <= self.frames[key][1]:
                    return  # Repeated timestamps do not refresh sensor health.
            if compressed:
                bgr = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"{key}: compressed image decode failed")
                array = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif key in ("left", "right") and self.config["tactile_format"] == "packed":
                array = decode_packed_shear(msg, self.layout, self.packed_encoding)
            else:
                array = decode_image(msg)
            if key in ("front", "wrist") and (array.dtype != np.uint8 or array.shape[-1] != 3):
                raise ValueError(f"{key}: expected RGB uint8")
            with self.lock:
                self.frames[key] = (array, stamp)
        except Exception as exc:
            with self.lock:
                self.fatal = f"{key}: {exc}"

    def spin(self):
        try:
            while not self.stopped.is_set() and rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.02)
        except Exception as exc:
            with self.lock:
                self.fatal = f"ROS: {exc}"

    def poll_state(self):
        while not self.stopped.is_set():
            start = time.monotonic()
            try:
                state, stamp = self.reader.read_state(self.config["max_sensor_age_sec"])
                with self.lock:
                    self.frames["state"] = (state, stamp)
            except Exception as exc:
                with self.lock:
                    self.fatal = f"State reader: {exc}"
                return
            self.stopped.wait(max(0, 1 / 30 - (time.monotonic() - start)))

    def sample_loop(self):
        next_tick = time.monotonic()
        last_tick = None
        while not self.stopped.is_set():
            now = time.monotonic()
            try:
                # A gap resets temporal history instead of pretending old frames are 100 ms apart.
                if last_tick is not None and now - last_tick > 0.15:
                    self.history.frames.clear()
                last_tick = now
                with self.lock:
                    frames = self.frames.copy()
                keys = ("front", "wrist", "state", *(("left", "right") if self.use_tactile else ()))
                stamps = [frames[k][1] for k in keys]
                if time.time() - min(stamps) > self.config["max_sensor_age_sec"]:
                    raise ValueError("Stale sensor stream")
                if max(stamps) - min(stamps) > self.config["max_sensor_skew_sec"]:
                    raise ValueError("Sensor capture timestamps are not synchronized")
                wrist = frames["wrist"][0]
                if list(wrist.shape) != self.config["wrist_shape"]:
                    raise ValueError("Wrist resolution differs from configured training resolution")
                data = {
                    "image": crop_front(frames["front"][0], self.conversion),
                    "wrist_image": wrist,
                    "state": frames["state"][0],
                    "prompt": self.config["prompt"],
                }
                if self.use_tactile:
                    data["tactile_marker_motion"] = self.history.append(frames["left"][0], frames["right"][0])
                sample = Sample(data, now, min(stamps))
                with self.lock:
                    self.sample, self.problem = sample, None
            except Exception as exc:
                self.history.frames.clear()
                with self.lock:
                    self.sample, self.problem = None, str(exc)
            next_tick += 0.1
            if next_tick < time.monotonic():
                next_tick = time.monotonic() + 0.1
            self.stopped.wait(max(0, next_tick - time.monotonic()))

    def snapshot(self):
        with self.lock:
            sample, problem, fatal = self.sample, self.problem, self.fatal
        if fatal:
            raise RuntimeError(fatal)
        if sample is None:
            raise ValueError(problem)
        if (
            time.time() - sample.oldest_capture > self.config["max_sensor_age_sec"]
            or time.monotonic() - sample.monotonic > 0.15
        ):
            raise ValueError("Observation sampler is stale")
        return sample

    def measured_state(self):
        with self.lock:
            state, stamp = self.frames["state"]
        if time.time() - stamp > self.config["max_sensor_age_sec"]:
            raise ValueError("Measured robot state is stale")
        return state

    def close(self):
        self.stopped.set()
        for thread in self.threads:
            thread.join(timeout=1)
        self.reader.close()
        self.node.destroy_node()
