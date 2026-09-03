"""Bounded network calls. Command requests are never retried."""

import time

from core import Chunk
from core import state_from_http
import numpy as np
from openpi_client import msgpack_numpy
import requests
from websockets.sync.client import connect


class RobotHttp:
    def __init__(self, url, timeout=0.15):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def post(self, route, payload=None):
        response = self.session.post(self.url + route, json=payload, timeout=self.timeout)
        response.raise_for_status()
        if "application/json" in response.headers.get("content-type", ""):
            result = response.json()
            if isinstance(result, dict) and (result.get("ok") is False or result.get("success") is False):
                raise RuntimeError(f"Robot rejected {route}: {result}")
            return result
        return response.text

    def read_state(self, max_age):
        start = time.monotonic()
        payload = self.post("/getstate")
        state = state_from_http(payload)
        # The attached server provides acquisition time in state.stamp.to_sec.
        stamp = float(payload["stamp"]["to_sec"])
        age = time.time() - stamp
        if not np.isfinite(stamp) or not -0.05 <= age <= max_age or time.monotonic() - start > max_age:
            raise ValueError(f"Robot state timestamp/latency is stale: age={age:.3f}s")
        return state, stamp

    def command_pose(self, pose):
        self.post("/pose", {"arr": np.asarray(pose).tolist()})

    def command_width(self, width):
        self.post("/move_gripper", {"gripper_width": float(width)})

    def close(self):
        self.session.close()


class RemotePolicy:
    def __init__(self, uri, timeout):
        self.timeout = timeout
        self.ws = connect(uri, compression=None, max_size=32 * 1024 * 1024, open_timeout=10, close_timeout=1)
        self.packer = msgpack_numpy.Packer()
        self.metadata = self.receive(timeout=10)

    def receive(self, timeout):
        result = self.ws.recv(timeout=timeout)
        if isinstance(result, str):
            raise RuntimeError(f"Policy server error: {result}")
        return msgpack_numpy.unpackb(result)

    def infer(self, sample, *, warmup=False):
        self.ws.send(self.packer.pack(sample.data))
        output = self.receive(timeout=120 if warmup else self.timeout)
        actions = np.asarray(output["actions"], dtype=np.float64)
        if actions.shape != (self.metadata["action_horizon"], 7) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite action chunk [{self.metadata['action_horizon']},7]")
        return Chunk(actions, sample.monotonic)

    def close(self):
        self.ws.close()
