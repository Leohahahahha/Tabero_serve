#!/usr/bin/env python3
"""ROS2 FR3 client. Defaults to shadow mode; --execute enables HTTP commands."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
from pathlib import Path
import signal
import threading
import time
from urllib.parse import urlparse

from core import TargetGuard
from core import action_distance_metrics
from core import action_to_http
from core import tactile_marker_summary
from core import validate_conversion
from core import validate_metadata
import numpy as np
from transport import RemotePolicy
from transport import RobotHttp


def load_config(path, *, robot_url=None, policy_url=None):
    config = json.loads(path.read_text())
    if robot_url is not None:
        config["robot_url"] = robot_url
    if policy_url is not None:
        config["policy_url"] = policy_url
    for key in (
        "http_timeout_sec",
        "policy_timeout_sec",
        "max_sensor_age_sec",
        "max_sensor_skew_sec",
        "enable_timeout_sec",
        "max_result_age_sec",
        "max_tick_delay_sec",
    ):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not isinstance(config["max_chunk_steps"], int) or not 1 <= config["max_chunk_steps"] <= 10:
        raise ValueError("max_chunk_steps must be an integer in [1,10]")
    if config["max_result_age_sec"] >= 0.1 * config["max_chunk_steps"]:
        raise ValueError("max_result_age_sec must be less than the executable chunk duration")
    if config["tactile_format"] not in ("packed", "shear"):
        raise ValueError("tactile_format must be packed or shear")
    if not isinstance(config["prompt"], str) or not config["prompt"].strip():
        raise ValueError("A nonempty training-compatible prompt is required")
    for key, schemes in (("robot_url", {"http", "https"}), ("policy_url", {"ws", "wss"})):
        parsed = urlparse(config[key])
        if parsed.scheme not in schemes or not parsed.netloc:
            raise ValueError(f"{key} must be an absolute {sorted(schemes)} URL")
    for name in ("front", "wrist", "tactile", "enable"):
        spec = config.get("qos", {}).get(name)
        if not isinstance(spec, dict):
            raise ValueError(f"qos.{name} must be configured")
        if spec.get("reliability") not in ("best_effort", "reliable"):
            raise ValueError(f"qos.{name}.reliability must be best_effort or reliable")
        if not isinstance(spec.get("depth"), int) or spec["depth"] <= 0:
            raise ValueError(f"qos.{name}.depth must be a positive integer")
    return config


def wait_ready(source, stopped):
    deadline = time.monotonic() + 30
    while not stopped.is_set():
        try:
            return source.snapshot()
        except ValueError:
            if time.monotonic() > deadline:
                raise
            stopped.wait(0.05)
    raise InterruptedError("Stopped during sensor startup")


def control_loop(source, policy, robot, config, *, execute, duration, stopped, log):
    """All commands belong to this one loop. Inference runs independently of the 10 Hz clock."""
    initial = wait_ready(source, stopped)
    logging.info("Warming up model (no robot commands); first compilation can take up to 120 seconds")
    policy.infer(initial, warmup=True)  # Discard warmup output and its stale observation.
    warmup_done = time.monotonic()
    if execute:
        logging.info("Waiting for fresh true heartbeat on %s (30 second timeout)", config["enable_topic"])
        while not source.enable_is_fresh(warmup_done):
            source.snapshot()
            if stopped.wait(0.02) or time.monotonic() - warmup_done > 30:
                raise InterruptedError("No fresh enable heartbeat after warmup")
        source.arm()
    guard = TargetGuard(config["limits"], source.measured_state())
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tabero-inference")
    future = None
    chunk = None
    started = time.monotonic()
    next_tick = started
    last_command_time = None
    last_width = 2 * guard.last[6]
    commands_sent = False
    count = 0
    chunk_id = 0
    active_chunk_id = None
    previous_prediction = None
    previous_chunk_id = None
    try:
        while not stopped.is_set() and time.monotonic() - started < duration:
            now = time.monotonic()
            if now < next_tick:
                stopped.wait(min(0.01, next_tick - now))
                # Deadman loss is noticed between ticks as well.
                if execute and not source.enable_is_fresh(warmup_done):
                    raise RuntimeError("Enable heartbeat released or expired")
                continue
            if now - next_tick > config["max_tick_delay_sec"]:
                raise RuntimeError("Control loop deadline missed; refusing catch-up commands")
            next_tick = now + 0.1
            sample = source.snapshot()
            if execute and not source.enable_is_fresh(warmup_done):
                raise RuntimeError("Enable heartbeat released or expired")
            if future is not None and future.done():
                candidate = future.result()
                future = None
                candidate_age = now - candidate.observation_time
                chunk_id += 1
                accepted = candidate_age <= config["max_result_age_sec"]
                log.write(
                    json.dumps(
                        {
                            "event": "inference_chunk",
                            "wall_time": time.time(),
                            "mode": "execute" if execute else "shadow",
                            "chunk_id": chunk_id,
                            "accepted": accepted,
                            "observation_age_sec_at_accept": candidate_age,
                            "observation_state": candidate.observation_state.tolist(),
                            "actions": candidate.actions.tolist(),
                        }
                    )
                    + "\n"
                )
                log.flush()
                if not accepted:
                    raise RuntimeError("Inference result too old; refusing stale targets")
                chunk = candidate
                active_chunk_id = chunk_id
            if future is None:
                future = pool.submit(policy.infer, sample)
            if chunk is None:
                if now - started > config["max_result_age_sec"] + 0.1:
                    raise RuntimeError("First runtime inference missed its deadline")
                continue
            raw, index = chunk.select(now, config["max_chunk_steps"])
            measured = source.measured_state()
            action0 = chunk.actions[0]
            dt = 0.1 if last_command_time is None else min(0.1, now - last_command_time)
            record = {
                "event": "control_tick",
                "wall_time": time.time(),
                "mode": "execute" if execute else "shadow",
                "chunk_id": active_chunk_id,
                "observation_age_sec": now - chunk.observation_time,
                "chunk_index": index,
                "observation_state": chunk.observation_state.tolist(),
                "measured": measured.tolist(),
                "action0": action0.tolist(),
                "prediction": raw.tolist(),
                "distances": {
                    "action0_vs_observation": action_distance_metrics(action0, chunk.observation_state),
                    "action0_vs_current": action_distance_metrics(action0, measured),
                    "selected_vs_current": action_distance_metrics(raw, measured),
                    "selected_vs_action0": action_distance_metrics(raw, action0),
                    "current_vs_observation": action_distance_metrics(measured, chunk.observation_state),
                    "selected_vs_previous_tick": (
                        None if previous_prediction is None else action_distance_metrics(raw, previous_prediction)
                    ),
                },
                "chunk_switched_since_previous_tick": (
                    previous_chunk_id is not None and active_chunk_id != previous_chunk_id
                ),
                "pose_sent": False,
                "gripper_sent": False,
            }
            if "tactile_marker_motion" in sample.data:
                record["tactile_marker"] = tactile_marker_summary(sample.data["tactile_marker_motion"])
            try:
                command = guard.prepare(raw, measured, dt)
                pose, width = action_to_http(command)
                record.update(limited_action=command.tolist(), pose_xyz_xyzw=pose.tolist(), gripper_width_m=width)
                if execute:
                    # Mark BEFORE sending: a timed-out request might already have moved the robot.
                    commands_sent = True
                    robot.command_pose(pose)
                    record["pose_sent"] = True
                    if abs(width - last_width) > 0.0001:
                        if stopped.is_set() or not source.enable_is_fresh(warmup_done):
                            raise RuntimeError("Enable lost between pose and gripper commands")
                        robot.command_width(width)
                        record["gripper_sent"] = True
                        last_width = width
                    guard.commit(command)
                    last_command_time = now
                else:
                    # Shadow mode has no simulated actuator; each prediction is checked against real state.
                    guard.commit(measured)
                record["ok"] = True
            except ValueError as exc:
                record.update(ok=False, error=str(exc))
                if execute:
                    raise
                logging.warning("Shadow target rejected: %s", exc)
                guard.commit(measured)
            except BaseException as exc:
                record.update(ok=False, error=str(exc))
                raise
            finally:
                log.write(json.dumps(record) + "\n")
                log.flush()
            previous_prediction = raw.copy()
            previous_chunk_id = active_chunk_id
            count += 1
            if count % 10 == 0:
                logging.info(
                    "%s: %d ticks, action[%d], age %.0f ms",
                    record["mode"],
                    count,
                    index,
                    1000 * record["observation_age_sec"],
                )
    finally:
        # No automatic reset, clearerr, gripper open/close, or retry on failure.
        if execute and commands_sent:
            try:
                state, _ = robot.read_state(config["max_sensor_age_sec"])
                pose, _ = action_to_http(state)
                robot.command_pose(pose)
                logging.warning("Sent best-effort measured-pose hold; this is not a hardware emergency stop")
            except Exception:
                logging.exception("Hold request failed; use the robot's hardware stop")
        policy.close()
        pool.shutdown(wait=False, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--conversion", type=Path, default=Path(__file__).with_name("tabero_conversion.json"))
    parser.add_argument("--robot-url", help="Override config robot_url, e.g. http://172.31.179.19:5000")
    parser.add_argument("--policy-url", help="Override config policy_url, e.g. ws://192.168.1.20:8000")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-tactile", action="store_true", help="Requires a separately trained RGB+state server")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--log", type=Path, required=True, help="New JSONL path; never overwrites an existing run")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    config = load_config(args.config, robot_url=args.robot_url, policy_url=args.policy_url)
    conversion_bytes = args.conversion.read_bytes()
    conversion = json.loads(conversion_bytes)
    validate_conversion(conversion)
    # Validate limits before importing ROS or contacting hardware.
    center = (np.array(config["limits"]["workspace_min"]) + config["limits"]["workspace_max"]) / 2
    TargetGuard(config["limits"], np.r_[center, [0.0, 0.0, 0.0, 0.02]])
    stopped = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopped.set())
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("x") as log:
        log.write(
            json.dumps(
                {
                    "event": "start",
                    "execute": args.execute,
                    "config": config,
                    "conversion_sha256": hashlib.sha256(conversion_bytes).hexdigest(),
                }
            )
            + "\n"
        )
        from observations import LiveObservations
        import rclpy

        rclpy.init(args=[])
        source = policy = robot = None
        try:
            policy = RemotePolicy(config["policy_url"], config["policy_timeout_sec"])
            validate_metadata(
                policy.metadata,
                use_tactile=not args.no_tactile,
                conversion_sha256=hashlib.sha256(conversion_bytes).hexdigest(),
            )
            log.write(
                json.dumps(
                    {
                        "event": "policy_metadata",
                        "config": policy.metadata.get("config"),
                        "checkpoint": policy.metadata.get("checkpoint"),
                        "norm_stats_sha256": policy.metadata.get("norm_stats_sha256"),
                        "conversion_sha256": policy.metadata.get("conversion_sha256"),
                        "action_horizon": policy.metadata.get("action_horizon"),
                        "dataset_fps": policy.metadata.get("dataset_fps"),
                        "use_tactile": policy.metadata.get("use_tactile"),
                    }
                )
                + "\n"
            )
            log.flush()
            source = LiveObservations(config, conversion, use_tactile=not args.no_tactile)
            robot = RobotHttp(config["robot_url"], config["http_timeout_sec"])
            control_loop(
                source, policy, robot, config, execute=args.execute, duration=args.seconds, stopped=stopped, log=log
            )
        except BaseException as exc:
            log.write(json.dumps({"event": "stopped", "reason": str(exc)}) + "\n")
            raise
        finally:
            if source is not None:
                source.close()
            if robot is not None:
                robot.close()
            if policy is not None:
                policy.close()
            rclpy.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
