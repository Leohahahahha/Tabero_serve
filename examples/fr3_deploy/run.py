#!/usr/bin/env python3
"""ROS2 FR3 client. Defaults to shadow mode; --execute enables HTTP commands."""

import argparse
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
        "control_period_sec",
    ):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
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
    """Run one blocking inference per observation and execute only action[0]."""
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
    started = time.monotonic()
    next_cycle = started
    last_command_time = None
    last_width = 2 * guard.last[6]
    commands_sent = False
    count = 0
    chunk_id = 0
    previous_prediction = None
    try:
        while not stopped.is_set() and time.monotonic() - started < duration:
            now = time.monotonic()
            if now < next_cycle:
                stopped.wait(min(0.01, next_cycle - now))
                if execute and not source.enable_is_fresh(warmup_done):
                    raise RuntimeError("Enable heartbeat released or expired")
                continue
            cycle_started = now
            next_cycle = cycle_started + config["control_period_sec"]
            sample = source.snapshot()
            if execute and not source.enable_is_fresh(warmup_done):
                raise RuntimeError("Enable heartbeat released or expired")

            inference_started = time.monotonic()
            chunk = policy.infer(sample)
            inference_finished = time.monotonic()
            inference_latency = inference_finished - inference_started
            result_age = inference_finished - chunk.observation_time
            chunk_id += 1
            accepted = result_age <= config["max_result_age_sec"]
            log.write(
                json.dumps(
                    {
                        "event": "inference_chunk",
                        "wall_time": time.time(),
                        "mode": "execute" if execute else "shadow",
                        "synchronous": True,
                        "chunk_id": chunk_id,
                        "accepted": accepted,
                        "inference_latency_sec": inference_latency,
                        "observation_age_sec_at_accept": result_age,
                        "selected_action_index": 0,
                        "observation_state": chunk.observation_state.tolist(),
                        "actions": chunk.actions.tolist(),
                    }
                )
                + "\n"
            )
            log.flush()
            if not accepted:
                raise RuntimeError("Inference result too old; refusing stale target")
            if stopped.is_set() or time.monotonic() - started >= duration:
                break
            if execute and not source.enable_is_fresh(warmup_done):
                raise RuntimeError("Enable heartbeat released or expired during inference")

            raw = chunk.actions[0]
            measured = source.measured_state()
            now = time.monotonic()
            dt = config["control_period_sec"] if last_command_time is None else now - last_command_time
            record = {
                "event": "control_tick",
                "wall_time": time.time(),
                "mode": "execute" if execute else "shadow",
                "synchronous": True,
                "chunk_id": chunk_id,
                "observation_age_sec": now - chunk.observation_time,
                "inference_latency_sec": inference_latency,
                "chunk_index": 0,
                "observation_state": chunk.observation_state.tolist(),
                "measured": measured.tolist(),
                "action0": raw.tolist(),
                "prediction": raw.tolist(),
                "distances": {
                    "action0_vs_observation": action_distance_metrics(raw, chunk.observation_state),
                    "action0_vs_current": action_distance_metrics(raw, measured),
                    "current_vs_observation": action_distance_metrics(measured, chunk.observation_state),
                    "action0_vs_previous_tick": (
                        None if previous_prediction is None else action_distance_metrics(raw, previous_prediction)
                    ),
                },
                "pose_sent": False,
                "gripper_sent": False,
            }
            if "tactile_marker_motion" in sample.data:
                record["tactile_marker"] = tactile_marker_summary(sample.data["tactile_marker_motion"])
            try:
                command = guard.prepare(raw, measured, dt)
                pose, width = action_to_http(command)
                record.update(
                    bounded_prediction=guard.last_bounded_target.tolist(),
                    limited_action=command.tolist(),
                    limits_applied=list(guard.last_limits_applied),
                    saturated=bool(guard.last_limits_applied),
                    pose_xyz_xyzw=pose.tolist(),
                    gripper_width_m=width,
                )
                record["distances"]["bounded_vs_current"] = action_distance_metrics(guard.last_bounded_target, measured)
                record["distances"]["limited_vs_current"] = action_distance_metrics(command, measured)
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
            count += 1
            if count % 5 == 0:
                logging.info(
                    "%s synchronous: %d inferences, action[0], latency %.0f ms, limits=%s",
                    record["mode"],
                    count,
                    1000 * inference_latency,
                    record["limits_applied"],
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
                    "control_mode": "synchronous_action0",
                    "prediction_limit_mode": "saturate",
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
