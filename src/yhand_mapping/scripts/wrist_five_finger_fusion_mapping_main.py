#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-node Manus wrist + five-finger fusion controller.

Fusion rule:
1. Map wrist quaternion to alpha/beta first.
2. Convert alpha/beta to 28-motor wrist delta.
3. Run five-finger delta models in de-wristed motor space.
4. Add wrist delta back, then apply finger/global offsets and limits.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from std_msgs.msg import Float64MultiArray

try:
    import torch
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError("This script requires torch for five-finger delta models") from exc

try:
    from devices_pkg.msg import YHandMsg
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError("Unable to import devices_pkg.msg.YHandMsg. Did you source install/setup.bash?") from exc

try:
    from replace_right_non_thumb_q_from_skeleton import compute_non_thumb_q16
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError("Unable to import compute_non_thumb_q16 from replace_right_non_thumb_q_from_skeleton.py") from exc

try:
    from five_finger_delta_mapping_main import (
        DEFAULT_FINGER_ORDER,
        FINGER_DEFAULTS_BY_HAND,
        MOTOR_CMD_OFFSET_MAP_DEFAULTS,
        MOTOR_FEEDBACK_LIMITS_RIGHT,
        Q_OFFSET_MAP_DEFAULTS,
        RIGHT_NON_THUMB_FINGERS,
        _load_delta_multihead_model,
        _mirror_motors_left_to_right,
        _parse_csv_ints,
        _parse_motor_cmd_offset_map_json,
        _parse_q_offset_map_json,
    )
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError("Unable to import five_finger_delta_mapping_main helpers") from exc

try:
    from wrist_angle_motor_control import (
        DEFAULT_OUTPUT_PUBLISH_RATE_HZ,
        DEFAULTS_PATH,
        MOTOR_COUNT,
        RIGHT_CMD_TOPIC,
        RIGHT_FEEDBACK_TOPIC,
        Motor,
        MotorFeedback,
        _compute_motor_targets_deg,
        _execute_all_zero,
        _format_nonzero_offsets,
        _load_defaults_from_py,
        _parse_motor_offsets_json,
    )
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError("Unable to import wrist_angle_motor_control helpers") from exc


from hand_motor_map import (
    WRIST_MOTOR_IDS_BY_HAND,
)

RAW_GLOVE_SKELETON_TOPIC = "/glove_joint_data_raw_skeleton"
GLOVE48_TOPIC = "/glove_joint_data"
DEFAULT_TARGET_HAND = "right"
DEFAULT_EXPECT_SIDE_CODE = 1.0
DEFAULT_CONTROL_HZ = 100.0
DEFAULT_STATUS_PERIOD_SEC = 1.0
DEFAULT_FEEDBACK_STALE_SEC = 0.5
DEFAULT_RAW_RECORD_STRIDE = 88
DEFAULT_Q_SEGMENT_LEN = 16
DEFAULT_ALPHA_LIMIT_DEG = 5.0
DEFAULT_BETA_LIMIT_DEG = 30.0
DEFAULT_WRIST_DEADBAND_DEG = 0.1
DEFAULT_MAX_STEP_ALPHA_DEG = 2.0
DEFAULT_MAX_STEP_BETA_DEG = 6.0
RIGHT_WRIST_MOTOR_IDS = WRIST_MOTOR_IDS_BY_HAND['right']


@dataclass
class FingerBundle:
    name: str
    q_indices: List[int]
    motor_ids: List[int]
    model: torch.nn.Module
    x_scaler: Any
    y_scaler: Any


def _now_sec() -> float:
    return time.monotonic()


def _normalize_quat_xyzw(q: Sequence[float]) -> Optional[np.ndarray]:
    arr = np.asarray(q, dtype=np.float64)
    if arr.shape != (4,) or not np.all(np.isfinite(arr)):
        return None
    n = float(np.linalg.norm(arr))
    if n < 1e-9:
        return None
    return arr / n


def _quat_conj_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _quat_to_matrix_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _extract_wrist_alpha_beta_deg(q_zero_xyzw: np.ndarray, q_now_xyzw: np.ndarray) -> Tuple[float, float]:
    q_rel = _quat_mul_xyzw(_quat_conj_xyzw(q_zero_xyzw), q_now_xyzw)
    q_rel = _normalize_quat_xyzw(q_rel)
    if q_rel is None:
        return 0.0, 0.0
    r = _quat_to_matrix_xyzw(q_rel)
    beta = math.asin(max(-1.0, min(1.0, -float(r[2, 0]))))
    alpha = math.atan2(-float(r[0, 1]), float(r[1, 1]))
    return math.degrees(alpha), math.degrees(beta)


def _apply_deadband(value: float, deadband: float) -> float:
    return 0.0 if abs(value) < deadband else value


def _clip_step(current: float, target: float, max_step: float) -> Tuple[float, bool]:
    delta = target - current
    if max_step <= 0.0 or abs(delta) <= max_step:
        return target, False
    return current + math.copysign(max_step, delta), True


def _load_config_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_python_defaults(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        return {}
    spec = importlib.util.spec_from_file_location("wrist_fusion_defaults", str(p))
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return {k: getattr(module, k) for k in dir(module) if k.isupper()}


def _nested_get(config: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _float_param(node: Node, name: str, default: float) -> float:
    node.declare_parameter(name, float(default))
    try:
        return float(node.get_parameter(name).value)
    except Exception:
        return float(default)


def _bool_param(node: Node, name: str, default: bool) -> bool:
    node.declare_parameter(name, bool(default))
    try:
        return bool(node.get_parameter(name).value)
    except Exception:
        return bool(default)


def _str_param(node: Node, name: str, default: str) -> str:
    node.declare_parameter(name, str(default))
    try:
        return str(node.get_parameter(name).value)
    except Exception:
        return str(default)


def _int_list_from_any(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return _parse_csv_ints(value)
    if isinstance(value, (list, tuple)):
        out: List[int] = []
        for item in value:
            try:
                out.append(int(item))
            except Exception:
                pass
        return out
    return []


class WristFiveFingerFusionMappingNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(f"wrist_five_finger_fusion_mapping_{args.hand}")
        self.args = args
        self.hand = args.hand.lower()
        if self.hand != "right":
            raise ValueError("This fusion script currently supports right hand only")

        self.config = _load_config_file(args.config)
        self.py_defaults = _load_python_defaults(args.defaults_path)
        self.wrist_defaults = _load_defaults_from_py(args.defaults_path)

        self.glove_raw_topic = _str_param(self, "glove_raw_topic", args.glove_raw_topic)
        self.glove48_topic = _str_param(self, "glove48_topic", args.glove48_topic)
        self.cmd_topic = _str_param(self, "cmd_topic", args.cmd_topic)
        self.feedback_topic = _str_param(self, "feedback_topic", args.feedback_topic)
        self.output_publish_rate_hz = _float_param(self, "output_publish_rate_hz", args.output_publish_rate_hz)
        self.status_period_sec = _float_param(self, "status_period_sec", args.status_period_sec)
        self.feedback_stale_sec = _float_param(self, "feedback_stale_sec", args.feedback_stale_sec)
        self.dry_run = _bool_param(self, "dry_run", args.dry_run)
        self.expect_side_code = _float_param(self, "right_side_code", args.right_side_code)
        self.raw_record_stride = int(_float_param(self, "raw_record_stride", args.raw_record_stride))

        self.alpha_limit_deg = _float_param(self, "alpha_limit_deg", args.alpha_limit_deg)
        self.beta_limit_deg = _float_param(self, "beta_limit_deg", args.beta_limit_deg)
        self.alpha_scale = _float_param(self, "alpha_scale", args.alpha_scale)
        self.beta_scale = _float_param(self, "beta_scale", args.beta_scale)
        self.alpha_offset_deg = _float_param(self, "alpha_offset_deg", args.alpha_offset_deg)
        self.beta_offset_deg = _float_param(self, "beta_offset_deg", args.beta_offset_deg)
        self.wrist_deadband_deg = _float_param(self, "wrist_deadband_deg", args.wrist_deadband_deg)
        self.max_step_alpha_deg = _float_param(self, "max_step_alpha_deg", args.max_step_alpha_deg)
        self.max_step_beta_deg = _float_param(self, "max_step_beta_deg", args.max_step_beta_deg)

        self.finger_deadband_deg = _float_param(self, "finger_deadband_deg", args.finger_deadband_deg)
        self.finger_max_delta_deg = _float_param(self, "finger_max_delta_deg", args.finger_max_delta_deg)
        self.skip_wrist_motor_finger_inference = _bool_param(
            self, "skip_wrist_motor_finger_inference", args.skip_wrist_motor_finger_inference
        )
        self.publish_idle_without_feedback = _bool_param(
            self, "publish_idle_without_feedback", args.publish_idle_without_feedback
        )
        self.use_right_non_thumb_raw_skeleton_q = _bool_param(
            self, "right_non_thumb_q_from_raw_skeleton", args.right_non_thumb_q_from_raw_skeleton
        )

        device_arg = str(args.device or "auto").strip().lower()
        self.device = "cuda" if (device_arg in ("", "auto") and torch.cuda.is_available()) else ("cpu" if device_arg in ("", "auto") else device_arg)
        self.finger_order = list(DEFAULT_FINGER_ORDER if not args.finger_order else [x.strip() for x in args.finger_order.split(",") if x.strip()])
        self.finger_bundles = self._load_finger_bundles()
        self.finger_motor_set = {int(mid) for b in self.finger_bundles.values() for mid in b.motor_ids}

        self.motor_offsets_deg = self._load_motor_offsets(args.motor_offsets_json)
        self.motor_cmd_offset_map = self._load_motor_cmd_offsets(args.motor_cmd_offset_map_json)
        self.q_offset_map = self._load_q_offsets(args.q_offset_map_json)
        self.motor_limits = self._load_motor_limits()

        self.default_matrix = self._build_default_matrix()
        self.idle_matrix = self._build_idle_tension_matrix()
        self.command_matrix = [row[:] for row in self.idle_matrix]
        self._motor_positions: List[Optional[float]] = [None] * MOTOR_COUNT
        self._motor_torques: List[Optional[float]] = [None] * MOTOR_COUNT
        self.feedback_count = 0
        self.last_feedback_time: Optional[float] = None
        self.manual_matrix_mode = False

        self.raw_lock = threading.Lock()
        self.latest_wrist_quat: Optional[np.ndarray] = None
        self.latest_raw_chunk: Optional[List[float]] = None
        self.latest_raw_msg_time: Optional[float] = None
        self.latest_raw_seq = 0
        self.processed_raw_seq = 0
        self.wrist_zero_quat: Optional[np.ndarray] = None
        self.last_side_codes: List[float] = []
        self.raw_cb_times: List[float] = []
        self.valid_raw_times: List[float] = []
        self.bad_raw_packets = 0

        self.glove_lock = threading.Lock()
        self.latest_glove20: Optional[np.ndarray] = None
        self.latest_right_non_thumb_q16: Optional[Dict[str, float]] = None
        self.latest_glove48_time: Optional[float] = None
        self.glove48_seq = 0
        self.processed_glove_seq = 0
        self.glove48_cb_times: List[float] = []
        self.bad_glove48_packets = 0

        self.current_alpha_deg = 0.0
        self.current_beta_deg = 0.0
        self.raw_alpha_deg = 0.0
        self.raw_beta_deg = 0.0
        self.wrist_zero_targets = np.asarray(_compute_motor_targets_deg(0.0, 0.0)[2], dtype=np.float32)
        self.last_wrist_delta = np.zeros(MOTOR_COUNT, dtype=np.float32)
        self.limit_hits = 0
        self.smooth_steps = 0

        self.prev_q_by_finger: Dict[str, Optional[np.ndarray]] = {name: None for name in self.finger_order}
        self.prev_decoupled_feedback: Optional[np.ndarray] = None
        self.has_last_finger_solution = False
        self.last_applied_wrist_delta = np.zeros(MOTOR_COUNT, dtype=np.float32)
        self.last_cmd_positions = np.asarray([row[0] for row in self.idle_matrix], dtype=np.float32)
        self.finger_cmd_counts: Dict[str, int] = {name: 0 for name in self.finger_order}
        self.finger_skips: Dict[str, int] = {name: 0 for name in self.finger_order}

        self.cmd_count = 0
        self.cmd_times: List[float] = []
        self.loop_times: List[float] = []
        self.late_loops = 0
        self.max_late_ms = 0.0
        self.last_status_time = _now_sec()

        cmd_qos = QoSProfile(depth=10)
        self.cmd_pub = self.create_publisher(YHandMsg, self.cmd_topic, cmd_qos)
        self.feedback_sub = self.create_subscription(YHandMsg, self.feedback_topic, self._feedback_cb, QoSProfile(depth=10))

        raw_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.raw_sub = self.create_subscription(Float64MultiArray, self.glove_raw_topic, self._raw_skeleton_cb, raw_qos)
        self.glove48_sub = self.create_subscription(Float64MultiArray, self.glove48_topic, self._glove48_cb, raw_qos)

        period = 1.0 / max(1.0, self.output_publish_rate_hz)
        self.publish_period_sec = period
        self.control_timer = self.create_timer(period, self._control_timer_cb)
        self.status_timer = self.create_timer(max(0.1, self.status_period_sec), self._status_timer_cb)

        self.get_logger().info(
            "[wrist_five_finger_fusion_mapping] started. "
            f"raw_topic={self.glove_raw_topic}, glove48_topic={self.glove48_topic}, "
            f"cmd_topic={self.cmd_topic}, feedback_topic={self.feedback_topic}, "
            f"rate={self.output_publish_rate_hz:.1f}Hz, dry_run={self.dry_run}, "
            f"models={list(self.finger_bundles.keys())}, device={self.device}"
        )
        if np.any(np.abs(self.motor_offsets_deg) > 1e-6):
            self.get_logger().info("global motor_offsets_deg: " + _format_nonzero_offsets(self.motor_offsets_deg))

    def _load_motor_offsets(self, cli_json: str) -> np.ndarray:
        if cli_json:
            values = _parse_motor_offsets_json(cli_json)
            return np.asarray(values, dtype=np.float32)
        defaults_value = self.wrist_defaults.get("MOTOR_OFFSETS_DEG") or self.py_defaults.get("MOTOR_OFFSETS_DEG")
        if defaults_value is not None:
            arr = np.asarray(list(defaults_value), dtype=np.float32)
            if arr.size == MOTOR_COUNT:
                return arr
        cfg_value = _nested_get(self.config, ["motor_offsets_deg"], None)
        if cfg_value is not None:
            arr = np.asarray(list(cfg_value), dtype=np.float32)
            if arr.size == MOTOR_COUNT:
                return arr
        return np.zeros(MOTOR_COUNT, dtype=np.float32)

    def _load_motor_cmd_offsets(self, cli_json: str) -> Dict[int, float]:
        merged: Dict[int, float] = dict(MOTOR_CMD_OFFSET_MAP_DEFAULTS.get(self.hand, {}))
        defaults_value = self.py_defaults.get("MOTOR_CMD_OFFSET_MAP")
        if isinstance(defaults_value, dict):
            hand_map = defaults_value.get(self.hand, defaults_value)
            if isinstance(hand_map, dict):
                for k, v in hand_map.items():
                    try:
                        merged[int(k)] = float(v)
                    except Exception:
                        pass
        if cli_json:
            parsed = _parse_motor_cmd_offset_map_json(cli_json, "--motor-cmd-offset-map-json")
            for k, v in parsed.get(self.hand, {}).items():
                merged[int(k)] = float(v)
        return merged

    def _load_q_offsets(self, cli_json: str) -> Dict[int, float]:
        merged: Dict[int, float] = {int(k): float(v) for k, v in Q_OFFSET_MAP_DEFAULTS.get(self.hand, {}).items()}
        defaults_value = self.py_defaults.get("Q_OFFSET_MAP")
        if isinstance(defaults_value, dict):
            hand_map = defaults_value.get(self.hand, defaults_value)
            if isinstance(hand_map, dict):
                for key, value in hand_map.items():
                    try:
                        merged[int(key)] = float(value)
                    except Exception:
                        pass
        if cli_json:
            parsed = _parse_q_offset_map_json(cli_json, "--q-offset-map-json")
            for key, value in parsed.get(self.hand, {}).items():
                merged[int(key)] = float(value)
        return merged

    def _load_motor_limits(self) -> Dict[int, Tuple[float, float]]:
        limits: Dict[int, Tuple[float, float]] = dict(MOTOR_FEEDBACK_LIMITS_RIGHT)
        defaults_value = self.py_defaults.get("MOTOR_LIMITS_DEG")
        if isinstance(defaults_value, dict):
            for key, value in defaults_value.items():
                try:
                    mid = int(key)
                    lo, hi = value
                    limits[mid] = (float(lo), float(hi))
                except Exception:
                    pass
        return limits

    def _load_finger_bundles(self) -> Dict[str, FingerBundle]:
        bundles: Dict[str, FingerBundle] = {}
        default_by_finger = FINGER_DEFAULTS_BY_HAND.get(self.hand, {})
        base_dir = Path(__file__).resolve().parent
        for finger in self.finger_order:
            cfg = dict(default_by_finger.get(finger, {}))
            enabled = str(cfg.get("enable_default", "true")).strip().lower() in ("1", "true", "yes", "on")
            if not enabled:
                continue
            model_override = str(getattr(self.args, f"{finger}_model", "") or "").strip()
            if model_override:
                model_path = Path(model_override).expanduser()
                if not model_path.is_absolute():
                    model_path = (base_dir / model_path).resolve()
            else:
                model_path = base_dir.joinpath(*cfg["model_ckpt_rel_parts"])
            q_indices = _int_list_from_any(getattr(self.args, f"{finger}_q_indices", None) or cfg.get("q_indices_csv"))
            motor_ids = _int_list_from_any(getattr(self.args, f"{finger}_motor_ids", None) or cfg.get("motor_ids_csv"))
            mirror = str(cfg.get("mirror_right_from_left_default", "false")).strip().lower() in ("1", "true", "yes", "on")
            if self.hand == "right" and mirror:
                motor_ids = _mirror_motors_left_to_right(motor_ids)
            if not q_indices or not motor_ids:
                self.get_logger().warn(f"skip finger {finger}: incomplete q_indices/motor_ids")
                continue
            bundle = _load_delta_multihead_model(Path(model_path), self.device)
            if len(motor_ids) != int(bundle["m_dim"]):
                raise RuntimeError(f"{finger}: len(motor_ids)={len(motor_ids)} != m_dim={bundle['m_dim']}")
            if len(q_indices) != int(bundle["q_dim"]):
                raise RuntimeError(f"{finger}: len(q_indices)={len(q_indices)} != q_dim={bundle['q_dim']}")
            bundles[finger] = FingerBundle(
                name=finger,
                q_indices=list(q_indices),
                motor_ids=list(motor_ids),
                model=bundle["model"],
                x_scaler=bundle["x_scaler"],
                y_scaler=bundle["y_scaler"],
            )
            self.get_logger().info(
                f"finger={finger}, ckpt={model_path}, motor_ids={motor_ids}, q_indices={q_indices}"
            )
        if not bundles:
            raise RuntimeError("No five-finger delta models were loaded")
        return bundles

    def _build_default_matrix(self) -> List[List[float]]:
        matrix: List[List[float]] = []
        for i in range(MOTOR_COUNT):
            matrix.append([0.0, 0.0, float(self.wrist_defaults["tor"][i]), 0.0, 0.0])
        return matrix

    def _build_idle_tension_matrix(self) -> List[List[float]]:
        return [row[:] for row in self.default_matrix]

    def _build_feedback_abs_28(self) -> Optional[np.ndarray]:
        if self.feedback_count <= 0:
            return None
        out = np.zeros(MOTOR_COUNT, dtype=np.float32)
        for idx, value in enumerate(self._motor_positions):
            if value is None or not math.isfinite(float(value)):
                return None
            out[idx] = float(value)
        return out

    def is_feedback_stale(self, stale_sec: Optional[float] = None) -> bool:
        if self.last_feedback_time is None:
            return True
        limit = self.feedback_stale_sec if stale_sec is None else stale_sec
        return (_now_sec() - self.last_feedback_time) > float(limit)

    def set_watchdog_fallback_enabled(self, enabled: bool) -> None:
        # Compatibility hook used by _execute_all_zero.
        return None

    def enable_manual_matrix_mode(self, enabled: bool) -> None:
        self.manual_matrix_mode = bool(enabled)

    def all_motor_positions(self) -> List[Optional[float]]:
        return list(self._motor_positions)

    def set_idle_tension(self) -> None:
        self.command_matrix = self._build_idle_tension_matrix()

    def _build_all_zero_matrix(self) -> List[List[float]]:
        matrix: List[List[float]] = []
        for i in range(MOTOR_COUNT):
            matrix.append([
                0.0,
                0.0,
                float(self.wrist_defaults["tor"][i]),
                float(self.wrist_defaults["kp"][i]),
                float(self.wrist_defaults["kd"][i]),
            ])
        return matrix

    def set_all_zero_control(self) -> None:
        self.command_matrix = self._build_all_zero_matrix()

    def set_all_zero_control_with_overrides(self, overrides: Dict[int, Sequence[float]]) -> None:
        matrix = self._build_all_zero_matrix()
        for motor_id, row in overrides.items():
            idx = int(motor_id) - 1
            if 0 <= idx < MOTOR_COUNT and len(row) >= 5:
                matrix[idx] = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])]
        self.command_matrix = matrix

    def set_all_motor_position_targets(self, positions: Sequence[float]) -> None:
        if len(positions) != MOTOR_COUNT:
            raise ValueError(f"target_positions length must be {MOTOR_COUNT}, got {len(positions)}")
        matrix: List[List[float]] = []
        for i, pos in enumerate(positions[:MOTOR_COUNT]):
            matrix.append([
                float(pos),
                float(self.wrist_defaults["vel"][i]),
                float(self.wrist_defaults["tor_active"][i]),
                float(self.wrist_defaults["kp"][i]),
                float(self.wrist_defaults["kd"][i]),
            ])
        self.command_matrix = matrix

    def publish_current_command(self) -> None:
        self._publish_matrix(self.command_matrix)

    def get_publish_count(self) -> int:
        return self.cmd_count

    def _feedback_cb(self, msg: YHandMsg) -> None:
        now = _now_sec()
        self.feedback_count += 1
        self.last_feedback_time = now
        motors = getattr(msg, "motors", [])
        for i, motor in enumerate(motors[:MOTOR_COUNT]):
            try:
                self._motor_positions[i] = float(motor.pos)
                self._motor_torques[i] = float(motor.tor)
            except Exception:
                pass

    def _raw_skeleton_cb(self, msg: Float64MultiArray) -> None:
        now = _now_sec()
        self.raw_cb_times.append(now)
        del self.raw_cb_times[:-512]
        data = list(msg.data)
        q, raw_chunk, side_codes = self._extract_right_wrist_quat(data)
        self.last_side_codes = side_codes
        if q is None:
            self.bad_raw_packets += 1
            return
        with self.raw_lock:
            self.latest_wrist_quat = q
            self.latest_raw_chunk = list(raw_chunk) if raw_chunk is not None else None
            self.latest_raw_msg_time = now
            self.latest_raw_seq += 1
        self.valid_raw_times.append(now)
        del self.valid_raw_times[:-512]

    def _extract_right_wrist_quat(self, data: Sequence[float]) -> Tuple[Optional[np.ndarray], Optional[List[float]], List[float]]:
        side_codes: List[float] = []
        best: Optional[np.ndarray] = None
        best_chunk: Optional[List[float]] = None
        stride = max(8, int(self.raw_record_stride))
        for base in range(0, len(data) - 7, stride):
            try:
                side = float(data[base])
                side_codes.append(side)
                if abs(side - self.expect_side_code) > 0.25:
                    continue
                # Raw skeleton record: side code at +0, wrist qx/qy/qz/qw at +4:+8.
                q = _normalize_quat_xyzw([data[base + 4], data[base + 5], data[base + 6], data[base + 7]])
                if q is not None:
                    best = q
                    best_chunk = list(data[base : base + stride])
                    break
            except Exception:
                continue
        if best is None and len(data) >= 8:
            q = _normalize_quat_xyzw([data[4], data[5], data[6], data[7]])
            if q is not None:
                best = q
                best_chunk = list(data[:stride]) if len(data) >= stride else list(data)
        return best, best_chunk, side_codes

    def _glove48_cb(self, msg: Float64MultiArray) -> None:
        now = _now_sec()
        self.glove48_cb_times.append(now)
        del self.glove48_cb_times[:-512]
        try:
            q20 = self._extract_q20_from_glove48(msg.data)
        except Exception:
            self.bad_glove48_packets += 1
            return
        if q20 is None:
            self.bad_glove48_packets += 1
            return
        with self.glove_lock:
            self.latest_glove20 = q20
            self.latest_glove48_time = now
            self.glove48_seq += 1

    def _extract_q20_from_glove48(self, data: Sequence[float]) -> Optional[np.ndarray]:
        arr = np.asarray(list(data), dtype=np.float32)
        if arr.size < 48 or not np.all(np.isfinite(arr[:48])):
            return None
        return arr[28:48].astype(np.float32)

    def _control_timer_cb(self) -> None:
        start = _now_sec()
        if self.manual_matrix_mode:
            if not self.dry_run:
                self.publish_current_command()
            return
        self.loop_times.append(start)
        del self.loop_times[:-512]

        raw_seq, q_now, raw_chunk, raw_time = self._read_latest_wrist_quat()
        new_wrist = bool(q_now is not None and raw_seq != self.processed_raw_seq)
        if q_now is not None and new_wrist:
            if self.wrist_zero_quat is None:
                self.wrist_zero_quat = q_now.copy()
                self.get_logger().info(f"Zero wrist frame captured, side_codes={self.last_side_codes}")
            self._update_wrist_angle(q_now, raw_seq)
            self._update_non_thumb_q16(raw_chunk)

        wrist_delta = self._compute_wrist_delta_28(self.current_alpha_deg, self.current_beta_deg)
        self.last_wrist_delta = wrist_delta

        fb_abs = self._build_feedback_abs_28()
        glove_seq, glove20, _glove_time = self._read_latest_glove20()
        new_glove = bool(glove20 is not None and glove_seq != self.processed_glove_seq)
        if fb_abs is None and not self.publish_idle_without_feedback:
            return

        if fb_abs is None or self.is_feedback_stale():
            # Before feedback is valid, only keep idle tension. Do not splice wrist delta into fingers.
            cmd_pos = np.asarray([row[0] for row in self.idle_matrix], dtype=np.float32)
            self.has_last_finger_solution = False
            mode = "idle_no_feedback"
        elif glove20 is None:
            # Reference logic: wrist motors follow wrist mapping; finger motors hold absolute feedback.
            cmd_pos = fb_abs.copy()
            for mid in RIGHT_WRIST_MOTOR_IDS:
                idx = int(mid) - 1
                cmd_pos[idx] = float(wrist_delta[idx] + self.motor_offsets_deg[idx])
            mode = "wrist_only_no_glove48"
        elif new_glove or (not self.has_last_finger_solution):
            cmd_pos = self._fuse_fingers_and_wrist(glove20, fb_abs, wrist_delta)
            self.processed_glove_seq = glove_seq
            self.has_last_finger_solution = True
            mode = "fusion"
        elif new_wrist:
            cmd_pos = self._shift_last_command_for_wrist_only(wrist_delta)
            mode = "wrist_shift_only"
        else:
            cmd_pos = self.last_cmd_positions.copy()
            mode = "hold_last"

        self.last_cmd_positions = cmd_pos.copy()
        self.last_applied_wrist_delta = wrist_delta.copy()
        self.set_all_motor_position_targets(cmd_pos.tolist())
        if not self.dry_run:
            self.publish_current_command()
        elapsed = _now_sec() - start
        period = 1.0 / max(1.0, self.output_publish_rate_hz)
        if elapsed > period:
            self.late_loops += 1
            self.max_late_ms = max(self.max_late_ms, elapsed * 1000.0)
        self._last_mode = mode

    def _read_latest_wrist_quat(self) -> Tuple[int, Optional[np.ndarray], Optional[List[float]], Optional[float]]:
        with self.raw_lock:
            if self.latest_wrist_quat is None:
                return self.latest_raw_seq, None, None, self.latest_raw_msg_time
            chunk = list(self.latest_raw_chunk) if self.latest_raw_chunk is not None else None
            return self.latest_raw_seq, self.latest_wrist_quat.copy(), chunk, self.latest_raw_msg_time

    def _read_latest_glove20(self) -> Tuple[int, Optional[np.ndarray], Optional[float]]:
        with self.glove_lock:
            if self.latest_glove20 is None:
                return self.glove48_seq, None, self.latest_glove48_time
            return self.glove48_seq, self.latest_glove20.copy(), self.latest_glove48_time

    def _update_non_thumb_q16(self, raw_chunk: Optional[Sequence[float]]) -> None:
        if raw_chunk is None:
            return
        try:
            q16_map = compute_non_thumb_q16(list(raw_chunk))
            cleaned: Dict[str, float] = {}
            for finger_name in RIGHT_NON_THUMB_FINGERS:
                for i in range(4):
                    key = f"{finger_name}_q_{i}"
                    if key in q16_map:
                        cleaned[key] = float(q16_map[key])
            if cleaned:
                self.latest_right_non_thumb_q16 = cleaned
        except Exception:
            return

    def _update_wrist_angle(self, q_now: np.ndarray, raw_seq: int) -> None:
        if self.wrist_zero_quat is None:
            return
        alpha_raw, beta_raw = _extract_wrist_alpha_beta_deg(self.wrist_zero_quat, q_now)
        self.raw_alpha_deg = alpha_raw
        self.raw_beta_deg = beta_raw
        alpha = _apply_deadband(alpha_raw, self.wrist_deadband_deg) * self.alpha_scale + self.alpha_offset_deg
        beta = _apply_deadband(beta_raw, self.wrist_deadband_deg) * self.beta_scale + self.beta_offset_deg
        alpha = max(-self.alpha_limit_deg, min(self.alpha_limit_deg, alpha))
        beta = max(-self.beta_limit_deg, min(self.beta_limit_deg, beta))
        if abs(alpha) >= self.alpha_limit_deg - 1e-6 or abs(beta) >= self.beta_limit_deg - 1e-6:
            self.limit_hits += 1
        next_alpha, smooth_a = _clip_step(self.current_alpha_deg, alpha, self.max_step_alpha_deg)
        next_beta, smooth_b = _clip_step(self.current_beta_deg, beta, self.max_step_beta_deg)
        if smooth_a or smooth_b:
            self.smooth_steps += 1
        self.current_alpha_deg = next_alpha
        self.current_beta_deg = next_beta
        self.processed_raw_seq = raw_seq

    def _compute_wrist_delta_28(self, alpha_deg: float, beta_deg: float) -> np.ndarray:
        targets = np.asarray(_compute_motor_targets_deg(float(alpha_deg), float(beta_deg))[2], dtype=np.float32)
        return targets - self.wrist_zero_targets

    def _fuse_fingers_and_wrist(self, glove20: np.ndarray, fb_abs: np.ndarray, wrist_delta: np.ndarray) -> np.ndarray:
        dec_cur = fb_abs - wrist_delta
        prev_dec = dec_cur if self.prev_decoupled_feedback is None else self.prev_decoupled_feedback
        cmd_pos = np.zeros((MOTOR_COUNT,), dtype=np.float32)
        delta_shift = wrist_delta - self.last_applied_wrist_delta

        # 4 wrist motors: use wrist mapping directly. Global offsets are added below.
        for mid in RIGHT_WRIST_MOTOR_IDS:
            idx = int(mid) - 1
            cmd_pos[idx] = float(wrist_delta[idx])

        for finger_name, bundle in self.finger_bundles.items():
            try:
                if self.use_right_non_thumb_raw_skeleton_q and finger_name in RIGHT_NON_THUMB_FINGERS:
                    if self.latest_right_non_thumb_q16 is None:
                        self._hold_last_cmd_for_finger_with_wrist_shift(cmd_pos, bundle.motor_ids, delta_shift)
                        self.finger_skips[finger_name] += 1
                        continue
                    vals = [self.latest_right_non_thumb_q16[f"{finger_name}_q_{i}"] for i in range(len(bundle.q_indices))]
                    q_now_raw = np.asarray(vals, dtype=np.float32)
                else:
                    q_now_raw = glove20[bundle.q_indices].astype(np.float32)
                q_now = self._apply_q_offsets(q_now_raw, bundle.q_indices)
            except Exception:
                self._hold_last_cmd_for_finger_with_wrist_shift(cmd_pos, bundle.motor_ids, delta_shift)
                self.finger_skips[finger_name] += 1
                continue

            q_prev = self.prev_q_by_finger.get(finger_name)
            if q_prev is None:
                q_prev = q_now.copy()
            dq = q_now - q_prev

            m_prev = np.asarray([float(prev_dec[int(mid) - 1]) for mid in bundle.motor_ids], dtype=np.float32)
            x = np.concatenate([q_now, q_prev.astype(np.float32), m_prev, dq.astype(np.float32)], axis=0).reshape(1, -1)
            try:
                x_norm = bundle.x_scaler.transform(x)
                x_tensor = torch.as_tensor(x_norm, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    pred_norm = bundle.model(x_tensor).detach().cpu().numpy()
                pred_delta = bundle.y_scaler.inverse(pred_norm)[0].astype(np.float32)
            except Exception as exc:
                self._hold_last_cmd_for_finger_with_wrist_shift(cmd_pos, bundle.motor_ids, delta_shift)
                self.finger_skips[finger_name] += 1
                self.get_logger().warn(f"finger {finger_name} inference failed: {exc}")
                continue

            if self.finger_max_delta_deg > 0.0:
                pred_delta = np.clip(pred_delta, -self.finger_max_delta_deg, self.finger_max_delta_deg)
            m_tilde_cmd = m_prev + pred_delta

            for out_idx, motor_id in enumerate(bundle.motor_ids):
                mid = int(motor_id)
                idx = mid - 1
                pos_abs = float(m_tilde_cmd[out_idx] + wrist_delta[idx])
                pos_abs += float(self.motor_cmd_offset_map.get(mid, 0.0))
                lim = self.motor_limits.get(mid)
                if lim is not None:
                    pos_abs = float(np.clip(pos_abs, lim[0], lim[1]))
                cmd_pos[idx] = pos_abs

            self.prev_q_by_finger[finger_name] = q_now.copy()
            self.finger_cmd_counts[finger_name] += 1

        # Final global motor offsets (28), exactly like the reference fusion logic.
        for i in range(MOTOR_COUNT):
            cmd_pos[i] = float(cmd_pos[i] + float(self.motor_offsets_deg[i]))

        self.prev_decoupled_feedback = dec_cur.copy()
        return cmd_pos

    def _shift_last_command_for_wrist_only(self, wrist_delta: np.ndarray) -> np.ndarray:
        cmd_pos = self.last_cmd_positions.copy()
        delta_shift = wrist_delta - self.last_applied_wrist_delta
        for mid in RIGHT_WRIST_MOTOR_IDS:
            idx = int(mid) - 1
            cmd_pos[idx] = float(wrist_delta[idx] + self.motor_offsets_deg[idx])
        for mid in self.finger_motor_set:
            if int(mid) in RIGHT_WRIST_MOTOR_IDS:
                continue
            idx = int(mid) - 1
            pos_abs = float(cmd_pos[idx] + delta_shift[idx])
            lim = self.motor_limits.get(int(mid))
            if lim is not None:
                pos_abs = float(np.clip(pos_abs, lim[0], lim[1]))
            cmd_pos[idx] = pos_abs
        return cmd_pos

    def _hold_last_cmd_for_finger_with_wrist_shift(
        self,
        cmd_pos: np.ndarray,
        motor_ids: Sequence[int],
        delta_shift: np.ndarray,
    ) -> None:
        for mid in motor_ids:
            idx = int(mid) - 1
            pos_abs = float(self.last_cmd_positions[idx] + delta_shift[idx])
            lim = self.motor_limits.get(int(mid))
            if lim is not None:
                pos_abs = float(np.clip(pos_abs, lim[0], lim[1]))
            cmd_pos[idx] = pos_abs

    def _apply_q_offsets(self, q_now: np.ndarray, q_indices: Sequence[int]) -> np.ndarray:
        out = q_now.astype(np.float32).copy()
        for i, qi in enumerate(q_indices):
            out[i] = float(out[i] + float(self.q_offset_map.get(int(qi), 0.0)))
        return out

    def _apply_global_offsets_and_limits(self, cmd_pos: np.ndarray) -> np.ndarray:
        out = np.asarray(cmd_pos, dtype=np.float32).copy()
        for mid, (lo, hi) in self.motor_limits.items():
            idx = int(mid) - 1
            if 0 <= idx < MOTOR_COUNT:
                out[idx] = max(float(lo), min(float(hi), float(out[idx])))
        out += self.motor_offsets_deg
        return out

    def _publish_matrix(self, matrix: Sequence[Sequence[float]]) -> None:
        msg = YHandMsg()
        msg.motors = []
        for i in range(MOTOR_COUNT):
            row = matrix[i]
            msg.motors.append(
                Motor(
                    pos=float(row[0]),
                    vel=float(row[1]),
                    tor=float(row[2]),
                    kp=float(row[3]),
                    kd=float(row[4]),
                )
            )
        msg.pretightenforce.flag = False
        msg.pretightenforce.torque = 0.0
        msg.pretightenforce.ms = 0
        self.cmd_pub.publish(msg)
        now = _now_sec()
        self.cmd_count += 1
        self.cmd_times.append(now)
        del self.cmd_times[:-512]

    def _rate(self, times: Sequence[float]) -> float:
        if len(times) < 2:
            return 0.0
        dt = float(times[-1] - times[0])
        if dt <= 1e-9:
            return 0.0
        return float(len(times) - 1) / dt

    def _status_timer_cb(self) -> None:
        raw_age = -1.0 if self.latest_raw_msg_time is None else _now_sec() - self.latest_raw_msg_time
        glove_age = -1.0 if self.latest_glove48_time is None else _now_sec() - self.latest_glove48_time
        fb_age = -1.0 if self.last_feedback_time is None else _now_sec() - self.last_feedback_time
        self.get_logger().info(
            "fusion: "
            f"raw_alpha={self.raw_alpha_deg:+.3f}deg, raw_beta={self.raw_beta_deg:+.3f}deg, "
            f"cmd_alpha={self.current_alpha_deg:+.3f}deg, cmd_beta={self.current_beta_deg:+.3f}deg, "
            f"raw_cb_hz={self._rate(self.raw_cb_times):.1f}, raw_valid_hz={self._rate(self.valid_raw_times):.1f}, "
            f"glove48_hz={self._rate(self.glove48_cb_times):.1f}, cmd_hz={self._rate(self.cmd_times):.1f}, "
            f"fb_count={self.feedback_count}, fb_age={fb_age:.3f}s, raw_age={raw_age:.3f}s, glove_age={glove_age:.3f}s, "
            f"cmd_count={self.cmd_count}, dry_run={self.dry_run}, mode={getattr(self, '_last_mode', 'init')}, "
            f"limit_hits={self.limit_hits}, smooth_steps={self.smooth_steps}, bad_raw={self.bad_raw_packets}, "
            f"bad_glove48={self.bad_glove48_packets}, late_loops={self.late_loops}, max_late_ms={self.max_late_ms:.2f}, "
            f"side_codes={self.last_side_codes}"
        )


def _add_finger_args(parser: argparse.ArgumentParser) -> None:
    for finger in DEFAULT_FINGER_ORDER:
        parser.add_argument(f"--{finger}-model", default="")
        parser.add_argument(f"--{finger}-x-scaler", default="")
        parser.add_argument(f"--{finger}-y-scaler", default="")
        parser.add_argument(f"--{finger}-q-indices", default="")
        parser.add_argument(f"--{finger}-motor-ids", default="")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manus wrist + five-finger fusion mapping controller")
    parser.add_argument("--hand", default=DEFAULT_TARGET_HAND, choices=["right"])
    parser.add_argument("--config", default="")
    parser.add_argument("--defaults-path", default=DEFAULTS_PATH)
    parser.add_argument("--glove-raw-topic", default=RAW_GLOVE_SKELETON_TOPIC)
    parser.add_argument("--glove48-topic", default=GLOVE48_TOPIC)
    parser.add_argument("--cmd-topic", default=RIGHT_CMD_TOPIC)
    parser.add_argument("--feedback-topic", default=RIGHT_FEEDBACK_TOPIC)
    parser.add_argument("--output-publish-rate-hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument("--status-period-sec", type=float, default=DEFAULT_STATUS_PERIOD_SEC)
    parser.add_argument("--feedback-stale-sec", type=float, default=DEFAULT_FEEDBACK_STALE_SEC)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--right-side-code", type=float, default=DEFAULT_EXPECT_SIDE_CODE)
    parser.add_argument("--raw-record-stride", type=int, default=DEFAULT_RAW_RECORD_STRIDE)
    parser.add_argument("--alpha-limit-deg", type=float, default=DEFAULT_ALPHA_LIMIT_DEG)
    parser.add_argument("--beta-limit-deg", type=float, default=DEFAULT_BETA_LIMIT_DEG)
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument("--beta-scale", type=float, default=1.0)
    parser.add_argument("--alpha-offset-deg", type=float, default=0.0)
    parser.add_argument("--beta-offset-deg", type=float, default=0.0)
    parser.add_argument("--wrist-deadband-deg", type=float, default=DEFAULT_WRIST_DEADBAND_DEG)
    parser.add_argument("--max-step-alpha-deg", type=float, default=DEFAULT_MAX_STEP_ALPHA_DEG)
    parser.add_argument("--max-step-beta-deg", type=float, default=DEFAULT_MAX_STEP_BETA_DEG)
    parser.add_argument("--finger-deadband-deg", type=float, default=0.0)
    parser.add_argument("--finger-max-delta-deg", type=float, default=0.0)
    parser.add_argument("--skip-wrist-motor-finger-inference", action="store_true", default=True)
    parser.add_argument("--allow-wrist-motor-finger-inference", dest="skip_wrist_motor_finger_inference", action="store_false")
    parser.add_argument("--publish-idle-without-feedback", action="store_true", default=True)
    parser.add_argument("--right-non-thumb-q-from-raw-skeleton", action="store_true", default=True)
    parser.add_argument("--no-right-non-thumb-q-from-raw-skeleton", dest="right_non_thumb_q_from_raw_skeleton", action="store_false")
    parser.add_argument("--motor-offsets-json", default="")
    parser.add_argument("--motor-cmd-offset-map-json", default="")
    parser.add_argument("--q-offset-map-json", default="")
    parser.add_argument("--finger-order", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--zero-on-exit", default="true", choices=["true", "false"])
    _add_finger_args(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    rclpy.init(args=None)
    node: Optional[WristFiveFingerFusionMappingNode] = None
    executor: Optional[SingleThreadedExecutor] = None
    spin_thread: Optional[threading.Thread] = None
    try:
        node = WristFiveFingerFusionMappingNode(args)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        def _spin_executor() -> None:
            try:
                executor.spin()
            except ExternalShutdownException:
                pass

        spin_thread = threading.Thread(target=_spin_executor, daemon=True)
        spin_thread.start()
        while rclpy.ok():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                zero_on_exit = str(getattr(args, "zero_on_exit", "true")).strip().lower() in ("1", "true", "yes", "on")
                if zero_on_exit and (not node.dry_run):
                    node.get_logger().info("开始全部回0...")
                    node.enable_manual_matrix_mode(True)
                    _execute_all_zero(node)
            except Exception as exc:
                node.get_logger().warn(f"回0过程异常: {exc}")
        if executor is not None:
            executor.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
