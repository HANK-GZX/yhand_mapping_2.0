#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-node Manus wrist + five-finger hybrid fusion controller.

Fusion rule:
1. Map wrist quaternion to alpha/beta and compute 28-motor wrist delta.
2. Keep wrist control logic identical to the existing wrist fusion path.
3. Replace the finger backend:
   - thumb: prev_m + target_q MLP
   - non-thumb: q0/curl MLP
4. Add the wrist delta back on top of the finger-side command.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray
try:
    from tf2_msgs.msg import TFMessage
except Exception:
    TFMessage = None

try:
    import torch
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError("This script requires torch for finger models") from exc

try:
    from devices_pkg.msg import YHandMsg
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError("Unable to import devices_pkg.msg.YHandMsg. Did you source install/setup.bash?") from exc

try:
    import five_finger_ring_curl_thumb_prev_m_target_q_mapping_main as hybrid_mod
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError("Unable to import five_finger hybrid mapping helpers") from exc

from hand_motor_map import (
    WRIST_MOTOR_IDS_BY_HAND,
)

thumb_mod = hybrid_mod.thumb_mod
ring_mod = hybrid_mod.curl_mod

try:
    from wrist_angle_motor_control import (
        DEFAULTS_PATH,
        MOTOR_COUNT,
        Motor,
        _compute_motor_targets_deg,
        _execute_all_zero,
        _parse_motor_offsets_json,
        RIGHT_TO_LEFT_MOTOR_MAP,
        get_hand_glove_side_code,
        get_hand_topics,
    )
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError("Unable to import wrist_angle_motor_control helpers") from exc


RAW_GLOVE_SKELETON_TOPIC = "/glove_joint_data_raw_skeleton"
GLOVE48_TOPIC = "/glove_joint_data"
MOCAP_TF_TOPIC = "/tf"
STATIC_TF_TOPIC = "/tf_static"
DEFAULT_TARGET_HAND = "right"
DEFAULT_EXPECT_SIDE_CODE = 1.0
DEFAULT_CONTROL_HZ = 100.0
DEFAULT_STATUS_PERIOD_SEC = 1.0
DEFAULT_FEEDBACK_STALE_SEC = 0.5
DEFAULT_GLOVE_SIDE_ACTIVITY_STALE_SEC = 0.5
DEFAULT_RAW_RECORD_STRIDE = 176
DEFAULT_ALPHA_LIMIT_DEG = 5.0
DEFAULT_BETA_LIMIT_DEG = 30.0
DEFAULT_WRIST_DEADBAND_DEG = 0.1
DEFAULT_MAX_STEP_ALPHA_DEG = 2.0
DEFAULT_MAX_STEP_BETA_DEG = 6.0
WRIST_INPUT_SOURCE_MANUS_RAW = "manus_raw_skeleton"
WRIST_INPUT_SOURCE_MOCAP_TF = "mocap_tf"
RIGHT_WRIST_MOTOR_IDS = WRIST_MOTOR_IDS_BY_HAND['right']
LEFT_WRIST_MOTOR_IDS = WRIST_MOTOR_IDS_BY_HAND['left']
WRIST_MOTOR_IDS_BY_HAND = {
    "left": LEFT_WRIST_MOTOR_IDS,
    "right": RIGHT_WRIST_MOTOR_IDS,
}


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


def _load_config_file(path: str) -> Dict[str, Any]:
    text = str(path or "").strip()
    if not text:
        return {}
    p = Path(text).expanduser()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


class WristFiveFingerHybridFusionNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.hand = str(args.hand).strip().lower()
        if self.hand not in ("left", "right"):
            raise ValueError(f"Unsupported hand={self.hand!r}, expected 'left' or 'right'")
        super().__init__(f"wrist_five_finger_hybrid_fusion_{self.hand}")

        self.config = _load_config_file(str(args.config))
        self.defaults_hand = thumb_mod._load_defaults_from_py(Path(args.defaults_path), self.hand)

        self.glove_raw_topic = str(args.glove_raw_topic)
        self.glove48_topic = str(args.glove48_topic)
        self.wrist_input_source = str(
            self.config.get("wrist_input_source", self.config.get("input_source", args.wrist_input_source))
        ).strip().lower()
        if self.wrist_input_source not in (WRIST_INPUT_SOURCE_MANUS_RAW, WRIST_INPUT_SOURCE_MOCAP_TF):
            raise ValueError(
                f"Unsupported wrist_input_source={self.wrist_input_source!r}, "
                f"expected one of {[WRIST_INPUT_SOURCE_MANUS_RAW, WRIST_INPUT_SOURCE_MOCAP_TF]}"
            )
        self.mocap_tf_topic = str(self.config.get("mocap_tf_topic", args.mocap_tf_topic)).strip() or MOCAP_TF_TOPIC
        self.mocap_hand_frame = str(self.config.get("mocap_hand_frame", args.mocap_hand_frame)).strip()
        if not self.mocap_hand_frame:
            self.mocap_hand_frame = "LeftHand" if self.hand == "left" else "RightHand"
        self.cmd_topic = str(args.cmd_topic)
        self.feedback_topic = str(args.feedback_topic)
        self.output_publish_rate_hz = max(float(args.output_publish_rate_hz), 1.0)
        self.status_period_sec = max(float(args.status_period_sec), 0.1)
        self.feedback_stale_sec = max(float(args.feedback_stale_sec), 0.05)
        self.glove_side_activity_stale_sec = DEFAULT_GLOVE_SIDE_ACTIVITY_STALE_SEC
        self.dry_run = bool(args.dry_run)
        self.expect_side_code = float(args.glove_side_code)
        self.raw_record_stride = max(int(args.raw_record_stride), 8)

        self.alpha_limit_deg = abs(float(args.alpha_limit_deg))
        beta_limit_deg = abs(float(args.beta_limit_deg))
        self.beta_min_deg = -beta_limit_deg if args.beta_min_deg is None else float(args.beta_min_deg)
        self.beta_max_deg = beta_limit_deg if args.beta_max_deg is None else float(args.beta_max_deg)
        if self.beta_min_deg > self.beta_max_deg:
            raise ValueError(
                f"Invalid beta range: beta_min_deg={self.beta_min_deg} > beta_max_deg={self.beta_max_deg}"
            )
        self.alpha_scale = float(args.alpha_scale)
        self.beta_scale = float(args.beta_scale)
        self.alpha_offset_deg = float(args.alpha_offset_deg)
        self.beta_offset_deg = float(args.beta_offset_deg)
        self.wrist_deadband_deg = max(float(args.wrist_deadband_deg), 0.0)
        self.max_step_alpha_deg = max(float(args.max_step_alpha_deg), 0.0)
        self.max_step_beta_deg = max(float(args.max_step_beta_deg), 0.0)
        self.use_non_thumb_raw_skeleton_q = bool(args.non_thumb_q_from_raw_skeleton)

        self.device = ring_mod._resolve_device(str(args.device or "auto").strip().lower())
        self.hand_glove_slice = (
            hybrid_mod.LEFT_HAND_GLOVE_SLICE if self.hand == "left" else hybrid_mod.RIGHT_HAND_GLOVE_SLICE
        )
        self.motor_offsets_deg = self._load_motor_offsets(str(args.motor_offsets_json))
        self.q_offsets_this_hand = self._load_q_offsets(str(args.q_offset_map_json))
        self.motor_cmd_offsets_this_hand = self._load_motor_cmd_offsets(str(args.motor_cmd_offset_map_json))
        self.motor_pos_limits = self._load_motor_limits(
            str(args.motor_feedback_limits_left_json),
            str(args.motor_feedback_limits_right_json),
        )
        self.motor_pos_limit_mode = getattr(self, "_motor_pos_limit_mode", "unknown")
        self.wrist_motor_ids = WRIST_MOTOR_IDS_BY_HAND[self.hand]
        self.wrist_motor_set = set(self.wrist_motor_ids)

        script_dir = Path(__file__).resolve().parent
        thumb_spec = hybrid_mod._build_thumb_spec(args=args, hand=self.hand, script_dir=script_dir)
        non_thumb_specs = hybrid_mod._build_non_thumb_specs(args=args, script_dir=script_dir)
        self.thumb_model = self._load_thumb_model(thumb_spec)
        self.non_thumb_models = self._load_non_thumb_models(non_thumb_specs)
        self.controlled_finger_motor_ids = self._build_controlled_motor_ids()
        if self.thumb_model is None and not self.non_thumb_models:
            raise RuntimeError("No finger model enabled for hybrid wrist fusion")

        self.raw_lock = threading.Lock()
        self.latest_raw_chunk: Optional[List[float]] = None
        self.latest_raw_msg_time: Optional[float] = None
        self.latest_raw_seq = 0
        self.last_side_codes: List[float] = []
        self.last_raw_side_accept_time: Optional[float] = None
        self.raw_cb_times: List[float] = []
        self.valid_raw_times: List[float] = []
        self.bad_raw_packets = 0
        self.processed_raw_chunk_seq = 0

        self.wrist_lock = threading.Lock()
        self.latest_wrist_quat: Optional[np.ndarray] = None
        self.latest_wrist_msg_time: Optional[float] = None
        self.latest_wrist_seq = 0
        self.processed_wrist_seq = 0
        self.wrist_zero_quat: Optional[np.ndarray] = None
        self.wrist_cb_times: List[float] = []
        self.valid_wrist_times: List[float] = []
        self.bad_wrist_packets = 0

        self.glove_lock = threading.Lock()
        self.latest_glove20: Optional[np.ndarray] = None
        self.latest_glove48_time: Optional[float] = None
        self.glove48_seq = 0
        self.processed_glove_seq = 0
        self.glove48_cb_times: List[float] = []
        self.bad_glove48_packets = 0
        self.latest_raw_q_by_finger: Dict[str, np.ndarray] = {}
        self.last_glove_raw_ts = 0.0

        self.current_alpha_deg = 0.0
        self.current_beta_deg = 0.0
        self.raw_alpha_deg = 0.0
        self.raw_beta_deg = 0.0
        self.wrist_zero_targets = np.asarray(
            _compute_motor_targets_deg(0.0, 0.0, hand=self.hand)[2],
            dtype=np.float32,
        )
        self.last_applied_wrist_delta = np.zeros(MOTOR_COUNT, dtype=np.float32)
        self.last_wrist_delta = np.zeros(MOTOR_COUNT, dtype=np.float32)
        self.limit_hits = 0
        self.smooth_steps = 0

        self.feedback_count = 0
        self.last_feedback_time: Optional[float] = None
        self.latest_feedback_matrix = np.full((MOTOR_COUNT, 5), np.nan, dtype=np.float32)
        self.manual_matrix_mode = False
        self.last_cmd_positions = np.asarray([0.0] * MOTOR_COUNT, dtype=np.float32)
        self.last_target_matrix = self.build_idle_tor_matrix()
        self.last_cmd_matrix = self._copy_matrix(self.last_target_matrix)
        self.has_last_finger_solution = False
        self._last_mode = "init"
        self.cmd_count = 0
        self.cmd_times: List[float] = []
        self.loop_times: List[float] = []
        self.late_loops = 0
        self.max_late_ms = 0.0
        self.tor_idle_latched = np.zeros((MOTOR_COUNT,), dtype=np.bool_)

        self.cmd_pub = self.create_publisher(YHandMsg, self.cmd_topic, QoSProfile(depth=10))
        self.feedback_sub = self.create_subscription(YHandMsg, self.feedback_topic, self._feedback_cb, QoSProfile(depth=10))
        raw_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.raw_sub = self.create_subscription(Float64MultiArray, self.glove_raw_topic, self._raw_skeleton_cb, raw_qos)
        self.glove48_sub = self.create_subscription(Float64MultiArray, self.glove48_topic, self._glove48_cb, raw_qos)
        if self.wrist_input_source == WRIST_INPUT_SOURCE_MOCAP_TF:
            if TFMessage is None:
                raise RuntimeError("Cannot import tf2_msgs.msg.TFMessage. Please source ROS2 workspace first.")
            tf_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
            static_tf_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.mocap_tf_sub = self.create_subscription(TFMessage, MOCAP_TF_TOPIC, self._mocap_tf_cb, tf_qos)
            if self.mocap_tf_topic != MOCAP_TF_TOPIC:
                self.mocap_tf_static_sub = self.create_subscription(
                    TFMessage,
                    self.mocap_tf_topic,
                    self._mocap_tf_cb,
                    static_tf_qos,
                )
            else:
                self.mocap_tf_static_sub = None
        else:
            self.mocap_tf_sub = None
            self.mocap_tf_static_sub = None

        self.publish_period_sec = 1.0 / self.output_publish_rate_hz
        self.control_timer = self.create_timer(self.publish_period_sec, self._control_timer_cb)
        self.status_timer = self.create_timer(self.status_period_sec, self._status_timer_cb)

        self.get_logger().info(
            "[wrist_five_finger_hybrid_fusion] started. "
            f"raw_topic={self.glove_raw_topic}, glove48_topic={self.glove48_topic}, "
            f"wrist_input_source={self.wrist_input_source}, mocap_tf_topic={self.mocap_tf_topic}, "
            f"mocap_hand_frame={self.mocap_hand_frame}, "
            f"cmd_topic={self.cmd_topic}, feedback_topic={self.feedback_topic}, "
            f"rate={self.output_publish_rate_hz:.1f}Hz, dry_run={self.dry_run}, device={self.device}, "
            f"thumb_enabled={self.thumb_model is not None}, non_thumb_enabled={list(self.non_thumb_models.keys())}"
        )
        if self.thumb_model is not None:
            self.get_logger().info(
                "[wrist_five_finger_hybrid_fusion] "
                f"thumb_ckpt={self.thumb_model['ckpt']}, thumb_motor_ids={self.thumb_model['motor_ids']}, "
                f"thumb_q_indices={self.thumb_model['q_indices']}, thumb_target_m_cols={self.thumb_model['target_m_cols']}"
            )

    def _load_motor_offsets(self, cli_json: str) -> np.ndarray:
        if cli_json.strip():
            return np.asarray(_parse_motor_offsets_json(cli_json), dtype=np.float32)
        cfg_value = self.config.get("motor_position_offsets_deg")
        if isinstance(cfg_value, list) and len(cfg_value) == MOTOR_COUNT:
            return np.asarray(list(cfg_value), dtype=np.float32)
        return np.zeros(MOTOR_COUNT, dtype=np.float32)

    def _load_q_offsets(self, cli_json: str) -> Dict[int, float]:
        if cli_json.strip():
            parsed = thumb_mod._parse_q_offset_map_json(cli_json, "--q-offset-map-json")
            return {int(k): float(v) for k, v in parsed.get(self.hand, {}).items()}
        defaults = hybrid_mod.Q_OFFSET_MAP_DEFAULT.get(self.hand, {})
        return {int(k): float(v) for k, v in dict(defaults).items()}

    def _load_motor_cmd_offsets(self, cli_json: str) -> Dict[int, float]:
        if cli_json.strip():
            parsed = thumb_mod._parse_motor_cmd_offset_map_json(cli_json, "--motor-cmd-offset-map-json")
            return {int(k): float(v) for k, v in parsed.get(self.hand, {}).items()}
        defaults = hybrid_mod.MOTOR_CMD_OFFSET_MAP_DEFAULT.get(self.hand, {})
        return {int(k): float(v) for k, v in dict(defaults).items()}

    def _load_motor_limits(
        self,
        left_limits_json: str,
        right_limits_json: str,
    ) -> Dict[int, Tuple[float, float]]:
        if self.hand == "left":
            if left_limits_json.strip():
                limits_left = thumb_mod._parse_motor_feedback_limits_json(
                    left_limits_json,
                    "--motor-feedback-limits-left-json",
                )
            else:
                limits_left = dict(hybrid_mod.MOTOR_FEEDBACK_LIMITS_LEFT_DEFAULT)
            raw_limits, mode = thumb_mod._build_motor_pos_limit_map(hand=self.hand, limits_left=limits_left)
        else:
            if right_limits_json.strip():
                limits_right = thumb_mod._parse_motor_feedback_limits_json(
                    right_limits_json,
                    "--motor-feedback-limits-right-json",
                )
            else:
                limits_right = dict(hybrid_mod.MOTOR_FEEDBACK_LIMITS_RIGHT_DEFAULT)
            raw_limits, mode = thumb_mod._build_motor_pos_limit_map(hand=self.hand, limits_right=limits_right)
        self._motor_pos_limit_mode = str(mode)
        return {int(k): (float(v[0]), float(v[1])) for k, v in raw_limits.items()}

    def _load_thumb_model(self, thumb_spec: dict) -> Optional[dict]:
        if not thumb_spec.get("enabled", False):
            return None
        loader = SimpleNamespace(hand=self.hand, device=self.device)
        bundle = thumb_mod.PrevMTargetQMLPMappingNode._load_finger_model(
            loader,
            name=hybrid_mod.THUMB_FINGER,
            spec=thumb_spec,
        )
        if any(int(mid) in self.wrist_motor_set for mid in bundle["motor_ids"]):
            raise RuntimeError(
                f"Thumb model motor_ids overlap reserved wrist motors: {bundle['motor_ids']}"
            )
        return bundle

    def _load_non_thumb_models(self, specs: Dict[str, dict]) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for finger in hybrid_mod.NON_THUMB_FINGERS:
            spec = specs.get(finger, {"enabled": False})
            if not spec.get("enabled", False):
                continue
            loader = SimpleNamespace(hand=self.hand, device=self.device)
            bundle = ring_mod.MultiFingerCurlMultiheadMLPMappingNode._load_finger_model(
                loader,
                finger=finger,
                spec=spec,
            )
            overlap = [int(mid) for mid in bundle["controlled_motor_ids"] if int(mid) in self.wrist_motor_set]
            if overlap:
                raise RuntimeError(
                    f"Non-thumb finger '{finger}' overlaps reserved wrist motors: {overlap}"
                )
            out[finger] = bundle
        return out

    def _build_controlled_motor_ids(self) -> List[int]:
        used: Dict[int, str] = {}
        if self.thumb_model is not None:
            for motor_id in self.thumb_model["motor_ids"]:
                used[int(motor_id)] = hybrid_mod.THUMB_FINGER
        for finger, bundle in self.non_thumb_models.items():
            for motor_id in bundle["controlled_motor_ids"]:
                prev = used.get(int(motor_id))
                if prev is not None:
                    self.get_logger().warn(
                        f"Motor overlap detected: motor {motor_id} is used by both '{prev}' and '{finger}'. "
                        "Later finger in order will overwrite previous command."
                    )
                used[int(motor_id)] = finger
        return sorted(used.keys())

    def build_idle_tor_matrix(self) -> List[List[float]]:
        return [
            [0.0, 0.0, float(self.defaults_hand["tor"][i]), 0.0, 0.0]
            for i in range(MOTOR_COUNT)
        ]

    @staticmethod
    def _copy_matrix(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
        return [
            [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])]
            for row in matrix
        ]

    def _build_target_matrix_from_positions(self, positions: Sequence[float]) -> List[List[float]]:
        if len(positions) != MOTOR_COUNT:
            raise ValueError(f"target_positions length must be {MOTOR_COUNT}, got {len(positions)}")
        matrix: List[List[float]] = []
        for i, pos in enumerate(positions[:MOTOR_COUNT]):
            matrix.append([
                float(pos),
                float(self.defaults_hand["vel"][i]),
                float(self.defaults_hand["tor_active"][i]),
                float(self.defaults_hand["kp"][i]),
                float(self.defaults_hand["kd"][i]),
            ])
        return matrix

    def _build_all_zero_matrix(self) -> List[List[float]]:
        matrix: List[List[float]] = []
        for i in range(MOTOR_COUNT):
            matrix.append([
                0.0,
                0.0,
                float(self.defaults_hand["tor"][i]),
                float(self.defaults_hand["kp"][i]),
                float(self.defaults_hand["kd"][i]),
            ])
        return matrix

    def set_all_zero_control(self) -> None:
        self.last_target_matrix = self._build_all_zero_matrix()

    def enable_manual_matrix_mode(self, enabled: bool) -> None:
        self.manual_matrix_mode = bool(enabled)

    def set_watchdog_fallback_enabled(self, enabled: bool) -> None:
        return None

    def is_feedback_stale(self) -> bool:
        if self.last_feedback_time is None:
            return True
        return (_now_sec() - self.last_feedback_time) > self.feedback_stale_sec

    def _feedback_cb(self, msg: YHandMsg) -> None:
        now = _now_sec()
        self.feedback_count += 1
        self.last_feedback_time = now
        matrix = np.full((MOTOR_COUNT, 5), np.nan, dtype=np.float32)
        motors = list(getattr(msg, "motors", []) or [])
        for i, motor in enumerate(motors[:MOTOR_COUNT]):
            try:
                matrix[i, 0] = float(motor.pos)
                matrix[i, 1] = float(motor.vel)
                matrix[i, 2] = float(motor.tor)
                matrix[i, 3] = float(motor.kp)
                matrix[i, 4] = float(motor.kd)
            except Exception:
                pass
        self.latest_feedback_matrix = matrix

    def _raw_skeleton_cb(self, msg: Float64MultiArray) -> None:
        now = _now_sec()
        self.raw_cb_times.append(now)
        del self.raw_cb_times[:-512]
        data = list(msg.data)
        q, raw_chunk, side_codes = self._extract_hand_raw_chunk_and_wrist_quat(data)
        self.last_side_codes = side_codes
        if raw_chunk is None:
            self.bad_raw_packets += 1
            return
        with self.raw_lock:
            self.latest_raw_chunk = list(raw_chunk)
            self.latest_raw_msg_time = now
            self.latest_raw_seq += 1
            self.last_raw_side_accept_time = now
        self.valid_raw_times.append(now)
        del self.valid_raw_times[:-512]
        if self.wrist_input_source == WRIST_INPUT_SOURCE_MANUS_RAW:
            self.wrist_cb_times.append(now)
            del self.wrist_cb_times[:-512]
            if q is None:
                self.bad_wrist_packets += 1
                return
            with self.wrist_lock:
                self.latest_wrist_quat = q
                self.latest_wrist_msg_time = now
                self.latest_wrist_seq += 1
            self.valid_wrist_times.append(now)
            del self.valid_wrist_times[:-512]

    def _extract_hand_raw_chunk_and_wrist_quat(
        self,
        data: Sequence[float],
    ) -> Tuple[Optional[np.ndarray], Optional[List[float]], List[float]]:
        side_codes: List[float] = []
        best: Optional[np.ndarray] = None
        best_chunk: Optional[List[float]] = None
        stride = self.raw_record_stride
        for base in range(0, len(data) - 7, stride):
            try:
                side = float(data[base])
                side_codes.append(side)
                if abs(side - self.expect_side_code) > 0.25:
                    continue
                q = _normalize_quat_xyzw([data[base + 4], data[base + 5], data[base + 6], data[base + 7]])
                if q is not None:
                    best = q
                    best_chunk = list(data[base: base + stride])
                    break
            except Exception:
                continue
        return best, best_chunk, side_codes

    def _has_recent_matching_raw_side(self, now: Optional[float] = None) -> bool:
        if now is None:
            now = _now_sec()
        last_accept = self.last_raw_side_accept_time
        if last_accept is None:
            return False
        return (now - last_accept) <= self.glove_side_activity_stale_sec

    def _extract_mocap_wrist_quat(self, msg: TFMessage) -> Tuple[Optional[np.ndarray], List[str]]:
        transforms = list(getattr(msg, "transforms", []) or [])
        if not transforms:
            return None, []
        seen_child_frames: List[str] = []
        for transform_stamped in transforms:
            child_frame = str(getattr(transform_stamped, "child_frame_id", "")).strip()
            if child_frame:
                seen_child_frames.append(child_frame)
            if child_frame != self.mocap_hand_frame:
                continue
            transform = getattr(transform_stamped, "transform", None)
            rotation = getattr(transform, "rotation", None)
            if rotation is None:
                continue
            q = _normalize_quat_xyzw(
                [
                    float(getattr(rotation, "x", 0.0)),
                    float(getattr(rotation, "y", 0.0)),
                    float(getattr(rotation, "z", 0.0)),
                    float(getattr(rotation, "w", 0.0)),
                ]
            )
            if q is not None:
                return q, seen_child_frames
        return None, seen_child_frames

    def _mocap_tf_cb(self, msg: TFMessage) -> None:
        now = _now_sec()
        self.wrist_cb_times.append(now)
        del self.wrist_cb_times[:-512]
        q, seen_child_frames = self._extract_mocap_wrist_quat(msg)
        if q is None:
            self.bad_wrist_packets += 1
            if self.bad_wrist_packets % 200 == 1:
                self.get_logger().warn(
                    f"Cannot select mocap wrist quaternion for hand={self.hand} "
                    f"(frame={self.mocap_hand_frame}, seen_children={seen_child_frames[:8]})"
                )
            return
        with self.wrist_lock:
            self.latest_wrist_quat = q
            self.latest_wrist_msg_time = now
            self.latest_wrist_seq += 1
        self.valid_wrist_times.append(now)
        del self.valid_wrist_times[:-512]

    def _glove48_cb(self, msg: Float64MultiArray) -> None:
        now = _now_sec()
        self.glove48_cb_times.append(now)
        del self.glove48_cb_times[:-512]
        if not self._has_recent_matching_raw_side(now):
            self.bad_glove48_packets += 1
            return
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
        return arr[self.hand_glove_slice].astype(np.float32)

    def _read_latest_raw_chunk(self) -> Tuple[int, Optional[List[float]], Optional[float]]:
        with self.raw_lock:
            chunk = list(self.latest_raw_chunk) if self.latest_raw_chunk is not None else None
            return self.latest_raw_seq, chunk, self.latest_raw_msg_time

    def _read_latest_wrist_quat(self) -> Tuple[int, Optional[np.ndarray], Optional[float]]:
        with self.wrist_lock:
            if self.latest_wrist_quat is None:
                return self.latest_wrist_seq, None, self.latest_wrist_msg_time
            return self.latest_wrist_seq, self.latest_wrist_quat.copy(), self.latest_wrist_msg_time

    def _read_latest_glove20(self) -> Tuple[int, Optional[np.ndarray], Optional[float]]:
        with self.glove_lock:
            if self.latest_glove20 is None:
                return self.glove48_seq, None, self.latest_glove48_time
            return self.glove48_seq, self.latest_glove20.copy(), self.latest_glove48_time

    def _update_non_thumb_raw_q(self, raw_chunk: Optional[Sequence[float]]) -> None:
        if raw_chunk is None or not self.use_non_thumb_raw_skeleton_q or not self.non_thumb_models:
            return
        try:
            q16_map = ring_mod.compute_non_thumb_q16(list(raw_chunk))
        except Exception:
            return
        now = _now_sec()
        for finger in self.non_thumb_models.keys():
            vals: List[float] = []
            ok = True
            for i in range(4):
                key = f"{finger}_q_{i}"
                if key not in q16_map:
                    ok = False
                    break
                vals.append(float(q16_map[key]))
            if ok:
                self.latest_raw_q_by_finger[finger] = np.asarray(vals, dtype=np.float32)
        self.last_glove_raw_ts = now

    def _update_wrist_angle(self, q_now: np.ndarray, raw_seq: int) -> None:
        if self.wrist_zero_quat is None:
            return
        alpha_raw, beta_raw = _extract_wrist_alpha_beta_deg(self.wrist_zero_quat, q_now)
        self.raw_alpha_deg = alpha_raw
        self.raw_beta_deg = beta_raw
        alpha = _apply_deadband(alpha_raw, self.wrist_deadband_deg) * self.alpha_scale + self.alpha_offset_deg
        beta = _apply_deadband(beta_raw, self.wrist_deadband_deg) * self.beta_scale + self.beta_offset_deg
        alpha = max(-self.alpha_limit_deg, min(self.alpha_limit_deg, alpha))
        beta = max(self.beta_min_deg, min(self.beta_max_deg, beta))
        if (
            abs(alpha) >= self.alpha_limit_deg - 1e-6
            or beta <= self.beta_min_deg + 1e-6
            or beta >= self.beta_max_deg - 1e-6
        ):
            self.limit_hits += 1
        next_alpha, smooth_a = _clip_step(self.current_alpha_deg, alpha, self.max_step_alpha_deg)
        next_beta, smooth_b = _clip_step(self.current_beta_deg, beta, self.max_step_beta_deg)
        if smooth_a or smooth_b:
            self.smooth_steps += 1
        self.current_alpha_deg = next_alpha
        self.current_beta_deg = next_beta
        self.processed_wrist_seq = raw_seq

    def _compute_wrist_delta_28(self, alpha_deg: float, beta_deg: float) -> np.ndarray:
        targets = np.asarray(
            _compute_motor_targets_deg(float(alpha_deg), float(beta_deg), hand=self.hand)[2],
            dtype=np.float32,
        )
        return targets - self.wrist_zero_targets

    def _build_feedback_abs_28(self) -> Optional[np.ndarray]:
        if self.feedback_count <= 0:
            return None
        out = np.zeros(MOTOR_COUNT, dtype=np.float32)
        for idx in range(MOTOR_COUNT):
            pos = float(self.latest_feedback_matrix[idx, 0])
            if not np.isfinite(pos):
                return None
            out[idx] = pos
        return out

    def _apply_q_offsets(self, q_now: np.ndarray, q_indices: Sequence[int]) -> np.ndarray:
        out = q_now.astype(np.float32).copy()
        for i, qi in enumerate(q_indices):
            out[i] = float(out[i] + float(self.q_offsets_this_hand.get(int(qi), 0.0)))
        return out

    def _non_thumb_cmd_offset(self, finger: str, motor_id: int) -> float:
        if str(finger) != "ring":
            return 0.0
        motor_id = int(motor_id)
        return float(hybrid_mod.RING_EXT_CMD_OFFSET_BY_HAND.get(self.hand, {}).get(motor_id, 0.0)) + float(
            hybrid_mod.RING_SIDE_SWING_CMD_OFFSET_BY_HAND.get(self.hand, {}).get(motor_id, 0.0)
        )

    def _hold_last_cmd_for_motor_ids(
        self,
        cmd_abs: np.ndarray,
        motor_ids: Sequence[int],
        delta_shift: np.ndarray,
        fb_abs: np.ndarray,
    ) -> None:
        for mid in motor_ids:
            idx = int(mid) - 1
            if self.has_last_finger_solution:
                pos_abs = float(self.last_target_matrix[idx][0] + delta_shift[idx])
            else:
                pos_abs = float(fb_abs[idx])
            lim = self.motor_pos_limits.get(int(mid))
            if lim is not None:
                pos_abs = float(np.clip(pos_abs, lim[0], lim[1]))
            cmd_abs[idx] = pos_abs

    def _infer_thumb_into_abs(
        self,
        cmd_abs: np.ndarray,
        glove20: np.ndarray,
        dec_cur: np.ndarray,
        wrist_delta: np.ndarray,
        delta_shift: np.ndarray,
        fb_abs: np.ndarray,
    ) -> None:
        if self.thumb_model is None:
            return
        bundle = self.thumb_model
        try:
            history_m = np.asarray(
                [float(dec_cur[int(mid) - 1]) for mid in bundle["motor_ids"]],
                dtype=np.float32,
            )
            target_q_raw = np.asarray([float(glove20[i]) for i in bundle["q_indices"]], dtype=np.float32)
            target_q = self._apply_q_offsets(target_q_raw, bundle["q_indices"])
            x = thumb_mod.PrevMTargetQMLPMappingNode._build_model_input(
                self,
                bundle=bundle,
                history_m=history_m,
                target_q=target_q,
            )
            x_norm = bundle["x_scaler"].transform(x)
            with torch.no_grad():
                xb = torch.tensor(x_norm, dtype=torch.float32, device=self.device)
                pred_m_norm = bundle["model"](xb).cpu().numpy()
            pred_m = bundle["y_scaler"].inverse(pred_m_norm)[0].astype(np.float32)
        except Exception as exc:
            self.get_logger().warn(f"thumb inference failed: {exc}")
            self._hold_last_cmd_for_motor_ids(cmd_abs, bundle["motor_ids"], delta_shift, fb_abs)
            return

        cmd_pos_by_model_motor_idx = np.zeros((bundle["m_dim"],), dtype=np.float32)
        for out_col_idx, (target_idx, _target_col) in enumerate(bundle["target_plan"]):
            motor_id = int(bundle["motor_ids"][int(target_idx)])
            idx = motor_id - 1
            pred_tilde = float(pred_m[out_col_idx])
            pos_abs = pred_tilde + float(wrist_delta[idx])
            pos_abs += float(self.motor_offsets_deg[idx])
            lim = self.motor_pos_limits.get(motor_id)
            if lim is not None:
                pos_abs = float(np.clip(pos_abs, lim[0], lim[1]))
            cmd_abs[idx] = pos_abs
            cmd_pos_by_model_motor_idx[int(target_idx)] = float(pos_abs)

        bundle["last_history_m"] = history_m.astype(np.float32)
        bundle["last_target_q"] = target_q.astype(np.float32)
        bundle["last_pred_m"] = pred_m.astype(np.float32)
        bundle["last_cmd_pos"] = cmd_pos_by_model_motor_idx.astype(np.float32)

    def _infer_non_thumb_into_abs(
        self,
        cmd_abs: np.ndarray,
        glove20: np.ndarray,
        wrist_delta: np.ndarray,
        delta_shift: np.ndarray,
        fb_abs: np.ndarray,
    ) -> None:
        for finger in hybrid_mod.NON_THUMB_FINGERS:
            if finger not in self.non_thumb_models:
                continue
            bundle = self.non_thumb_models[finger]
            try:
                if self.use_non_thumb_raw_skeleton_q:
                    raw_q = self.latest_raw_q_by_finger.get(finger)
                    if raw_q is None:
                        raise RuntimeError("raw q not ready")
                else:
                    raw_q = np.asarray(
                        [float(glove20[i]) for i in bundle["q_indices"]],
                        dtype=np.float32,
                    )
                raw_q = np.asarray(raw_q, dtype=np.float32)
                q0 = float(raw_q[0])
                if self.hand == "left" and bundle["mirror_left_from_right"]:
                    q0 = -q0
                curl, q_cal = ring_mod.compute_curl(raw_q[1], raw_q[2], raw_q[3], bundle["curl_meta"])
                model_bundle = bundle["model_bundle"]
                if model_bundle["model_kind"] == "q0_curl_multihead":
                    x_curl = model_bundle["scalers"]["x_curl"].transform(np.asarray([[curl]], dtype=np.float32))
                    x_q0curl = model_bundle["scalers"]["x_q0curl"].transform(
                        np.asarray([[q0, curl]], dtype=np.float32)
                    )
                    with torch.no_grad():
                        xb_curl = torch.tensor(x_curl, dtype=torch.float32, device=self.device)
                        xb_q0curl = torch.tensor(x_q0curl, dtype=torch.float32, device=self.device)
                        pred_curl_norm, pred_q0curl_norm = model_bundle["model"](xb_curl, xb_q0curl)
                    pred_curl = model_bundle["scalers"][model_bundle["curl_output_scaler_name"]].inverse(
                        pred_curl_norm.cpu().numpy()
                    )[0]
                    pred_q0curl = model_bundle["scalers"][model_bundle["q0curl_output_scaler_name"]].inverse(
                        pred_q0curl_norm.cpu().numpy()
                    )[0]
                    pred_by_col: Dict[str, float] = {}
                    for col, value in zip(model_bundle["curl_head_target_m_cols"], pred_curl):
                        pred_by_col[str(col)] = float(value)
                    for col, value in zip(model_bundle["q0curl_head_target_m_cols"], pred_q0curl):
                        pred_by_col[str(col)] = float(value)
                    pred_m = np.asarray(
                        [float(pred_by_col[col]) for col in model_bundle["target_m_cols"]],
                        dtype=np.float32,
                    )
                else:
                    x = np.asarray([[curl]], dtype=np.float32)
                    x_norm = model_bundle["x_scaler"].transform(x)
                    with torch.no_grad():
                        xb = torch.tensor(x_norm, dtype=torch.float32, device=self.device)
                        pred_m_norm = model_bundle["model"](xb).cpu().numpy()
                    pred_m = model_bundle["y_scaler"].inverse(pred_m_norm)[0].astype(np.float32)
            except Exception as exc:
                if not bundle["raw_not_ready_warned"]:
                    self.get_logger().warn(f"{finger} inference skipped: {exc}")
                    bundle["raw_not_ready_warned"] = True
                self._hold_last_cmd_for_motor_ids(cmd_abs, bundle["controlled_motor_ids"], delta_shift, fb_abs)
                continue

            bundle["raw_not_ready_warned"] = False
            cmd_pos = []
            for out_col_idx, (target_idx, _target_col) in enumerate(model_bundle["target_plan"]):
                motor_id = int(bundle["model_motor_ids"][int(target_idx)])
                idx = motor_id - 1
                pred_tilde = float(pred_m[out_col_idx])
                pred_cmd_local = pred_tilde + float(self._non_thumb_cmd_offset(finger, motor_id))
                prev_cmd_pos_abs = float(self.last_target_matrix[idx][0])
                prev_cmd_local = (
                    prev_cmd_pos_abs
                    - float(self.last_applied_wrist_delta[idx])
                    - float(self.motor_offsets_deg[idx])
                )
                if abs(pred_cmd_local - prev_cmd_local) <= ring_mod.CMD_POSITION_HOLD_DEADBAND_DEG:
                    pred_cmd_local = prev_cmd_local
                pos_abs = pred_cmd_local + float(wrist_delta[idx])
                pos_abs += float(self.motor_offsets_deg[idx])
                lim = self.motor_pos_limits.get(motor_id)
                if lim is not None:
                    pos_abs = float(np.clip(pos_abs, lim[0], lim[1]))
                cmd_abs[idx] = pos_abs
                cmd_pos.append(float(pos_abs))

            bundle["last_raw_q"] = raw_q.astype(np.float32)
            bundle["last_q_cal"] = np.asarray(q_cal, dtype=np.float32)
            bundle["last_curl"] = float(curl)
            bundle["last_pred_m"] = pred_m.astype(np.float32)
            bundle["last_cmd_pos"] = np.asarray(cmd_pos, dtype=np.float32)

    def _fuse_fingers_and_wrist(self, glove20: np.ndarray, fb_abs: np.ndarray, wrist_delta: np.ndarray) -> np.ndarray:
        dec_cur = fb_abs - wrist_delta
        delta_shift = wrist_delta - self.last_applied_wrist_delta
        cmd_abs = fb_abs.copy()

        for mid in self.wrist_motor_ids:
            idx = int(mid) - 1
            cmd_abs[idx] = float(wrist_delta[idx] + self.motor_offsets_deg[idx])

        self._infer_thumb_into_abs(cmd_abs, glove20, dec_cur, wrist_delta, delta_shift, fb_abs)
        self._infer_non_thumb_into_abs(cmd_abs, glove20, wrist_delta, delta_shift, fb_abs)
        return cmd_abs

    def _shift_last_command_for_wrist_only(self, wrist_delta: np.ndarray, fb_abs: np.ndarray) -> np.ndarray:
        if not self.has_last_finger_solution:
            cmd_abs = fb_abs.copy()
        else:
            cmd_abs = np.asarray([float(row[0]) for row in self.last_target_matrix], dtype=np.float32)
        delta_shift = wrist_delta - self.last_applied_wrist_delta
        for mid in self.wrist_motor_ids:
            idx = int(mid) - 1
            cmd_abs[idx] = float(wrist_delta[idx] + self.motor_offsets_deg[idx])
        for mid in self.controlled_finger_motor_ids:
            idx = int(mid) - 1
            if self.has_last_finger_solution:
                pos_abs = float(self.last_target_matrix[idx][0] + delta_shift[idx])
            else:
                pos_abs = float(fb_abs[idx])
            lim = self.motor_pos_limits.get(int(mid))
            if lim is not None:
                pos_abs = float(np.clip(pos_abs, lim[0], lim[1]))
            cmd_abs[idx] = pos_abs
        return cmd_abs

    def _apply_directional_tor_active(self, matrix: List[List[float]]) -> List[List[float]]:
        if self.feedback_count <= 0:
            return matrix
        for motor_id in self.controlled_finger_motor_ids:
            idx = int(motor_id) - 1
            row = matrix[idx]
            kp = float(row[3])
            kd = float(row[4])
            if abs(kp) <= 1e-9 and abs(kd) <= 1e-9:
                self.tor_idle_latched[idx] = False
                continue

            if ring_mod._is_side_swing_motor_id(motor_id):
                enter_deadband_deg = float(ring_mod.SIDE_SWING_TOR_ACTIVE_DIRECTION_ENTER_DEADBAND_DEG)
                exit_deadband_deg = float(ring_mod.SIDE_SWING_TOR_ACTIVE_DIRECTION_EXIT_DEADBAND_DEG)
            else:
                enter_deadband_deg = float(ring_mod.TOR_ACTIVE_DIRECTION_ENTER_DEADBAND_DEG)
                exit_deadband_deg = float(ring_mod.TOR_ACTIVE_DIRECTION_EXIT_DEADBAND_DEG)

            current_pos = float(self.latest_feedback_matrix[idx, 0])
            if not np.isfinite(current_pos):
                continue

            position_error = float(row[0]) - current_pos
            abs_error = abs(position_error)
            if bool(self.tor_idle_latched[idx]):
                if abs_error > exit_deadband_deg:
                    self.tor_idle_latched[idx] = False
                else:
                    if ring_mod._is_side_swing_motor_id(motor_id):
                        row[:] = [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    else:
                        row[:] = [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    continue

            if abs_error <= enter_deadband_deg:
                self.tor_idle_latched[idx] = True
                if ring_mod._is_side_swing_motor_id(motor_id):
                    row[:] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                else:
                    row[:] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                continue

            row[2] = ring_mod._directional_tor_active(
                self.defaults_hand["tor_active"][idx],
                self.defaults_hand["tor"][idx],
                row[0],
                current_pos,
            )
        return matrix

    def _publish_matrix(self, matrix: Sequence[Sequence[float]]) -> None:
        msg = YHandMsg()
        msg.motors = [
            Motor(
                pos=float(row[0]),
                vel=float(row[1]),
                tor=float(row[2]),
                kp=float(row[3]),
                kd=float(row[4]),
            )
            for row in matrix[:MOTOR_COUNT]
        ]
        msg.pretightenforce.flag = False
        msg.pretightenforce.torque = 0.0
        msg.pretightenforce.ms = 0
        self.cmd_pub.publish(msg)
        now = _now_sec()
        self.cmd_count += 1
        self.cmd_times.append(now)
        del self.cmd_times[:-512]

    def _control_timer_cb(self) -> None:
        start = _now_sec()
        if self.manual_matrix_mode:
            if not self.dry_run:
                self._publish_matrix(self.last_target_matrix)
            return

        self.loop_times.append(start)
        del self.loop_times[:-512]

        raw_chunk_seq, raw_chunk, _raw_time = self._read_latest_raw_chunk()
        if raw_chunk is not None and raw_chunk_seq != self.processed_raw_chunk_seq:
            self._update_non_thumb_raw_q(raw_chunk)
            self.processed_raw_chunk_seq = raw_chunk_seq

        wrist_seq, q_now, _wrist_time = self._read_latest_wrist_quat()
        new_wrist = bool(q_now is not None and wrist_seq != self.processed_wrist_seq)
        if q_now is not None and new_wrist:
            if self.wrist_zero_quat is None:
                self.wrist_zero_quat = q_now.copy()
                self.get_logger().info(f"Zero wrist frame captured, side_codes={self.last_side_codes}")
            self._update_wrist_angle(q_now, wrist_seq)

        wrist_delta = self._compute_wrist_delta_28(self.current_alpha_deg, self.current_beta_deg)
        self.last_wrist_delta = wrist_delta

        fb_abs = self._build_feedback_abs_28()
        glove_seq, glove20, _glove_time = self._read_latest_glove20()
        new_glove = bool(glove20 is not None and glove_seq != self.processed_glove_seq)

        if fb_abs is None or self.is_feedback_stale():
            target_matrix = self.build_idle_tor_matrix()
            self.has_last_finger_solution = False
            mode = "idle_no_feedback"
        elif glove20 is None:
            cmd_abs = fb_abs.copy()
            for mid in self.wrist_motor_ids:
                idx = int(mid) - 1
                cmd_abs[idx] = float(wrist_delta[idx] + self.motor_offsets_deg[idx])
            target_matrix = self._build_target_matrix_from_positions(cmd_abs.tolist())
            mode = "wrist_only_no_glove48"
        elif new_glove or (not self.has_last_finger_solution):
            cmd_abs = self._fuse_fingers_and_wrist(glove20, fb_abs, wrist_delta)
            self.processed_glove_seq = glove_seq
            self.has_last_finger_solution = True
            target_matrix = self._build_target_matrix_from_positions(cmd_abs.tolist())
            mode = "fusion"
        elif new_wrist:
            cmd_abs = self._shift_last_command_for_wrist_only(wrist_delta, fb_abs)
            target_matrix = self._build_target_matrix_from_positions(cmd_abs.tolist())
            mode = "wrist_shift_only"
        else:
            target_matrix = self._copy_matrix(self.last_target_matrix)
            mode = "hold_last"

        matrix = self._copy_matrix(target_matrix)
        if mode != "idle_no_feedback":
            matrix = self._apply_directional_tor_active(matrix)

        self.last_target_matrix = self._copy_matrix(target_matrix)
        self.last_cmd_matrix = self._copy_matrix(matrix)
        self.last_cmd_positions = np.asarray([float(row[0]) for row in matrix], dtype=np.float32)
        self.last_applied_wrist_delta = wrist_delta.copy()
        self._last_mode = mode

        if not self.dry_run:
            self._publish_matrix(matrix)

        elapsed = _now_sec() - start
        if elapsed > self.publish_period_sec:
            self.late_loops += 1
            self.max_late_ms = max(self.max_late_ms, elapsed * 1000.0)

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
            "[wrist_five_finger_hybrid_fusion] "
            f"raw_alpha={self.raw_alpha_deg:+.3f}deg, raw_beta={self.raw_beta_deg:+.3f}deg, "
            f"cmd_alpha={self.current_alpha_deg:+.3f}deg, cmd_beta={self.current_beta_deg:+.3f}deg, "
            f"raw_cb_hz={self._rate(self.raw_cb_times):.1f}, raw_valid_hz={self._rate(self.valid_raw_times):.1f}, "
            f"wrist_cb_hz={self._rate(self.wrist_cb_times):.1f}, wrist_valid_hz={self._rate(self.valid_wrist_times):.1f}, "
            f"glove48_hz={self._rate(self.glove48_cb_times):.1f}, cmd_hz={self._rate(self.cmd_times):.1f}, "
            f"fb_count={self.feedback_count}, fb_age={fb_age:.3f}s, raw_age={raw_age:.3f}s, glove_age={glove_age:.3f}s, "
            f"cmd_count={self.cmd_count}, dry_run={self.dry_run}, mode={self._last_mode}, "
            f"wrist_input_source={self.wrist_input_source}, "
            f"limit_hits={self.limit_hits}, smooth_steps={self.smooth_steps}, bad_raw={self.bad_raw_packets}, bad_wrist={self.bad_wrist_packets}, "
            f"bad_glove48={self.bad_glove48_packets}, late_loops={self.late_loops}, max_late_ms={self.max_late_ms:.2f}, "
            f"side_codes={self.last_side_codes}, raw_q_keys={sorted(self.latest_raw_q_by_finger.keys())}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manus wrist + five-finger hybrid fusion controller"
    )
    parser.add_argument("--hand", default=DEFAULT_TARGET_HAND, choices=["left", "right"])
    parser.add_argument("--config", default="")
    parser.add_argument("--defaults-path", default=str(DEFAULTS_PATH))
    parser.add_argument("--glove-raw-topic", default=RAW_GLOVE_SKELETON_TOPIC)
    parser.add_argument("--glove48-topic", default=GLOVE48_TOPIC)
    parser.add_argument("--cmd-topic", default="")
    parser.add_argument("--feedback-topic", default="")
    parser.add_argument("--output-publish-rate-hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument("--status-period-sec", type=float, default=DEFAULT_STATUS_PERIOD_SEC)
    parser.add_argument("--feedback-stale-sec", type=float, default=DEFAULT_FEEDBACK_STALE_SEC)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--glove-side-code", type=float, default=None)
    parser.add_argument("--raw-record-stride", type=int, default=DEFAULT_RAW_RECORD_STRIDE)
    parser.add_argument(
        "--wrist-input-source",
        default=WRIST_INPUT_SOURCE_MANUS_RAW,
        choices=[WRIST_INPUT_SOURCE_MANUS_RAW, WRIST_INPUT_SOURCE_MOCAP_TF],
        help="Wrist quaternion source only. Finger data paths remain unchanged.",
    )
    parser.add_argument("--mocap-tf-topic", default=MOCAP_TF_TOPIC)
    parser.add_argument("--mocap-hand-frame", default="")
    parser.add_argument("--alpha-limit-deg", type=float, default=DEFAULT_ALPHA_LIMIT_DEG)
    parser.add_argument("--beta-limit-deg", type=float, default=DEFAULT_BETA_LIMIT_DEG)
    parser.add_argument("--beta-min-deg", type=float, default=None)
    parser.add_argument("--beta-max-deg", type=float, default=None)
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument("--beta-scale", type=float, default=1.0)
    parser.add_argument("--alpha-offset-deg", type=float, default=0.0)
    parser.add_argument("--beta-offset-deg", type=float, default=0.0)
    parser.add_argument("--wrist-deadband-deg", type=float, default=DEFAULT_WRIST_DEADBAND_DEG)
    parser.add_argument("--max-step-alpha-deg", type=float, default=DEFAULT_MAX_STEP_ALPHA_DEG)
    parser.add_argument("--max-step-beta-deg", type=float, default=DEFAULT_MAX_STEP_BETA_DEG)
    parser.add_argument("--non-thumb-q-from-raw-skeleton", action="store_true", default=True)
    parser.add_argument(
        "--no-non-thumb-q-from-raw-skeleton",
        dest="non_thumb_q_from_raw_skeleton",
        action="store_false",
    )
    parser.add_argument("--motor-offsets-json", default="")
    parser.add_argument(
        "--motor-feedback-limits-left-json",
        default=json.dumps(hybrid_mod.MOTOR_FEEDBACK_LIMITS_LEFT_DEFAULT, ensure_ascii=False),
    )
    parser.add_argument(
        "--motor-feedback-limits-right-json",
        default=json.dumps(hybrid_mod.MOTOR_FEEDBACK_LIMITS_RIGHT_DEFAULT, ensure_ascii=False),
    )
    parser.add_argument(
        "--q-offset-map-json",
        default=json.dumps(hybrid_mod.Q_OFFSET_MAP_DEFAULT, ensure_ascii=False),
    )
    parser.add_argument(
        "--motor-cmd-offset-map-json",
        default=json.dumps(hybrid_mod.MOTOR_CMD_OFFSET_MAP_DEFAULT, ensure_ascii=False),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--zero-on-exit", default="true", choices=["true", "false"])
    hybrid_mod._add_thumb_args(parser)
    hybrid_mod._add_non_thumb_args(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.hand = str(args.hand).strip().lower()
    default_cmd_topic, default_feedback_topic = get_hand_topics(args.hand)
    args.cmd_topic = str(args.cmd_topic).strip() or default_cmd_topic
    args.feedback_topic = str(args.feedback_topic).strip() or default_feedback_topic
    if args.glove_side_code is None:
        args.glove_side_code = float(get_hand_glove_side_code(args.hand))
    rclpy.init(args=None)
    node: Optional[WristFiveFingerHybridFusionNode] = None
    executor: Optional[SingleThreadedExecutor] = None
    spin_thread: Optional[threading.Thread] = None
    try:
        node = WristFiveFingerHybridFusionNode(args)
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
