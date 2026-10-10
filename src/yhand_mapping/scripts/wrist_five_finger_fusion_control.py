#!/usr/bin/env python3
import argparse
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray

try:
    import torch
except ImportError as exc:  # pragma: no cover
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    from replace_right_non_thumb_q_from_skeleton import compute_non_thumb_q16
except Exception as exc:  # pragma: no cover
    compute_non_thumb_q16 = None
    NON_THUMB_Q_IMPORT_ERROR = exc
else:
    NON_THUMB_Q_IMPORT_ERROR = None

try:
    from wrist_five_finger_fusion_mapping_main import WristFiveFingerFusionMappingNode
except Exception as exc:  # pragma: no cover
    WristFiveFingerFusionMappingNode = None
    FUSION_MAPPING_IMPORT_ERROR = exc
else:
    FUSION_MAPPING_IMPORT_ERROR = None

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
from wrist_angle_motor_control import (
    DEFAULT_OUTPUT_PUBLISH_RATE_HZ,
    DEFAULTS_PATH,
    MOTOR_COUNT,
    RIGHT_CMD_TOPIC,
    RIGHT_FEEDBACK_TOPIC,
    WristAngleControlNode,
    _compute_motor_targets_deg,
    _execute_all_zero,
    _format_nonzero_offsets,
    _load_defaults_from_py,
    _parse_motor_offsets_json,
)


from hand_motor_map import (
    WRIST_MOTOR_IDS_BY_HAND,
)

RAW_GLOVE_SKELETON_TOPIC = "/glove_joint_data_raw_skeleton"
GLOVE48_TOPIC = "/glove_joint_data"
RIGHT_GLOVE_SIDE_CODE = 1.0
SKELETON_CHUNK_LEN = 176
DEFAULT_CONTROL_PERIOD_SEC = 0.01
DEFAULT_STATUS_PERIOD_SEC = 1.0
DEFAULT_MAX_COMMAND_STEP_ALPHA_DEG = 2.0
DEFAULT_MAX_COMMAND_STEP_BETA_DEG = 6.0

RIGHT_WRIST_MOTOR_IDS = WRIST_MOTOR_IDS_BY_HAND['right']
RIGHT_WRIST_MOTOR_SET = set(RIGHT_WRIST_MOTOR_IDS)


def _finite_float(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _parse_bool_arg(raw: str, arg_name: str):
    v = str(raw).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid value for {arg_name}: {raw!r}, expected true/false")


def _parse_motor_feedback_limits_right_json(raw_json: str):
    text = str(raw_json).strip()
    if not text:
        return {int(k): (float(v[0]), float(v[1])) for k, v in MOTOR_FEEDBACK_LIMITS_RIGHT.items()}
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise ValueError(f"Invalid JSON for --motor-feedback-limits-right-json: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--motor-feedback-limits-right-json must be object {\"motor_id\": [lo, hi], ...}")
    out = {}
    for k, v in parsed.items():
        mid = int(k)
        if (not isinstance(v, (list, tuple))) or len(v) != 2:
            raise ValueError(f"motor {mid} limit must be [lo, hi], got {v!r}")
        lo = float(v[0])
        hi = float(v[1])
        if lo > hi:
            raise ValueError(f"motor {mid} invalid limit: lo={lo} > hi={hi}")
        out[mid] = (lo, hi)
    return out


def _normalize_quaternion(x, y, z, w):
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return None
    return (x / norm, y / norm, z / norm, w / norm)


def _quat_conjugate(q):
    return (-float(q[0]), -float(q[1]), -float(q[2]), float(q[3]))


def _quat_multiply(q1, q2):
    x1, y1, z1, w1 = [float(v) for v in q1]
    x2, y2, z2, w2 = [float(v) for v in q2]
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _relative_quaternion(q_zero, q_now):
    q_rel = _quat_multiply(_quat_conjugate(q_zero), q_now)
    return _normalize_quaternion(*q_rel)


def _quat_to_rotmat(qx, qy, qz, qw):
    qn = _normalize_quaternion(qx, qy, qz, qw)
    if qn is None:
        return None
    x, y, z, w = qn
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw)),
        (2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw)),
        (2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy)),
    )


def _clamp(value: float, low: float, high: float):
    return max(float(low), min(float(high), float(value)))


def _step_toward(current: float, target: float, max_step: float):
    max_step = abs(float(max_step))
    if max_step <= 0.0:
        return float(target)
    delta = float(target) - float(current)
    if abs(delta) <= max_step:
        return float(target)
    return float(current) + math.copysign(max_step, delta)


def _ema_ms(prev_ms: float, new_ms: float, alpha: float = 0.1):
    if prev_ms <= 0.0:
        return float(new_ms)
    return float((1.0 - float(alpha)) * float(prev_ms) + float(alpha) * float(new_ms))


def _relative_quat_to_alpha_beta_deg(q_rel):
    r = _quat_to_rotmat(*q_rel)
    if r is None:
        return (0.0, 0.0)
    sin_beta = _clamp(-r[2][0], -1.0, 1.0)
    beta_rad = math.asin(sin_beta)
    alpha_rad = math.atan2(-r[0][1], r[1][1])
    return (math.degrees(alpha_rad), math.degrees(beta_rad))


class WristFiveFingerFusionNode(Node):
    def __init__(
        self,
        control_node: WristAngleControlNode,
        defaults_hand: dict,
        motor_offsets_deg: list[float],
        glove_topic: str,
        glove_raw_skeleton_topic: str,
        right_non_thumb_q_from_raw_skeleton: bool,
        right_side_code: float,
        alpha_scale: float,
        beta_scale: float,
        alpha_offset_deg: float,
        beta_offset_deg: float,
        alpha_limit_deg: float,
        beta_limit_deg: float,
        deadband_deg: float,
        max_command_step_alpha_deg: float,
        max_command_step_beta_deg: float,
        allow_single_chunk_fallback: bool,
        fallback_right_chunk_index: int,
        control_period_sec: float,
        status_print_period_sec: float,
        device: str,
        q_offset_map: dict,
        motor_cmd_offset_map: dict,
        motor_pos_limits_right: dict,
        dry_run: bool,
    ):
        super().__init__("wrist_five_finger_fusion_right")
        if torch is None:
            raise RuntimeError(
                "Cannot import torch. Please install torch in this Python env.\n"
                f"Import error: {TORCH_IMPORT_ERROR}"
            )

        self.control_node = control_node
        self.defaults_hand = defaults_hand
        self.motor_offsets_deg = list(motor_offsets_deg)
        self.glove_topic = str(glove_topic)
        self.glove_raw_skeleton_topic = str(glove_raw_skeleton_topic)
        self.use_right_non_thumb_raw_skeleton_q = bool(right_non_thumb_q_from_raw_skeleton)
        self.right_side_code = float(right_side_code)
        self.alpha_scale = float(alpha_scale)
        self.beta_scale = float(beta_scale)
        self.alpha_offset_deg = float(alpha_offset_deg)
        self.beta_offset_deg = float(beta_offset_deg)
        self.alpha_limit_deg = abs(float(alpha_limit_deg))
        self.beta_limit_deg = abs(float(beta_limit_deg))
        self.deadband_deg = max(0.0, float(deadband_deg))
        self.max_command_step_alpha_deg = max(0.0, float(max_command_step_alpha_deg))
        self.max_command_step_beta_deg = max(0.0, float(max_command_step_beta_deg))
        self.allow_single_chunk_fallback = bool(allow_single_chunk_fallback)
        self.fallback_right_chunk_index = int(fallback_right_chunk_index)
        self.control_period_sec = max(0.002, float(control_period_sec))
        self.status_print_period_sec = max(0.2, float(status_print_period_sec))
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
        self.dry_run = bool(dry_run)

        self.q_offsets_this_hand = dict(q_offset_map.get("right", {}))
        self.finger_cmd_offsets_this_hand = dict(motor_cmd_offset_map.get("right", {}))
        self.motor_pos_limits_right = {int(k): (float(v[0]), float(v[1])) for k, v in motor_pos_limits_right.items()}
        if self.use_right_non_thumb_raw_skeleton_q and (not callable(compute_non_thumb_q16)):
            raise RuntimeError(
                "Cannot import replace_right_non_thumb_q_from_skeleton.py while "
                "--right-non-thumb-q-from-raw-skeleton is enabled.\n"
                f"Import error: {NON_THUMB_Q_IMPORT_ERROR}"
            )

        self._lock = threading.Lock()
        self.fusion_enabled = True
        self.feedback_received = False
        self.calibrated_wrist_zero = False
        self.wrist_zero_quat = None
        self.latest_glove20 = None
        self.latest_wrist_quat = None
        self.latest_raw_chunk = None
        self.latest_right_non_thumb_q16 = None
        self.latest_glove_seq = 0
        self.latest_wrist_seq = 0
        self.last_applied_glove_seq = 0
        self.last_applied_wrist_seq = 0
        self.last_glove_ts = 0.0
        self.last_raw_ts = 0.0
        self.last_rx_side_codes = []
        self.desired_alpha_deg = 0.0
        self.desired_beta_deg = 0.0
        self.smoothed_initialized = False
        self.smoothed_alpha_deg = 0.0
        self.smoothed_beta_deg = 0.0
        self.last_cmd_alpha_deg = 0.0
        self.last_cmd_beta_deg = 0.0
        self.last_raw_alpha_deg = 0.0
        self.last_raw_beta_deg = 0.0
        self.rx_glove48_count = 0
        self.rx_raw_count = 0
        self.step_count = 0
        self.ctrl_tick_count = 0
        self.ctrl_recompute_count = 0
        self.ctrl_hold_no_new_count = 0
        self.ctrl_feedback_guard_count = 0
        self.ctrl_not_calibrated_count = 0
        self.ctrl_recompute_shift_only_count = 0
        self.ctrl_recompute_wrist_only_count = 0
        self.ctrl_recompute_full_infer_count = 0
        self.ctrl_infer_forward_total = 0
        self.ctrl_last_ms = 0.0
        self.ctrl_ema_ms = 0.0
        self.infer_last_ms = 0.0
        self.infer_ema_ms = 0.0
        self.raw_cb_last_ms = 0.0
        self.raw_cb_ema_ms = 0.0
        self.raw_qsolve_last_ms = 0.0
        self.raw_qsolve_ema_ms = 0.0
        self.clip_count_total = 0
        self.limit_hit_count = 0
        self.smooth_step_count = 0
        self.last_cmd_positions = [0.0] * MOTOR_COUNT
        self.prev_decoupled_feedback = None
        self.last_applied_wrist_delta = np.zeros((MOTOR_COUNT,), dtype=np.float32)
        self.has_last_finger_solution = False
        self.last_no_glove_warn_ts = 0.0
        self.last_feedback_guard_warn_ts = 0.0
        self.feedback_guard_active = False
        self._raw_non_thumb_fallback_warned = set()
        self.last_status_ts = time.monotonic()
        self.last_status_ctrl_tick_count = 0
        self.last_status_step_count = 0
        self.last_status_recompute_count = 0
        self.last_status_hold_no_new_count = 0
        self.last_status_feedback_guard_count = 0
        self.last_status_not_calibrated_count = 0
        self.last_status_raw_count = 0
        self.last_status_glove_count = 0
        self.last_status_out_count = 0

        _, _, w0 = _compute_motor_targets_deg(0.0, 0.0)
        self.wrist_zero_targets = np.asarray(w0, dtype=np.float32)

        self.finger_models = self._load_right_finger_models()
        finger_motor_ids = sorted(
            {int(mid) for b in self.finger_models.values() for mid in b["motor_ids"]}
        )
        self.finger_motor_set = set(finger_motor_ids)
        if len(finger_motor_ids) != 24:
            self.get_logger().warn(
                f"Expected 24 unique finger motors, got {len(finger_motor_ids)}: {finger_motor_ids}"
            )
        overlap = self.finger_motor_set.intersection(RIGHT_WRIST_MOTOR_SET)
        if overlap:
            raise RuntimeError(f"Finger motors overlap wrist motors: {sorted(overlap)}")

        # Decouple subscription callbacks and control timer callback.
        # Default callback group is mutually-exclusive; splitting groups allows
        # MultiThreadedExecutor to schedule these paths in parallel.
        self.sub_cb_group = ReentrantCallbackGroup()
        self.ctrl_cb_group = MutuallyExclusiveCallbackGroup()
        self.status_cb_group = MutuallyExclusiveCallbackGroup()

        latest_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.glove_sub = self.create_subscription(
            Float64MultiArray,
            self.glove_topic,
            self._glove48_cb,
            latest_qos,
            callback_group=self.sub_cb_group,
        )
        self.raw_sub = self.create_subscription(
            Float64MultiArray,
            self.glove_raw_skeleton_topic,
            self._raw_skeleton_cb,
            latest_qos,
            callback_group=self.sub_cb_group,
        )
        self.ctrl_timer = self.create_timer(
            self.control_period_sec,
            self._control_step_cb,
            callback_group=self.ctrl_cb_group,
        )
        self.status_timer = self.create_timer(
            self.status_print_period_sec,
            self._status_cb,
            callback_group=self.status_cb_group,
        )

        self.get_logger().info(
            "[wrist_five_finger_fusion] started. "
            f"glove_topic={self.glove_topic}, raw_topic={self.glove_raw_skeleton_topic}, "
            f"cmd_rate={self.control_node.output_publish_rate_hz:.1f}Hz, "
            f"control_period={self.control_period_sec:.3f}s, "
            f"alpha_limit=±{self.alpha_limit_deg:.1f}deg, beta_limit=±{self.beta_limit_deg:.1f}deg, "
            f"max_step_alpha={self.max_command_step_alpha_deg:.3f}deg/tick, "
            f"max_step_beta={self.max_command_step_beta_deg:.3f}deg/tick, "
            f"dry_run={self.dry_run}, "
            f"right_non_thumb_raw_skeleton={self.use_right_non_thumb_raw_skeleton_q}, "
            "raw_cb=cache_latest_frame_only, control_timer=consume_latest_or_hold, "
            "fusion=cmd[wrist4]=wrist, cmd[finger24]=finger_decouple_recover, "
            "publish=control_timer_fixed_rate"
        )

    def _load_right_finger_models(self):
        script_dir = Path(__file__).resolve().parent
        hand_cfg = FINGER_DEFAULTS_BY_HAND["right"]
        out = {}
        for name in DEFAULT_FINGER_ORDER:
            cfg = hand_cfg[name]
            enabled = _parse_bool_arg(cfg["enable_default"], f"right default enable {name}")
            if not enabled:
                continue
            model_ckpt = str(script_dir.joinpath(*cfg["model_ckpt_rel_parts"]))
            motor_ids = _parse_csv_ints(cfg["motor_ids_csv"])
            q_indices = _parse_csv_ints(cfg["q_indices_csv"])
            mirror = _parse_bool_arg(
                cfg["mirror_right_from_left_default"],
                f"right default mirror {name}",
            )
            if mirror:
                motor_ids = _mirror_motors_left_to_right(motor_ids)

            bundle = _load_delta_multihead_model(Path(model_ckpt), self.device)
            if len(motor_ids) != int(bundle["m_dim"]):
                raise RuntimeError(
                    f"{name}: len(motor_ids)={len(motor_ids)} != m_dim={bundle['m_dim']}"
                )
            if len(q_indices) != int(bundle["q_dim"]):
                raise RuntimeError(
                    f"{name}: len(q_indices)={len(q_indices)} != q_dim={bundle['q_dim']}"
                )
            out[name] = {
                "name": name,
                "ckpt": model_ckpt,
                "model": bundle["model"],
                "x_scaler": bundle["x_scaler"],
                "y_scaler": bundle["y_scaler"],
                "motor_ids": list(motor_ids),
                "q_indices": list(q_indices),
                "prev_q": None,
                "last_pred_delta": np.zeros((int(bundle["m_dim"]),), dtype=np.float32),
            }
            self.get_logger().info(
                f"[wrist_five_finger_fusion] finger={name}, ckpt={model_ckpt}, "
                f"motor_ids={motor_ids}, q_indices={q_indices}"
            )
        if not out:
            raise RuntimeError("No finger model enabled by right-hand defaults.")
        return out

    def _extract_right_wrist_chunk_and_quat(self, arr: list[float]):
        n = len(arr)
        if n < SKELETON_CHUNK_LEN:
            return None, None, []
        full_chunks = n // SKELETON_CHUNK_LEN
        parsed_chunks = []
        for ci in range(full_chunks):
            base = ci * SKELETON_CHUNK_LEN
            chunk = arr[base : base + SKELETON_CHUNK_LEN]
            side = _finite_float(arr[base + 0])
            qx = _finite_float(arr[base + 4])
            qy = _finite_float(arr[base + 5])
            qz = _finite_float(arr[base + 6])
            qw = _finite_float(arr[base + 7])
            if None in (qx, qy, qz, qw):
                continue
            qn = _normalize_quaternion(float(qx), float(qy), float(qz), float(qw))
            if qn is None:
                continue
            parsed_chunks.append((ci, side, qn, chunk))

        if not parsed_chunks:
            return None, None, []

        side_codes = []
        for _ci, side, _qn, _chunk in parsed_chunks:
            side_codes.append(None if side is None else float(side))

        for _ci, side, qn, chunk in parsed_chunks:
            if side is None:
                continue
            if abs(float(side) - self.right_side_code) <= 0.25:
                return chunk, qn, side_codes

        if self.allow_single_chunk_fallback and len(parsed_chunks) == 1:
            return parsed_chunks[0][3], parsed_chunks[0][2], side_codes

        if self.fallback_right_chunk_index >= 0:
            for ci, _side, qn, chunk in parsed_chunks:
                if ci == self.fallback_right_chunk_index:
                    return chunk, qn, side_codes

        return None, None, side_codes

    def _apply_q_offsets(self, q_now: np.ndarray, q_indices):
        q_adj = q_now.copy()
        for i, qi in enumerate(q_indices):
            off = float(self.q_offsets_this_hand.get(int(qi), 0.0))
            if abs(off) > 1e-9:
                q_adj[i] = float(q_adj[i] + off)
        return q_adj

    def _glove48_cb(self, msg: Float64MultiArray):
        data = list(msg.data) if msg is not None else []
        if len(data) < 48:
            return
        glove20 = np.asarray(data[28:48], dtype=np.float32)
        if glove20.shape[0] != 20:
            return
        now = time.monotonic()
        with self._lock:
            self.latest_glove20 = glove20
            self.last_glove_ts = now
            self.rx_glove48_count += 1
            self.latest_glove_seq += 1

    def _raw_skeleton_cb(self, msg: Float64MultiArray):
        cb_t0 = time.monotonic()
        data = list(msg.data) if msg is not None else []
        if not data:
            return
        raw_chunk, quat, side_codes = self._extract_right_wrist_chunk_and_quat(data)
        if quat is None:
            return
        now = time.monotonic()
        with self._lock:
            self.rx_raw_count += 1
            self.last_raw_ts = now
            self.last_rx_side_codes = list(side_codes)
            self.latest_wrist_quat = tuple(quat)
            self.latest_raw_chunk = list(raw_chunk) if raw_chunk is not None else None
            self.latest_wrist_seq += 1
        cb_ms = (time.monotonic() - cb_t0) * 1000.0
        with self._lock:
            self.raw_cb_last_ms = float(cb_ms)
            self.raw_cb_ema_ms = float(_ema_ms(self.raw_cb_ema_ms, cb_ms))

    def _process_wrist_frame(
        self,
        quat,
        raw_chunk,
        calibrated: bool,
        wrist_zero_quat,
        smoothed_initialized: bool,
        smoothed_alpha_deg: float,
        smoothed_beta_deg: float,
    ):
        out = {
            "valid": False,
            "just_calibrated": False,
            "calibrated": bool(calibrated),
            "wrist_zero_quat": wrist_zero_quat,
            "raw_alpha_deg": 0.0,
            "raw_beta_deg": 0.0,
            "desired_alpha_deg": 0.0,
            "desired_beta_deg": 0.0,
            "cmd_alpha_deg": 0.0,
            "cmd_beta_deg": 0.0,
            "smoothed_initialized": bool(smoothed_initialized),
            "smoothed_alpha_deg": float(smoothed_alpha_deg),
            "smoothed_beta_deg": float(smoothed_beta_deg),
            "limit_hit_inc": 0,
            "smooth_step_inc": 0,
            "q16_clean": None,
            "qsolve_ms": 0.0,
        }
        if quat is None:
            return out

        if self.use_right_non_thumb_raw_skeleton_q and raw_chunk is not None:
            try:
                qsolve_t0 = time.monotonic()
                q16_map = compute_non_thumb_q16(raw_chunk)
                out["qsolve_ms"] = float((time.monotonic() - qsolve_t0) * 1000.0)
                q16_clean = {}
                for finger_name in RIGHT_NON_THUMB_FINGERS:
                    for i in range(4):
                        key = f"{finger_name}_q_{i}"
                        if key not in q16_map:
                            raise KeyError(f"Missing key from compute_non_thumb_q16: {key}")
                        q16_clean[key] = float(q16_map[key])
                out["q16_clean"] = q16_clean
            except Exception as exc:
                self.get_logger().warn(
                    f"[wrist_five_finger_fusion] failed to compute right non-thumb q from raw skeleton: {exc}"
                )
                out["q16_clean"] = None

        if not calibrated:
            out["valid"] = True
            out["just_calibrated"] = True
            out["calibrated"] = True
            out["wrist_zero_quat"] = tuple(quat)
            out["smoothed_initialized"] = True
            out["smoothed_alpha_deg"] = 0.0
            out["smoothed_beta_deg"] = 0.0
            out["cmd_alpha_deg"] = 0.0
            out["cmd_beta_deg"] = 0.0
            return out

        if wrist_zero_quat is None:
            return out

        q_rel = _relative_quaternion(tuple(wrist_zero_quat), tuple(quat))
        if q_rel is None:
            return out
        raw_alpha_deg, raw_beta_deg = _relative_quat_to_alpha_beta_deg(q_rel)
        if abs(raw_alpha_deg) < self.deadband_deg:
            raw_alpha_deg = 0.0
        if abs(raw_beta_deg) < self.deadband_deg:
            raw_beta_deg = 0.0

        desired_alpha = raw_alpha_deg * self.alpha_scale + self.alpha_offset_deg
        desired_beta = raw_beta_deg * self.beta_scale + self.beta_offset_deg
        cmd_alpha = _clamp(desired_alpha, -self.alpha_limit_deg, self.alpha_limit_deg)
        cmd_beta = _clamp(desired_beta, -self.beta_limit_deg, self.beta_limit_deg)
        if (abs(cmd_alpha - desired_alpha) > 1e-9) or (abs(cmd_beta - desired_beta) > 1e-9):
            out["limit_hit_inc"] = 1

        if not smoothed_initialized:
            smoothed_initialized = True
            smoothed_alpha_deg = 0.0
            smoothed_beta_deg = 0.0

        smooth_alpha = _step_toward(smoothed_alpha_deg, cmd_alpha, self.max_command_step_alpha_deg)
        smooth_beta = _step_toward(smoothed_beta_deg, cmd_beta, self.max_command_step_beta_deg)
        if (abs(smooth_alpha - cmd_alpha) > 1e-9) or (abs(smooth_beta - cmd_beta) > 1e-9):
            out["smooth_step_inc"] = 1

        out["valid"] = True
        out["calibrated"] = True
        out["raw_alpha_deg"] = float(raw_alpha_deg)
        out["raw_beta_deg"] = float(raw_beta_deg)
        out["desired_alpha_deg"] = float(desired_alpha)
        out["desired_beta_deg"] = float(desired_beta)
        out["cmd_alpha_deg"] = float(smooth_alpha)
        out["cmd_beta_deg"] = float(smooth_beta)
        out["smoothed_initialized"] = True
        out["smoothed_alpha_deg"] = float(smooth_alpha)
        out["smoothed_beta_deg"] = float(smooth_beta)
        return out

    def _try_get_non_thumb_raw_q(self, finger_name: str, q_dim: int):
        q16_map = self.latest_right_non_thumb_q16
        if q16_map is None:
            return None
        q_vals = []
        for i in range(int(q_dim)):
            key = f"{finger_name}_q_{i}"
            if key not in q16_map:
                return None
            q_vals.append(float(q16_map[key]))
        return np.asarray(q_vals, dtype=np.float32)

    def _hold_last_cmd_for_finger_with_wrist_shift(self, cmd_pos: np.ndarray, motor_ids, delta_shift: np.ndarray):
        for mid in motor_ids:
            idx = int(mid) - 1
            pos_abs = float(self.last_cmd_positions[idx] + delta_shift[idx])
            lim = self.motor_pos_limits_right.get(int(mid))
            if lim is not None:
                pos_abs = float(np.clip(pos_abs, lim[0], lim[1]))
            cmd_pos[idx] = pos_abs

    def _build_feedback_abs_28(self):
        pos_list = self.control_node.all_motor_positions()
        arr = np.zeros((MOTOR_COUNT,), dtype=np.float32)
        for i in range(MOTOR_COUNT):
            p = pos_list[i] if i < len(pos_list) else None
            if p is None:
                p = self.last_cmd_positions[i]
            arr[i] = float(p)
        return arr

    def _compute_wrist_delta_28(self, alpha_deg: float, beta_deg: float):
        _lengths, _deltas, targets = _compute_motor_targets_deg(alpha_deg, beta_deg)
        t = np.asarray(targets, dtype=np.float32)
        return (t - self.wrist_zero_targets)

    def _publish_cmd_pos(self, cmd_pos: np.ndarray):
        if self.dry_run:
            return
        self.control_node.set_all_motor_position_targets([float(v) for v in cmd_pos.tolist()])

    def _control_step_cb(self):
        tick_t0 = time.monotonic()
        with self._lock:
            if not self.fusion_enabled:
                return
            self.ctrl_tick_count += 1
            glove20 = None if self.latest_glove20 is None else self.latest_glove20.copy()
            latest_wrist_quat = None if self.latest_wrist_quat is None else tuple(self.latest_wrist_quat)
            latest_raw_chunk = None if self.latest_raw_chunk is None else list(self.latest_raw_chunk)
            calibrated = bool(self.calibrated_wrist_zero)
            wrist_zero_quat = None if self.wrist_zero_quat is None else tuple(self.wrist_zero_quat)
            alpha = float(self.last_cmd_alpha_deg)
            beta = float(self.last_cmd_beta_deg)
            smoothed_initialized = bool(self.smoothed_initialized)
            smoothed_alpha_deg = float(self.smoothed_alpha_deg)
            smoothed_beta_deg = float(self.smoothed_beta_deg)
            glove_seq = int(self.latest_glove_seq)
            wrist_seq = int(self.latest_wrist_seq)
            last_applied_glove_seq = int(self.last_applied_glove_seq)
            last_applied_wrist_seq = int(self.last_applied_wrist_seq)
            has_last_finger_solution = bool(self.has_last_finger_solution)
            last_applied_wrist_delta = self.last_applied_wrist_delta.copy()
            feedback_guard_active = bool(self.feedback_guard_active)

        fb_ready = bool(self.control_node.feedback_received)
        fb_stale = self.control_node.is_feedback_stale()
        if (not fb_ready) or fb_stale:
            now = time.monotonic()
            if not self.dry_run:
                self.control_node.set_idle_tension()
                self.control_node.publish_current_command(manual=False)
            with self._lock:
                if (now - self.last_feedback_guard_warn_ts) >= 2.0:
                    self.last_feedback_guard_warn_ts = now
                    self.get_logger().warn(
                        "Feedback not ready/stale, fusion step is paused and command is held at idle tension. "
                        f"feedback_ready={fb_ready}, feedback_stale={fb_stale}"
                    )
                self.feedback_guard_active = True
                self.ctrl_feedback_guard_count += 1
                ctrl_ms = (time.monotonic() - tick_t0) * 1000.0
                self.ctrl_last_ms = float(ctrl_ms)
                self.ctrl_ema_ms = float(_ema_ms(self.ctrl_ema_ms, ctrl_ms))
            return
        elif feedback_guard_active:
            with self._lock:
                self.feedback_guard_active = False
            self.get_logger().info("Feedback recovered, fusion control resumed.")

        new_glove = glove_seq != last_applied_glove_seq
        new_wrist = wrist_seq != last_applied_wrist_seq
        just_calibrated = False
        if new_wrist and latest_wrist_quat is not None:
            wrist_out = self._process_wrist_frame(
                quat=latest_wrist_quat,
                raw_chunk=latest_raw_chunk,
                calibrated=calibrated,
                wrist_zero_quat=wrist_zero_quat,
                smoothed_initialized=smoothed_initialized,
                smoothed_alpha_deg=smoothed_alpha_deg,
                smoothed_beta_deg=smoothed_beta_deg,
            )
            with self._lock:
                q16_clean = wrist_out.get("q16_clean", None)
                if self.use_right_non_thumb_raw_skeleton_q:
                    self.latest_right_non_thumb_q16 = q16_clean
                qsolve_ms = float(wrist_out.get("qsolve_ms", 0.0))
                self.raw_qsolve_last_ms = qsolve_ms
                self.raw_qsolve_ema_ms = float(_ema_ms(self.raw_qsolve_ema_ms, qsolve_ms))

                if wrist_out.get("valid", False):
                    self.calibrated_wrist_zero = bool(wrist_out["calibrated"])
                    self.wrist_zero_quat = wrist_out["wrist_zero_quat"]
                    self.last_raw_alpha_deg = float(wrist_out["raw_alpha_deg"])
                    self.last_raw_beta_deg = float(wrist_out["raw_beta_deg"])
                    self.desired_alpha_deg = float(wrist_out["desired_alpha_deg"])
                    self.desired_beta_deg = float(wrist_out["desired_beta_deg"])
                    self.smoothed_initialized = bool(wrist_out["smoothed_initialized"])
                    self.smoothed_alpha_deg = float(wrist_out["smoothed_alpha_deg"])
                    self.smoothed_beta_deg = float(wrist_out["smoothed_beta_deg"])
                    self.last_cmd_alpha_deg = float(wrist_out["cmd_alpha_deg"])
                    self.last_cmd_beta_deg = float(wrist_out["cmd_beta_deg"])
                    self.limit_hit_count += int(wrist_out["limit_hit_inc"])
                    self.smooth_step_count += int(wrist_out["smooth_step_inc"])
                    just_calibrated = bool(wrist_out.get("just_calibrated", False))

            if wrist_out.get("valid", False):
                calibrated = bool(wrist_out["calibrated"])
                alpha = float(wrist_out["cmd_alpha_deg"])
                beta = float(wrist_out["cmd_beta_deg"])
                if just_calibrated:
                    self.get_logger().info("Captured wrist zero quaternion (first valid right-hand frame).")

        if not calibrated:
            if not self.dry_run:
                # Keep fixed-rate output before wrist zero is captured.
                self.control_node.publish_current_command(manual=False)
            with self._lock:
                self.ctrl_not_calibrated_count += 1
                ctrl_ms = (time.monotonic() - tick_t0) * 1000.0
                self.ctrl_last_ms = float(ctrl_ms)
                self.ctrl_ema_ms = float(_ema_ms(self.ctrl_ema_ms, ctrl_ms))
            return

        if just_calibrated:
            if not self.dry_run:
                self.control_node.publish_current_command(manual=False)
            with self._lock:
                self.ctrl_hold_no_new_count += 1
                ctrl_ms = (time.monotonic() - tick_t0) * 1000.0
                self.ctrl_last_ms = float(ctrl_ms)
                self.ctrl_ema_ms = float(_ema_ms(self.ctrl_ema_ms, ctrl_ms))
            return

        should_recompute = bool(new_glove or new_wrist or (not has_last_finger_solution))
        if not should_recompute:
            # No new glove/wrist input: hold last command and still publish at fixed rate.
            if not self.dry_run:
                self.control_node.publish_current_command(manual=False)
            with self._lock:
                self.ctrl_hold_no_new_count += 1
                ctrl_ms = (time.monotonic() - tick_t0) * 1000.0
                self.ctrl_last_ms = float(ctrl_ms)
                self.ctrl_ema_ms = float(_ema_ms(self.ctrl_ema_ms, ctrl_ms))
            return
        with self._lock:
            self.ctrl_recompute_count += 1

        wrist_delta = self._compute_wrist_delta_28(alpha, beta)
        fb_abs = self._build_feedback_abs_28()
        dec_cur = fb_abs - wrist_delta
        prev_dec = dec_cur if self.prev_decoupled_feedback is None else self.prev_decoupled_feedback

        cmd_pos = np.zeros((MOTOR_COUNT,), dtype=np.float32)
        clip_count = 0

        if (not new_glove) and has_last_finger_solution:
            with self._lock:
                self.ctrl_recompute_shift_only_count += 1
            cmd_pos = np.asarray(self.last_cmd_positions, dtype=np.float32)
            delta_shift = wrist_delta - last_applied_wrist_delta

            for mid in RIGHT_WRIST_MOTOR_IDS:
                idx = int(mid) - 1
                cmd_pos[idx] = float(wrist_delta[idx]) + float(self.motor_offsets_deg[idx])

            for mid in self.finger_motor_set:
                idx = int(mid) - 1
                pos_abs = float(cmd_pos[idx] + delta_shift[idx])
                lim = self.motor_pos_limits_right.get(int(mid))
                if lim is not None:
                    clipped = float(np.clip(pos_abs, lim[0], lim[1]))
                    if abs(clipped - pos_abs) > 1e-6:
                        clip_count += 1
                    pos_abs = clipped
                cmd_pos[idx] = pos_abs
        else:
            if glove20 is None:
                with self._lock:
                    self.ctrl_recompute_wrist_only_count += 1
                now = time.monotonic()
                with self._lock:
                    if (now - self.last_no_glove_warn_ts) >= 2.0:
                        self.last_no_glove_warn_ts = now
                        self.get_logger().warn(
                            "No /glove_joint_data(48D) received yet. "
                            "Fallback to wrist-only mode (finger hold)."
                        )
                # Wrist-only fallback:
                # - wrist motors follow wrist mapping
                # - finger motors hold current absolute feedback
                cmd_pos = fb_abs.copy()
                for mid in RIGHT_WRIST_MOTOR_IDS:
                    idx = int(mid) - 1
                    cmd_pos[idx] = float(wrist_delta[idx]) + float(self.motor_offsets_deg[idx])
            else:
                infer_t0 = time.monotonic()
                infer_forward_count = 0
                # 4 wrist motors: use wrist mapping directly.
                for mid in RIGHT_WRIST_MOTOR_IDS:
                    idx = int(mid) - 1
                    cmd_pos[idx] = float(wrist_delta[idx])

                # 24 finger motors: infer in "de-wristed" feedback space, then add current wrist delta back.
                delta_shift = wrist_delta - last_applied_wrist_delta
                for name in DEFAULT_FINGER_ORDER:
                    if name not in self.finger_models:
                        continue
                    b = self.finger_models[name]

                    if self.use_right_non_thumb_raw_skeleton_q and name in RIGHT_NON_THUMB_FINGERS:
                        q_now_raw = self._try_get_non_thumb_raw_q(name, len(b["q_indices"]))
                        if q_now_raw is None:
                            if name not in self._raw_non_thumb_fallback_warned:
                                self.get_logger().warn(
                                    "[wrist_five_finger_fusion] right non-thumb raw skeleton q is not ready yet "
                                    f"for finger={name}; hold previous command for this finger."
                                )
                                self._raw_non_thumb_fallback_warned.add(name)
                            self._hold_last_cmd_for_finger_with_wrist_shift(
                                cmd_pos=cmd_pos,
                                motor_ids=b["motor_ids"],
                                delta_shift=delta_shift,
                            )
                            continue
                        self._raw_non_thumb_fallback_warned.discard(name)
                    else:
                        q_now_raw = np.asarray([float(glove20[i]) for i in b["q_indices"]], dtype=np.float32)
                    q_now = self._apply_q_offsets(q_now_raw, b["q_indices"])
                    q_prev = q_now if b["prev_q"] is None else b["prev_q"]
                    dq = (q_now - q_prev)

                    m_prev = np.asarray(
                        [float(prev_dec[int(mid) - 1]) for mid in b["motor_ids"]],
                        dtype=np.float32,
                    )

                    x = np.concatenate([q_now, q_prev, m_prev, dq], axis=0).reshape(1, -1)
                    x_norm = b["x_scaler"].transform(x)
                    with torch.no_grad():
                        xb = torch.tensor(x_norm, dtype=torch.float32, device=self.device)
                        pred_delta_norm = b["model"](xb).cpu().numpy()
                    infer_forward_count += 1
                    pred_delta = b["y_scaler"].inverse(pred_delta_norm)[0]
                    m_tilde_cmd = m_prev + pred_delta

                    for out_idx, mid in enumerate(b["motor_ids"]):
                        idx = int(mid) - 1
                        pos_abs = float(m_tilde_cmd[out_idx] + wrist_delta[idx])
                        pos_abs += float(self.finger_cmd_offsets_this_hand.get(int(mid), 0.0))
                        lim = self.motor_pos_limits_right.get(int(mid))
                        if lim is not None:
                            clipped = float(np.clip(pos_abs, lim[0], lim[1]))
                            if abs(clipped - pos_abs) > 1e-6:
                                clip_count += 1
                            pos_abs = clipped
                        cmd_pos[idx] = pos_abs

                    b["prev_q"] = q_now
                    b["last_pred_delta"] = pred_delta.astype(np.float32)

                # Final global motor offsets (28).
                for i in range(MOTOR_COUNT):
                    cmd_pos[i] = float(cmd_pos[i] + float(self.motor_offsets_deg[i]))
                infer_ms = (time.monotonic() - infer_t0) * 1000.0
                with self._lock:
                    self.ctrl_recompute_full_infer_count += 1
                    self.ctrl_infer_forward_total += int(infer_forward_count)
                    self.infer_last_ms = float(infer_ms)
                    self.infer_ema_ms = float(_ema_ms(self.infer_ema_ms, infer_ms))

        # Update latest command matrix; publish once in this control tick.
        self._publish_cmd_pos(cmd_pos)
        if not self.dry_run:
            self.control_node.publish_current_command(manual=False)

        with self._lock:
            self.last_cmd_positions = [float(v) for v in cmd_pos.tolist()]
            self.prev_decoupled_feedback = dec_cur.copy()
            self.last_applied_wrist_delta = wrist_delta.copy()
            self.last_applied_glove_seq = glove_seq
            self.last_applied_wrist_seq = wrist_seq
            self.has_last_finger_solution = True
            self.step_count += 1
            self.clip_count_total += int(clip_count)
            ctrl_ms = (time.monotonic() - tick_t0) * 1000.0
            self.ctrl_last_ms = float(ctrl_ms)
            self.ctrl_ema_ms = float(_ema_ms(self.ctrl_ema_ms, ctrl_ms))

    def _status_cb(self):
        now = time.monotonic()
        out_count = self.control_node.get_publish_count()
        fb_ready = bool(self.control_node.feedback_received)
        fb_stale = self.control_node.is_feedback_stale()
        with self._lock:
            glove_age = (now - self.last_glove_ts) if self.last_glove_ts > 0.0 else float("inf")
            raw_age = (now - self.last_raw_ts) if self.last_raw_ts > 0.0 else float("inf")
            calibrated = bool(self.calibrated_wrist_zero)
            raw_non_thumb_ready = (
                self.latest_right_non_thumb_q16 is not None
                if self.use_right_non_thumb_raw_skeleton_q
                else "disabled"
            )
            raw_alpha = float(self.last_raw_alpha_deg)
            raw_beta = float(self.last_raw_beta_deg)
            desired_alpha = float(self.desired_alpha_deg)
            desired_beta = float(self.desired_beta_deg)
            cmd_alpha = float(self.last_cmd_alpha_deg)
            cmd_beta = float(self.last_cmd_beta_deg)
            side_codes = list(self.last_rx_side_codes)
            rx_glove = int(self.rx_glove48_count)
            rx_raw = int(self.rx_raw_count)
            ctrl_ticks = int(self.ctrl_tick_count)
            steps = int(self.step_count)
            recompute = int(self.ctrl_recompute_count)
            hold_no_new = int(self.ctrl_hold_no_new_count)
            fb_guard = int(self.ctrl_feedback_guard_count)
            not_cal = int(self.ctrl_not_calibrated_count)
            shift_only = int(self.ctrl_recompute_shift_only_count)
            wrist_only = int(self.ctrl_recompute_wrist_only_count)
            full_infer = int(self.ctrl_recompute_full_infer_count)
            infer_fwd_total = int(self.ctrl_infer_forward_total)
            ctrl_last_ms = float(self.ctrl_last_ms)
            ctrl_ema_ms = float(self.ctrl_ema_ms)
            infer_last_ms = float(self.infer_last_ms)
            infer_ema_ms = float(self.infer_ema_ms)
            raw_cb_last_ms = float(self.raw_cb_last_ms)
            raw_cb_ema_ms = float(self.raw_cb_ema_ms)
            raw_qsolve_last_ms = float(self.raw_qsolve_last_ms)
            raw_qsolve_ema_ms = float(self.raw_qsolve_ema_ms)
            clip_total = int(self.clip_count_total)
            limit_hits = int(self.limit_hit_count)
            smooth_steps = int(self.smooth_step_count)
            dt = max(now - self.last_status_ts, 1e-6)
            glove_hz = (rx_glove - self.last_status_glove_count) / dt
            raw_hz = (rx_raw - self.last_status_raw_count) / dt
            ctrl_tick_hz = (ctrl_ticks - self.last_status_ctrl_tick_count) / dt
            step_hz = (steps - self.last_status_step_count) / dt
            recompute_hz = (recompute - self.last_status_recompute_count) / dt
            hold_no_new_hz = (hold_no_new - self.last_status_hold_no_new_count) / dt
            fb_guard_hz = (fb_guard - self.last_status_feedback_guard_count) / dt
            not_cal_hz = (not_cal - self.last_status_not_calibrated_count) / dt
            out_hz = (out_count - self.last_status_out_count) / dt
            ctrl_load = (
                ctrl_ema_ms / (self.control_period_sec * 1000.0)
                if self.control_period_sec > 0.0
                else float("nan")
            )
            infer_load = (
                infer_ema_ms / (self.control_period_sec * 1000.0)
                if self.control_period_sec > 0.0
                else float("nan")
            )
            self.last_status_ts = now
            self.last_status_glove_count = rx_glove
            self.last_status_raw_count = rx_raw
            self.last_status_ctrl_tick_count = ctrl_ticks
            self.last_status_step_count = steps
            self.last_status_recompute_count = recompute
            self.last_status_hold_no_new_count = hold_no_new
            self.last_status_feedback_guard_count = fb_guard
            self.last_status_not_calibrated_count = not_cal
            self.last_status_out_count = out_count

        self.get_logger().info(
            "[wrist_five_finger_fusion] "
            f"calibrated={calibrated}, raw_alpha={raw_alpha:+.3f}, raw_beta={raw_beta:+.3f}, "
            f"desired_alpha={desired_alpha:+.3f}, desired_beta={desired_beta:+.3f}, "
            f"cmd_alpha={cmd_alpha:+.3f}, cmd_beta={cmd_beta:+.3f}, "
            f"feedback_ready={fb_ready}, feedback_stale={fb_stale}, raw_non_thumb_q_ready={raw_non_thumb_ready}, "
            f"glove_age={glove_age:.3f}s, raw_age={raw_age:.3f}s, "
            f"glove_hz={glove_hz:.1f}, raw_hz={raw_hz:.1f}, ctrl_tick_hz={ctrl_tick_hz:.1f}, "
            f"recompute_hz={recompute_hz:.1f}, step_hz={step_hz:.1f}, out_hz={out_hz:.1f}, "
            f"hold_no_new_hz={hold_no_new_hz:.1f}, fb_guard_hz={fb_guard_hz:.1f}, not_cal_hz={not_cal_hz:.1f}, "
            f"ctrl_last_ms={ctrl_last_ms:.3f}, ctrl_ema_ms={ctrl_ema_ms:.3f}, ctrl_load={ctrl_load:.2f}, "
            f"infer_last_ms={infer_last_ms:.3f}, infer_ema_ms={infer_ema_ms:.3f}, infer_load={infer_load:.2f}, "
            f"raw_cb_last_ms={raw_cb_last_ms:.3f}, raw_cb_ema_ms={raw_cb_ema_ms:.3f}, "
            f"raw_qsolve_last_ms={raw_qsolve_last_ms:.3f}, raw_qsolve_ema_ms={raw_qsolve_ema_ms:.3f}, "
            f"rx_glove={rx_glove}, rx_raw={rx_raw}, ctrl_ticks={ctrl_ticks}, recompute={recompute}, "
            f"steps={steps}, out_count={out_count}, infer_fwds={infer_fwd_total}, "
            f"shift_only={shift_only}, wrist_only={wrist_only}, full_infer={full_infer}, "
            f"clip_total={clip_total}, limit_hits={limit_hits}, smooth_steps={smooth_steps}, "
            f"dry_run={self.dry_run}, side_codes={side_codes}"
        )

    def set_fusion_enabled(self, enabled: bool):
        with self._lock:
            self.fusion_enabled = bool(enabled)


def _spin_executor(executor: MultiThreadedExecutor):
    try:
        executor.spin()
    except ExternalShutdownException:
        return
    except KeyboardInterrupt:
        return


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Right-hand wrist + five-finger fusion controller.\n"
            "Logic: cmd[wrist4]=wrist_infer; cmd[finger24]=finger_infer(decoupled by wrist delta)."
        )
    )
    parser.add_argument("--defaults-path", default=str(DEFAULTS_PATH))
    parser.add_argument("--cmd-topic", default=RIGHT_CMD_TOPIC)
    parser.add_argument("--feedback-topic", default=RIGHT_FEEDBACK_TOPIC)
    parser.add_argument("--glove-topic", default=GLOVE48_TOPIC, help="48D glove topic.")
    parser.add_argument(
        "--glove-raw-skeleton-topic",
        default=RAW_GLOVE_SKELETON_TOPIC,
        help="Raw skeleton topic for wrist quaternion extraction.",
    )
    parser.add_argument(
        "--right-index-q-from-raw-skeleton",
        default="false",
        choices=["true", "false"],
        help=(
            "Deprecated alias. Same as --right-non-thumb-q-from-raw-skeleton: "
            "replace right non-thumb(index/middle/ring/little) q with raw-skeleton solve."
        ),
    )
    parser.add_argument(
        "--right-non-thumb-q-from-raw-skeleton",
        default="false",
        choices=["true", "false"],
        help=(
            "If true, replace right non-thumb(index/middle/ring/little) glove q by "
            "raw-skeleton q solved from /glove_joint_data_raw_skeleton."
        ),
    )
    parser.add_argument("--right-side-code", type=float, default=RIGHT_GLOVE_SIDE_CODE)
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument("--beta-scale", type=float, default=1.0)
    parser.add_argument("--alpha-offset-deg", type=float, default=0.0)
    parser.add_argument("--beta-offset-deg", type=float, default=0.0)
    parser.add_argument("--alpha-limit-deg", type=float, default=5.0)
    parser.add_argument("--beta-limit-deg", type=float, default=30.0)
    parser.add_argument("--deadband-deg", type=float, default=0.1)
    parser.add_argument(
        "--max-command-step-alpha-deg",
        type=float,
        default=DEFAULT_MAX_COMMAND_STEP_ALPHA_DEG,
        help="Maximum alpha command change per control tick (deg). Set 0 to disable.",
    )
    parser.add_argument(
        "--max-command-step-beta-deg",
        type=float,
        default=DEFAULT_MAX_COMMAND_STEP_BETA_DEG,
        help="Maximum beta command change per control tick (deg). Set 0 to disable.",
    )
    parser.add_argument(
        "--allow-single-chunk-fallback",
        default="true",
        choices=["true", "false"],
    )
    parser.add_argument("--fallback-right-chunk-index", type=int, default=-1)
    parser.add_argument(
        "--output-publish-rate-hz",
        type=float,
        default=DEFAULT_OUTPUT_PUBLISH_RATE_HZ,
        help="Final motor command publish rate to cmd topic.",
    )
    parser.add_argument(
        "--control-period-sec",
        type=float,
        default=DEFAULT_CONTROL_PERIOD_SEC,
        help=(
            "Deprecated. Kept for CLI compatibility; actual control period is "
            "derived from --output-publish-rate-hz."
        ),
    )
    parser.add_argument(
        "--status-print-period-sec",
        type=float,
        default=DEFAULT_STATUS_PERIOD_SEC,
    )
    parser.add_argument(
        "--motor-offsets-json",
        default="",
        help="Global per-motor command offset in deg (len=28 list or dict by motor id).",
    )
    parser.add_argument(
        "--q-offset-map-json",
        default=json.dumps(Q_OFFSET_MAP_DEFAULTS, ensure_ascii=False),
        help="Glove q offset map JSON by hand and q_index.",
    )
    parser.add_argument(
        "--motor-cmd-offset-map-json",
        default=json.dumps(MOTOR_CMD_OFFSET_MAP_DEFAULTS, ensure_ascii=False),
        help="Finger motor cmd offset map JSON by hand and motor_id.",
    )
    parser.add_argument(
        "--motor-feedback-limits-right-json",
        default="",
        help="Override right-hand finger motor position limits JSON.",
    )
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument(
        "--zero-on-exit",
        default="true",
        choices=["true", "false"],
        help="Execute full return-zero flow before exit.",
    )
    parser.add_argument(
        "--dry-run",
        default="false",
        choices=["true", "false"],
        help="Run fusion inference without publishing /yhand_right_cmd and without exit return-zero.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if WristFiveFingerFusionMappingNode is None:
        raise RuntimeError(
            "Unable to import WristFiveFingerFusionMappingNode from "
            "wrist_five_finger_fusion_mapping_main.py\n"
            f"Import error: {FUSION_MAPPING_IMPORT_ERROR}"
        )

    output_publish_rate_hz = max(float(args.output_publish_rate_hz), 1e-6)
    zero_on_exit = _parse_bool_arg(args.zero_on_exit, "--zero-on-exit")
    dry_run = _parse_bool_arg(args.dry_run, "--dry-run")
    use_right_non_thumb_raw = (
        _parse_bool_arg(args.right_index_q_from_raw_skeleton, "--right-index-q-from-raw-skeleton")
        or _parse_bool_arg(
            args.right_non_thumb_q_from_raw_skeleton,
            "--right-non-thumb-q-from-raw-skeleton",
        )
    )

    # Keep legacy CLI compatibility but run the single-node publish architecture
    # from wrist_five_finger_fusion_mapping_main.py.
    mapping_args = argparse.Namespace(
        hand="right",
        config="",
        defaults_path=str(args.defaults_path),
        glove_raw_topic=str(args.glove_raw_skeleton_topic),
        glove48_topic=str(args.glove_topic),
        cmd_topic=str(args.cmd_topic),
        feedback_topic=str(args.feedback_topic),
        output_publish_rate_hz=float(output_publish_rate_hz),
        status_period_sec=float(args.status_print_period_sec),
        feedback_stale_sec=0.5,
        dry_run=bool(dry_run),
        right_side_code=float(args.right_side_code),
        # Raw skeleton packet per-hand stride in this project is 176.
        raw_record_stride=176,
        alpha_limit_deg=float(args.alpha_limit_deg),
        beta_limit_deg=float(args.beta_limit_deg),
        alpha_scale=float(args.alpha_scale),
        beta_scale=float(args.beta_scale),
        alpha_offset_deg=float(args.alpha_offset_deg),
        beta_offset_deg=float(args.beta_offset_deg),
        wrist_deadband_deg=float(args.deadband_deg),
        max_step_alpha_deg=float(args.max_command_step_alpha_deg),
        max_step_beta_deg=float(args.max_command_step_beta_deg),
        finger_deadband_deg=0.0,
        finger_max_delta_deg=0.0,
        skip_wrist_motor_finger_inference=True,
        publish_idle_without_feedback=True,
        right_non_thumb_q_from_raw_skeleton=bool(use_right_non_thumb_raw),
        motor_offsets_json=str(args.motor_offsets_json),
        motor_cmd_offset_map_json=str(args.motor_cmd_offset_map_json),
        q_offset_map_json=str(args.q_offset_map_json),
        finger_order="",
        device=str(args.device).strip().lower(),
        zero_on_exit=("true" if zero_on_exit else "false"),
    )

    for finger_name in DEFAULT_FINGER_ORDER:
        setattr(mapping_args, f"{finger_name}_model", "")
        setattr(mapping_args, f"{finger_name}_x_scaler", "")
        setattr(mapping_args, f"{finger_name}_y_scaler", "")
        setattr(mapping_args, f"{finger_name}_q_indices", "")
        setattr(mapping_args, f"{finger_name}_motor_ids", "")

    rclpy.init()
    fusion_node = WristFiveFingerFusionMappingNode(mapping_args)
    if not use_right_non_thumb_raw:
        fusion_node.latest_right_non_thumb_q16 = None
        fusion_node._update_non_thumb_q16 = lambda _raw_chunk: None
        fusion_node.get_logger().info(
            "[wrist_five_finger_fusion] right non-thumb raw-skeleton q solve is disabled by CLI flag."
        )

    executor = SingleThreadedExecutor()
    executor.add_node(fusion_node)

    spin_thread = threading.Thread(target=_spin_executor, args=(executor,), daemon=True)
    spin_thread.start()

    try:
        print(
            "进入 wrist+five-finger 融合控制模式(右手, single-node publish)\n"
            f"cmd_topic={args.cmd_topic}, feedback_topic={args.feedback_topic}\n"
            f"glove48_topic={args.glove_topic}, raw_skeleton_topic={args.glove_raw_skeleton_topic}\n"
            f"output_publish_rate_hz={output_publish_rate_hz:.1f}, "
            f"control_period_sec={max(0.002, 1.0 / output_publish_rate_hz):.3f}, "
            f"dry_run={dry_run}\n"
            f"max_step_alpha={max(0.0, float(args.max_command_step_alpha_deg)):.3f}deg/tick, "
            f"max_step_beta={max(0.0, float(args.max_command_step_beta_deg)):.3f}deg/tick\n"
            "融合逻辑: cmd[wrist4]=wrist, cmd[finger24]=finger(decouple-by-wrist-delta)\n"
            "其中 finger decouple: m_tilde = feedback - delta_wrist; "
            "recover: m_abs = m_tilde_cmd + delta_wrist\n"
            "控制结构: raw/glove 回调只缓存最新帧，100Hz control_timer 消费并发布。\n"
            "启动不阻塞等待feedback；feedback未就绪时按固定频率发送idle tension。\n"
            "按 Ctrl+C 退出。"
        )
        nonzero_text = _format_nonzero_offsets(fusion_node.motor_offsets_deg)
        if nonzero_text:
            print(f"当前启用 global motor offset(deg): {nonzero_text}")
        else:
            print("当前启用 global motor offset(deg): 全部为0")

        while rclpy.ok():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("收到 Ctrl+C，准备退出。")
    finally:
        try:
            fusion_node.enable_manual_matrix_mode(True)
        except Exception:
            pass
        time.sleep(max(0.02, (1.0 / output_publish_rate_hz) * 2.0))
        if zero_on_exit and (not dry_run):
            try:
                _execute_all_zero(fusion_node)
            except Exception as exc:
                print(f"退出回0失败: {exc}")

        executor.shutdown(timeout_sec=1.0)
        fusion_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
