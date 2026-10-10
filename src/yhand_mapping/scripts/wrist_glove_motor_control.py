#!/usr/bin/env python3
import argparse
import math
import threading
import time
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray
try:
    from tf2_msgs.msg import TFMessage
except Exception:
    TFMessage = None

from wrist_angle_motor_control import (
    DEFAULT_HAND,
    DEFAULT_OUTPUT_PUBLISH_RATE_HZ,
    DEFAULTS_PATH,
    FEEDBACK_STALE_SEC,
    MOTOR_COUNT,
    Motor,
    MotorFeedback,
    SUPPORTED_HANDS,
    YHandMsg,
    _compute_motor_targets_deg,
    _execute_all_zero,
    _format_nonzero_offsets,
    _load_defaults_from_py,
    _normalize_hand,
    _parse_motor_offsets_json,
    _to_float_or_none,
    get_hand_glove_side_code,
    get_hand_topics,
)


RAW_GLOVE_TOPIC = "/glove_joint_data_raw_skeleton"
MOCAP_TF_TOPIC = "/tf_static"
DYNAMIC_TF_TOPIC = "/tf"
SKELETON_CHUNK_LEN = 176
STATUS_PRINT_PERIOD_SEC = 1.0
INPUT_SOURCE_MANUS_RAW = "manus_raw_skeleton"
INPUT_SOURCE_MOCAP_TF = "mocap_tf"
MOCAP_HAND_FRAME_BY_HAND = {
    "left": "LeftHand",
    "right": "RightHand",
}


def _finite_float(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


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


def _relative_quat_to_alpha_beta_deg(q_rel):
    r = _quat_to_rotmat(*q_rel)
    if r is None:
        return (0.0, 0.0)

    # Manus and formula share the same coordinate frame.
    # Extract alpha/beta directly from R = Rz(alpha) * Ry(beta).
    # R = Rz(alpha) * Ry(beta).
    sin_beta = _clamp(-r[2][0], -1.0, 1.0)
    beta_rad = math.asin(sin_beta)
    alpha_rad = math.atan2(-r[0][1], r[1][1])
    return (math.degrees(alpha_rad), math.degrees(beta_rad))


class GloveToWristController(Node):
    def __init__(
        self,
        hand: str,
        cmd_topic: str,
        feedback_topic: str,
        glove_topic: str,
        input_source: str,
        mocap_tf_topic: str,
        mocap_hand_frame: str,
        defaults_hand: dict,
        output_publish_rate_hz: float,
        motor_offsets_deg: list[float],
        glove_side_code: float,
        alpha_scale: float,
        beta_scale: float,
        alpha_offset_deg: float,
        beta_offset_deg: float,
        status_print_period_sec: float,
        deadband_deg: float,
        allow_single_chunk_fallback: bool,
        fallback_chunk_index: int,
        alpha_limit_deg: float,
        beta_min_deg: float,
        beta_max_deg: float,
        max_command_step_alpha_deg: float,
        max_command_step_beta_deg: float,
        dry_run: bool,
    ):
        self.hand = _normalize_hand(hand)
        super().__init__(f"wrist_glove_motor_control_{self.hand}")
        if (not dry_run) and (YHandMsg is None or Motor is None):
            raise RuntimeError("Cannot import devices_pkg.msg. Please source ROS2 workspace first.")

        self.cmd_topic = str(cmd_topic)
        self.feedback_topic = str(feedback_topic)
        self.glove_topic = str(glove_topic)
        self.input_source = str(input_source).strip().lower()
        if self.input_source not in (INPUT_SOURCE_MANUS_RAW, INPUT_SOURCE_MOCAP_TF):
            raise ValueError(
                f"Unsupported input_source={input_source!r}; "
                f"expected one of {[INPUT_SOURCE_MANUS_RAW, INPUT_SOURCE_MOCAP_TF]}"
            )
        self.mocap_tf_topic = str(mocap_tf_topic)
        self.mocap_hand_frame = str(mocap_hand_frame).strip()
        self.input_topics = [self.glove_topic] if self.input_source == INPUT_SOURCE_MANUS_RAW else []
        self.input_topic = self.glove_topic if self.input_source == INPUT_SOURCE_MANUS_RAW else self.mocap_tf_topic
        self.input_debug_label = ""
        self.defaults_hand = defaults_hand
        self.output_publish_rate_hz = max(float(output_publish_rate_hz), 1e-6)
        self.publish_period_sec = 1.0 / self.output_publish_rate_hz
        self.motor_offsets_deg = list(motor_offsets_deg)
        self.glove_side_code = float(glove_side_code)
        self.alpha_scale = float(alpha_scale)
        self.beta_scale = float(beta_scale)
        self.alpha_offset_deg = float(alpha_offset_deg)
        self.beta_offset_deg = float(beta_offset_deg)
        self.status_print_period_sec = max(float(status_print_period_sec), 0.2)
        self.deadband_deg = max(0.0, float(deadband_deg))
        self.allow_single_chunk_fallback = bool(allow_single_chunk_fallback)
        self.fallback_chunk_index = int(fallback_chunk_index)
        self.alpha_limit_deg = abs(float(alpha_limit_deg))
        self.beta_min_deg = float(beta_min_deg)
        self.beta_max_deg = float(beta_max_deg)
        if self.beta_min_deg > self.beta_max_deg:
            raise ValueError(
                f"Invalid beta range: beta_min_deg={self.beta_min_deg} > beta_max_deg={self.beta_max_deg}"
            )
        self.max_command_step_alpha_deg = max(0.0, float(max_command_step_alpha_deg))
        self.max_command_step_beta_deg = max(0.0, float(max_command_step_beta_deg))
        self.dry_run = bool(dry_run)

        self.calibrated = False
        self.zero_quat = None
        self.latest_quat = None
        self.latest_quat_seq = 0
        self.processed_quat_seq = 0
        self.latest_side_codes = []
        self.smoothed_initialized = False
        self.smoothed_alpha_deg = 0.0
        self.smoothed_beta_deg = 0.0
        self.desired_alpha_deg = 0.0
        self.desired_beta_deg = 0.0
        self.last_cmd_alpha_deg = 0.0
        self.last_cmd_beta_deg = 0.0
        self.last_raw_alpha_deg = 0.0
        self.last_raw_beta_deg = 0.0
        self.last_rx_ts = 0.0
        self.cb_count = 0
        self.rx_count = 0
        self.bad_packet_count = 0
        self.limit_hit_count = 0
        self.fallback_pick_count = 0
        self.smooth_step_count = 0
        self.cmd_count = 0
        self.publish_count = 0
        self.last_status_ts = time.monotonic()
        self.last_status_cb_count = 0
        self.last_status_rx_count = 0
        self.last_status_cmd_count = 0
        self.last_status_publish_count = 0
        self.feedback_received = False
        self.last_feedback_ts = 0.0
        self.feedback_stale = False
        self.watchdog_fallback_enabled = True
        self.latest_feedback = [MotorFeedback() for _ in range(MOTOR_COUNT)]
        self.manual_matrix_mode = False
        self.command_matrix = self._build_idle_tension_matrix()

        glove_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        if self.input_source == INPUT_SOURCE_MANUS_RAW:
            self.glove_sub = self.create_subscription(
                Float64MultiArray,
                self.glove_topic,
                self._glove_cb,
                glove_qos,
            )
            self.mocap_tf_sub = None
            self.mocap_tf_static_sub = None
        else:
            if TFMessage is None:
                raise RuntimeError("Cannot import tf2_msgs.msg.TFMessage. Please source ROS2 workspace first.")
            self.glove_sub = None
            dynamic_tf_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            static_tf_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.mocap_tf_sub = self.create_subscription(
                TFMessage,
                DYNAMIC_TF_TOPIC,
                self._mocap_tf_cb,
                dynamic_tf_qos,
            )
            if self.mocap_tf_topic != DYNAMIC_TF_TOPIC:
                self.mocap_tf_static_sub = self.create_subscription(
                    TFMessage,
                    self.mocap_tf_topic,
                    self._mocap_tf_cb,
                    static_tf_qos,
                )
            else:
                self.mocap_tf_static_sub = None
            self.input_topics = []
            for topic in (self.mocap_tf_topic, DYNAMIC_TF_TOPIC):
                if topic not in self.input_topics:
                    self.input_topics.append(topic)
            self.input_topic = ",".join(self.input_topics)
        if self.dry_run:
            self.publisher = None
            self.feedback_sub = None
        else:
            # Match five_finger_delta_mapping_main.py / Y_Hand_Node default queue style.
            self.publisher = self.create_publisher(YHandMsg, self.cmd_topic, 10)
            self.feedback_sub = self.create_subscription(YHandMsg, self.feedback_topic, self._feedback_cb, 10)

        self.control_timer = self.create_timer(self.publish_period_sec, self._control_timer_cb)
        self.status_timer = self.create_timer(self.status_print_period_sec, self._status_timer_cb)
        self.watchdog_timer = self.create_timer(0.1, self._watchdog_cb)

        self.get_logger().info(
            "[wrist_glove_motor_control] started. "
            f"cmd_topic={self.cmd_topic}, feedback_topic={self.feedback_topic}, input_topic={self.input_topic}, "
            f"output_publish_rate_hz={self.output_publish_rate_hz:.1f}, dry_run={self.dry_run}, "
            f"input_source={self.input_source}, "
            "mode=single_node_latest_wrist_quat_timer_publish, "
            "input_cb=cache_latest_quat_only, control_timer=consume_latest_or_hold, "
            "startup=no_feedback_wait_publish_idle, cmd_qos=queue10_default, "
            f"hand={self.hand}, glove_side_code={self.glove_side_code:.3f}, deadband={self.deadband_deg:.3f}deg, "
            f"alpha_limit=±{self.alpha_limit_deg:.1f}deg, beta_range=[{self.beta_min_deg:.1f},{self.beta_max_deg:.1f}]deg, "
            f"max_step_alpha={self.max_command_step_alpha_deg:.3f}deg/tick, "
            f"max_step_beta={self.max_command_step_beta_deg:.3f}deg/tick"
        )

    def _build_idle_tension_matrix(self):
        matrix = []
        for i in range(MOTOR_COUNT):
            matrix.append([0.0, 0.0, float(self.defaults_hand["tor"][i]), 0.0, 0.0])
        return matrix

    def _build_all_zero_matrix(self):
        matrix = []
        for i in range(MOTOR_COUNT):
            matrix.append(
                [
                    0.0,
                    0.0,
                    float(self.defaults_hand["tor"][i]),
                    float(self.defaults_hand["kp"][i]),
                    float(self.defaults_hand["kd"][i]),
                ]
            )
        return matrix

    def _extract_hand_wrist_quat(self, arr: list[float]):
        n = len(arr)
        if n < SKELETON_CHUNK_LEN:
            return None, []

        full_chunks = n // SKELETON_CHUNK_LEN
        parsed_chunks = []
        for ci in range(full_chunks):
            base = ci * SKELETON_CHUNK_LEN
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
            parsed_chunks.append((ci, side, qn))

        if not parsed_chunks:
            return None, []

        side_codes = [None if side is None else float(side) for _ci, side, _q in parsed_chunks]
        for _ci, side, qn in parsed_chunks:
            if side is not None and abs(float(side) - self.glove_side_code) <= 0.25:
                return qn, side_codes

        if self.allow_single_chunk_fallback and len(parsed_chunks) == 1:
            return parsed_chunks[0][2], side_codes

        if self.fallback_chunk_index >= 0:
            for ci, _side, qn in parsed_chunks:
                if ci == self.fallback_chunk_index:
                    return qn, side_codes

        return None, side_codes

    def _glove_cb(self, msg: Float64MultiArray):
        now = time.monotonic()
        self.cb_count += 1
        data = msg.data if msg is not None else []
        if len(data) == 0:
            return

        quat, side_codes = self._extract_hand_wrist_quat(data)
        self.latest_side_codes = list(side_codes)
        if quat is None:
            self.bad_packet_count += 1
            if self.bad_packet_count % 200 == 1:
                self.get_logger().warn(
                    f"Cannot select {self.hand} wrist quaternion from raw skeleton packet "
                    f"(len={len(data)}, side_codes={side_codes}, expect_side={self.glove_side_code})."
                )
            return

        self.latest_quat = tuple(quat)
        self.latest_quat_seq += 1
        self.last_rx_ts = now
        self.rx_count += 1
        self.input_debug_label = f"side_codes={side_codes}"
        if side_codes and all(
            (sc is None) or (abs(float(sc) - self.glove_side_code) > 0.25) for sc in side_codes
        ):
            self.fallback_pick_count += 1

    def _extract_mocap_wrist_quat(self, msg):
        transforms = list(getattr(msg, "transforms", []) or [])
        if not transforms:
            return None, []

        seen_child_frames = []
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
            qx = _finite_float(getattr(rotation, "x", None))
            qy = _finite_float(getattr(rotation, "y", None))
            qz = _finite_float(getattr(rotation, "z", None))
            qw = _finite_float(getattr(rotation, "w", None))
            if None in (qx, qy, qz, qw):
                continue
            qn = _normalize_quaternion(float(qx), float(qy), float(qz), float(qw))
            if qn is None:
                continue
            return qn, seen_child_frames
        return None, seen_child_frames

    def _mocap_tf_cb(self, msg):
        now = time.monotonic()
        self.cb_count += 1
        quat, seen_child_frames = self._extract_mocap_wrist_quat(msg)
        self.latest_side_codes = [float(self.glove_side_code)]
        self.input_debug_label = (
            f"mocap_frame={self.mocap_hand_frame}, seen_children={seen_child_frames[:8]}"
        )
        if quat is None:
            self.bad_packet_count += 1
            if self.bad_packet_count % 200 == 1:
                self.get_logger().warn(
                    f"Cannot select mocap wrist quaternion for hand={self.hand} "
                    f"(frame={self.mocap_hand_frame}, seen_children={seen_child_frames[:8]})."
                )
            return

        self.latest_quat = tuple(quat)
        self.latest_quat_seq += 1
        self.last_rx_ts = now
        self.rx_count += 1

    def _feedback_cb(self, msg):
        motors = list(msg.motors) if getattr(msg, "motors", None) is not None else []
        for i in range(MOTOR_COUNT):
            if i < len(motors):
                m = motors[i]
                self.latest_feedback[i] = MotorFeedback(
                    pos=_to_float_or_none(m.pos),
                    vel=_to_float_or_none(m.vel),
                    tor=_to_float_or_none(m.tor),
                    kp=_to_float_or_none(m.kp),
                    kd=_to_float_or_none(m.kd),
                )
            else:
                self.latest_feedback[i] = MotorFeedback()
        self.feedback_received = True
        self.last_feedback_ts = time.monotonic()
        self.feedback_stale = False

    def _watchdog_cb(self):
        if self.dry_run or (not self.watchdog_fallback_enabled) or (not self.feedback_received):
            return
        if (time.monotonic() - self.last_feedback_ts) > FEEDBACK_STALE_SEC:
            self.feedback_stale = True
            if not self.manual_matrix_mode:
                self.command_matrix = self._build_idle_tension_matrix()

    def _control_timer_cb(self):
        if self.manual_matrix_mode:
            if not self.dry_run:
                self.publish_current_command(manual=False)
            return

        if (not self.dry_run) and (not self.feedback_received):
            self.command_matrix = self._build_idle_tension_matrix()
            self.publish_current_command(manual=False)
            return

        latest_quat = self.latest_quat
        latest_seq = int(self.latest_quat_seq)
        if latest_quat is None:
            if not self.dry_run:
                self.command_matrix = self._build_idle_tension_matrix()
                self.publish_current_command(manual=False)
            return

        if not self.calibrated:
            self.zero_quat = tuple(latest_quat)
            self.processed_quat_seq = latest_seq
            self.calibrated = True
            self.smoothed_initialized = True
            self.smoothed_alpha_deg = 0.0
            self.smoothed_beta_deg = 0.0
            self.desired_alpha_deg = 0.0
            self.desired_beta_deg = 0.0
            self.last_raw_alpha_deg = 0.0
            self.last_raw_beta_deg = 0.0
            self.last_cmd_alpha_deg = 0.0
            self.last_cmd_beta_deg = 0.0
            self.get_logger().info(
                f"Zero frame captured from {self.hand} wrist quaternion "
                f"(input_source={self.input_source}, debug={self.input_debug_label or self.latest_side_codes})"
            )
            if not self.dry_run:
                self.command_matrix = self._build_idle_tension_matrix()
                self.publish_current_command(manual=False)
            return

        if latest_seq != self.processed_quat_seq:
            self._update_command_from_quat(latest_quat, latest_seq)

        if not self.dry_run:
            self.publish_current_command(manual=False)

    def _update_command_from_quat(self, quat, seq: int):
        q_rel = _relative_quaternion(tuple(self.zero_quat), tuple(quat))
        if q_rel is None:
            return

        raw_alpha_deg, raw_beta_deg = _relative_quat_to_alpha_beta_deg(q_rel)
        if abs(raw_alpha_deg) < self.deadband_deg:
            raw_alpha_deg = 0.0
        if abs(raw_beta_deg) < self.deadband_deg:
            raw_beta_deg = 0.0

        desired_alpha_deg = raw_alpha_deg * self.alpha_scale + self.alpha_offset_deg
        desired_beta_deg = raw_beta_deg * self.beta_scale + self.beta_offset_deg
        cmd_alpha_deg = _clamp(desired_alpha_deg, -self.alpha_limit_deg, self.alpha_limit_deg)
        cmd_beta_deg = _clamp(desired_beta_deg, self.beta_min_deg, self.beta_max_deg)
        if (abs(cmd_alpha_deg - desired_alpha_deg) > 1e-9) or (abs(cmd_beta_deg - desired_beta_deg) > 1e-9):
            self.limit_hit_count += 1

        if not self.smoothed_initialized:
            self.smoothed_alpha_deg = 0.0
            self.smoothed_beta_deg = 0.0
            self.smoothed_initialized = True

        smooth_alpha_deg = _step_toward(
            self.smoothed_alpha_deg,
            cmd_alpha_deg,
            self.max_command_step_alpha_deg,
        )
        smooth_beta_deg = _step_toward(
            self.smoothed_beta_deg,
            cmd_beta_deg,
            self.max_command_step_beta_deg,
        )
        if (abs(smooth_alpha_deg - cmd_alpha_deg) > 1e-9) or (abs(smooth_beta_deg - cmd_beta_deg) > 1e-9):
            self.smooth_step_count += 1

        self.smoothed_alpha_deg = float(smooth_alpha_deg)
        self.smoothed_beta_deg = float(smooth_beta_deg)
        self.desired_alpha_deg = float(desired_alpha_deg)
        self.desired_beta_deg = float(desired_beta_deg)
        self.last_raw_alpha_deg = float(raw_alpha_deg)
        self.last_raw_beta_deg = float(raw_beta_deg)
        self.last_cmd_alpha_deg = float(smooth_alpha_deg)
        self.last_cmd_beta_deg = float(smooth_beta_deg)
        self.processed_quat_seq = int(seq)

        if not self.dry_run:
            _lengths, _deltas, base_targets = _compute_motor_targets_deg(
                smooth_alpha_deg,
                smooth_beta_deg,
                hand=self.hand,
            )
            targets = [float(base_targets[i]) + float(self.motor_offsets_deg[i]) for i in range(MOTOR_COUNT)]
            self.set_all_motor_position_targets(targets)

    def _publish_matrix(self, matrix):
        if self.dry_run or self.publisher is None:
            return
        msg = YHandMsg()
        msg.motors = [
            Motor(
                pos=float(r[0]),
                vel=float(r[1]),
                tor=float(r[2]),
                kp=float(r[3]),
                kd=float(r[4]),
            )
            for r in matrix
        ]
        msg.pretightenforce.flag = False
        msg.pretightenforce.torque = 0.0
        msg.pretightenforce.ms = 0
        self.publisher.publish(msg)

    def publish_current_command(self, manual: bool = False):
        _ = manual
        matrix = [row[:] for row in self.command_matrix]
        self._publish_matrix(matrix)
        self.publish_count += 1
        self.cmd_count += 1

    def publish_idle_once(self):
        self._publish_matrix(self._build_idle_tension_matrix())

    def set_idle_tension(self):
        self.command_matrix = self._build_idle_tension_matrix()

    def set_all_zero_output(self):
        self.command_matrix = [[0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(MOTOR_COUNT)]

    def set_all_zero_control(self):
        self.command_matrix = self._build_all_zero_matrix()

    def set_all_zero_control_with_overrides(self, overrides: dict[int, list]):
        matrix = self._build_all_zero_matrix()
        for override_motor_id, row in overrides.items():
            if override_motor_id < 1 or override_motor_id > MOTOR_COUNT:
                continue
            ridx = override_motor_id - 1
            matrix[ridx] = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])]
        self.command_matrix = matrix

    def set_all_motor_position_targets(self, target_positions):
        if len(target_positions) != MOTOR_COUNT:
            raise ValueError(f"target_positions length must be {MOTOR_COUNT}, got {len(target_positions)}")
        matrix = []
        for i, target in enumerate(target_positions):
            target_f = _to_float_or_none(target)
            if target_f is None:
                raise ValueError(f"target_positions[{i}] is invalid: {target}")
            matrix.append(
                [
                    float(target_f),
                    float(self.defaults_hand["vel"][i]),
                    float(self.defaults_hand["tor_active"][i]),
                    float(self.defaults_hand["kp"][i]),
                    float(self.defaults_hand["kd"][i]),
                ]
            )
        self.command_matrix = matrix

    def set_watchdog_fallback_enabled(self, enabled: bool):
        self.watchdog_fallback_enabled = bool(enabled)

    def enable_manual_matrix_mode(self, enabled: bool):
        self.manual_matrix_mode = bool(enabled)

    def get_publish_count(self):
        return int(self.publish_count)

    def all_motor_positions(self):
        return [fb.pos for fb in self.latest_feedback]

    def is_feedback_stale(self):
        if not self.feedback_received:
            return True
        return (time.monotonic() - self.last_feedback_ts) > FEEDBACK_STALE_SEC

    def _status_timer_cb(self):
        now = time.monotonic()
        dt = max(now - self.last_status_ts, 1e-6)
        cb_hz = (self.cb_count - self.last_status_cb_count) / dt
        rx_hz = (self.rx_count - self.last_status_rx_count) / dt
        cmd_hz = (self.cmd_count - self.last_status_cmd_count) / dt
        out_hz = (self.publish_count - self.last_status_publish_count) / dt
        self.last_status_ts = now
        self.last_status_cb_count = int(self.cb_count)
        self.last_status_rx_count = int(self.rx_count)
        self.last_status_cmd_count = int(self.cmd_count)
        self.last_status_publish_count = int(self.publish_count)

        rx_age = now - self.last_rx_ts if self.last_rx_ts > 0.0 else float("inf")
        fb_age = now - self.last_feedback_ts if self.last_feedback_ts > 0.0 else float("inf")

        if not self.calibrated:
            pub_details = ", ".join(
                f"{topic}:{self.count_publishers(topic)}" for topic in self.input_topics
            )
            self.get_logger().info(
                f"Waiting first valid {self.hand}-hand wrist frame on "
                f"{self.input_topic} ... "
                f"(pubs={pub_details}, feedback_ready={self.feedback_received}, "
                f"cb_hz={cb_hz:.1f}, rx_hz={rx_hz:.1f}, out_hz={out_hz:.1f}, "
                f"expect_side={self.glove_side_code}, debug={self.input_debug_label or self.latest_side_codes})"
            )
            return

        self.get_logger().info(
            "running: "
            f"raw_alpha={self.last_raw_alpha_deg:+.3f}deg, raw_beta={self.last_raw_beta_deg:+.3f}deg, "
            f"desired_alpha={self.desired_alpha_deg:+.3f}deg, desired_beta={self.desired_beta_deg:+.3f}deg, "
            f"cmd_alpha={self.last_cmd_alpha_deg:+.3f}deg, cmd_beta={self.last_cmd_beta_deg:+.3f}deg, "
            f"feedback_ready={self.feedback_received}, fb_age={fb_age:.3f}s, "
            f"cb_hz={cb_hz:.1f}, rx_hz={rx_hz:.1f}, cmd_hz={cmd_hz:.1f}, out_hz={out_hz:.1f}, "
            f"cmd_count={self.cmd_count}, bad_packets={self.bad_packet_count}, dry_run={self.dry_run}, "
            f"limit_hits={self.limit_hit_count}, smooth_steps={self.smooth_step_count}, "
            f"fallback_picks={self.fallback_pick_count}, debug={self.input_debug_label or self.latest_side_codes}, "
            f"rx_age={rx_age:.3f}s"
        )


def _spin_executor(executor):
    try:
        executor.spin()
    except ExternalShutdownException:
        return
    except KeyboardInterrupt:
        return


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Left/right glove-driven wrist controller (relative quaternion, alpha=z, beta=y)."
    )
    parser.add_argument("--hand", default=DEFAULT_HAND, choices=SUPPORTED_HANDS)
    parser.add_argument("--defaults-path", default=str(DEFAULTS_PATH))
    parser.add_argument("--cmd-topic", default="")
    parser.add_argument("--feedback-topic", default="")
    parser.add_argument("--glove-topic", default=RAW_GLOVE_TOPIC)
    parser.add_argument(
        "--input-source",
        default=INPUT_SOURCE_MANUS_RAW,
        choices=[INPUT_SOURCE_MANUS_RAW, INPUT_SOURCE_MOCAP_TF],
        help=(
            "Input wrist source. "
            "'manus_raw_skeleton' reads /glove_joint_data_raw_skeleton Float64MultiArray; "
            "'mocap_tf' reads TF wrist quaternion from mocap_main_stickman_ros2.py."
        ),
    )
    parser.add_argument(
        "--mocap-tf-topic",
        default=MOCAP_TF_TOPIC,
        help="TF topic that carries mocap LeftHand/RightHand local wrist transforms.",
    )
    parser.add_argument(
        "--mocap-hand-frame",
        default="",
        help="Override mocap wrist child frame. Default: LeftHand/RightHand chosen by --hand.",
    )
    parser.add_argument("--glove-side-code", type=float, default=None)
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument("--beta-scale", type=float, default=1.0)
    parser.add_argument("--alpha-offset-deg", type=float, default=0.0)
    parser.add_argument("--beta-offset-deg", type=float, default=0.0)
    parser.add_argument(
        "--output-publish-rate-hz",
        type=float,
        default=DEFAULT_OUTPUT_PUBLISH_RATE_HZ,
        help="Final motor command publish rate in Hz.",
    )
    parser.add_argument("--status-print-period-sec", type=float, default=STATUS_PRINT_PERIOD_SEC)
    parser.add_argument(
        "--deadband-deg",
        type=float,
        default=0.1,
        help="Set tiny relative-angle magnitudes to 0 to reduce jitter.",
    )
    parser.add_argument(
        "--alpha-limit-deg",
        type=float,
        default=5.0,
        help="Safety clamp for alpha(side-sway), command range [-limit,+limit] deg.",
    )
    parser.add_argument(
        "--beta-limit-deg",
        type=float,
        default=30.0,
        help="Safety clamp for beta(flexion), command range [-limit,+limit] deg.",
    )
    parser.add_argument(
        "--beta-min-deg",
        type=float,
        default=None,
        help="Asymmetric lower clamp for beta(flexion). Overrides symmetric --beta-limit-deg when set.",
    )
    parser.add_argument(
        "--beta-max-deg",
        type=float,
        default=None,
        help="Asymmetric upper clamp for beta(flexion). Overrides symmetric --beta-limit-deg when set.",
    )
    parser.add_argument(
        "--max-command-step-alpha-deg",
        type=float,
        default=2.0,
        help=(
            "Maximum alpha command change per received glove frame. "
            "Set 0 to disable step limiting."
        ),
    )
    parser.add_argument(
        "--max-command-step-beta-deg",
        type=float,
        default=6.0,
        help=(
            "Maximum beta command change per received glove frame. "
            "Set 0 to disable step limiting."
        ),
    )
    parser.add_argument(
        "--allow-single-chunk-fallback",
        default="true",
        choices=["true", "false"],
        help=(
            "If true and packet contains exactly one valid skeleton chunk, use it even if side_code "
            "does not match the requested hand side code."
        ),
    )
    parser.add_argument(
        "--fallback-chunk-index",
        type=int,
        default=-1,
        help=(
            "Fallback raw-skeleton chunk index (0-based) when side_code match fails. "
            "Set to 0/1 when a fixed left/right chunk order is known; -1 disables."
        ),
    )
    parser.add_argument(
        "--motor-offsets-json",
        default="",
        help=(
            "Per-motor command offset in deg. "
            "JSON list(length=28) or JSON dict keyed by motor id(1-28). "
            "Example dict: '{\"1\":1.5,\"28\":-2.0}'"
        ),
    )
    parser.add_argument(
        "--zero-on-exit",
        default="true",
        choices=["true", "false"],
        help="Execute full return-zero flow (with pre-release + timeout) before exit.",
    )
    parser.add_argument(
        "--dry-run",
        default="false",
        choices=["true", "false"],
        help="Subscribe and solve glove angles, but do not publish motor commands.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    hand = _normalize_hand(args.hand)
    input_source = str(args.input_source).strip().lower()
    default_cmd_topic, default_feedback_topic = get_hand_topics(hand)
    cmd_topic = str(args.cmd_topic).strip() or default_cmd_topic
    feedback_topic = str(args.feedback_topic).strip() or default_feedback_topic
    glove_side_code = (
        get_hand_glove_side_code(hand)
        if args.glove_side_code is None
        else float(args.glove_side_code)
    )
    defaults_path = Path(args.defaults_path).expanduser()
    defaults_hand = _load_defaults_from_py(defaults_path, hand=hand)
    motor_offsets_deg = _parse_motor_offsets_json(args.motor_offsets_json)
    zero_on_exit = str(args.zero_on_exit).strip().lower() == "true"
    dry_run = str(args.dry_run).strip().lower() == "true"
    allow_single_chunk_fallback = str(args.allow_single_chunk_fallback).strip().lower() == "true"
    beta_limit_deg = abs(float(args.beta_limit_deg))
    beta_min_deg = -beta_limit_deg if args.beta_min_deg is None else float(args.beta_min_deg)
    beta_max_deg = beta_limit_deg if args.beta_max_deg is None else float(args.beta_max_deg)
    mocap_hand_frame = str(args.mocap_hand_frame).strip() or MOCAP_HAND_FRAME_BY_HAND[hand]
    input_topic = str(args.glove_topic) if input_source == INPUT_SOURCE_MANUS_RAW else str(args.mocap_tf_topic)

    rclpy.init()
    node = GloveToWristController(
        hand=hand,
        cmd_topic=cmd_topic,
        feedback_topic=feedback_topic,
        glove_topic=str(args.glove_topic),
        input_source=input_source,
        mocap_tf_topic=str(args.mocap_tf_topic),
        mocap_hand_frame=mocap_hand_frame,
        defaults_hand=defaults_hand,
        output_publish_rate_hz=float(args.output_publish_rate_hz),
        motor_offsets_deg=motor_offsets_deg,
        glove_side_code=glove_side_code,
        alpha_scale=float(args.alpha_scale),
        beta_scale=float(args.beta_scale),
        alpha_offset_deg=float(args.alpha_offset_deg),
        beta_offset_deg=float(args.beta_offset_deg),
        status_print_period_sec=float(args.status_print_period_sec),
        deadband_deg=float(args.deadband_deg),
        allow_single_chunk_fallback=allow_single_chunk_fallback,
        fallback_chunk_index=int(args.fallback_chunk_index),
        alpha_limit_deg=float(args.alpha_limit_deg),
        beta_min_deg=beta_min_deg,
        beta_max_deg=beta_max_deg,
        max_command_step_alpha_deg=float(args.max_command_step_alpha_deg),
        max_command_step_beta_deg=float(args.max_command_step_beta_deg),
        dry_run=dry_run,
    )
    node.set_idle_tension()

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=_spin_executor, args=(executor,), daemon=True)
    spin_thread.start()

    try:
        print(
            f"进入 wrist glove 控制模式({hand})\n"
            f"cmd_topic={cmd_topic}, feedback_topic={feedback_topic}\n"
            f"最终命令发布频率: output_publish_rate_hz={float(args.output_publish_rate_hz):.1f}\n"
            f"dry_run={dry_run}（true时不发布手腕命令，退出也不回零）\n"
            f"input_source={input_source}, input_topic={input_topic}\n"
            "quat_rel_to_zero + direct_extract(Rz(alpha)*Ry(beta))\n"
            f"首帧有效 {hand} wrist 四元数作为零位，后续使用相对零位差值控制。\n"
            f"alpha_limit=±{abs(float(args.alpha_limit_deg)):.1f}deg, "
            f"beta_range=[{beta_min_deg:.1f},{beta_max_deg:.1f}]deg, "
            f"deadband={max(0.0, float(args.deadband_deg)):.3f}deg, "
            f"max_step_alpha={max(0.0, float(args.max_command_step_alpha_deg)):.3f}deg/control_tick, "
            f"max_step_beta={max(0.0, float(args.max_command_step_beta_deg)):.3f}deg/control_tick, "
            "command_mode=single_node_latest_wrist_quat_100hz_timer_hold_last\n"
            f"glove_side_code={float(glove_side_code):.3f}, "
            f"allow_single_chunk_fallback={allow_single_chunk_fallback}, "
            f"fallback_chunk_index={int(args.fallback_chunk_index)}\n"
            f"mocap_hand_frame={mocap_hand_frame}\n"
            "启动不阻塞等待feedback；feedback未就绪时按固定频率发送idle tension。\n"
            "按 Ctrl+C 退出。"
        )
        nonzero_text = _format_nonzero_offsets(motor_offsets_deg)
        if nonzero_text:
            print(f"当前启用 motor offset(deg): {nonzero_text}")
        else:
            print("当前启用 motor offset(deg): 全部为0")

        while rclpy.ok():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("收到 Ctrl+C，准备退出。")
    finally:
        if zero_on_exit and (not dry_run):
            try:
                node.enable_manual_matrix_mode(True)
                _execute_all_zero(node)
            except Exception as exc:
                print(f"退出回0失败: {exc}")

        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
