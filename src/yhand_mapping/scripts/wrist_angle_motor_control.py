#!/usr/bin/env python3
import argparse
import copy
import importlib.util
import json
import math
import threading
import time
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

try:
    from devices_pkg.msg import Motor, YHandMsg
except ImportError as exc:  # pragma: no cover
    Motor = None
    YHandMsg = None
    DEVICES_MSG_IMPORT_ERROR = exc
else:
    DEVICES_MSG_IMPORT_ERROR = None


from hand_motor_map import (
    FLEXOR_MOTOR_IDS_BY_HAND, LEFT_TO_RIGHT_MOTOR_MAP, RIGHT_TO_LEFT_MOTOR_MAP, SIDE_SWING_MOTOR_IDS_BY_HAND,
)

MOTOR_COUNT = 28
DEFAULT_OUTPUT_PUBLISH_RATE_HZ = 50.0
DEFAULT_CONTROL_PERIOD_SEC = 1.0 / DEFAULT_OUTPUT_PUBLISH_RATE_HZ
ARRIVAL_POS_TOL = 7.0
ARRIVAL_CONFIRM_FRAMES = 30
ANGLE_APPLY_TIMEOUT_SEC = 6.0
RETURN_ZERO_TIMEOUT_SEC = 6.0
FEEDBACK_STALE_SEC = 1.0
TOPIC_CONFLICT_CONFIRM_SEC = 2.0

ZERO_SWAY_RELEASE_TOR_A = 0.1
ZERO_SWAY_RELEASE_SEC = 0.0
ZERO_FLEXOR_RELEASE_TOR_A = 0.2
ZERO_FLEXOR_RELEASE_SEC = 0.5

PULLEY_DIAMETER_MM = 4.0
PULLEY_CIRCUMFERENCE_MM = math.pi * PULLEY_DIAMETER_MM

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULTS_PATH = SCRIPT_DIR / "muscle_adjustment_defaults.py"
LEFT_CMD_TOPIC = "/yhand_left_cmd"
RIGHT_CMD_TOPIC = "/yhand_right_cmd"
LEFT_FEEDBACK_TOPIC = "/yhand_left_feedback"
RIGHT_FEEDBACK_TOPIC = "/yhand_right_feedback"
SUPPORTED_HANDS = ("left", "right")
DEFAULT_HAND = "right"
HAND_CMD_TOPICS = {
    "left": LEFT_CMD_TOPIC,
    "right": RIGHT_CMD_TOPIC,
}
HAND_FEEDBACK_TOPICS = {
    "left": LEFT_FEEDBACK_TOPIC,
    "right": RIGHT_FEEDBACK_TOPIC,
}
HAND_GLOVE_SIDE_CODES = {
    "left": 0.0,
    "right": 1.0,
}

# From motor_position_test right-hand assist categories
RIGHT_SWAY_MOTOR_IDS = sorted(SIDE_SWING_MOTOR_IDS_BY_HAND['right'])
RIGHT_FLEXOR_MOTOR_IDS = sorted(FLEXOR_MOTOR_IDS_BY_HAND['right'])
# Shared anatomical correspondence is imported above.
# Shared anatomical correspondence is imported above.
LEFT_SWAY_MOTOR_IDS = sorted(SIDE_SWING_MOTOR_IDS_BY_HAND['left'])
LEFT_FLEXOR_MOTOR_IDS = sorted(FLEXOR_MOTOR_IDS_BY_HAND['left'])
PRE_RELEASE_MOTOR_IDS_BY_HAND = {
    "left": {
        "sway": LEFT_SWAY_MOTOR_IDS,
        "flexor": LEFT_FLEXOR_MOTOR_IDS,
    },
    "right": {
        "sway": RIGHT_SWAY_MOTOR_IDS,
        "flexor": RIGHT_FLEXOR_MOTOR_IDS,
    },
}

# Channel IDs follow hand_motor_map.py; geometry values migrate with each muscle.
# Geometry extracted from
# Yhand_wrist_coordinates_and_distance_formulas_combined_LR_code_input_deg.xlsx.
# Values are:
#   motor_id: (band_y, band_z, base_y, base_z, initial_l0_mm)
WRIST_GEOMETRY_BY_HAND = {'right': {17: (0.0, -17.75, 11.691, -6.75, 34.91580630602707),
           2: (-1.5, -15.25, -12.697, -8.89, 33.57464010231532),
           3: (-3.0, -17.75, -11.691, -6.75, 34.02892196352979),
           27: (-4.5, -15.25, -14.048, -6.551, 33.589744774261085),
           19: (-19.25, -9.25, -15.441, -1.351, 32.22323278319542),
           7: (-11.569, 11.326, -14.048, 6.551, 31.4703052892723),
           24: (-6.121, 14.837, -13.295, 2.344, 34.190281864880845),
           8: (-2.165, 11.326, -12.686, 4.617, 33.42358704567779),
           9: (0.335, 14.779, 11.691, 6.75, 33.98319622990162),
           11: (7.36, 13.735, 12.697, 8.89, 31.833812888185417),
           26: (11.078, 13.735, 14.048, 6.551, 31.966620168544566),
           20: (19.0, 11.326, 15.441, 1.351, 32.76585349109649),
           14: (3.0, -12.75, 12.697, -8.89, 32.716440179212654),
           18: (4.5, -15.25, 14.048, -6.551, 33.589744774261085),
           15: (0.0, -12.75, -12.686, -4.617, 34.4748362432659),
           16: (1.5, -15.25, 12.686, -4.617, 34.63555014721146),
           1: (3.0, -17.75, 13.295, -2.344, 36.12154357166925),
           5: (13.5, -10.25, 14.972, -4.012, 31.662493221475785),
           13: (19.0, -17.0, 15.441, -1.351, 34.914062081058404),
           21: (12.635, 11.326, 14.972, 4.012, 31.943547298319892),
           22: (9.684, 11.326, 13.295, 2.344, 32.483067804627076),
           10: (5.61, 11.326, 12.686, 4.617, 32.50406906834897),
           23: (1.835, 11.326, -11.691, 6.75, 34.136878899512766),
           25: (-10.0, 15.0, -12.697, 8.89, 31.71813295261876),
           12: (-18.0, 10.25, -14.972, 4.012, 31.772873288388634),
           6: (-15.38, 11.326, -15.441, 1.351, 32.572049290764625),
           4: (-10.0, -17.0, -14.972, -4.012, 33.982980696224985),
           28: (-3.0, -12.75, -13.295, -2.344, 34.288568211577456)},
 'left': {1: (0.0, -12.75, -12.686, -4.617, 34.4748362432659),
          2: (0.0, -17.75, 12.686, -4.617, 35.98394550351587),
          15: (-3.0, -17.75, -12.697, -8.89, 33.674403602736604),
          5: (19.25, -9.25, 15.441, -1.351, 32.22323278319542),
          27: (-9.5, -20.211, -14.972, -4.012, 35.40881859085389),
          7: (-11.569, 11.326, -14.048, 6.551, 31.4703052892723),
          8: (-7.664, 11.326, -13.295, 2.344, 32.76917048080406),
          24: (-2.165, 11.326, -12.686, 4.617, 33.42358704567779),
          9: (1.835, 11.326, -11.691, 6.75, 34.136878899512766),
          11: (14.546, 13.735, 14.048, 6.551, 31.832246370622357),
          12: (-19.25, 7.25, -15.441, 1.351, 31.792148889309132),
          20: (19.0, 11.326, 14.972, 4.012, 32.11157780302924),
          18: (6.0, -12.75, 14.972, -4.012, 33.44074576022491),
          14: (3.0, -12.75, 13.295, -2.344, 34.288568211577456),
          3: (-1.5, -15.25, -11.691, -6.75, 33.72744475942404),
          16: (1.5, -15.25, 11.691, -6.75, 33.72744475942404),
          17: (3.0, -17.75, 12.697, -8.89, 33.674403602736604),
          13: (4.5, -15.25, 14.048, -6.551, 33.589744774261085),
          19: (-19.25, -9.25, -15.441, -1.351, 32.22323278319542),
          21: (9.684, 11.326, 13.295, 2.344, 32.483067804627076),
          10: (7.36, 13.735, 12.697, 8.89, 31.833812888185417),
          22: (5.61, 11.326, 12.686, 4.617, 32.50406906834897),
          23: (0.335, 14.779, 11.691, 6.75, 33.98319622990162),
          25: (-10.0, 15.0, -12.697, 8.89, 31.71813295261876),
          26: (19.25, 7.25, 15.441, 1.351, 31.792148889309132),
          6: (-15.38, 11.326, -14.972, 4.012, 31.86055726129096),
          28: (-3.0, -12.75, -13.295, -2.344, 34.288568211577456),
          4: (-4.5, -15.25, -14.048, -6.551, 33.589744774261085)}}


class MotorFeedback:
    def __init__(self, pos=None, vel=None, tor=None, kp=None, kd=None):
        self.pos = pos
        self.vel = vel
        self.tor = tor
        self.kp = kp
        self.kd = kd


def _to_float_or_none(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _sanitize_rate_hz(rate_hz, default_hz: float):
    rv = _to_float_or_none(rate_hz)
    if rv is None or rv <= 0.0:
        rv = float(default_hz)
    return float(rv)


def _normalize_hand(hand: str) -> str:
    value = str(hand or DEFAULT_HAND).strip().lower()
    if value not in SUPPORTED_HANDS:
        raise ValueError(f"Unsupported hand={hand!r}, expected one of {SUPPORTED_HANDS}")
    return value


def get_hand_topics(hand: str) -> tuple[str, str]:
    hand_key = _normalize_hand(hand)
    return HAND_CMD_TOPICS[hand_key], HAND_FEEDBACK_TOPICS[hand_key]


def get_hand_glove_side_code(hand: str) -> float:
    return float(HAND_GLOVE_SIDE_CODES[_normalize_hand(hand)])


def _normalize_motor_offsets(offsets_raw):
    offsets = [0.0] * MOTOR_COUNT
    if offsets_raw is None:
        return offsets

    if isinstance(offsets_raw, list):
        if len(offsets_raw) != MOTOR_COUNT:
            raise ValueError(
                f"motor offsets list length must be {MOTOR_COUNT}, got {len(offsets_raw)}"
            )
        for i, v in enumerate(offsets_raw):
            fv = _to_float_or_none(v)
            if fv is None:
                raise ValueError(f"motor offsets list index {i} is invalid: {v!r}")
            offsets[i] = float(fv)
        return offsets

    if isinstance(offsets_raw, dict):
        for key, value in offsets_raw.items():
            try:
                motor_id = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid motor id in offsets dict: {key!r}") from exc
            if motor_id < 1 or motor_id > MOTOR_COUNT:
                raise ValueError(f"motor id out of range in offsets dict: {motor_id}")
            fv = _to_float_or_none(value)
            if fv is None:
                raise ValueError(f"invalid offset for motor {motor_id}: {value!r}")
            offsets[motor_id - 1] = float(fv)
        return offsets

    raise ValueError(
        "motor offsets must be either a 28-length JSON list or a JSON object keyed by motor id(1-28)"
    )


def _parse_motor_offsets_json(offsets_text: str):
    text = str(offsets_text or "").strip()
    if text == "":
        return [0.0] * MOTOR_COUNT
    try:
        raw = json.loads(text)
    except Exception as exc:
        raise ValueError(f"failed to parse --motor-offsets-json: {exc}") from exc
    return _normalize_motor_offsets(raw)


def _format_nonzero_offsets(offsets):
    parts = []
    for i, v in enumerate(offsets, start=1):
        if abs(float(v)) > 1e-12:
            parts.append(f"m{i}:{float(v):+.3f}")
    return ", ".join(parts)


def _ensure_defaults_shape(defaults_map, hand: str = DEFAULT_HAND):
    hand_key = _normalize_hand(hand)
    hand_map = defaults_map[hand_key]
    for key, fallback in (("vel", 0.0), ("kp", 500.0), ("kd", 10.0), ("tor", -0.01)):
        arr = list(hand_map.get(key, []))
        if len(arr) < MOTOR_COUNT:
            arr += [fallback] * (MOTOR_COUNT - len(arr))
        elif len(arr) > MOTOR_COUNT:
            arr = arr[:MOTOR_COUNT]
        hand_map[key] = [fallback if _to_float_or_none(v) is None else float(v) for v in arr]

    tor_active = list(hand_map.get("tor_active", hand_map["tor"]))
    if len(tor_active) < MOTOR_COUNT:
        for i in range(len(tor_active), MOTOR_COUNT):
            tor_active.append(hand_map["tor"][i])
    elif len(tor_active) > MOTOR_COUNT:
        tor_active = tor_active[:MOTOR_COUNT]
    hand_map["tor_active"] = [
        hand_map["tor"][i] if _to_float_or_none(v) is None else float(v)
        for i, v in enumerate(tor_active)
    ]


def _load_defaults_from_py(defaults_path: Path, hand: str = DEFAULT_HAND):
    hand_key = _normalize_hand(hand)
    spec = importlib.util.spec_from_file_location("muscle_adjustment_defaults_dynamic", str(defaults_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load defaults file: {defaults_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    defaults = getattr(mod, "DEFAULTS", None)
    if not isinstance(defaults, dict) or hand_key not in defaults:
        raise RuntimeError(f"DEFAULTS['{hand_key}'] not found in {defaults_path}")
    defaults = copy.deepcopy(defaults)
    _ensure_defaults_shape(defaults, hand=hand_key)
    return defaults[hand_key]


class WristAngleControlNode(Node):
    def __init__(
        self,
        hand: str,
        cmd_topic: str,
        feedback_topic: str,
        defaults_hand: dict,
        output_publish_rate_hz: float = DEFAULT_OUTPUT_PUBLISH_RATE_HZ,
        suppress_timer_after_manual_publish: bool = False,
        enable_publish_timer: bool = True,
    ):
        self.hand = _normalize_hand(hand)
        super().__init__(f"wrist_angle_control_{self.hand}")
        if YHandMsg is None or Motor is None:
            raise RuntimeError("Cannot import devices_pkg.msg. Please source ROS2 workspace first.")

        self.cmd_topic = cmd_topic
        self.feedback_topic = feedback_topic
        self.defaults_hand = defaults_hand
        self.output_publish_rate_hz = _sanitize_rate_hz(
            output_publish_rate_hz, DEFAULT_OUTPUT_PUBLISH_RATE_HZ
        )
        self.publish_period_sec = 1.0 / self.output_publish_rate_hz
        self.suppress_timer_after_manual_publish = bool(suppress_timer_after_manual_publish)
        self.enable_publish_timer = bool(enable_publish_timer)

        self._lock = threading.Lock()
        self.latest_feedback = [MotorFeedback() for _ in range(MOTOR_COUNT)]
        self.feedback_received = False
        self.last_feedback_ts = 0.0
        self.feedback_stale = False
        self.watchdog_fallback_enabled = True
        self.command_matrix = self._build_idle_tension_matrix()
        self.publish_count = 0
        self.last_manual_publish_ts = 0.0

        latest_cmd_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        latest_feedback_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.publisher = self.create_publisher(YHandMsg, self.cmd_topic, latest_cmd_qos)
        self.subscription = self.create_subscription(
            YHandMsg, self.feedback_topic, self._feedback_cb, latest_feedback_qos
        )
        self.publish_timer = (
            self.create_timer(self.publish_period_sec, self._publish_cb)
            if self.enable_publish_timer
            else None
        )
        self.watchdog_timer = self.create_timer(0.1, self._watchdog_cb)

        self.get_logger().info(
            "[wrist_angle_control] started. "
            f"cmd_topic={self.cmd_topic}, feedback_topic={self.feedback_topic}, "
            f"output_publish_rate_hz={self.output_publish_rate_hz:.1f}, "
            f"publish_timer_enabled={self.enable_publish_timer}"
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

    def _feedback_cb(self, msg):
        motors = list(msg.motors) if getattr(msg, "motors", None) is not None else []
        now = time.monotonic()
        with self._lock:
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
            self.last_feedback_ts = now
            self.feedback_stale = False

    def _publish_cb(self):
        if self.suppress_timer_after_manual_publish:
            with self._lock:
                last_manual_publish_ts = float(self.last_manual_publish_ts)
            if (
                last_manual_publish_ts > 0.0
                and (time.monotonic() - last_manual_publish_ts) < (self.publish_period_sec * 1.5)
            ):
                return

        self.publish_current_command(manual=False)

    def publish_current_command(self, manual: bool = False):
        with self._lock:
            matrix = [row[:] for row in self.command_matrix]
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
        with self._lock:
            self.publish_count += 1
            if manual:
                self.last_manual_publish_ts = time.monotonic()

    def _watchdog_cb(self):
        with self._lock:
            if not self.watchdog_fallback_enabled:
                return
            if not self.feedback_received:
                return
            age = time.monotonic() - self.last_feedback_ts
        if age > FEEDBACK_STALE_SEC:
            with self._lock:
                self.feedback_stale = True
                self.command_matrix = self._build_idle_tension_matrix()

    def wait_feedback(self, timeout_sec: float):
        start = time.monotonic()
        while time.monotonic() - start < timeout_sec:
            with self._lock:
                if self.feedback_received:
                    return True
            time.sleep(0.05)
        return False

    def is_feedback_stale(self):
        with self._lock:
            if not self.feedback_received:
                return True
            return (time.monotonic() - self.last_feedback_ts) > FEEDBACK_STALE_SEC

    def check_topic_safety(self, timeout_sec: float = 10.0):
        conflict_start_ts = None
        latest_pub_count = 0
        latest_pub_nodes_text = "<unknown>"
        start = time.monotonic()
        while time.monotonic() - start < timeout_sec:
            now = time.monotonic()
            pub_count = self.count_publishers(self.cmd_topic)
            cmd_sub_count = self.count_subscribers(self.cmd_topic)
            fb_pub_count = self.count_publishers(self.feedback_topic)
            latest_pub_count = pub_count

            if pub_count > 1:
                pub_infos = self.get_publishers_info_by_topic(self.cmd_topic)
                pub_nodes = []
                for info in pub_infos:
                    ns = info.node_namespace or "/"
                    name = info.node_name or "<unknown>"
                    pub_nodes.append(f"{ns}{name}")
                latest_pub_nodes_text = ", ".join(sorted(set(pub_nodes))) if pub_nodes else "<unknown>"
                if conflict_start_ts is None:
                    conflict_start_ts = now
                if (now - conflict_start_ts) >= TOPIC_CONFLICT_CONFIRM_SEC:
                    return False, (
                        f"Command topic conflict on {self.cmd_topic}. "
                        f"publishers={pub_count}, publisher_nodes=[{latest_pub_nodes_text}]"
                    )
            else:
                conflict_start_ts = None

            if cmd_sub_count >= 1 and fb_pub_count >= 1:
                return True, ""
            time.sleep(0.1)

        if latest_pub_count > 1:
            return False, (
                f"Command topic conflict on {self.cmd_topic}. "
                f"publishers={latest_pub_count}, publisher_nodes=[{latest_pub_nodes_text}]"
            )
        return False, (
            f"Driver topic not ready. cmd_subscribers={self.count_subscribers(self.cmd_topic)}, "
            f"feedback_publishers={self.count_publishers(self.feedback_topic)}."
        )

    def all_motor_positions(self):
        with self._lock:
            return [fb.pos for fb in self.latest_feedback]

    def set_idle_tension(self):
        with self._lock:
            self.command_matrix = self._build_idle_tension_matrix()

    def set_all_zero_output(self):
        with self._lock:
            self.command_matrix = [[0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(MOTOR_COUNT)]

    def set_all_zero_control(self):
        with self._lock:
            self.command_matrix = self._build_all_zero_matrix()

    def set_all_zero_control_with_overrides(self, overrides: dict[int, list]):
        with self._lock:
            matrix = self._build_all_zero_matrix()
            for override_motor_id, row in overrides.items():
                if override_motor_id < 1 or override_motor_id > MOTOR_COUNT:
                    continue
                ridx = override_motor_id - 1
                matrix[ridx] = [
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                ]
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
        with self._lock:
            self.command_matrix = matrix

    def set_watchdog_fallback_enabled(self, enabled: bool):
        with self._lock:
            self.watchdog_fallback_enabled = bool(enabled)

    def get_publish_count(self):
        with self._lock:
            return int(self.publish_count)


def _compute_motor_lengths_mm(alpha_deg: float, beta_deg: float, hand: str = DEFAULT_HAND):
    hand_key = _normalize_hand(hand)
    alpha = math.radians(float(alpha_deg))
    beta = math.radians(float(beta_deg))
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    cb = math.cos(beta)
    sb = math.sin(beta)

    lengths = [0.0] * MOTOR_COUNT
    geom = WRIST_GEOMETRY_BY_HAND[hand_key]
    for motor_id in range(1, MOTOR_COUNT + 1):
        band_y, band_z, base_y, base_z, _l0 = geom[motor_id]
        a_term = 16.000 + 7.010 * cb + float(band_z) * sb
        x_rot = 7.997 + a_term * ca - float(band_y) * sa
        y_rot = a_term * sa + float(band_y) * ca
        z_rot = -7.010 * sb + float(band_z) * cb
        lengths[motor_id - 1] = math.sqrt(
            x_rot * x_rot
            + (y_rot - float(base_y)) ** 2
            + (z_rot - float(base_z)) ** 2
        )
    return lengths


def _length_to_motor_pos_deg(length_mm: float, initial_length_mm: float):
    delta_mm = float(length_mm) - float(initial_length_mm)
    return delta_mm * 360.0 / PULLEY_CIRCUMFERENCE_MM


def _compute_motor_targets_deg(alpha_deg: float, beta_deg: float, hand: str = DEFAULT_HAND):
    hand_key = _normalize_hand(hand)
    lengths = _compute_motor_lengths_mm(alpha_deg, beta_deg, hand=hand_key)
    targets = []
    deltas = []
    geom = WRIST_GEOMETRY_BY_HAND[hand_key]
    for i in range(1, MOTOR_COUNT + 1):
        li = lengths[i - 1]
        l0 = float(geom[i][4])
        delta = float(li) - float(l0)
        pos = _length_to_motor_pos_deg(li, l0)
        deltas.append(delta)
        targets.append(pos)
    return lengths, deltas, targets


def _evaluate_reached(node: WristAngleControlNode, target_positions: list[float]):
    pos_list = node.all_motor_positions()
    missing_ids = []
    failed_ids = []
    max_err = 0.0

    for i, target in enumerate(target_positions, start=1):
        cur = pos_list[i - 1] if (i - 1) < len(pos_list) else None
        if cur is None:
            missing_ids.append(i)
            failed_ids.append(i)
            continue
        err = abs(float(cur) - float(target))
        if err > max_err:
            max_err = err
        if err > ARRIVAL_POS_TOL:
            failed_ids.append(i)
    reached = (not missing_ids) and (not failed_ids)
    return reached, failed_ids, missing_ids, max_err


def _apply_targets_and_wait(node: WristAngleControlNode, target_positions: list[float], timeout_sec: float):
    node.set_all_motor_position_targets(target_positions)
    start = time.monotonic()
    ok_frames = 0
    result = "timeout"
    last_failed_ids = []
    last_missing_ids = []
    last_max_err = 0.0

    node.set_watchdog_fallback_enabled(False)
    try:
        while True:
            if node.is_feedback_stale():
                result = "feedback_stale"
                break
            reached, failed_ids, missing_ids, max_err = _evaluate_reached(node, target_positions)
            last_failed_ids = failed_ids
            last_missing_ids = missing_ids
            last_max_err = max_err

            if reached:
                ok_frames += 1
            else:
                ok_frames = 0
            if ok_frames >= ARRIVAL_CONFIRM_FRAMES:
                result = "reached"
                break
            if (time.monotonic() - start) >= timeout_sec:
                break
            time.sleep(max(0.001, float(getattr(node, "publish_period_sec", DEFAULT_CONTROL_PERIOD_SEC))))
    finally:
        node.set_watchdog_fallback_enabled(True)

    return {
        "result": result,
        "elapsed_sec": float(time.monotonic() - start),
        "failed_ids": list(last_failed_ids),
        "missing_ids": list(last_missing_ids),
        "max_error": float(last_max_err),
    }


def _all_reached_zero(node: WristAngleControlNode):
    pos_list = node.all_motor_positions()
    available = [p for p in pos_list if p is not None]
    if not available:
        return False
    for p in available:
        if abs(float(p)) > ARRIVAL_POS_TOL:
            return False
    return True


def _zero_failure_details(node: WristAngleControlNode):
    pos_list = node.all_motor_positions()
    failed_ids = []
    for i, p in enumerate(pos_list, start=1):
        if p is None:
            failed_ids.append(i)
            continue
        if abs(float(p)) > ARRIVAL_POS_TOL:
            failed_ids.append(i)
    pos_fmt = ", ".join(
        [f"m{i + 1}:{('None' if p is None else f'{float(p):.3f}')}" for i, p in enumerate(pos_list)]
    )
    return failed_ids, pos_fmt


def _execute_all_zero(node: WristAngleControlNode):
    print("开始全部回0...")
    zero_start = None
    ok_frames = 0
    result = "timeout"
    pre_release_groups = PRE_RELEASE_MOTOR_IDS_BY_HAND.get(node.hand, {})
    sway_motor_ids = list(pre_release_groups.get("sway", []))
    flexor_motor_ids = list(pre_release_groups.get("flexor", []))
    node.set_watchdog_fallback_enabled(False)
    try:
        pre_release_total_sec = max(float(ZERO_SWAY_RELEASE_SEC), float(ZERO_FLEXOR_RELEASE_SEC))
        if pre_release_total_sec > 0.0 and (sway_motor_ids or flexor_motor_ids):
            if sway_motor_ids:
                print(
                    "回0前侧摆放线: "
                    f"motors={sway_motor_ids}, tor={ZERO_SWAY_RELEASE_TOR_A:.3f}A, "
                    f"duration={ZERO_SWAY_RELEASE_SEC:.3f}s"
                )
            if flexor_motor_ids:
                print(
                    "回0前屈肌放线: "
                    f"motors={flexor_motor_ids}, tor={ZERO_FLEXOR_RELEASE_TOR_A:.3f}A, "
                    f"duration={ZERO_FLEXOR_RELEASE_SEC:.3f}s"
                )
            release_start = time.monotonic()
            while (time.monotonic() - release_start) < pre_release_total_sec:
                if node.is_feedback_stale():
                    result = "feedback_stale"
                    break
                elapsed = time.monotonic() - release_start
                overrides = {}
                if sway_motor_ids and elapsed < float(ZERO_SWAY_RELEASE_SEC):
                    for mid in sway_motor_ids:
                        overrides[int(mid)] = [0.0, 0.0, float(ZERO_SWAY_RELEASE_TOR_A), 0.0, 0.0]
                if flexor_motor_ids and elapsed < float(ZERO_FLEXOR_RELEASE_SEC):
                    for mid in flexor_motor_ids:
                        cur_row = overrides.get(int(mid))
                        cur = float(cur_row[2]) if cur_row is not None else 0.0
                        overrides[int(mid)] = [0.0, 0.0, max(cur, float(ZERO_FLEXOR_RELEASE_TOR_A)), 0.0, 0.0]
                node.set_all_zero_control_with_overrides(overrides)
                time.sleep(max(0.001, float(getattr(node, "publish_period_sec", DEFAULT_CONTROL_PERIOD_SEC))))

        if result != "feedback_stale":
            node.set_all_zero_control()
            zero_start = time.monotonic()
            while True:
                if node.is_feedback_stale():
                    result = "feedback_stale"
                    break
                if _all_reached_zero(node):
                    ok_frames += 1
                else:
                    ok_frames = 0
                if ok_frames >= ARRIVAL_CONFIRM_FRAMES:
                    result = "reached"
                    break
                if (time.monotonic() - zero_start) >= RETURN_ZERO_TIMEOUT_SEC:
                    break
                time.sleep(max(0.001, float(getattr(node, "publish_period_sec", DEFAULT_CONTROL_PERIOD_SEC))))
    finally:
        node.set_watchdog_fallback_enabled(True)

    node.set_idle_tension()
    elapsed = time.monotonic() - zero_start if zero_start is not None else 0.0
    if result == "reached":
        print(f"全部回0完成：已达到阈值（{elapsed:.2f}s）并切回默认张紧。")
    elif result == "feedback_stale":
        print("全部回0中断：反馈超时，已切回默认张紧。")
        failed_ids, pos_fmt = _zero_failure_details(node)
        print(f"回0失败电机ID: {failed_ids if failed_ids else '[]'}")
        print(f"当前28个电机位置: {pos_fmt}")
    else:
        print(f"全部回0超时（{elapsed:.2f}s），已切回默认张紧。")
        failed_ids, pos_fmt = _zero_failure_details(node)
        print(f"回0失败电机ID: {failed_ids if failed_ids else '[]'}")
        print(f"当前28个电机位置: {pos_fmt}")
    return result


def _print_compact_result(
    alpha_deg: float,
    beta_deg: float,
    deltas: list[float],
    base_targets: list[float],
    final_targets: list[float],
):
    print(f"角度命令: alpha={alpha_deg:.3f}deg, beta={beta_deg:.3f}deg")
    print("m01-m28 ΔL(mm):")
    print(", ".join([f"m{i + 1}:{deltas[i]:+.3f}" for i in range(MOTOR_COUNT)]))
    print("m01-m28 pos_base(deg):")
    print(", ".join([f"m{i + 1}:{base_targets[i]:+.2f}" for i in range(MOTOR_COUNT)]))
    print("m01-m28 pos_cmd(deg, after offset):")
    print(", ".join([f"m{i + 1}:{final_targets[i]:+.2f}" for i in range(MOTOR_COUNT)]))


def _run_interactive_loop(node: WristAngleControlNode, apply_timeout_sec: float, motor_offsets_deg: list[float]):
    help_text = (
        "\n输入格式:\n"
        "  <alpha_deg> <beta_deg>   例如: 10 -8\n"
        "  z                         全部回0\n"
        "  x                         退出(退出前自动回0)\n"
        "  h                         显示帮助\n"
    )
    print(help_text)
    while True:
        raw = input("请输入腕部角度或指令: ").strip().lower()
        if raw in ("x", "quit", "exit"):
            _execute_all_zero(node)
            return
        if raw in ("h", "help"):
            print(help_text)
            continue
        if raw in ("z", "zero"):
            _execute_all_zero(node)
            continue

        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            print("输入无效，请输入: <alpha_deg> <beta_deg>，或 z/x/h")
            continue
        try:
            alpha_deg = float(parts[0])
            beta_deg = float(parts[1])
        except ValueError:
            print("角度必须是数字，请重试。")
            continue

        lengths, deltas, base_targets = _compute_motor_targets_deg(alpha_deg, beta_deg, hand=node.hand)
        targets = [float(base_targets[i]) + float(motor_offsets_deg[i]) for i in range(MOTOR_COUNT)]
        _ = lengths  # keep for potential future logging
        _print_compact_result(alpha_deg, beta_deg, deltas, base_targets, targets)

        result = _apply_targets_and_wait(node, targets, timeout_sec=apply_timeout_sec)
        if result["result"] == "reached":
            print(
                f"执行完成: reached, elapsed={result['elapsed_sec']:.2f}s, max_error={result['max_error']:.3f}"
            )
        elif result["result"] == "feedback_stale":
            node.set_idle_tension()
            print("执行中断: 反馈超时，已切回默认张紧。")
        else:
            node.set_idle_tension()
            print(
                f"执行超时: elapsed={result['elapsed_sec']:.2f}s, max_error={result['max_error']:.3f}, "
                f"failed_ids={result['failed_ids']}"
            )


def _run_continuous_trajectory_test(
    node: WristAngleControlNode,
    motor_offsets_deg: list[float],
    count: int,
    rate_hz: float,
    alpha_min_deg: float,
    alpha_max_deg: float,
    beta_min_deg: float,
    beta_max_deg: float,
):
    count = max(0, int(count))
    rate_hz = _sanitize_rate_hz(rate_hz, 100.0)
    period_sec = 1.0 / rate_hz
    alpha_center = (float(alpha_min_deg) + float(alpha_max_deg)) * 0.5
    beta_center = (float(beta_min_deg) + float(beta_max_deg)) * 0.5
    alpha_amp = abs(float(alpha_max_deg) - float(alpha_min_deg)) * 0.5
    beta_amp = abs(float(beta_max_deg) - float(beta_min_deg)) * 0.5

    print(
        f"进入 wrist angle 连续曲线轨迹测试模式({node.hand})\n"
        f"轨迹点数: {count}, 发布频率: {rate_hz:.1f}Hz, 周期: {period_sec * 1000.0:.3f}ms\n"
        f"alpha侧摆范围: [{alpha_min_deg:.3f}, {alpha_max_deg:.3f}]deg\n"
        f"beta弯曲范围: [{beta_min_deg:.3f}, {beta_max_deg:.3f}]deg\n"
        "轨迹: alpha=sin, beta=cos, 5秒一圈(默认500点@100Hz)"
    )
    if count <= 0:
        print("轨迹点数为0，跳过测试。")
        return

    start_ts = time.monotonic()
    next_ts = start_ts
    min_sleep_margin = float("inf")
    late_count = 0
    last_alpha = 0.0
    last_beta = 0.0

    for idx in range(1, count + 1):
        u = (idx - 1) / max(count - 1, 1)
        phase = 2.0 * math.pi * u
        alpha_deg = alpha_center + alpha_amp * math.sin(phase)
        beta_deg = beta_center + beta_amp * math.cos(phase)
        _lengths, _deltas, base_targets = _compute_motor_targets_deg(alpha_deg, beta_deg, hand=node.hand)
        targets = [float(base_targets[i]) + float(motor_offsets_deg[i]) for i in range(MOTOR_COUNT)]
        node.set_all_motor_position_targets(targets)
        node.publish_current_command(manual=True)

        last_alpha = float(alpha_deg)
        last_beta = float(beta_deg)
        if idx == 1 or idx == count or idx % 50 == 0:
            elapsed = time.monotonic() - start_ts
            actual_hz = idx / max(elapsed, 1e-9)
            print(
                f"[trajectory_test] {idx:04d}/{count}: "
                f"alpha={last_alpha:+.3f}deg, beta={last_beta:+.3f}deg, "
                f"elapsed={elapsed:.3f}s, avg_publish={actual_hz:.1f}Hz"
            )

        next_ts += period_sec
        sleep_sec = next_ts - time.monotonic()
        min_sleep_margin = min(min_sleep_margin, sleep_sec)
        if sleep_sec > 0.0:
            time.sleep(sleep_sec)
        else:
            late_count += 1

    total_sec = time.monotonic() - start_ts
    avg_hz = count / max(total_sec, 1e-9)
    print(
        "连续曲线轨迹测试完成: "
        f"sent={count}, elapsed={total_sec:.3f}s, avg_publish={avg_hz:.1f}Hz, "
        f"late_frames={late_count}, min_sleep_margin={min_sleep_margin * 1000.0:.3f}ms, "
        f"last_alpha={last_alpha:+.3f}deg, last_beta={last_beta:+.3f}deg"
    )


def _spin_node(node: Node):
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        return
    except KeyboardInterrupt:
        return


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Left/right wrist-angle to 28-motor position controller."
    )
    parser.add_argument("--hand", default=DEFAULT_HAND, choices=SUPPORTED_HANDS)
    parser.add_argument("--defaults-path", default=str(DEFAULTS_PATH))
    parser.add_argument("--cmd-topic", default="")
    parser.add_argument("--feedback-topic", default="")
    parser.add_argument(
        "--output-publish-rate-hz",
        type=float,
        default=DEFAULT_OUTPUT_PUBLISH_RATE_HZ,
        help="Final motor command publish rate in Hz.",
    )
    parser.add_argument("--apply-timeout-sec", type=float, default=ANGLE_APPLY_TIMEOUT_SEC)
    parser.add_argument(
        "--motor-offsets-json",
        default="",
        help=(
            "Per-motor command offset in deg. "
            "JSON list(length=28) or JSON dict keyed by motor id(1-28). "
            "Example list: '[0,0,...]'; example dict: '{\"1\":1.5,\"28\":-2.0}'"
        ),
    )
    parser.add_argument(
        "--random-test",
        default="false",
        choices=["true", "false"],
        help="Run continuous in-range trajectory streaming test instead of interactive input.",
    )
    parser.add_argument("--random-test-count", type=int, default=500)
    parser.add_argument("--random-test-rate-hz", type=float, default=100.0)
    parser.add_argument("--random-alpha-min-deg", type=float, default=-5.0)
    parser.add_argument("--random-alpha-max-deg", type=float, default=5.0)
    parser.add_argument("--random-beta-min-deg", type=float, default=-20.0)
    parser.add_argument("--random-beta-max-deg", type=float, default=20.0)
    parser.add_argument(
        "--zero-on-exit",
        default="true",
        choices=["true", "false"],
        help="Execute return-zero before exiting random test or after Ctrl+C.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    hand = _normalize_hand(args.hand)
    default_cmd_topic, default_feedback_topic = get_hand_topics(hand)
    cmd_topic = str(args.cmd_topic).strip() or default_cmd_topic
    feedback_topic = str(args.feedback_topic).strip() or default_feedback_topic
    defaults_path = Path(args.defaults_path).expanduser()
    defaults_hand = _load_defaults_from_py(defaults_path, hand=hand)
    motor_offsets_deg = _parse_motor_offsets_json(args.motor_offsets_json)
    random_test = str(args.random_test).strip().lower() == "true"
    zero_on_exit = str(args.zero_on_exit).strip().lower() == "true"

    rclpy.init()
    node = WristAngleControlNode(
        hand=hand,
        cmd_topic=cmd_topic,
        feedback_topic=feedback_topic,
        defaults_hand=defaults_hand,
        output_publish_rate_hz=float(args.output_publish_rate_hz),
        suppress_timer_after_manual_publish=random_test,
    )
    node.set_idle_tension()
    spin_thread = threading.Thread(target=_spin_node, args=(node,), daemon=True)
    spin_thread.start()

    try:
        ok, msg = node.check_topic_safety(timeout_sec=12.0)
        if not ok:
            print(f"错误: {msg}")
            return

        if not node.wait_feedback(timeout_sec=6.0):
            cmd_pub = node.count_publishers(cmd_topic)
            cmd_sub = node.count_subscribers(cmd_topic)
            fb_pub = node.count_publishers(feedback_topic)
            fb_sub = node.count_subscribers(feedback_topic)
            print(
                "错误: 启动后未收到反馈数据，请检查设备与话题连接。"
                f"\n当前话题状态: {cmd_topic} pubs={cmd_pub}, subs={cmd_sub}; "
                f"{feedback_topic} pubs={fb_pub}, subs={fb_sub}"
            )
            return

        print(
            f"进入 wrist angle 控制模式({hand})\n"
            f"cmd_topic={cmd_topic}, feedback_topic={feedback_topic}\n"
            f"最终命令发布频率: output_publish_rate_hz={float(args.output_publish_rate_hz):.1f}\n"
            "规则: 输入绝对腕角(alpha,beta,单位deg)，按28条独立公式计算线长并转换为电机角度指令。\n"
            f"换算: pos_deg = (L-L0) * 360 / (pi*{PULLEY_DIAMETER_MM:.3f})\n"
            f"到位判定: |error|<={ARRIVAL_POS_TOL} 且连续 {ARRIVAL_CONFIRM_FRAMES} 帧, "
            f"动作超时 {float(args.apply_timeout_sec):.2f}s, 回0超时 {RETURN_ZERO_TIMEOUT_SEC}s"
        )
        nonzero_text = _format_nonzero_offsets(motor_offsets_deg)
        if nonzero_text:
            print(f"当前启用 motor offset(deg): {nonzero_text}")
        else:
            print("当前启用 motor offset(deg): 全部为0")

        if random_test:
            _run_continuous_trajectory_test(
                node=node,
                motor_offsets_deg=motor_offsets_deg,
                count=int(args.random_test_count),
                rate_hz=float(args.random_test_rate_hz),
                alpha_min_deg=float(args.random_alpha_min_deg),
                alpha_max_deg=float(args.random_alpha_max_deg),
                beta_min_deg=float(args.random_beta_min_deg),
                beta_max_deg=float(args.random_beta_max_deg),
            )
            if zero_on_exit:
                _execute_all_zero(node)
        else:
            _run_interactive_loop(
                node,
                apply_timeout_sec=float(args.apply_timeout_sec),
                motor_offsets_deg=motor_offsets_deg,
            )
    except KeyboardInterrupt:
        if zero_on_exit:
            print("收到 Ctrl+C，执行全部回0后退出。")
            _execute_all_zero(node)
        else:
            print("收到 Ctrl+C，直接退出。")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    if DEVICES_MSG_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Cannot import devices_pkg.msg. Please source ROS2 workspace before running."
        ) from DEVICES_MSG_IMPORT_ERROR
    main()
