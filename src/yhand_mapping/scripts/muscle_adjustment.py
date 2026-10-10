#!/usr/bin/env python3
import argparse
import copy
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

try:
    from devices_pkg.msg import Motor, YHandMsg
except ImportError as exc:  # pragma: no cover - runtime dependency check
    Motor = None
    YHandMsg = None
    DEVICES_MSG_IMPORT_ERROR = exc
else:
    DEVICES_MSG_IMPORT_ERROR = None

try:
    from openpyxl import Workbook, load_workbook
except ImportError:  # pragma: no cover - runtime dependency check
    Workbook = None
    load_workbook = None

from muscle_adjustment_defaults import DEFAULTS

from hand_motor_map import (
    extensor_assist_map,
)

MOTOR_COUNT = 28
CONTROL_PERIOD_SEC = 0.02
ARRIVAL_POS_TOL = 7.0
ARRIVAL_CONFIRM_FRAMES = 30
SINGLE_MOTOR_TIMEOUT_SEC = 5.0
RETURN_ZERO_TIMEOUT_SEC = 6.0
# Keep selected motor position after "reached" so it does not bounce back
# immediately when returning to idle pure-tension state.
HOLD_SELECTED_AFTER_REACHED = True
FEEDBACK_STALE_SEC = 1.0
DRIFT_WARN_THRESHOLD = 40.0
DRIFT_LOG_INTERVAL_SEC = 2.0
TOPIC_CONFLICT_CONFIRM_SEC = 2.0
GLOBAL_PRETIGHTEN_TOR = -0.05
TOR_PAIR_MAX_DIFF_A = 0.1
# Keep async callback logs quiet to avoid breaking interactive terminal input.
QUIET_ASYNC_TERMINAL_LOGS = True
ABNORMAL_CONFIRM_THRESHOLD = {
    "pos": 2000.0,
    "vel": 400.0,
    "tor": 1.0,
    "kp": 4000.0,
    "kd": 200.0,
}

# Bi-directional extensor/flexor assist modes.
ASSIST_MODE_OFF = "off"
ASSIST_MODE_RELEASE = "release_tor"
ASSIST_MODE_ORDER = [ASSIST_MODE_OFF, ASSIST_MODE_RELEASE]
ASSIST_RELEASE_TOR = 0.5
ASSIST_RELEASE_SEC = 0.4
ASSIST_AUTO_DEFAULT_MODE = ASSIST_MODE_RELEASE
SWITCH_RELEASE_SEC = 0.8

# Per-hand extensor -> antagonist flexor map.
# Runtime logic is bi-directional:
# - extensor active => release mapped flexor(s)
# - flexor active   => release mapped extensor
EXTENSOR_ASSIST_MAP = {hand: extensor_assist_map(hand) for hand in ("left", "right")}


def _build_tor_safety_pairs(hand: str):
    hand_map = EXTENSOR_ASSIST_MAP.get(hand, {})
    pairs = []
    seen = set()
    for motor_id, info in hand_map.items():
        for target_id in info.get("flexors", []):
            try:
                a_id = int(motor_id)
                b_id = int(target_id)
            except (TypeError, ValueError):
                continue
            if not (1 <= a_id <= MOTOR_COUNT and 1 <= b_id <= MOTOR_COUNT):
                continue
            pair = tuple(sorted((a_id, b_id)))
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
    return sorted(pairs)


def _validate_tor_pair_safety(hand: str, tor_values: list[float], threshold_a: float = TOR_PAIR_MAX_DIFF_A):
    values = list(tor_values or [])
    if len(values) < MOTOR_COUNT:
        values += [0.0] * (MOTOR_COUNT - len(values))
    elif len(values) > MOTOR_COUNT:
        values = values[:MOTOR_COUNT]

    violations = []
    for a_id, b_id in _build_tor_safety_pairs(hand):
        a_val = float(values[a_id - 1])
        b_val = float(values[b_id - 1])
        diff = abs(a_val - b_val)
        if diff > float(threshold_a):
            violations.append(
                {
                    "a_id": a_id,
                    "b_id": b_id,
                    "a_tor": a_val,
                    "b_tor": b_val,
                    "diff": diff,
                }
            )
    return violations


def _print_tor_safety_violations(hand: str, violations: list[dict], threshold_a: float = TOR_PAIR_MAX_DIFF_A):
    print(
        f"[SAFETY] 检测到 {hand} 手 tor 对抗肌配对差值超限（阈值 {threshold_a:.3f}A）。"
        " 已禁止动作并强制全部电机输出 0。"
    )
    for item in violations:
        print(
            f"  - m{item['a_id']:02d}({item['a_tor']:.3f}A) vs "
            f"m{item['b_id']:02d}({item['b_tor']:.3f}A), diff={item['diff']:.3f}A"
        )


@dataclass
class MotorFeedback:
    pos: float | None = None
    vel: float | None = None
    tor: float | None = None
    kp: float | None = None
    kd: float | None = None


def _to_float_or_none(value):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def _topics_for_hand(hand: str):
    if hand == "left":
        return "/yhand_left_cmd", "/yhand_left_feedback"
    return "/yhand_right_cmd", "/yhand_right_feedback"


def _ensure_hand_defaults(defaults_map, hand: str):
    if hand not in defaults_map:
        defaults_map[hand] = {}
    hand_map = defaults_map[hand]

    for key, default_value in (("vel", 0.0), ("kp", 500.0), ("kd", 10.0), ("tor", -0.01)):
        values = list(hand_map.get(key, []))
        if len(values) < MOTOR_COUNT:
            values += [default_value] * (MOTOR_COUNT - len(values))
        elif len(values) > MOTOR_COUNT:
            values = values[:MOTOR_COUNT]
        normalized = []
        for v in values:
            fv = _to_float_or_none(v)
            normalized.append(default_value if fv is None else float(fv))
        hand_map[key] = normalized

    # Optional field used by data_collection active motors.
    # If missing, default to idle torque profile for backward compatibility.
    tor_active_values = list(hand_map.get("tor_active", hand_map.get("tor", [])))
    if len(tor_active_values) < MOTOR_COUNT:
        base_tor = list(hand_map.get("tor", []))
        for i in range(len(tor_active_values), MOTOR_COUNT):
            if i < len(base_tor):
                tor_active_values.append(base_tor[i])
            else:
                tor_active_values.append(-0.01)
    elif len(tor_active_values) > MOTOR_COUNT:
        tor_active_values = tor_active_values[:MOTOR_COUNT]
    normalized_tor_active = []
    for v in tor_active_values:
        fv = _to_float_or_none(v)
        normalized_tor_active.append(-0.01 if fv is None else float(fv))
    hand_map["tor_active"] = normalized_tor_active


def _format_float_list(values, indent_spaces=12, per_line=8):
    indent = " " * indent_spaces
    rows = []
    for i in range(0, len(values), per_line):
        chunk = values[i : i + per_line]
        rows.append(indent + ", ".join(f"{float(v):.6g}" for v in chunk) + ",")
    return "\n".join(rows)


def _write_defaults_python(path: Path, defaults_map):
    for hand in ("left", "right"):
        _ensure_hand_defaults(defaults_map, hand)

    lines = [
        '"""',
        "Persistent defaults for muscle adjustment.",
        "",
        "This file is updated by scripts/muscle_adjustment.py when the user saves",
        "new per-motor vel/KP/KD/tor values.",
        "tor: idle/default tension current",
        "tor_active: active-motor feedforward current for data_collection",
        '"""',
        "",
        "DEFAULTS = {",
    ]
    for hand in ("left", "right"):
        lines.append(f'    "{hand}": {{')
        for key in ("kp", "kd", "vel", "tor", "tor_active"):
            lines.append(f'        "{key}": [')
            lines.append(_format_float_list(defaults_map[hand][key], indent_spaces=12, per_line=8))
            lines.append("        ],")
        lines.append("    },")
    lines.append("}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _save_excel(
    excel_path: Path,
    hand: str,
    defaults_hand,
):
    if Workbook is None or load_workbook is None:
        raise RuntimeError("openpyxl is required for Excel saving. Please install openpyxl.")

    excel_path.parent.mkdir(parents=True, exist_ok=True)
    if excel_path.exists():
        wb = load_workbook(excel_path)
    else:
        wb = Workbook()

    defaults_sheet_name = f"{hand}_defaults"
    if defaults_sheet_name in wb.sheetnames:
        ws = wb[defaults_sheet_name]
        if ws.max_row > 0:
            ws.delete_rows(1, ws.max_row)
    else:
        ws = wb.create_sheet(defaults_sheet_name)

    ws.append(["motor_id", "vel_default", "kp", "kd", "tor_default", "tor_active_default", "updated_at"])
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for motor_id in range(1, MOTOR_COUNT + 1):
        idx = motor_id - 1
        ws.append(
            [
                motor_id,
                float(defaults_hand["vel"][idx]),
                float(defaults_hand["kp"][idx]),
                float(defaults_hand["kd"][idx]),
                float(defaults_hand["tor"][idx]),
                float(defaults_hand["tor_active"][idx]),
                ts,
            ]
        )

    # Keep excel clean: do not persist verbose interaction history.
    history_sheet_name = f"{hand}_history"
    if history_sheet_name in wb.sheetnames:
        del wb[history_sheet_name]

    wb.save(excel_path)


class MuscleAdjustmentNode(Node):
    def __init__(self, hand: str, cmd_topic: str, feedback_topic: str, defaults_hand):
        super().__init__(f"muscle_adjustment_{hand}")
        if YHandMsg is None or Motor is None:
            raise RuntimeError(
                "Cannot import devices_pkg.msg (YHandMsg/Motor). "
                "Please source the ROS2 workspace before running."
            )

        self.hand = hand
        self.cmd_topic = cmd_topic
        self.feedback_topic = feedback_topic
        self.defaults_hand = defaults_hand

        self._lock = threading.Lock()
        self.latest_feedback = [MotorFeedback() for _ in range(MOTOR_COUNT)]
        self.feedback_received = False
        self.feedback_stale = False
        self.last_feedback_ts = 0.0
        self.active_motor_id = None
        self.idle_reference_pos = [None] * MOTOR_COUNT
        self.last_drift_log_ts = 0.0
        self.quiet_async_terminal_logs = QUIET_ASYNC_TERMINAL_LOGS
        self.watchdog_fallback_enabled = True
        self.global_pretighten_enabled = False

        self.command_matrix = self._build_idle_matrix()

        self.publisher = self.create_publisher(YHandMsg, self.cmd_topic, 10)
        self.subscription = self.create_subscription(YHandMsg, self.feedback_topic, self._feedback_cb, 10)
        self.publish_timer = self.create_timer(CONTROL_PERIOD_SEC, self._publish_cb)
        self.watchdog_timer = self.create_timer(0.1, self._watchdog_cb)
        self.drift_timer = self.create_timer(0.5, self._drift_cb)

        self.get_logger().info(
            f"[muscle_adjustment] started. hand={self.hand}, cmd_topic={self.cmd_topic}, "
            f"feedback_topic={self.feedback_topic}"
        )

    def _build_idle_matrix(self):
        matrix = []
        for i in range(MOTOR_COUNT):
            tor = (
                float(GLOBAL_PRETIGHTEN_TOR)
                if self.global_pretighten_enabled
                else float(self.defaults_hand["tor"][i])
            )
            matrix.append([0.0, 0.0, tor, 0.0, 0.0])
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
            if self.feedback_stale:
                self.feedback_stale = False
                if not self.quiet_async_terminal_logs:
                    self.get_logger().warn("[muscle_adjustment] feedback recovered.")

    def _publish_cb(self):
        with self._lock:
            matrix = [row[:] for row in self.command_matrix]

        msg = YHandMsg()
        motor_msgs = []
        for row in matrix:
            motor_msgs.append(
                Motor(
                    pos=float(row[0]),
                    vel=float(row[1]),
                    tor=float(row[2]),
                    kp=float(row[3]),
                    kd=float(row[4]),
                )
            )
        msg.motors = motor_msgs
        msg.pretightenforce.flag = False
        msg.pretightenforce.torque = 0.0
        msg.pretightenforce.ms = 0
        self.publisher.publish(msg)

    def _watchdog_cb(self):
        with self._lock:
            feedback_received = self.feedback_received
            stale = self.feedback_stale
            age = time.monotonic() - self.last_feedback_ts
            watchdog_fallback_enabled = self.watchdog_fallback_enabled
        if not feedback_received:
            return
        if age > FEEDBACK_STALE_SEC and not stale:
            with self._lock:
                self.feedback_stale = True
            if not self.quiet_async_terminal_logs:
                self.get_logger().error(
                    f"[muscle_adjustment] feedback stale ({age:.3f}s), switching to pure tension."
                )
            if watchdog_fallback_enabled:
                self.set_idle_tension(update_idle_reference=False)

    def _drift_cb(self):
        now = time.monotonic()
        if now - self.last_drift_log_ts < DRIFT_LOG_INTERVAL_SEC:
            return

        with self._lock:
            if self.active_motor_id is not None:
                return
            if not self.feedback_received:
                return
            drift_items = []
            for i in range(MOTOR_COUNT):
                ref = self.idle_reference_pos[i]
                cur = self.latest_feedback[i].pos
                if ref is None or cur is None:
                    continue
                drift = abs(cur - ref)
                if drift > DRIFT_WARN_THRESHOLD:
                    drift_items.append((i + 1, drift))

        if drift_items:
            self.last_drift_log_ts = now
            if not self.quiet_async_terminal_logs:
                text = ", ".join([f"m{mid}:{drift:.1f}" for mid, drift in drift_items[:8]])
                self.get_logger().warn(
                    "[muscle_adjustment] non-selected motor drift observed (warn-only): " + text
                )

    def wait_feedback(self, timeout_sec: float):
        start = time.monotonic()
        while time.monotonic() - start < timeout_sec:
            with self._lock:
                if self.feedback_received:
                    return True
            time.sleep(0.05)
        return False

    def check_topic_safety(self, timeout_sec: float = 8.0):
        conflict_start_ts = None
        latest_pub_count = 0
        latest_pub_nodes_text = "<unknown>"
        latest_same_name_only = False
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
                same_name_only = True
                for info in pub_infos:
                    ns = info.node_namespace or "/"
                    name = info.node_name or "<unknown>"
                    full_name = f"{ns}{name}"
                    pub_nodes.append(full_name)
                    if not (name == self.get_name() and ns == self.get_namespace()):
                        same_name_only = False
                pub_nodes = sorted(set(pub_nodes))
                pub_nodes_text = ", ".join(pub_nodes) if pub_nodes else "<unknown>"
                latest_pub_nodes_text = pub_nodes_text
                latest_same_name_only = same_name_only
                if conflict_start_ts is None:
                    conflict_start_ts = now
                # Discovery graph may temporarily retain stale endpoints right after restart.
                # Confirm conflict only if it persists for a short window.
                if (now - conflict_start_ts) >= TOPIC_CONFLICT_CONFIRM_SEC:
                    hint = (
                        " Detected duplicate publishers with same node name; "
                        "likely previous instance not fully cleaned up yet or another "
                        "muscle_adjustment process still running."
                        if same_name_only
                        else ""
                    )
                    return False, (
                        f"Command topic conflict detected on {self.cmd_topic}. "
                        f"publishers={pub_count} (expected 1). "
                        f"publisher_nodes=[{pub_nodes_text}].{hint}"
                    )
            else:
                conflict_start_ts = None
            if cmd_sub_count >= 1 and fb_pub_count >= 1:
                return True, ""
            time.sleep(0.1)
        if latest_pub_count > 1:
            hint = (
                " Detected duplicate publishers with same node name; "
                "likely previous instance not fully cleaned up yet or another "
                "muscle_adjustment process still running."
                if latest_same_name_only
                else ""
            )
            return False, (
                f"Command topic conflict detected on {self.cmd_topic}. "
                f"publishers={latest_pub_count} (expected 1). "
                f"publisher_nodes=[{latest_pub_nodes_text}].{hint}"
            )
        return False, (
            f"Driver topic not ready. cmd_subscribers={self.count_subscribers(self.cmd_topic)}, "
            f"feedback_publishers={self.count_publishers(self.feedback_topic)}."
        )

    def set_idle_tension(self, update_idle_reference: bool = True):
        with self._lock:
            self.command_matrix = self._build_idle_matrix()
            self.active_motor_id = None
            if update_idle_reference:
                self.idle_reference_pos = [fb.pos for fb in self.latest_feedback]

    def set_single_motor_command(self, motor_id: int, command_row):
        with self._lock:
            matrix = self._build_idle_matrix()
            idx = motor_id - 1
            matrix[idx] = [
                float(command_row[0]),
                float(command_row[1]),
                float(command_row[2]),
                float(command_row[3]),
                float(command_row[4]),
            ]
            self.command_matrix = matrix
            self.active_motor_id = motor_id

    def set_single_motor_command_with_overrides(self, motor_id: int, command_row, overrides: dict[int, list]):
        with self._lock:
            matrix = self._build_idle_matrix()
            active_idx = motor_id - 1
            matrix[active_idx] = [
                float(command_row[0]),
                float(command_row[1]),
                float(command_row[2]),
                float(command_row[3]),
                float(command_row[4]),
            ]
            for override_motor_id, row in overrides.items():
                if override_motor_id < 1 or override_motor_id > MOTOR_COUNT:
                    continue
                idx = override_motor_id - 1
                matrix[idx] = [
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                ]
            self.command_matrix = matrix
            self.active_motor_id = motor_id

    def set_custom_matrix(self, matrix):
        with self._lock:
            self.command_matrix = [row[:] for row in matrix]
            self.active_motor_id = None

    def set_all_zero_output(self):
        with self._lock:
            self.command_matrix = [[0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(MOTOR_COUNT)]
            self.active_motor_id = None

    def get_motor_command_row(self, motor_id: int):
        idx = motor_id - 1
        with self._lock:
            row = self.command_matrix[idx]
            return [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])]

    def get_motor_feedback(self, motor_id: int):
        idx = motor_id - 1
        with self._lock:
            fb = self.latest_feedback[idx]
            return MotorFeedback(pos=fb.pos, vel=fb.vel, tor=fb.tor, kp=fb.kp, kd=fb.kd)

    def all_motor_positions(self):
        with self._lock:
            return [fb.pos for fb in self.latest_feedback]

    def is_feedback_stale(self):
        with self._lock:
            if not self.feedback_received:
                return True
            return (time.monotonic() - self.last_feedback_ts) > FEEDBACK_STALE_SEC

    def set_watchdog_fallback_enabled(self, enabled: bool):
        with self._lock:
            self.watchdog_fallback_enabled = bool(enabled)

    def set_global_pretighten_enabled(self, enabled: bool):
        with self._lock:
            self.global_pretighten_enabled = bool(enabled)
            self.command_matrix = self._build_idle_matrix()
            self.active_motor_id = None
            self.idle_reference_pos = [fb.pos for fb in self.latest_feedback]

    def get_idle_torque_profile(self):
        with self._lock:
            if self.global_pretighten_enabled:
                return [float(GLOBAL_PRETIGHTEN_TOR)] * MOTOR_COUNT
            return [float(v) for v in self.defaults_hand["tor"]]

    def is_global_pretighten_enabled(self):
        with self._lock:
            return bool(self.global_pretighten_enabled)


def _print_feedback(prefix: str, motor_id: int, fb: MotorFeedback):
    print(
        f"{prefix} motor={motor_id} "
        f"pos={fb.pos} vel={fb.vel} tor={fb.tor} kp={fb.kp} kd={fb.kd}"
    )


def _print_all_motor_feedback(node: MuscleAdjustmentNode, prefix: str = "全部电机反馈"):
    print(prefix)
    for motor_id in range(1, MOTOR_COUNT + 1):
        fb = node.get_motor_feedback(motor_id)
        cmd = node.get_motor_command_row(motor_id)
        print(
            f"  motor={motor_id:02d} "
            f"pos={fb.pos} vel={fb.vel} tor={fb.tor} kp={fb.kp} kd={fb.kd} "
            f"| cmd=[{cmd[0]:.3f},{cmd[1]:.3f},{cmd[2]:.3f},{cmd[3]:.3f},{cmd[4]:.3f}]"
        )


def _assist_mode_label(mode: str) -> str:
    if mode == ASSIST_MODE_RELEASE:
        return (
            "release_tor(主动肌运动前0.5s，对抗肌给正扭矩放线, "
            f"tor={ASSIST_RELEASE_TOR:.3g})"
        )
    return "off(关闭)"


def _cycle_assist_mode(current_mode: str) -> str:
    try:
        idx = ASSIST_MODE_ORDER.index(current_mode)
    except ValueError:
        return ASSIST_MODE_OFF
    return ASSIST_MODE_ORDER[(idx + 1) % len(ASSIST_MODE_ORDER)]


def _is_assist_motor(hand: str, motor_id: int) -> bool:
    hand_map = EXTENSOR_ASSIST_MAP.get(hand, {})
    if motor_id in hand_map:
        return True
    for info in hand_map.values():
        if motor_id in info.get("flexors", []):
            return True
    return False


def _resolve_assist_mode_for_motor(hand: str, motor_id: int, assist_mode_preference: str):
    if not _is_assist_motor(hand, motor_id):
        return ASSIST_MODE_OFF, False
    if assist_mode_preference == ASSIST_MODE_OFF:
        return ASSIST_AUTO_DEFAULT_MODE, True
    return assist_mode_preference, False


def _get_assist_target_motors(hand: str, motor_id: int, assist_mode: str):
    if assist_mode != ASSIST_MODE_RELEASE:
        return []

    hand_map = EXTENSOR_ASSIST_MAP.get(hand, {})
    # Extensor active: release mapped flexors.
    if motor_id in hand_map:
        return list(hand_map[motor_id].get("flexors", []))

    # Flexor active: release corresponding extensor(s).
    ext_targets = []
    for ext_id, info in hand_map.items():
        if motor_id in info.get("flexors", []):
            ext_targets.append(ext_id)
    return sorted(set(ext_targets))


def _apply_extensor_assist_phase(node: MuscleAdjustmentNode, motor_id: int, cmd_row, assist_mode: str):
    if assist_mode == ASSIST_MODE_OFF:
        return {"applied": False, "targets": []}

    target_ids = _get_assist_target_motors(node.hand, motor_id, assist_mode)
    if not target_ids:
        return {"applied": False, "targets": []}

    overrides = {}
    assist_duration_sec = ASSIST_RELEASE_SEC
    if assist_mode != ASSIST_MODE_RELEASE:
        return {"applied": False, "targets": []}
    for tid in target_ids:
        overrides[tid] = [0.0, 0.0, float(ASSIST_RELEASE_TOR), 0.0, 0.0]

    node.set_single_motor_command_with_overrides(motor_id, cmd_row, overrides)
    start = time.monotonic()
    while (time.monotonic() - start) < assist_duration_sec:
        if node.is_feedback_stale():
            return {"applied": True, "targets": target_ids, "feedback_stale": True}
        time.sleep(CONTROL_PERIOD_SEC)

    # Restore to standard single-motor command after the 1s assist phase.
    node.set_single_motor_command(motor_id, cmd_row)
    return {"applied": True, "targets": target_ids, "feedback_stale": False}


def _is_abnormal_command(command_row):
    pos, vel, tor, kp, kd = command_row
    checks = [
        abs(pos) > ABNORMAL_CONFIRM_THRESHOLD["pos"],
        abs(vel) > ABNORMAL_CONFIRM_THRESHOLD["vel"],
        abs(tor) > ABNORMAL_CONFIRM_THRESHOLD["tor"],
        abs(kp) > ABNORMAL_CONFIRM_THRESHOLD["kp"],
        abs(kd) > ABNORMAL_CONFIRM_THRESHOLD["kd"],
    ]
    return any(checks)


def _prompt_motor_id():
    while True:
        raw = input(
            "选择电机编号(1-28)，输入 p 全局预紧(-0.05A), n 恢复默认张紧, a 切换伸屈肌辅助模式, x 退出: "
        ).strip()
        raw_l = raw.lower()
        if raw_l in ("x", "quit", "exit"):
            return "quit", None
        if raw_l in ("p", "pretighten"):
            return "pretighten_on", None
        if raw_l in ("n", "normal", "default"):
            return "pretighten_off", None
        if raw_l in ("a", "assist"):
            return "toggle_assist", None
        if raw_l in ("q", "back"):
            print("当前已在电机选择界面，请输入 1-28 或 x。")
            continue
        try:
            mid = int(raw_l)
        except ValueError:
            print("输入无效，请输入 1-28 / p / n / a / x")
            continue
        if 1 <= mid <= MOTOR_COUNT:
            return "motor", mid
        print("电机编号超范围，请输入 1-28")


def _prompt_command_request():
    while True:
        raw = input(
            "输入指令: 5维(pos vel tor kp kd) / tor 电流模式(tor <tor> <sec>) / "
            "vel 速度模式(vel <vel> <sec> [tor] [kp] [kd]) "
            "(q 回选择 / x 退出 / m 切换电机 / a 切换伸屈肌辅助): "
        ).strip()
        raw_l = raw.lower()
        if raw_l in ("q", "back"):
            return "back_to_select", None
        if raw_l in ("x", "quit", "exit"):
            return "quit_program", None
        if raw_l in ("m", "motor"):
            return "change_motor", None
        if raw_l in ("a", "assist"):
            return "toggle_assist", None

        parts = raw_l.split()
        if not parts:
            print("输入为空，请重试。")
            continue

        mode_token = parts[0]

        if mode_token in ("tor", "torque", "i", "cur", "current"):
            if len(parts) != 3:
                print("电流模式格式: tor <tor> <sec>")
                continue
            try:
                tor = float(parts[1])
                duration_sec = float(parts[2])
            except ValueError:
                print("包含非数字，请重新输入。")
                continue
            if duration_sec <= 0.0:
                print("持续时间必须 > 0 秒。")
                continue
            probe_cmd = [0.0, 0.0, tor, 0.0, 0.0]
            if _is_abnormal_command(probe_cmd):
                confirm = input(
                    "检测到异常大电流指令值，是否继续执行？输入 yes 继续，其他任意键取消: "
                ).strip().lower()
                if confirm != "yes":
                    print("已取消该条指令。")
                    continue
            return "ok", {"mode": "torque", "tor": tor, "duration_sec": duration_sec}

        if mode_token in ("vel", "velocity", "v"):
            if len(parts) not in (3, 4, 5, 6):
                print("速度模式格式: vel <vel> <sec> [tor] [kp] [kd]")
                continue
            try:
                vel = float(parts[1])
                duration_sec = float(parts[2])
                tor = float(parts[3]) if len(parts) >= 4 else -0.01
                kp = float(parts[4]) if len(parts) >= 5 else 0.0
                kd = float(parts[5]) if len(parts) >= 6 else 0.0
            except ValueError:
                print("包含非数字，请重新输入。")
                continue
            if duration_sec <= 0.0:
                print("持续时间必须 > 0 秒。")
                continue
            if kp < 0.0 or kd < 0.0:
                print("警告: kp/kd 建议使用非负值，负值通常会导致不稳定或不可预测行为。")
            probe_cmd = [0.0, vel, tor, kp, kd]
            if _is_abnormal_command(probe_cmd):
                confirm = input(
                    "检测到异常大速度/电流/增益指令值，是否继续执行？输入 yes 继续，其他任意键取消: "
                ).strip().lower()
                if confirm != "yes":
                    print("已取消该条指令。")
                    continue
            return "ok", {
                "mode": "velocity",
                "vel": vel,
                "duration_sec": duration_sec,
                "tor": tor,
                "kp": kp,
                "kd": kd,
            }

        if len(parts) != 5:
            print(
                "格式错误。位置模式请输入 5 个数字，或使用 tor/vel 模式命令。"
            )
            continue
        try:
            cmd = [float(parts[i]) for i in range(5)]
        except ValueError:
            print("包含非数字，请重新输入。")
            continue
        if cmd[3] < 0.0 or cmd[4] < 0.0:
            print("警告: kp/kd 建议使用非负值，负值通常会导致不稳定或不可预测行为。")
        if _is_abnormal_command(cmd):
            confirm = input(
                "检测到异常大指令值，是否继续执行？输入 yes 继续，其他任意键取消: "
            ).strip().lower()
            if confirm != "yes":
                print("已取消该条指令。")
                continue
        return "ok", {"mode": "position", "row": cmd}


def _execute_position_command(
    node: MuscleAdjustmentNode,
    motor_id: int,
    cmd_row,
    assist_mode: str,
):
    node.set_single_motor_command(motor_id, cmd_row)
    node.set_watchdog_fallback_enabled(False)
    assist_info = _apply_extensor_assist_phase(node, motor_id, cmd_row, assist_mode)
    if assist_info.get("feedback_stale"):
        post_fb = node.get_motor_feedback(motor_id)
        node.set_watchdog_fallback_enabled(True)
        if node.is_global_pretighten_enabled():
            node.set_global_pretighten_enabled(False)
        node.set_idle_tension(update_idle_reference=True)
        return {
            "motor_id": motor_id,
            "mode": "position",
            "cmd": cmd_row[:],
            "result": "feedback_stale",
            "position_error": None,
            "duration_sec": None,
            "elapsed_sec": 0.0,
            "assist_mode": assist_mode,
            "assist_applied": bool(assist_info.get("applied", False)),
            "assist_targets": assist_info.get("targets", []),
            "post_fb": {
                "pos": post_fb.pos,
                "vel": post_fb.vel,
                "tor": post_fb.tor,
                "kp": post_fb.kp,
                "kd": post_fb.kd,
            },
        }

    start = time.monotonic()
    ok_frames = 0
    result = "timeout"
    last_err = None

    try:
        while True:
            if node.is_feedback_stale():
                result = "feedback_stale"
                break
            fb = node.get_motor_feedback(motor_id)
            if fb.pos is not None:
                last_err = abs(float(fb.pos) - float(cmd_row[0]))
                if last_err <= ARRIVAL_POS_TOL:
                    ok_frames += 1
                else:
                    ok_frames = 0
                if ok_frames >= ARRIVAL_CONFIRM_FRAMES:
                    result = "reached"
                    break
            if (time.monotonic() - start) >= SINGLE_MOTOR_TIMEOUT_SEC:
                result = "timeout"
                break
            time.sleep(CONTROL_PERIOD_SEC)
    finally:
        node.set_watchdog_fallback_enabled(True)

    post_fb = node.get_motor_feedback(motor_id)
    if result == "reached" and HOLD_SELECTED_AFTER_REACHED:
        hold_pos = float(cmd_row[0])
        hold_tor = float(cmd_row[2])
        hold_kp = float(cmd_row[3])
        hold_kd = float(cmd_row[4])
        node.set_single_motor_command(
            motor_id,
            [hold_pos, 0.0, hold_tor, hold_kp, hold_kd],
        )
    else:
        node.set_idle_tension(update_idle_reference=True)
    return {
        "motor_id": motor_id,
        "mode": "position",
        "cmd": cmd_row[:],
        "result": result,
        "position_error": last_err,
        "duration_sec": None,
        "elapsed_sec": time.monotonic() - start,
        "assist_mode": assist_mode,
        "assist_applied": bool(assist_info.get("applied", False)),
        "assist_targets": assist_info.get("targets", []),
        "post_fb": {
            "pos": post_fb.pos,
            "vel": post_fb.vel,
            "tor": post_fb.tor,
            "kp": post_fb.kp,
            "kd": post_fb.kd,
        },
    }


def _execute_timed_command(node: MuscleAdjustmentNode, motor_id: int, command_request, assist_mode: str):
    start_fb = node.get_motor_feedback(motor_id)
    base_pos = 0.0 if start_fb.pos is None else float(start_fb.pos)
    mode = command_request["mode"]
    duration_sec = float(command_request["duration_sec"])

    if mode == "torque":
        cmd_row = [base_pos, 0.0, float(command_request["tor"]), 0.0, 0.0]
    else:
        cmd_row = [
            base_pos,
            float(command_request["vel"]),
            float(command_request["tor"]),
            float(command_request["kp"]),
            float(command_request["kd"]),
        ]

    node.set_single_motor_command(motor_id, cmd_row)
    node.set_watchdog_fallback_enabled(False)
    assist_info = _apply_extensor_assist_phase(node, motor_id, cmd_row, assist_mode)
    if assist_info.get("feedback_stale"):
        post_fb = node.get_motor_feedback(motor_id)
        node.set_watchdog_fallback_enabled(True)
        if node.is_global_pretighten_enabled():
            node.set_global_pretighten_enabled(False)
        node.set_idle_tension(update_idle_reference=True)
        return {
            "motor_id": motor_id,
            "mode": mode,
            "cmd": cmd_row[:],
            "result": "feedback_stale",
            "position_error": None,
            "duration_sec": duration_sec,
            "elapsed_sec": 0.0,
            "assist_mode": assist_mode,
            "assist_applied": bool(assist_info.get("applied", False)),
            "assist_targets": assist_info.get("targets", []),
            "post_fb": {
                "pos": post_fb.pos,
                "vel": post_fb.vel,
                "tor": post_fb.tor,
                "kp": post_fb.kp,
                "kd": post_fb.kd,
            },
        }

    start = time.monotonic()
    result = "finished"
    try:
        while True:
            if node.is_feedback_stale():
                result = "feedback_stale"
                break
            if (time.monotonic() - start) >= duration_sec:
                break
            time.sleep(CONTROL_PERIOD_SEC)
    finally:
        node.set_watchdog_fallback_enabled(True)

    post_fb = node.get_motor_feedback(motor_id)
    node.set_idle_tension(update_idle_reference=True)
    return {
        "motor_id": motor_id,
        "mode": mode,
        "cmd": cmd_row[:],
        "result": result,
        "position_error": None,
        "duration_sec": duration_sec,
        "elapsed_sec": time.monotonic() - start,
        "assist_mode": assist_mode,
        "assist_applied": bool(assist_info.get("applied", False)),
        "assist_targets": assist_info.get("targets", []),
        "post_fb": {
            "pos": post_fb.pos,
            "vel": post_fb.vel,
            "tor": post_fb.tor,
            "kp": post_fb.kp,
            "kd": post_fb.kd,
        },
    }


def _execute_command(node: MuscleAdjustmentNode, motor_id: int, command_request, assist_mode: str):
    mode = command_request["mode"]
    if mode == "position":
        return _execute_position_command(node, motor_id, command_request["row"], assist_mode)
    return _execute_timed_command(node, motor_id, command_request, assist_mode)


def _save_motor_defaults(
    full_defaults,
    hand: str,
    motor_id: int,
    vel: float,
    kp: float,
    kd: float,
    tor: float,
):
    idx = motor_id - 1
    full_defaults[hand]["vel"][idx] = float(vel)
    full_defaults[hand]["kp"][idx] = float(kp)
    full_defaults[hand]["kd"][idx] = float(kd)
    full_defaults[hand]["tor"][idx] = float(tor)


def _all_reached_zero(node: MuscleAdjustmentNode):
    pos_list = node.all_motor_positions()
    available = [p for p in pos_list if p is not None]
    if not available:
        return False
    for p in available:
        if abs(float(p)) > ARRIVAL_POS_TOL:
            return False
    return True


def _return_all_to_zero_then_tension(node: MuscleAdjustmentNode, defaults_hand):
    print("开始执行退出流程：全部电机回零并保持张紧...")
    idle_tor_profile = node.get_idle_torque_profile()

    start_ts = time.monotonic()
    reached = False

    node.set_watchdog_fallback_enabled(False)
    try:
        # Faster return-to-zero: command all motors directly to 0 and wait.
        matrix = []
        for i in range(MOTOR_COUNT):
            matrix.append(
                [
                    0.0,
                    0.0,
                    float(idle_tor_profile[i]),
                    float(defaults_hand["kp"][i]),
                    float(defaults_hand["kd"][i]),
                ]
            )
        node.set_custom_matrix(matrix)

        while True:
            if _all_reached_zero(node):
                reached = True
                break
            if (time.monotonic() - start_ts) >= RETURN_ZERO_TIMEOUT_SEC:
                break
            time.sleep(CONTROL_PERIOD_SEC)
    finally:
        node.set_watchdog_fallback_enabled(True)

    node.set_idle_tension(update_idle_reference=True)
    time.sleep(0.2)

    if reached:
        print("退出流程完成：全部电机达到0位阈值并回到纯张紧。")
    else:
        print("退出流程超时：已切回纯张紧后退出。")


def _switch_motor_with_release(node: MuscleAdjustmentNode, previous_motor_id: int):
    if node.is_global_pretighten_enabled():
        node.set_global_pretighten_enabled(False)
        print("已自动关闭全局预紧，切回 defaults 小张紧。")

    idle_tor_profile = node.get_idle_torque_profile()
    matrix = [[0.0, 0.0, float(idle_tor_profile[i]), 0.0, 0.0] for i in range(MOTOR_COUNT)]

    if 1 <= previous_motor_id <= MOTOR_COUNT:
        fb = node.get_motor_feedback(previous_motor_id)
        prev_pos = 0.0 if fb.pos is None else float(fb.pos)
        # Briefly set previous active motor to pure zero-current to release residual load.
        matrix[previous_motor_id - 1] = [prev_pos, 0.0, 0.0, 0.0, 0.0]
        print(
            f"切换电机缓释: motor={previous_motor_id} 先执行 {SWITCH_RELEASE_SEC:.1f}s 零电流，再回 defaults 小张紧。"
        )
        node.set_custom_matrix(matrix)
        start = time.monotonic()
        while (time.monotonic() - start) < SWITCH_RELEASE_SEC:
            if node.is_feedback_stale():
                break
            time.sleep(CONTROL_PERIOD_SEC)

    node.set_idle_tension(update_idle_reference=True)
    # Wait a few publish/feedback cycles so printed feedback reflects new command.
    time.sleep(max(0.12, CONTROL_PERIOD_SEC * 4))


def _run_interactive_loop(
    node: MuscleAdjustmentNode,
    full_defaults,
    defaults_path: Path,
    excel_path: Path,
):
    hand = node.hand
    assist_mode_preference = ASSIST_MODE_OFF

    while True:
        select_action, selected_value = _prompt_motor_id()
        if select_action == "quit":
            _return_all_to_zero_then_tension(node, full_defaults[hand])
            return
        if select_action == "pretighten_on":
            node.set_global_pretighten_enabled(True)
            print(
                f"已开启全局预紧：全部电机 tor={GLOBAL_PRETIGHTEN_TOR:.2f}A, kp=0, kd=0。"
            )
            continue
        if select_action == "pretighten_off":
            node.set_global_pretighten_enabled(False)
            print("已恢复默认张紧（每电机 tor 使用 muscle_adjustment_defaults.py 配置）。")
            continue
        if select_action == "toggle_assist":
            assist_mode_preference = _cycle_assist_mode(assist_mode_preference)
            print(f"伸屈肌辅助偏好已切换为: {_assist_mode_label(assist_mode_preference)}")
            continue
        motor_id = selected_value
        effective_assist_mode, auto_enabled = _resolve_assist_mode_for_motor(
            hand, motor_id, assist_mode_preference
        )
        if auto_enabled:
            print(
                f"检测到可辅助电机 m{motor_id}，已自动启用辅助模式: {_assist_mode_label(effective_assist_mode)}"
            )
        elif _is_assist_motor(hand, motor_id):
            print(f"当前伸屈肌辅助模式: {_assist_mode_label(effective_assist_mode)}")

        while True:
            pre_fb = node.get_motor_feedback(motor_id)
            _print_feedback("当前反馈", motor_id, pre_fb)

            status, command_request = _prompt_command_request()
            if status == "quit_program":
                _return_all_to_zero_then_tension(node, full_defaults[hand])
                return
            if status == "back_to_select":
                _return_all_to_zero_then_tension(node, full_defaults[hand])
                break
            if status == "toggle_assist":
                assist_mode_preference = _cycle_assist_mode(assist_mode_preference)
                effective_assist_mode, auto_enabled = _resolve_assist_mode_for_motor(
                    hand, motor_id, assist_mode_preference
                )
                if auto_enabled:
                    print(
                        f"伸屈肌辅助偏好为 off，但当前是可辅助电机 m{motor_id}，自动启用: "
                        f"{_assist_mode_label(effective_assist_mode)}"
                    )
                else:
                    print(f"当前伸屈肌辅助模式: {_assist_mode_label(effective_assist_mode)}")
                continue
            if status == "change_motor":
                _switch_motor_with_release(node, motor_id)
                _print_all_motor_feedback(node, prefix="切换电机后全部电机反馈:")
                break

            effective_assist_mode, _ = _resolve_assist_mode_for_motor(
                hand, motor_id, assist_mode_preference
            )
            result = _execute_command(node, motor_id, command_request, effective_assist_mode)
            if result["mode"] == "position":
                print(
                    f"执行结果: {result['result']}, "
                    f"position_error={result['position_error']}, "
                    f"sent_cmd={result['cmd']}"
                )
            else:
                print(
                    f"执行结果: {result['result']}, mode={result['mode']}, "
                    f"elapsed={result['elapsed_sec']:.2f}s/{result['duration_sec']:.2f}s, "
                    f"sent_cmd={result['cmd']}"
                )
            fb_dict = result["post_fb"]
            print(
                "停止后反馈: "
                f"pos={fb_dict['pos']} vel={fb_dict['vel']} tor={fb_dict['tor']} "
                f"kp={fb_dict['kp']} kd={fb_dict['kd']}"
            )
            if result.get("assist_applied"):
                print(
                    f"伸屈肌辅助已执行: mode={_assist_mode_label(result.get('assist_mode', ASSIST_MODE_OFF))}, "
                    f"targets={result.get('assist_targets', [])}"
                )

            action_raw = input(
                "下一步: [c]继续调该电机 [s]保存当前vel/KP/KD/tor [m]切换电机 "
                "[a]切换伸屈肌辅助模式 [q]回到电机选择 [x]退出程序: "
            ).strip()
            action = action_raw.lower()

            if action in ("x", "quit", "exit"):
                _return_all_to_zero_then_tension(node, full_defaults[hand])
                return
            if action in ("q", "back"):
                _return_all_to_zero_then_tension(node, full_defaults[hand])
                break
            if action in ("m", "motor"):
                _switch_motor_with_release(node, motor_id)
                _print_all_motor_feedback(node, prefix="切换电机后全部电机反馈:")
                break
            if action in ("a", "assist"):
                assist_mode_preference = _cycle_assist_mode(assist_mode_preference)
                effective_assist_mode, auto_enabled = _resolve_assist_mode_for_motor(
                    hand, motor_id, assist_mode_preference
                )
                if auto_enabled:
                    print(
                        f"伸屈肌辅助偏好为 off，但当前是可辅助电机 m{motor_id}，自动启用: "
                        f"{_assist_mode_label(effective_assist_mode)}"
                    )
                else:
                    print(f"当前伸屈肌辅助模式: {_assist_mode_label(effective_assist_mode)}")
                continue
            if action in ("s", "save"):
                mode = result["mode"]
                save_vel = float(full_defaults[hand]["vel"][motor_id - 1])
                save_kp = float(full_defaults[hand]["kp"][motor_id - 1])
                save_kd = float(full_defaults[hand]["kd"][motor_id - 1])
                save_tor = float(full_defaults[hand]["tor"][motor_id - 1])
                if mode == "position":
                    save_vel = float(result["cmd"][1])
                    save_kp = float(result["cmd"][3])
                    save_kd = float(result["cmd"][4])
                    save_tor = float(result["cmd"][2])
                elif mode == "torque":
                    save_tor = float(result["cmd"][2])
                elif mode == "velocity":
                    save_vel = float(result["cmd"][1])
                    save_kp = float(result["cmd"][3])
                    save_kd = float(result["cmd"][4])
                    save_tor = float(result["cmd"][2])
                _save_motor_defaults(
                    full_defaults=full_defaults,
                    hand=hand,
                    motor_id=motor_id,
                    vel=save_vel,
                    kp=save_kp,
                    kd=save_kd,
                    tor=save_tor,
                )
                _write_defaults_python(defaults_path, full_defaults)
                _save_excel(
                    excel_path=excel_path,
                    hand=hand,
                    defaults_hand=full_defaults[hand],
                )
                if mode == "torque":
                    print(
                        f"已保存 motor={motor_id} 的默认 tor 到: {defaults_path.name} "
                        f"并写入 Excel: {excel_path}"
                    )
                else:
                    print(
                        f"已保存 motor={motor_id} 的默认 vel/KP/KD/tor 到: {defaults_path.name} "
                        f"并写入 Excel: {excel_path}"
                    )
                continue
            if action in ("", "c", "continue"):
                continue
            print("输入无效，请输入 c/s/m/a/q/x。")


def _parse_args():
    parser = argparse.ArgumentParser(description="Interactive muscle adjustment tool for Y-hand.")
    parser.add_argument("--hand", required=True, choices=["left", "right"], help="Target hand to adjust.")
    parser.add_argument(
        "--defaults-path",
        default="",
        help="Path to muscle_adjustment_defaults.py. Default: scripts/muscle_adjustment_defaults.py",
    )
    parser.add_argument(
        "--excel-path",
        default="",
        help="Path to output Excel file. Default: scripts/motor_test_logs/muscle_adjustment_<hand>.xlsx",
    )
    parser.add_argument(
        "--cmd-topic",
        default="",
        help="Override command topic. Default derived from hand.",
    )
    parser.add_argument(
        "--feedback-topic",
        default="",
        help="Override feedback topic. Default derived from hand.",
    )
    return parser.parse_args()


def _spin_node(node: Node):
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        # Normal path during shutdown; keep terminal clean.
        return
    except KeyboardInterrupt:
        return


def _hold_zero_output_until_exit(node: MuscleAdjustmentNode):
    print("[SAFETY] 安全模式已激活：持续输出全 0 指令。按 Ctrl+C 退出。")
    try:
        while rclpy.ok():
            node.set_all_zero_output()
            time.sleep(CONTROL_PERIOD_SEC)
    except KeyboardInterrupt:
        pass
    finally:
        node.set_all_zero_output()


def main():
    args = _parse_args()

    script_dir = Path(__file__).resolve().parent
    defaults_path = Path(args.defaults_path) if args.defaults_path else (script_dir / "muscle_adjustment_defaults.py")
    excel_path = (
        Path(args.excel_path)
        if args.excel_path
        else (script_dir / "motor_test_logs" / f"muscle_adjustment_{args.hand}.xlsx")
    )

    cmd_topic, feedback_topic = _topics_for_hand(args.hand)
    if args.cmd_topic:
        cmd_topic = args.cmd_topic
    if args.feedback_topic:
        feedback_topic = args.feedback_topic

    full_defaults = copy.deepcopy(DEFAULTS)
    _ensure_hand_defaults(full_defaults, "left")
    _ensure_hand_defaults(full_defaults, "right")

    tor_safety_violations = _validate_tor_pair_safety(args.hand, full_defaults[args.hand]["tor"])
    runtime_defaults_hand = full_defaults[args.hand]
    if tor_safety_violations:
        runtime_defaults_hand = copy.deepcopy(full_defaults[args.hand])
        runtime_defaults_hand["tor"] = [0.0] * MOTOR_COUNT
        runtime_defaults_hand["tor_active"] = [0.0] * MOTOR_COUNT

    rclpy.init()
    node = MuscleAdjustmentNode(
        hand=args.hand,
        cmd_topic=cmd_topic,
        feedback_topic=feedback_topic,
        defaults_hand=runtime_defaults_hand,
    )
    node.set_idle_tension(update_idle_reference=False)

    spin_thread = threading.Thread(target=_spin_node, args=(node,), daemon=True)
    spin_thread.start()

    try:
        ok, msg = node.check_topic_safety(timeout_sec=12.0)
        if not ok:
            print(f"错误: {msg}")
            _return_all_to_zero_then_tension(node, full_defaults[args.hand])
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
            _return_all_to_zero_then_tension(node, full_defaults[args.hand])
            return

        if tor_safety_violations:
            node.set_all_zero_output()
            _print_tor_safety_violations(args.hand, tor_safety_violations)
            _hold_zero_output_until_exit(node)
            return

        print(
            f"进入调参模式 hand={args.hand}, cmd_topic={cmd_topic}, feedback_topic={feedback_topic}\n"
            f"默认纯张紧: kp=0, kd=0, tor=-0.01(每电机独立可配置)\n"
            f"可选全局预紧: tor={GLOBAL_PRETIGHTEN_TOR:.2f}A（电机选择界面输入 p 开启, n 关闭）\n"
            "伸屈肌辅助: 相关电机自动启用；a 可切换偏好 off -> release_tor\n"
            "指令模式: 位置(5维) / 电流(tor <tor> <sec>) / 速度(vel <vel> <sec> [tor] [kp] [kd])\n"
            f"到位后保持: {'开启' if HOLD_SELECTED_AFTER_REACHED else '关闭'}（仅当前电机）\n"
            f"到位判定: |error|<={ARRIVAL_POS_TOL} 且连续 {ARRIVAL_CONFIRM_FRAMES} 帧, "
            f"单电机超时 {SINGLE_MOTOR_TIMEOUT_SEC}s, 回0超时 {RETURN_ZERO_TIMEOUT_SEC}s"
        )
        _run_interactive_loop(
            node=node,
            full_defaults=full_defaults,
            defaults_path=defaults_path,
            excel_path=excel_path,
        )
    except KeyboardInterrupt:
        print("收到 Ctrl+C，执行回零并退出。")
        _return_all_to_zero_then_tension(node, full_defaults[args.hand])
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    if DEVICES_MSG_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Cannot import devices_pkg.msg (YHandMsg/Motor). "
            "Please source the ROS2 workspace before running this script."
        ) from DEVICES_MSG_IMPORT_ERROR
    main()
