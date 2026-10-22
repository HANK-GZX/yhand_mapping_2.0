#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import os
import random
import select
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

try:
    from devices_pkg.msg import Motor, YHandMsg
except ImportError as exc:  # pragma: no cover
    Motor = None
    YHandMsg = None
    DEVICES_MSG_IMPORT_ERROR = exc
else:
    DEVICES_MSG_IMPORT_ERROR = None

try:
    from openpyxl import Workbook
except ImportError:  # pragma: no cover
    Workbook = None

from data_collection_config import (
    ARRIVAL_CONFIRM_SEC,
    ANTAGONIST_RELEASE_ENABLE,
    BIN_COVERAGE_INV_WEIGHT_POWER,
    BOUNDARY_EPS,
    COMBO_STALL_COMPLETE_RATIO,
    COMBO_STALL_COMPLETE_HIT_WINDOWS,
    COMBO_STALL_FAIL_TRIGGER,
    COMBO_STALL_MAX_ZERO_RETRY,
    COMBO_WEIGHT_COVERAGE_GAIN,
    COMBO_WEIGHT_QUOTA_GAP_GAIN,
    COMBO_WEIGHT_REMAIN_GAIN,
    CONTROL_PERIOD_SEC,
    DATA_COLLECTION_RELEASE_SEC_PER_DEG_BY_ACTIVE_TARGET,
    DATA_COLLECTION_RELEASE_SEC_PER_DEG_DEFAULT,
    DATA_COLLECTION_RELEASE_TOR_BY_ACTIVE_TARGET,
    DATA_COLLECTION_RELEASE_TOR_DEFAULT_A,
    THUMB_FLEXOR_EXTENSOR_RELEASE_TOR_A,
    THUMB_FLEXOR_EXTENSOR_RELEASE_SEC,
    DEFAULT_GLOVE_TOPIC,
    DEFAULT_GLOVE_RAW_SKELETON_TOPIC,
    DEFAULT_HAND,
    DEGENERATE_AXIS_SPAN_EPS,
    DISABLE_RANDOM_SINGLE_MOTOR_MODES,
    DIRECTION_BY_ROOM_WEIGHTED,
    ENFORCE_ACTIVE_TARGET_DECREASE,
    EXTENSOR_FIRST_ENABLE,
    EXTENSOR_FIRST_RELEASE_CURRENT_A,
    ACTIVE_EXTENSOR_FEEDFORWARD_TOR_A,
    EXTENSOR_FIRST_TIMEOUT_SEC,
    FEEDBACK_STALE_SEC,
    FINGER_COMPLETE_MIN_RATIO,
    FINGER_UNFREEZE_MAX_NO_PROGRESS_ROUNDS,
    FINGER_UNFREEZE_MAX_ROUNDS,
    FINGER_ORDER_FIXED,
    FLEXION_EDGE_BIAS_PROB,
    FLEXION_EDGE_REGION_RATIO,
    FOUR_FINGER_FLEXION_EXTRA_POINTS_PER_FINGER,
    FOUR_FINGER_ACTIVE_MODE_RATIO,
    FOUR_FINGER_QUOTA_BASE_PER_COMBO,
    HAND_CHOICES,
    MIN_STEP_ABS,
    MOTOR_COUNT,
    MOVE_TIMEOUT_SEC,
    POINT_ARRIVAL_TOL,
    POSITION_STABLE_MAX_RANGE_DEG,
    POSITION_STABLE_WINDOW_SEC,
    POST_ARRIVAL_TIGHTEN_SEC,
    POST_ARRIVAL_TIGHTEN_TOR_A,
    PROCESS_SAMPLE_HZ,
    QUOTA_BASE_PER_COMBO,
    QUOTA_FLOAT_RATIO,
    READY_WAIT_TIMEOUT_SEC,
    RESULT_WINDOW_END_SEC,
    RESULT_WINDOW_START_SEC,
    RETURN_ZERO_TIMEOUT_SEC,
    RNG_SEED,
    PRE_RANDOM_EXTENSOR_STEP,
    PRE_RANDOM_EXTENSOR_PASSIVE_RELEASE_TOR_A,
    PRE_RANDOM_EXTENSOR_PASSIVE_RELEASE_SEC,
    PRE_RANDOM_EXTENSOR_SWAY_TIGHTEN_TOR_A,
    THUMB_PRE_RANDOM_PASSIVE_RELEASE_TOR_A,
    THUMB_PRE_RANDOM_PASSIVE_RELEASE_SEC,
    THUMB_PRE_RANDOM_TIGHTEN_TOR_A,
    THUMB_PRE_RANDOM_TIGHTEN_SEC,
    THUMB_PRE_RANDOM_RECORD_SEC,
    THUMB_COMBO_RELEASE_TOR_A,
    THUMB_COMBO_RELEASE_SEC,
    THUMB_COMBO_PASSIVE_HOLD_TOR_A,
    THUMB_COMBO_FLEXOR_EXTENSOR_RELEASE_TOR_A,
    THUMB_COMBO_TIGHTEN_TOR_A,
    THUMB_COMBO_TIGHTEN_SEC,
    THUMB_FLEXOR_TIGHTEN_SEC,
    THUMB_COMBO_RECORD_SEC,
    PRE_RANDOM_FLEX_EXTENSOR_RELEASE_TOR_A,
    PRE_RANDOM_SWAY_OPPOSITE_RELEASE_TOR_A,
    PRE_RANDOM_PASSIVE_RELEASE_SEC,
    PRE_RANDOM_FLEX_STEP,
    PRE_RANDOM_OTHER_STEP,
    PRE_RANDOM_SWAY_STEP,
    PRE_RANDOM_UNIFORM_ENABLE,
    STABLE_HOLD_SEC,
    STATUS_PRINT_EVERY_POINTS,
    TARGET_DIRECTION_EPS,
    TARGET_NONE_COOLDOWN_CYCLES,
    TARGET_NONE_MAX_CONSEC_PER_COMBO,
    TARGET_NONE_WARN_EVERY,
    TARGET_GENERATION_RETRY,
    THUMB_OPPOSITION_ON_RATIO,
    THUMB_EXCLUDED_COMBO_IDS,
    THUMB_ACTIVE_MODE_RATIO,
    THUMB_ABDUCTION_EXTRA_POINTS,
    THUMB_ABDUCTION_EDGE_REGION_RATIO,
    THUMB_ABDUCTION_EDGE_BIAS_PROB,
    THUMB_FLEXION_EXTRA_POINTS,
    ZERO_FLEXOR_RELEASE_SEC,
    ZERO_FLEXOR_RELEASE_TOR_A,
    ZERO_FLEX_SWAY_TIGHTEN_TOR_A,
    ZERO_EXTENSOR_TOR_A,
    ZERO_THUMB_OPPOSITION_TOR_A,
    ZERO_CALIBRATE_STABLE_SEC,
    ZERO_CALIBRATE_MAX_ATTEMPTS,
    ZERO_CALIBRATE_EVERY_POINTS,
    ZERO_SWAY_RELEASE_SEC,
    ZERO_SWAY_RELEASE_TOR_A,
    default_output_root,
)
from workspace_boundary_config import HAND_MOTOR_MAP, get_hand_topics


MOTOR_FIELDS = ("pos", "vel", "tor", "kp", "kd")


@dataclass
class DirectionBoundary:
    dims: int
    x: np.ndarray
    y_min: np.ndarray
    y_max: np.ndarray
    z: np.ndarray | None = None


@dataclass
class ComboSpec:
    combo_id: str
    finger: str
    model_id: str
    dims: int
    direction: str
    opposition_mode: str
    x_motor: int
    y_motor: int
    z_motor: int | None
    active_motors: list[int]
    boundary: DirectionBoundary


@dataclass
class ComboCoverageState:
    mode: str  # "x" or "xz"
    x_edges: np.ndarray
    z_edges: np.ndarray | None
    counts: np.ndarray


@dataclass
class TargetPlan:
    target_by_motor: dict[int, float]
    coverage_key: tuple[int, ...] | None
    active_motors: list[int]
    active_mode: str


def _select_hand_raw_skeleton_from_msg(raw_data, hand_side_code: float):
    if not raw_data:
        return None
    try:
        arr = [float(v) for v in raw_data]
    except Exception:
        return None
    if len(arr) < 176:
        return None
    if len(arr) % 176 != 0:
        return None
    chunk_count = len(arr) // 176
    for i in range(chunk_count):
        start = i * 176
        chunk = arr[start : start + 176]
        if not chunk:
            continue
        side_code = float(chunk[0])
        if abs(side_code - float(hand_side_code)) <= 1e-6:
            return chunk
    return None


class DataCollectionNode(Node):
    def __init__(
        self,
        hand: str,
        cmd_topic: str,
        feedback_topic: str,
        glove_topic: str,
        glove_raw_skeleton_topic: str,
        defaults_hand,
    ):
        super().__init__(f"data_collection_{hand}")
        if YHandMsg is None or Motor is None:
            raise RuntimeError("Cannot import devices_pkg.msg. Please source workspace first.")

        self.hand = hand
        self.cmd_topic = cmd_topic
        self.feedback_topic = feedback_topic
        self.glove_topic = glove_topic
        self.glove_raw_skeleton_topic = glove_raw_skeleton_topic
        self.defaults_hand = defaults_hand
        self.hand_side_code = 0.0 if hand == "left" else 1.0

        self._lock = threading.Lock()
        self.latest_pos = np.zeros(MOTOR_COUNT, dtype=np.float64)
        self.latest_vel = np.zeros(MOTOR_COUNT, dtype=np.float64)
        self.latest_tor = np.zeros(MOTOR_COUNT, dtype=np.float64)
        self.latest_kp = np.zeros(MOTOR_COUNT, dtype=np.float64)
        self.latest_kd = np.zeros(MOTOR_COUNT, dtype=np.float64)
        self.feedback_received = False
        self.last_feedback_ts = 0.0

        self.latest_glove48 = np.full(48, np.nan, dtype=np.float64)
        self.glove_received = False
        self.last_glove_ts = 0.0
        self.latest_glove_skeleton176 = np.full(176, np.nan, dtype=np.float64)
        self.glove_raw_received = False
        self.last_glove_raw_ts = 0.0

        self.command_matrix = self.build_idle_tor_matrix()

        self.publisher = self.create_publisher(YHandMsg, self.cmd_topic, 10)
        self.feedback_sub = self.create_subscription(YHandMsg, self.feedback_topic, self._feedback_cb, 10)
        self.glove_sub = self.create_subscription(Float64MultiArray, self.glove_topic, self._glove_cb, 10)
        self.glove_raw_sub = self.create_subscription(
            Float64MultiArray,
            self.glove_raw_skeleton_topic,
            self._glove_raw_cb,
            10,
        )
        self.publish_timer = self.create_timer(CONTROL_PERIOD_SEC, self._publish_cb)

        self.get_logger().info(
            f"[data_collection] started hand={self.hand}, cmd={self.cmd_topic}, "
            f"feedback={self.feedback_topic}, glove={self.glove_topic}, "
            f"glove_raw={self.glove_raw_skeleton_topic}"
        )

    def _feedback_cb(self, msg):
        motors = list(msg.motors) if getattr(msg, "motors", None) is not None else []
        now = time.monotonic()
        with self._lock:
            for i in range(MOTOR_COUNT):
                if i < len(motors):
                    m = motors[i]
                    self.latest_pos[i] = float(m.pos)
                    self.latest_vel[i] = float(m.vel)
                    self.latest_tor[i] = float(m.tor)
                    self.latest_kp[i] = float(m.kp)
                    self.latest_kd[i] = float(m.kd)
            self.feedback_received = True
            self.last_feedback_ts = now

    def _glove_cb(self, msg):
        data = list(msg.data) if getattr(msg, "data", None) is not None else []
        if len(data) < 48:
            return
        now = time.monotonic()
        with self._lock:
            self.latest_glove48 = np.asarray(data[:48], dtype=np.float64)
            self.glove_received = True
            self.last_glove_ts = now

    def _glove_raw_cb(self, msg):
        data = list(msg.data) if getattr(msg, "data", None) is not None else []
        selected = _select_hand_raw_skeleton_from_msg(data, self.hand_side_code)
        if selected is None:
            return
        now = time.monotonic()
        with self._lock:
            self.latest_glove_skeleton176 = np.asarray(selected, dtype=np.float64)
            self.glove_raw_received = True
            self.last_glove_raw_ts = now

    def _publish_cb(self):
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

    def wait_ready(
        self,
        timeout_sec: float,
        require_feedback: bool = True,
        require_glove: bool = True,
        require_glove_raw: bool = True,
    ) -> tuple[bool, str]:
        start = time.monotonic()
        while (time.monotonic() - start) < timeout_sec:
            with self._lock:
                fb_ok = (not bool(require_feedback)) or bool(self.feedback_received)
                glove_ok = (not bool(require_glove)) or bool(self.glove_received)
                raw_ok = (not bool(require_glove_raw)) or bool(self.glove_raw_received)
                if fb_ok and glove_ok and raw_ok:
                    return True, ""
            time.sleep(0.05)
        with self._lock:
            return False, (
                f"feedback_received={self.feedback_received}, glove_received={self.glove_received}, "
                f"glove_raw_received={self.glove_raw_received}, "
                f"require_feedback={bool(require_feedback)}, "
                f"require_glove={bool(require_glove)}, "
                f"require_glove_raw={bool(require_glove_raw)}, "
                f"cmd_sub={self.count_subscribers(self.cmd_topic)}, "
                f"fb_pub={self.count_publishers(self.feedback_topic)}, "
                f"glove_pub={self.count_publishers(self.glove_topic)}, "
                f"glove_raw_pub={self.count_publishers(self.glove_raw_skeleton_topic)}"
            )

    def build_zero_matrix_full_defaults(self):
        matrix = []
        for i in range(MOTOR_COUNT):
            matrix.append(
                [
                    0.0,
                    float(self.defaults_hand["vel"][i]),
                    float(self.defaults_hand["tor"][i]),
                    float(self.defaults_hand["kp"][i]),
                    float(self.defaults_hand["kd"][i]),
                ]
            )
        return matrix

    def build_idle_tor_matrix(self):
        matrix = []
        for i in range(MOTOR_COUNT):
            matrix.append(
                [
                    0.0,
                    0.0,
                    float(self.defaults_hand["tor"][i]),
                    0.0,
                    0.0,
                ]
            )
        return matrix

    def set_command_matrix(self, matrix):
        with self._lock:
            self.command_matrix = [row[:] for row in matrix]

    def snapshot(self):
        with self._lock:
            if not self.feedback_received:
                return None
            return {
                "mono": time.monotonic(),
                "wall": time.time(),
                "feedback_age": time.monotonic() - self.last_feedback_ts,
                "glove_age": (time.monotonic() - self.last_glove_ts) if self.glove_received else float("inf"),
                "glove_raw_age": (
                    time.monotonic() - self.last_glove_raw_ts
                ) if self.glove_raw_received else float("inf"),
                "pos": self.latest_pos.copy(),
                "vel": self.latest_vel.copy(),
                "tor": self.latest_tor.copy(),
                "kp": self.latest_kp.copy(),
                "kd": self.latest_kd.copy(),
                "glove48": self.latest_glove48.copy() if self.glove_received else None,
                "glove_skeleton176": (
                    self.latest_glove_skeleton176.copy() if self.glove_raw_received else None
                ),
                "cmd_matrix": [row[:] for row in self.command_matrix],
            }


def _spin_node(node: Node):
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        return
    except KeyboardInterrupt:
        return


def _load_defaults_from_py(defaults_path: Path, hand: str):
    spec = importlib.util.spec_from_file_location("muscle_adjustment_defaults_dynamic", str(defaults_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load defaults file: {defaults_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    defaults = getattr(mod, "DEFAULTS", None)
    if defaults is None or hand not in defaults:
        raise RuntimeError(f"DEFAULTS or hand '{hand}' not found in {defaults_path}")
    hand_defaults = defaults[hand]
    out = {}
    for key, fallback in (("vel", 0.0), ("kp", 500.0), ("kd", 10.0), ("tor", -0.03)):
        values = list(hand_defaults.get(key, []))
        if len(values) < MOTOR_COUNT:
            values += [fallback] * (MOTOR_COUNT - len(values))
        elif len(values) > MOTOR_COUNT:
            values = values[:MOTOR_COUNT]
        out[key] = [float(v) for v in values]

    # Optional active feedforward torque profile for active motors.
    # Backward compatible: if missing, reuse idle/default tor profile.
    tor_active_values = list(hand_defaults.get("tor_active", out["tor"]))
    if len(tor_active_values) < MOTOR_COUNT:
        for i in range(len(tor_active_values), MOTOR_COUNT):
            tor_active_values.append(out["tor"][i])
    elif len(tor_active_values) > MOTOR_COUNT:
        tor_active_values = tor_active_values[:MOTOR_COUNT]
    out["tor_active"] = [float(v) for v in tor_active_values]
    return out


def _load_boundary_json(boundary_path: Path, hand: str):
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary json not found: {boundary_path}")
    obj = json.loads(boundary_path.read_text(encoding="utf-8"))
    if obj.get("hand") != hand:
        raise RuntimeError(
            f"Boundary hand mismatch: json={obj.get('hand')} runtime={hand}, file={boundary_path}"
        )
    combos = obj.get("combos", {})
    if not isinstance(combos, dict) or not combos:
        raise RuntimeError(f"Invalid combos in boundary json: {boundary_path}")
    return obj, combos


def _finger_motor_sets(hand: str):
    hand_map = HAND_MOTOR_MAP[hand]
    out = {}
    for finger, muscles in hand_map.items():
        mids = sorted({int(v) for v in muscles.values()})
        out[finger] = mids
    return out


def _zero_sway_motor_ids(hand: str):
    hand_map = HAND_MOTOR_MAP.get(hand, {})
    out = set()
    for finger, muscle_map in hand_map.items():
        if finger == "thumb":
            continue
        if not isinstance(muscle_map, dict):
            continue
        for muscle_name, motor_id in muscle_map.items():
            key = str(muscle_name).lower()
            if ("abduction" in key) or ("adduction" in key):
                mid = int(motor_id)
                if 1 <= mid <= MOTOR_COUNT:
                    out.add(mid)
    return sorted(out)


def _zero_flexor_motor_ids(hand: str):
    hand_map = HAND_MOTOR_MAP.get(hand, {})
    out = set()
    for finger, muscle_map in hand_map.items():
        if finger == "thumb":
            continue
        if not isinstance(muscle_map, dict):
            continue
        for muscle_name, motor_id in muscle_map.items():
            key = str(muscle_name).lower()
            if "flex" in key:
                mid = int(motor_id)
                if 1 <= mid <= MOTOR_COUNT:
                    out.add(mid)
    return sorted(out)


def _zero_feedback_positions(node: DataCollectionNode):
    snap = node.snapshot()
    if snap is None:
        return None
    age = float(snap.get("feedback_age", float("inf")))
    if not math.isfinite(age) or age > FEEDBACK_STALE_SEC:
        return None
    positions = np.asarray(snap["pos"], dtype=np.float64)
    if positions.shape != (MOTOR_COUNT,) or not np.all(np.isfinite(positions)):
        return None
    return positions


def _run_zero_prerelease(node: DataCollectionNode, defaults_hand):
    sway_motor_ids = _zero_sway_motor_ids(node.hand)
    flexor_motor_ids = _zero_flexor_motor_ids(node.hand)
    total_sec = max(float(ZERO_SWAY_RELEASE_SEC), float(ZERO_FLEXOR_RELEASE_SEC))
    if total_sec <= 0.0 or (not sway_motor_ids and not flexor_motor_ids):
        return

    # All non-releasing motors, including extensors, hold their initial pose.
    positions = _zero_feedback_positions(node)
    if positions is not None:
        hold_matrix = _build_hold_current_matrix(defaults_hand, positions)
    else:
        hold_matrix = _build_idle_tor_matrix(defaults_hand)

    if sway_motor_ids:
        print(
            "[calibrate] 回0前侧摆放线: "
            f"motors={sway_motor_ids}, tor={float(ZERO_SWAY_RELEASE_TOR_A):.3f}A, "
            f"duration={float(ZERO_SWAY_RELEASE_SEC):.3f}s"
        )
    if flexor_motor_ids:
        print(
            "[calibrate] 回0前屈肌放线: "
            f"motors={flexor_motor_ids}, tor={float(ZERO_FLEXOR_RELEASE_TOR_A):.3f}A, "
            f"duration={float(ZERO_FLEXOR_RELEASE_SEC):.3f}s"
        )

    release_start = time.monotonic()
    while (time.monotonic() - release_start) < total_sec:
        elapsed = time.monotonic() - release_start
        overrides = {}
        if sway_motor_ids and elapsed < float(ZERO_SWAY_RELEASE_SEC):
            for mid in sway_motor_ids:
                overrides[int(mid)] = [0.0, 0.0, float(ZERO_SWAY_RELEASE_TOR_A), 0.0, 0.0]
        if flexor_motor_ids and elapsed < float(ZERO_FLEXOR_RELEASE_SEC):
            for mid in flexor_motor_ids:
                cur_row = overrides.get(int(mid))
                cur_tor = float(cur_row[2]) if cur_row is not None else 0.0
                overrides[int(mid)] = [
                    0.0,
                    0.0,
                    max(cur_tor, float(ZERO_FLEXOR_RELEASE_TOR_A)),
                    0.0,
                    0.0,
                ]

        matrix = [row[:] for row in hold_matrix]
        for mid, row in overrides.items():
            matrix[int(mid) - 1] = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])]
        node.set_command_matrix(matrix)
        time.sleep(CONTROL_PERIOD_SEC)


def _target_direction_ok(hand: str, current_pos: np.ndarray, target_by_motor: dict[int, float]) -> bool:
    if not ENFORCE_ACTIVE_TARGET_DECREASE:
        return True
    eps = float(TARGET_DIRECTION_EPS)
    for motor_id, target in target_by_motor.items():
        mid = int(motor_id)
        if not (1 <= mid <= MOTOR_COUNT):
            continue
        cur = float(current_pos[mid - 1])
        tgt = float(target)
        # Active-motor tighten direction: target must be strictly smaller than current.
        if not (tgt < (cur - eps)):
            return False
    return True


def _finger_motor_named_ids(hand: str, finger: str):
    hand_map = HAND_MOTOR_MAP.get(hand, {})
    muscle_map = hand_map.get(finger, {})
    out = []
    seen = set()
    for muscle_name, motor_id in muscle_map.items():
        try:
            mid = int(motor_id)
        except Exception:
            continue
        if not (1 <= mid <= MOTOR_COUNT) or mid in seen:
            continue
        seen.add(mid)
        out.append((str(muscle_name), int(mid)))
    return out


def _print_point_stage(point_id: int, nominal_total: int, stage: str):
    total = int(max(1, int(nominal_total)))
    print(f"[point-stage] {int(point_id)}/{total} stage={stage}")


def _build_uniform_current_matrix(current_a: float):
    tor = float(current_a)
    return [[0.0, 0.0, tor, 0.0, 0.0] for _ in range(MOTOR_COUNT)]


def _keep_four_fingers_at_default_current(hand, defaults_hand, matrix):
    out = [row[:] for row in matrix]
    for finger in ("index", "middle", "ring", "little"):
        for mid in HAND_MOTOR_MAP[hand][finger].values():
            out[mid - 1] = [0.0, 0.0, float(defaults_hand["tor"][mid - 1]), 0.0, 0.0]
    return out


def _poll_stdin_line_nonblocking():
    if not sys.stdin.isatty():
        return None
    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    except Exception:
        return None
    if not readable:
        return None
    line = sys.stdin.readline()
    if line is None:
        return None
    line = str(line).strip()
    if not line:
        return None
    return line


def _resolve_control_file_path(control_file_raw: str, output_root: Path, hand: str) -> Path:
    raw = str(control_file_raw or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(output_root) / f"{str(hand).strip().lower()}_control_commands.log"


def _append_control_command(control_file: Path, cmd_text: str):
    p = Path(control_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": time.time(),
        "cmd": str(cmd_text).strip(),
    }
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _init_control_reader(control_file: Path):
    p = Path(control_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    try:
        if p.exists():
            offset = int(p.stat().st_size)
    except Exception:
        offset = 0
    return {"path": p, "offset": offset}


def _poll_control_commands(reader_state: dict):
    if not isinstance(reader_state, dict):
        return []
    p = Path(reader_state.get("path", ""))
    if not str(p):
        return []
    if not p.exists():
        return []
    offset = int(reader_state.get("offset", 0))
    text = ""
    try:
        with p.open("r", encoding="utf-8") as f:
            f.seek(max(0, offset))
            text = f.read()
            reader_state["offset"] = int(f.tell())
    except Exception:
        return []
    if not text:
        return []
    out = []
    for line in str(text).splitlines():
        line = str(line).strip()
        if not line:
            continue
        cmd = ""
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                cmd = str(obj.get("cmd", "")).strip()
            else:
                cmd = str(obj).strip()
        except Exception:
            cmd = line
        if cmd:
            out.append(cmd)
    return out


def _print_runtime_auto_help():
    print(
        "[auto-control] commands: "
        "p=pause+zero, f <finger>=switch finger, r=resume interrupted finger, "
        "z=zero, t=tighten(-0.1A), i=current0, m=manual supplement, e=emergency stop(current0), "
        "q=safe exit, h=help"
    )


def _prepare_combo_boundary(combo: dict):
    dims = int(combo["dims"])
    if combo.get("status") != "ok":
        raise RuntimeError(f"Combo not ready: combo_id={combo.get('combo_id', '<unknown>')}")
    x = np.asarray(combo.get("x_grid", []), dtype=np.float64)
    y_min = np.asarray(combo.get("y_min", []), dtype=np.float64)
    y_max = np.asarray(combo.get("y_max", []), dtype=np.float64)
    if x.size < 2:
        raise RuntimeError("Invalid combo boundary: x_grid must contain at least 2 points")
    if dims == 2:
        if y_min.shape != x.shape or y_max.shape != x.shape:
            raise RuntimeError("Invalid 2D combo boundary: shape mismatch between x_grid and y bounds")
        return DirectionBoundary(dims=2, x=x, y_min=y_min, y_max=y_max, z=None)
    z = np.asarray(combo.get("z_grid", []), dtype=np.float64)
    if z.size < 2:
        raise RuntimeError("Invalid 3D combo boundary: z_grid must contain at least 2 points")
    if y_min.shape != (x.size, z.size) or y_max.shape != (x.size, z.size):
        raise RuntimeError("Invalid 3D combo boundary: y_min/y_max shape mismatch with x_grid/z_grid")
    return DirectionBoundary(dims=3, x=x, y_min=y_min, y_max=y_max, z=z)


def _build_combo_specs(combos: dict):
    combos_by_finger = {finger: [] for finger in FINGER_ORDER_FIXED}
    for combo_id, combo in combos.items():
        finger = str(combo.get("finger"))
        if finger not in combos_by_finger:
            continue
        if combo.get("status") != "ok":
            continue
        if finger == "thumb" and combo_id in THUMB_EXCLUDED_COMBO_IDS:
            continue
        dims = int(combo["dims"])
        axes = combo.get("axes", {})
        x_motor = int(axes["x_motor"])
        y_motor = int(axes["y_motor"])
        z_motor = int(axes["z_motor"]) if axes.get("z_motor") is not None else None
        direction = str(combo.get("direction", ""))
        opposition_mode = str(combo.get("opposition_mode", "none"))
        if dims == 2:
            z_motor = None
            opposition_mode = "none"
            active_motors = [x_motor, y_motor]
        else:
            active_motors = [x_motor, y_motor]
            if opposition_mode == "on" and z_motor is not None:
                active_motors.append(z_motor)

        boundary = _prepare_combo_boundary(combo)
        combo_spec = ComboSpec(
            combo_id=str(combo_id),
            finger=finger,
            model_id=str(combo.get("model_id", combo_id)),
            dims=dims,
            direction=direction,
            opposition_mode=opposition_mode,
            x_motor=x_motor,
            y_motor=y_motor,
            z_motor=z_motor,
            active_motors=active_motors,
            boundary=boundary,
        )
        combos_by_finger[finger].append(combo_spec)

    for finger in combos_by_finger:
        combos_by_finger[finger] = sorted(combos_by_finger[finger], key=lambda c: c.combo_id)
    return combos_by_finger


def _interp_1d(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    if xq <= float(x[0]):
        return float(y[0])
    if xq >= float(x[-1]):
        return float(y[-1])
    return float(np.interp(xq, x, y))


def _bilinear_interp(x: np.ndarray, z: np.ndarray, grid: np.ndarray, xq: float, zq: float) -> float:
    xq = float(np.clip(xq, x[0], x[-1]))
    zq = float(np.clip(zq, z[0], z[-1]))
    ix = int(np.searchsorted(x, xq) - 1)
    iz = int(np.searchsorted(z, zq) - 1)
    ix = max(0, min(ix, len(x) - 2))
    iz = max(0, min(iz, len(z) - 2))
    x1, x2 = float(x[ix]), float(x[ix + 1])
    z1, z2 = float(z[iz]), float(z[iz + 1])
    q11 = float(grid[ix, iz])
    q12 = float(grid[ix, iz + 1])
    q21 = float(grid[ix + 1, iz])
    q22 = float(grid[ix + 1, iz + 1])
    tx = 0.0 if abs(x2 - x1) < 1e-12 else (xq - x1) / (x2 - x1)
    tz = 0.0 if abs(z2 - z1) < 1e-12 else (zq - z1) / (z2 - z1)
    a = q11 * (1.0 - tx) + q21 * tx
    b = q12 * (1.0 - tx) + q22 * tx
    return a * (1.0 - tz) + b * tz


def _build_combo_coverage_state(combo: ComboSpec) -> ComboCoverageState:
    x_edges = np.asarray(combo.boundary.x, dtype=np.float64)
    nx = max(0, len(x_edges) - 1)
    if combo.dims == 3 and combo.opposition_mode == "on" and combo.boundary.z is not None:
        z_edges = np.asarray(combo.boundary.z, dtype=np.float64)
        nz = max(0, len(z_edges) - 1)
        if nx > 0 and nz > 0:
            return ComboCoverageState(
                mode="xz",
                x_edges=x_edges,
                z_edges=z_edges,
                counts=np.zeros((nx, nz), dtype=np.int64),
            )
    return ComboCoverageState(
        mode="x",
        x_edges=x_edges,
        z_edges=None,
        counts=np.zeros((nx,), dtype=np.int64),
    )


def _coverage_deficit(coverage: ComboCoverageState) -> float:
    total = int(coverage.counts.size)
    if total <= 0:
        return 1.0
    visited = int(np.count_nonzero(coverage.counts))
    return 1.0 - (visited / float(total))


def _coverage_stats(coverage: ComboCoverageState) -> dict:
    total = int(coverage.counts.size)
    visited = int(np.count_nonzero(coverage.counts))
    return {
        "mode": coverage.mode,
        "visited_bins": visited,
        "total_bins": total,
        "coverage_ratio": (visited / total) if total > 0 else 0.0,
        "min_bin_count": int(np.min(coverage.counts)) if total > 0 else 0,
        "max_bin_count": int(np.max(coverage.counts)) if total > 0 else 0,
    }


def _coverage_bin_weight(bin_count: float) -> float:
    power = max(1e-6, float(BIN_COVERAGE_INV_WEIGHT_POWER))
    return 1.0 / ((1.0 + float(bin_count)) ** power)


def _direction_feasible(cur: float, lo: float, hi: float, center: float) -> bool:
    can_neg = (cur - lo) > BOUNDARY_EPS
    can_pos = (hi - cur) > BOUNDARY_EPS
    if center < (cur - BOUNDARY_EPS):
        return can_neg
    if center > (cur + BOUNDARY_EPS):
        return can_pos
    return can_neg or can_pos


def _direction_room_weight(cur: float, lo: float, hi: float, center: float) -> float:
    if not DIRECTION_BY_ROOM_WEIGHTED:
        return 1.0
    neg_room = max(0.0, cur - lo)
    pos_room = max(0.0, hi - cur)
    if center < (cur - BOUNDARY_EPS):
        return max(1e-6, neg_room)
    if center > (cur + BOUNDARY_EPS):
        return max(1e-6, pos_room)
    return max(1e-6, max(neg_room, pos_room))


def _choose_x_bin(coverage: ComboCoverageState, x_cur: float, rng: random.Random):
    if coverage.counts.size <= 0 or len(coverage.x_edges) < 2:
        return None
    x_lo = float(coverage.x_edges[0])
    x_hi = float(coverage.x_edges[-1])
    x_ref = float(np.clip(x_cur, x_lo, x_hi))
    centers = 0.5 * (coverage.x_edges[:-1] + coverage.x_edges[1:])

    indices = []
    weights = []
    for ix in range(len(centers)):
        center = float(centers[ix])
        if coverage.mode == "xz" and getattr(coverage.counts, "ndim", 1) == 2:
            # For x-only planning on xz-coverage combos, aggregate z bins to one x weight.
            x_cov_count = float(np.mean(coverage.counts[ix, :]))
        else:
            x_cov_count = float(coverage.counts[ix])
        w_cov = _coverage_bin_weight(x_cov_count)
        if _direction_feasible(x_ref, x_lo, x_hi, center):
            w_dir = _direction_room_weight(x_ref, x_lo, x_hi, center)
        else:
            w_dir = 0.0
        indices.append(ix)
        weights.append(w_cov * w_dir)
    if not indices:
        return None
    if sum(weights) <= 0.0:
        weights = []
        for ix in indices:
            if coverage.mode == "xz" and getattr(coverage.counts, "ndim", 1) == 2:
                x_cov_count = float(np.mean(coverage.counts[ix, :]))
            else:
                x_cov_count = float(coverage.counts[ix])
            weights.append(_coverage_bin_weight(x_cov_count))
    return rng.choices(indices, weights=weights, k=1)[0]


def _nearest_z_bin(z_edges: np.ndarray, zq: float):
    if len(z_edges) < 2:
        return None
    centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    if centers.size <= 0:
        return None
    return int(np.argmin(np.abs(centers - float(zq))))


def _choose_xz_bin(coverage: ComboCoverageState, x_cur: float, z_cur: float, rng: random.Random):
    if coverage.counts.size <= 0 or coverage.z_edges is None:
        return None
    if len(coverage.x_edges) < 2 or len(coverage.z_edges) < 2:
        return None
    x_lo = float(coverage.x_edges[0])
    x_hi = float(coverage.x_edges[-1])
    z_lo = float(coverage.z_edges[0])
    z_hi = float(coverage.z_edges[-1])
    x_ref = float(np.clip(x_cur, x_lo, x_hi))
    z_ref = float(np.clip(z_cur, z_lo, z_hi))
    x_centers = 0.5 * (coverage.x_edges[:-1] + coverage.x_edges[1:])
    z_centers = 0.5 * (coverage.z_edges[:-1] + coverage.z_edges[1:])

    flat = []
    weights = []
    for ix in range(len(x_centers)):
        for iz in range(len(z_centers)):
            cx = float(x_centers[ix])
            cz = float(z_centers[iz])
            w_cov = _coverage_bin_weight(float(coverage.counts[ix, iz]))
            feasible_x = _direction_feasible(x_ref, x_lo, x_hi, cx)
            feasible_z = _direction_feasible(z_ref, z_lo, z_hi, cz)
            if feasible_x and feasible_z:
                wx = _direction_room_weight(x_ref, x_lo, x_hi, cx)
                wz = _direction_room_weight(z_ref, z_lo, z_hi, cz)
                w_dir = wx * wz
            else:
                w_dir = 0.0
            flat.append((ix, iz))
            weights.append(w_cov * w_dir)
    if not flat:
        return None
    if sum(weights) <= 0.0:
        weights = [_coverage_bin_weight(float(coverage.counts[ix, iz])) for ix, iz in flat]
    return rng.choices(flat, weights=weights, k=1)[0]


def _sample_from_edges(edges: np.ndarray, idx: int, rng: random.Random):
    lo = float(edges[idx])
    hi = float(edges[idx + 1])
    if hi < lo:
        lo, hi = hi, lo
    if abs(hi - lo) <= BOUNDARY_EPS:
        return lo
    return float(rng.uniform(lo, hi))


def _is_thumb_out_combo(combo: ComboSpec) -> bool:
    return (str(combo.finger).strip().lower() == "thumb") and (str(combo.direction).strip().lower() == "out")


def _is_flex_combo(combo: ComboSpec) -> bool:
    model = str(combo.model_id).strip().lower()
    return "flex" in model


def _is_thumb_flex_combo(combo: ComboSpec) -> bool:
    return str(combo.finger).strip().lower() == "thumb" and _is_flex_combo(combo)


def _is_four_finger_flex_combo(combo: ComboSpec) -> bool:
    return str(combo.finger).strip().lower() != "thumb" and _is_flex_combo(combo)


def _x_bin_index_from_edges(x_edges: np.ndarray, xq: float):
    if len(x_edges) < 2:
        return None
    return int(np.clip(np.searchsorted(x_edges, float(xq), side="right") - 1, 0, len(x_edges) - 2))


def _sample_x_with_flex_edge_bias(
    combo: ComboSpec,
    x_lo: float,
    x_hi: float,
    rng: random.Random,
    enable_bias: bool = False,
) -> float:
    if x_hi < x_lo:
        x_lo, x_hi = x_hi, x_lo
    span = float(x_hi - x_lo)
    if span <= DEGENERATE_AXIS_SPAN_EPS:
        return 0.5 * (x_lo + x_hi)

    if enable_bias and _is_flex_combo(combo):
        edge_ratio = min(1.0, max(0.0, float(FLEXION_EDGE_REGION_RATIO)))
        bias_prob = min(1.0, max(0.0, float(FLEXION_EDGE_BIAS_PROB)))
        if edge_ratio > 0.0 and rng.random() < bias_prob:
            edge_hi = x_lo + span * edge_ratio
            if edge_hi > x_lo + float(BOUNDARY_EPS):
                return float(rng.uniform(x_lo, edge_hi))

    return float(rng.uniform(x_lo, x_hi))


def _sample_y_with_thumb_out_edge_bias(
    combo: ComboSpec,
    y_lo: float,
    y_hi: float,
    rng: random.Random,
    enable_bias: bool = False,
) -> float:
    if y_hi < y_lo:
        y_lo, y_hi = y_hi, y_lo
    y_span = float(y_hi - y_lo)
    if y_span <= DEGENERATE_AXIS_SPAN_EPS:
        return 0.5 * (y_lo + y_hi)

    # Thumb out (abduction) enhancement:
    # sample more points near the abduction-side boundary.
    if enable_bias and _is_thumb_out_combo(combo):
        edge_ratio = min(1.0, max(0.0, float(THUMB_ABDUCTION_EDGE_REGION_RATIO)))
        bias_prob = min(1.0, max(0.0, float(THUMB_ABDUCTION_EDGE_BIAS_PROB)))
        if edge_ratio > 0.0 and rng.random() < bias_prob:
            # By current tendon direction convention, tighter abduction target tends to be
            # the smaller position side, so we bias to the lower-edge band.
            edge_hi = y_lo + y_span * edge_ratio
            if edge_hi > y_lo + float(BOUNDARY_EPS):
                return float(rng.uniform(y_lo, edge_hi))

    return float(rng.uniform(y_lo, y_hi))


def _max_active_step(current_pos: np.ndarray, target_by_motor: dict[int, float], active_motors: list[int]) -> float:
    step = 0.0
    for motor_id in active_motors:
        if motor_id not in target_by_motor:
            continue
        cur = float(current_pos[motor_id - 1])
        tgt = float(target_by_motor[motor_id])
        step = max(step, abs(tgt - cur))
    return step


def _min_target_step_required() -> float:
    # Keep random targets meaningfully farther than arrival threshold.
    # Example: if POINT_ARRIVAL_TOL=14, required step is >14.
    return max(float(MIN_STEP_ABS), float(POINT_ARRIVAL_TOL) + float(TARGET_DIRECTION_EPS))


def _generate_target_2d(
    combo: ComboSpec,
    hand: str,
    current_pos: np.ndarray,
    coverage: ComboCoverageState,
    rng: random.Random,
    edge_bias: bool = False,
):
    b = combo.boundary
    if len(b.x) < 2:
        return None
    x_cur = float(current_pos[combo.x_motor - 1])
    min_step_required = _min_target_step_required()
    for _ in range(TARGET_GENERATION_RETRY):
        ix = _choose_x_bin(coverage, x_cur, rng)
        if ix is None:
            continue
        x_new = _sample_x_with_flex_edge_bias(
            combo,
            float(coverage.x_edges[0]),
            float(coverage.x_edges[-1]),
            rng,
            enable_bias=edge_bias,
        )
        ix_new = _x_bin_index_from_edges(coverage.x_edges, x_new)
        if ix_new is not None:
            ix = int(ix_new)
        y_lo = _interp_1d(b.x, b.y_min, x_new)
        y_hi = _interp_1d(b.x, b.y_max, x_new)
        if y_hi < y_lo:
            continue
        y_span = y_hi - y_lo
        if y_span <= DEGENERATE_AXIS_SPAN_EPS:
            y_new = 0.5 * (y_lo + y_hi)
        else:
            y_new = float(rng.uniform(y_lo, y_hi))
        target_by_motor = {
            combo.x_motor: float(x_new),
            combo.y_motor: float(y_new),
        }
        if not _target_direction_ok(hand, current_pos, target_by_motor):
            continue
        if _max_active_step(current_pos, target_by_motor, combo.active_motors) < min_step_required:
            continue
        return TargetPlan(
            target_by_motor=target_by_motor,
            coverage_key=(ix,),
            active_motors=sorted(int(mid) for mid in target_by_motor.keys()),
            active_mode=("xy" if combo.dims == 2 else "full"),
        )
    return None


def _generate_target_3d(
    combo: ComboSpec,
    hand: str,
    current_pos: np.ndarray,
    coverage: ComboCoverageState,
    rng: random.Random,
    edge_bias: bool = False,
):
    b = combo.boundary
    if b.z is None or len(b.x) < 2 or len(b.z) < 2:
        return None
    x_cur = float(current_pos[combo.x_motor - 1])
    z_cur = float(current_pos[combo.z_motor - 1]) if combo.z_motor is not None else 0.0
    z_lo_all = float(b.z[0])
    z_hi_all = float(b.z[-1])
    z_ref = float(np.clip(z_cur, z_lo_all, z_hi_all))
    min_step_required = _min_target_step_required()

    for _ in range(TARGET_GENERATION_RETRY):
        if combo.opposition_mode == "on":
            if coverage.mode != "xz":
                return None
            key = _choose_xz_bin(coverage, x_cur, z_ref, rng)
            if key is None:
                continue
            ix, iz = key
            x_new = _sample_x_with_flex_edge_bias(
                combo,
                float(coverage.x_edges[0]),
                float(coverage.x_edges[-1]),
                rng,
                enable_bias=edge_bias,
            )
            ix_new = _x_bin_index_from_edges(coverage.x_edges, x_new)
            if ix_new is not None:
                ix = int(ix_new)
            z_new = _sample_from_edges(coverage.z_edges, iz, rng)
            coverage_key = (ix, iz)
        else:
            ix = _choose_x_bin(coverage, x_cur, rng)
            if ix is None:
                continue
            x_new = _sample_x_with_flex_edge_bias(
                combo,
                float(coverage.x_edges[0]),
                float(coverage.x_edges[-1]),
                rng,
                enable_bias=edge_bias,
            )
            ix_new = _x_bin_index_from_edges(coverage.x_edges, x_new)
            if ix_new is not None:
                ix = int(ix_new)
            # Requested behavior for opposition_mode=off:
            # command opposition motor to a larger position than current (within boundary).
            if combo.z_motor is not None:
                z_lo_candidate = z_ref + float(TARGET_DIRECTION_EPS)
                if z_hi_all <= z_lo_candidate:
                    # No room to choose a strictly larger in-bound opposition target.
                    continue
                z_new = float(rng.uniform(z_lo_candidate, z_hi_all))
            else:
                z_span = z_hi_all - z_lo_all
                if z_span <= DEGENERATE_AXIS_SPAN_EPS:
                    z_new = 0.5 * (z_lo_all + z_hi_all)
                else:
                    z_new = z_ref
            coverage_key = (ix,)

        y_lo = _bilinear_interp(b.x, b.z, b.y_min, x_new, z_new)
        y_hi = _bilinear_interp(b.x, b.z, b.y_max, x_new, z_new)
        if y_hi < y_lo:
            continue
        y_new = _sample_y_with_thumb_out_edge_bias(combo, y_lo, y_hi, rng, enable_bias=edge_bias)

        target_by_motor = {
            combo.x_motor: float(x_new),
            combo.y_motor: float(y_new),
        }
        if combo.z_motor is not None:
            target_by_motor[combo.z_motor] = float(z_new)

        direction_check_targets = dict(target_by_motor)
        if combo.opposition_mode != "on" and combo.z_motor is not None:
            # opposition_mode=off special case: opposition motor is allowed to move to larger target.
            direction_check_targets.pop(int(combo.z_motor), None)

        if not _target_direction_ok(hand, current_pos, direction_check_targets):
            continue
        if _max_active_step(current_pos, target_by_motor, combo.active_motors) < min_step_required:
            continue
        return TargetPlan(
            target_by_motor=target_by_motor,
            coverage_key=coverage_key,
            active_motors=sorted(int(mid) for mid in target_by_motor.keys()),
            active_mode="full",
        )
    return None


def _generate_target(
    hand: str,
    combo: ComboSpec,
    current_pos: np.ndarray,
    coverage: ComboCoverageState,
    rng: random.Random,
    active_mode: str | None = None,
    edge_bias: bool = False,
):
    mode = str(active_mode or "").strip().lower()
    if combo.dims == 2:
        if mode in ("", "xy", "full"):
            return _generate_target_2d(combo, hand, current_pos, coverage, rng, edge_bias=edge_bias)
        if mode == "x_only":
            active_ids = [int(combo.x_motor)]
        elif mode == "y_only":
            active_ids = [int(combo.y_motor)]
        else:
            active_ids = [int(combo.x_motor), int(combo.y_motor)]
        return _generate_target_2d_subset(
            combo=combo,
            hand=hand,
            current_pos=current_pos,
            coverage=coverage,
            rng=rng,
            active_motors=active_ids,
            active_mode=mode if mode else "xy",
            edge_bias=edge_bias,
        )

    if mode in ("", "full", "xyz"):
        return _generate_target_3d(combo, hand, current_pos, coverage, rng, edge_bias=edge_bias)

    z_motor = int(combo.z_motor) if combo.z_motor is not None else None
    if z_motor is None:
        return _generate_target_3d(combo, hand, current_pos, coverage, rng, edge_bias=edge_bias)

    x_motor = int(combo.x_motor)
    y_motor = int(combo.y_motor)
    if mode == "single":
        candidates = [
            [x_motor],
            [y_motor],
            [z_motor],
        ]
    elif mode == "double":
        candidates = [
            [x_motor, y_motor],
            [x_motor, z_motor],
            [y_motor, z_motor],
        ]
    else:
        return _generate_target_3d(combo, hand, current_pos, coverage, rng)

    rng.shuffle(candidates)
    for active_ids in candidates:
        plan = _generate_target_3d_subset(
            combo=combo,
            hand=hand,
            current_pos=current_pos,
            coverage=coverage,
            rng=rng,
            active_motors=active_ids,
            active_mode=mode,
            edge_bias=edge_bias,
        )
        if plan is not None:
            return plan
    return None


def _nearest_x_bin(coverage: ComboCoverageState, xq: float):
    if coverage.counts.size <= 0 or len(coverage.x_edges) < 2:
        return None
    centers = 0.5 * (coverage.x_edges[:-1] + coverage.x_edges[1:])
    if centers.size <= 0:
        return None
    return int(np.argmin(np.abs(centers - float(xq))))


def _generate_target_2d_subset(
    combo: ComboSpec,
    hand: str,
    current_pos: np.ndarray,
    coverage: ComboCoverageState,
    rng: random.Random,
    active_motors: list[int],
    active_mode: str,
    edge_bias: bool = False,
):
    b = combo.boundary
    if len(b.x) < 2:
        return None
    active_set = {int(mid) for mid in active_motors if 1 <= int(mid) <= MOTOR_COUNT}
    x_active = int(combo.x_motor) in active_set
    y_active = int(combo.y_motor) in active_set
    if not (x_active or y_active):
        return None

    x_lo = float(b.x[0])
    x_hi = float(b.x[-1])
    x_cur = float(current_pos[combo.x_motor - 1])
    y_cur = float(current_pos[combo.y_motor - 1])
    x_ref = float(np.clip(x_cur, x_lo, x_hi))
    min_step_required = _min_target_step_required()

    for _ in range(TARGET_GENERATION_RETRY):
        ix = None
        if x_active:
            ix = _choose_x_bin(coverage, x_ref, rng)
            if ix is None:
                continue
            x_new = _sample_x_with_flex_edge_bias(
                combo,
                float(coverage.x_edges[0]),
                float(coverage.x_edges[-1]),
                rng,
                enable_bias=edge_bias,
            )
            ix_new = _x_bin_index_from_edges(coverage.x_edges, x_new)
            if ix_new is not None:
                ix = int(ix_new)
        else:
            x_new = x_ref
            ix = _nearest_x_bin(coverage, x_ref)

        y_lo = _interp_1d(b.x, b.y_min, x_new)
        y_hi = _interp_1d(b.x, b.y_max, x_new)
        if y_hi < y_lo:
            continue

        if y_active:
            y_new = _sample_y_with_thumb_out_edge_bias(combo, y_lo, y_hi, rng, enable_bias=edge_bias)
        else:
            if y_cur < (y_lo - float(BOUNDARY_EPS)) or y_cur > (y_hi + float(BOUNDARY_EPS)):
                continue
            y_new = y_cur

        target_by_motor = {}
        if x_active:
            target_by_motor[int(combo.x_motor)] = float(x_new)
        if y_active:
            target_by_motor[int(combo.y_motor)] = float(y_new)
        if not target_by_motor:
            continue

        if not _target_direction_ok(hand, current_pos, target_by_motor):
            continue
        active_sorted = sorted(int(mid) for mid in target_by_motor.keys())
        if _max_active_step(current_pos, target_by_motor, active_sorted) < min_step_required:
            continue
        coverage_key = (int(ix),) if ix is not None else None
        return TargetPlan(
            target_by_motor=target_by_motor,
            coverage_key=coverage_key,
            active_motors=active_sorted,
            active_mode=str(active_mode),
        )
    return None


def _generate_target_3d_subset(
    combo: ComboSpec,
    hand: str,
    current_pos: np.ndarray,
    coverage: ComboCoverageState,
    rng: random.Random,
    active_motors: list[int],
    active_mode: str,
    edge_bias: bool = False,
):
    b = combo.boundary
    if b.z is None or len(b.x) < 2 or len(b.z) < 2 or combo.z_motor is None:
        return None
    active_set = {int(mid) for mid in active_motors if 1 <= int(mid) <= MOTOR_COUNT}
    x_motor = int(combo.x_motor)
    y_motor = int(combo.y_motor)
    z_motor = int(combo.z_motor)
    x_active = x_motor in active_set
    y_active = y_motor in active_set
    z_active = z_motor in active_set
    if not (x_active or y_active or z_active):
        return None

    x_lo_all = float(b.x[0])
    x_hi_all = float(b.x[-1])
    z_lo_all = float(b.z[0])
    z_hi_all = float(b.z[-1])
    x_cur = float(current_pos[x_motor - 1])
    y_cur = float(current_pos[y_motor - 1])
    z_cur = float(current_pos[z_motor - 1])
    x_ref = float(np.clip(x_cur, x_lo_all, x_hi_all))
    z_ref = float(np.clip(z_cur, z_lo_all, z_hi_all))
    min_step_required = _min_target_step_required()

    for _ in range(TARGET_GENERATION_RETRY):
        coverage_key = None

        if x_active and z_active and combo.opposition_mode == "on" and coverage.mode == "xz":
            key = _choose_xz_bin(coverage, x_ref, z_ref, rng)
            if key is None:
                continue
            ix, iz = key
            x_new = _sample_x_with_flex_edge_bias(
                combo,
                float(coverage.x_edges[0]),
                float(coverage.x_edges[-1]),
                rng,
                enable_bias=edge_bias,
            )
            ix_new = _x_bin_index_from_edges(coverage.x_edges, x_new)
            if ix_new is not None:
                ix = int(ix_new)
            z_new = _sample_from_edges(coverage.z_edges, iz, rng)
            coverage_key = (int(ix), int(iz))
        else:
            ix = None
            if x_active:
                ix = _choose_x_bin(coverage, x_ref, rng)
                if ix is None:
                    continue
                x_new = _sample_x_with_flex_edge_bias(
                    combo,
                    float(coverage.x_edges[0]),
                    float(coverage.x_edges[-1]),
                    rng,
                    enable_bias=edge_bias,
                )
                ix_new = _x_bin_index_from_edges(coverage.x_edges, x_new)
                if ix_new is not None:
                    ix = int(ix_new)
            else:
                x_new = x_ref
                ix = _nearest_x_bin(coverage, x_ref)
            if ix is not None:
                coverage_key = (int(ix),)

            if z_active:
                if combo.opposition_mode == "off":
                    z_lo_candidate = z_ref + float(TARGET_DIRECTION_EPS)
                    if z_hi_all <= z_lo_candidate:
                        continue
                    z_new = float(rng.uniform(z_lo_candidate, z_hi_all))
                else:
                    if len(b.z) >= 2:
                        iz = rng.randrange(len(b.z) - 1)
                        z_new = _sample_from_edges(b.z, iz, rng)
                    else:
                        z_new = z_ref
            else:
                z_new = z_ref

            if ix is not None:
                if coverage.mode == "xz" and coverage.z_edges is not None and len(coverage.z_edges) >= 2:
                    iz_key = _nearest_z_bin(
                        coverage.z_edges,
                        (z_new if z_active else z_ref),
                    )
                    if iz_key is not None:
                        coverage_key = (int(ix), int(iz_key))

        y_lo = _bilinear_interp(b.x, b.z, b.y_min, x_new, z_new)
        y_hi = _bilinear_interp(b.x, b.z, b.y_max, x_new, z_new)
        if y_hi < y_lo:
            continue

        if y_active:
            y_new = _sample_y_with_thumb_out_edge_bias(combo, y_lo, y_hi, rng, enable_bias=edge_bias)
        else:
            if y_cur < (y_lo - float(BOUNDARY_EPS)) or y_cur > (y_hi + float(BOUNDARY_EPS)):
                continue
            y_new = y_cur

        target_by_motor = {}
        if x_active:
            target_by_motor[x_motor] = float(x_new)
        if y_active:
            target_by_motor[y_motor] = float(y_new)
        if z_active:
            target_by_motor[z_motor] = float(z_new)
        if not target_by_motor:
            continue

        direction_check_targets = dict(target_by_motor)
        if combo.opposition_mode != "on" and z_active:
            # opposition_mode=off keeps historical behavior: opposition can go to larger target.
            direction_check_targets.pop(int(z_motor), None)
        if not _target_direction_ok(hand, current_pos, direction_check_targets):
            continue

        active_sorted = sorted(int(mid) for mid in target_by_motor.keys())
        if _max_active_step(current_pos, target_by_motor, active_sorted) < min_step_required:
            continue
        return TargetPlan(
            target_by_motor=target_by_motor,
            coverage_key=coverage_key,
            active_motors=active_sorted,
            active_mode=str(active_mode),
        )
    return None


def _mark_coverage_hit(coverage: ComboCoverageState, coverage_key: tuple[int, ...] | None):
    if coverage_key is None:
        return
    if coverage.mode == "x":
        if len(coverage_key) != 1:
            return
        ix = int(coverage_key[0])
        if 0 <= ix < coverage.counts.shape[0]:
            coverage.counts[ix] += 1
        return
    if coverage.mode == "xz":
        if len(coverage_key) != 2:
            return
        ix = int(coverage_key[0])
        iz = int(coverage_key[1])
        if 0 <= ix < coverage.counts.shape[0] and 0 <= iz < coverage.counts.shape[1]:
            coverage.counts[ix, iz] += 1


def _extract_glove48(glove48: np.ndarray | None):
    if glove48 is None or len(glove48) < 48:
        return [float("nan")] * 48
    return [float(glove48[i]) for i in range(48)]


def _extract_glove_skeleton176(glove_skeleton176: np.ndarray | None):
    if glove_skeleton176 is None or len(glove_skeleton176) < 176:
        return [float("nan")] * 176
    return [float(glove_skeleton176[i]) for i in range(176)]


def _save_startup_first_frame(node: DataCollectionNode, hand_root: Path, wait_timeout_sec: float = 2.0):
    def _safe(v):
        try:
            fv = float(v)
        except Exception:
            return None
        if not math.isfinite(fv):
            return None
        return fv

    deadline = time.monotonic() + max(0.0, float(wait_timeout_sec))
    snap = node.snapshot()
    while snap is None and time.monotonic() < deadline:
        time.sleep(0.01)
        snap = node.snapshot()

    if snap is None:
        print("[WARN] startup first frame snapshot unavailable, skip saving.")
        return None

    glove48 = _extract_glove48(snap.get("glove48"))
    motor_pos = []
    pos_arr = snap.get("pos")
    for i in range(MOTOR_COUNT):
        if pos_arr is not None and len(pos_arr) > i:
            motor_pos.append(_safe(pos_arr[i]))
        else:
            motor_pos.append(None)

    motor_feedback = []
    vel_arr = snap.get("vel")
    tor_arr = snap.get("tor")
    kp_arr = snap.get("kp")
    kd_arr = snap.get("kd")
    for i in range(MOTOR_COUNT):
        motor_feedback.append(
            {
                "motor_id": i + 1,
                "pos": motor_pos[i],
                "vel": _safe(vel_arr[i]) if vel_arr is not None and len(vel_arr) > i else None,
                "tor": _safe(tor_arr[i]) if tor_arr is not None and len(tor_arr) > i else None,
                "kp": _safe(kp_arr[i]) if kp_arr is not None and len(kp_arr) > i else None,
                "kd": _safe(kd_arr[i]) if kd_arr is not None and len(kd_arr) > i else None,
            }
        )

    payload = {
        "captured_at": datetime.now().isoformat(),
        "feedback_age_sec": _safe(snap.get("feedback_age")),
        "glove_age_sec": _safe(snap.get("glove_age")),
        "glove_raw_age_sec": _safe(snap.get("glove_raw_age")),
        "glove48": [_safe(v) for v in glove48],
        "motor_pos": motor_pos,
        "motor_feedback": motor_feedback,
    }

    json_path = hand_root / "startup_first_frame.json"
    glove_csv_path = hand_root / "startup_first_frame_glove48.csv"
    motor_csv_path = hand_root / "startup_first_frame_motor_pos.csv"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_csv_row(
        glove_csv_path,
        [f"q_{i}" for i in range(48)],
        payload["glove48"],
    )
    _append_csv_row(
        motor_csv_path,
        [f"m{i:02d}_pos" for i in range(1, MOTOR_COUNT + 1)],
        payload["motor_pos"],
    )

    return {
        "json": str(json_path),
        "glove48_csv": str(glove_csv_path),
        "motor_pos_csv": str(motor_csv_path),
    }


def _skeleton176_to_node_rows(skeleton_row):
    vals = [float(v) for v in skeleton_row]
    vals += [float("nan")] * max(0, 176 - len(vals))
    vals = vals[:176]
    side_code = float(vals[0])
    rows = []
    for node_id in range(25):
        base = 1 + node_id * 7
        rows.append(
            [
                side_code,
                float(node_id),
                float(vals[base + 0]),
                float(vals[base + 1]),
                float(vals[base + 2]),
                float(vals[base + 3]),
                float(vals[base + 4]),
                float(vals[base + 5]),
                float(vals[base + 6]),
            ]
        )
    return rows


def _flatten_motor_matrix(matrix):
    out = []
    for m in matrix:
        out.extend([float(v) for v in m])
    return out


def _flatten_feedback(snapshot):
    out = []
    for i in range(MOTOR_COUNT):
        out.append(float(snapshot["pos"][i]))
        out.append(float(snapshot["vel"][i]))
        out.append(float(snapshot["tor"][i]))
        out.append(float(snapshot["kp"][i]))
        out.append(float(snapshot["kd"][i]))
    return out


def _motor_headers():
    headers = []
    for m in range(1, MOTOR_COUNT + 1):
        for field in MOTOR_FIELDS:
            headers.append(f"m{m:02d}_{field}")
    return headers


def _split_motor_flat_row(row):
    t_rel = float("nan")
    if row and len(row) >= 1:
        t_rel = float(row[0])
    vals = list(row[1:]) if row and len(row) > 1 else []
    need = MOTOR_COUNT * 5
    if len(vals) < need:
        vals += [float("nan")] * (need - len(vals))
    mats = []
    for i in range(MOTOR_COUNT):
        base = i * 5
        mats.append(
            [
                float(vals[base + 0]),
                float(vals[base + 1]),
                float(vals[base + 2]),
                float(vals[base + 3]),
                float(vals[base + 4]),
            ]
        )
    return t_rel, mats


def _write_xlsx(path: Path, headers, rows):
    if Workbook is None:
        raise RuntimeError("openpyxl is required to save xlsx files. Please install openpyxl.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    wb = Workbook()
    ws = wb.active
    ws.title = "data"
    ws.append(list(headers))
    for row in rows:
        ws.append(list(row))
    wb.save(tmp_path)
    tmp_path.replace(path)


def _write_xlsx_rows(path: Path, rows):
    if Workbook is None:
        raise RuntimeError("openpyxl is required to save xlsx files. Please install openpyxl.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    wb = Workbook()
    ws = wb.active
    ws.title = "data"
    for row in rows:
        ws.append(list(row))
    wb.save(tmp_path)
    tmp_path.replace(path)


def _append_csv_row(csv_path: Path, headers, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(headers)
        writer.writerow(row)


def _merge_release_plan_entry(
    release_sec_by_motor: dict[int, float],
    release_tor_by_motor: dict[int, float],
    motor_id: int,
    release_sec: float,
    release_tor: float,
):
    mid = int(motor_id)
    if not (1 <= mid <= MOTOR_COUNT):
        return
    sec_v = float(release_sec)
    if (not math.isfinite(sec_v)) or sec_v <= 0.0:
        return
    tor_v = float(release_tor)
    if not math.isfinite(tor_v):
        tor_v = float(DATA_COLLECTION_RELEASE_TOR_DEFAULT_A)

    prev_sec = float(release_sec_by_motor.get(mid, 0.0))
    prev_tor = float(release_tor_by_motor.get(mid, 0.0))
    release_sec_by_motor[mid] = max(prev_sec, sec_v)
    release_tor_by_motor[mid] = max(prev_tor, tor_v)


def _dc_release_tor_for_pair(hand: str, active_motor_id: int, target_motor_id: int) -> float:
    hand_map = DATA_COLLECTION_RELEASE_TOR_BY_ACTIVE_TARGET.get(str(hand), {})
    active_map = hand_map.get(int(active_motor_id), {})
    tor_v = float(active_map.get(int(target_motor_id), DATA_COLLECTION_RELEASE_TOR_DEFAULT_A))
    if not math.isfinite(tor_v):
        tor_v = float(DATA_COLLECTION_RELEASE_TOR_DEFAULT_A)
    return float(tor_v)


def _dc_release_sec_for_pair(hand: str, active_motor_id: int, target_motor_id: int, delta_deg: float) -> float:
    hand_map = DATA_COLLECTION_RELEASE_SEC_PER_DEG_BY_ACTIVE_TARGET.get(str(hand), {})
    active_map = hand_map.get(int(active_motor_id), {})
    per_deg = float(
        active_map.get(int(target_motor_id), DATA_COLLECTION_RELEASE_SEC_PER_DEG_DEFAULT)
    )
    if (not math.isfinite(per_deg)) or per_deg <= 0.0:
        per_deg = float(DATA_COLLECTION_RELEASE_SEC_PER_DEG_DEFAULT)
    delta_abs = max(0.0, float(delta_deg))
    return float(delta_abs * per_deg)


def _extensor_first_motor_sets(hand, finger, target_by_motor, active_motor_ids):
    """Use muscle roles so the same staged motion works for both hands."""
    if not (ANTAGONIST_RELEASE_ENABLE and EXTENSOR_FIRST_ENABLE):
        return [], []
    muscles = HAND_MOTOR_MAP.get(hand, {}).get(finger, {})
    active = set(active_motor_ids) & set(target_by_motor)
    extensors = sorted(int(mid) for role, mid in muscles.items()
                       if role.startswith("ext") and int(mid) in active)
    if not extensors:
        return [], []
    released = sorted(int(mid) for role, mid in muscles.items()
                      if "flex" in role or role in ("abduction", "adduction"))
    return extensors, released


def _preuniform_passive_currents(hand: str, finger: str, target_by_motor: dict[int, float]):
    if not ANTAGONIST_RELEASE_ENABLE or len(target_by_motor) != 1:
        return {}
    muscles = HAND_MOTOR_MAP.get(hand, {}).get(finger, {})
    active = next(iter(target_by_motor))
    if finger == "thumb":
        return {int(mid): float(THUMB_PRE_RANDOM_PASSIVE_RELEASE_TOR_A)
                for mid in muscles.values() if int(mid) != int(active)}
    role = next((name for name, mid in muscles.items() if int(mid) == int(active)), "")
    if role.startswith("ext"):
        return {int(mid): float(PRE_RANDOM_EXTENSOR_PASSIVE_RELEASE_TOR_A)
                for name, mid in muscles.items()
                if "flex" in name or name in ("abduction", "adduction")}
    if "flex" in role:
        return {int(mid): float(PRE_RANDOM_FLEX_EXTENSOR_RELEASE_TOR_A)
                for name, mid in muscles.items() if name.startswith("ext")}
    if role in ("abduction", "adduction"):
        currents = {int(mid): 0.0 for name, mid in muscles.items()
                    if name.startswith("ext") or "flex" in name}
        opposite = "adduction" if role == "abduction" else "abduction"
        currents[int(muscles[opposite])] = float(PRE_RANDOM_SWAY_OPPOSITE_RELEASE_TOR_A)
        return currents
    return {}


def _build_combo_release_plan(
    hand: str,
    current_pos: np.ndarray,
    target_by_motor: dict[int, float],
    active_motor_ids: list[int] | None = None,
):
    if active_motor_ids is None:
        active_set = sorted({int(mid) for mid in target_by_motor.keys() if 1 <= int(mid) <= MOTOR_COUNT})
    else:
        active_set = sorted({int(mid) for mid in active_motor_ids if 1 <= int(mid) <= MOTOR_COUNT})
    release_sec_by_motor = {}
    release_tor_by_motor = {}

    sec_cfg_hand = DATA_COLLECTION_RELEASE_SEC_PER_DEG_BY_ACTIVE_TARGET.get(str(hand), {})
    tor_cfg_hand = DATA_COLLECTION_RELEASE_TOR_BY_ACTIVE_TARGET.get(str(hand), {})

    for motor_id in active_set:
        target_pos = float(target_by_motor[motor_id])
        cur_pos = float(current_pos[motor_id - 1]) if current_pos.size >= motor_id else 0.0
        delta_deg = abs(target_pos - cur_pos)

        sec_targets = set(sec_cfg_hand.get(int(motor_id), {}).keys())
        tor_targets = set(tor_cfg_hand.get(int(motor_id), {}).keys())
        target_ids = sorted({int(tid) for tid in (sec_targets | tor_targets)})

        for tid in target_ids:
            _merge_release_plan_entry(
                release_sec_by_motor=release_sec_by_motor,
                release_tor_by_motor=release_tor_by_motor,
                motor_id=int(tid),
                release_sec=_dc_release_sec_for_pair(hand, int(motor_id), int(tid), delta_deg),
                release_tor=_dc_release_tor_for_pair(hand, int(motor_id), int(tid)),
            )

    # Multi-motor policy: keep only opposite-side release, never release active motors.
    for motor_id in active_set:
        release_sec_by_motor.pop(int(motor_id), None)
        release_tor_by_motor.pop(int(motor_id), None)

    return release_sec_by_motor, release_tor_by_motor


def _build_command_matrix_for_point(
    hand: str,
    defaults_hand,
    finger_motor_set: dict[str, list[int]],
    finger: str,
    target_by_motor: dict[int, float],
    release_tor_by_motor: dict[int, float] | None = None,
    hold_pos_by_motor: dict[int, float] | None = None,
):
    matrix = []
    same_finger_motors = set(finger_motor_set[finger])
    extensor_motors = {int(mid) for role, mid in HAND_MOTOR_MAP[hand][finger].items()
                       if role.startswith("ext")}
    if release_tor_by_motor is None:
        release_tor_by_motor = {}
    else:
        release_tor_by_motor = {
            int(mid): float(tor)
            for mid, tor in release_tor_by_motor.items()
            if 1 <= int(mid) <= MOTOR_COUNT
        }
    if hold_pos_by_motor is None:
        hold_pos_by_motor = {}
    else:
        hold_pos_by_motor = {
            int(mid): float(pos)
            for mid, pos in hold_pos_by_motor.items()
            if 1 <= int(mid) <= MOTOR_COUNT
        }
    for motor_id in range(1, MOTOR_COUNT + 1):
        idx = motor_id - 1
        vel_d = float(defaults_hand["vel"][idx])
        tor_idle_d = float(defaults_hand["tor"][idx])
        kp_d = float(defaults_hand["kp"][idx])
        kd_d = float(defaults_hand["kd"][idx])
        if motor_id in target_by_motor:
            tor_d = (float(ACTIVE_EXTENSOR_FEEDFORWARD_TOR_A)
                     if motor_id in extensor_motors else tor_idle_d)
            row = [float(target_by_motor[motor_id]), vel_d, tor_d, kp_d, kd_d]
        elif motor_id in release_tor_by_motor:
            # Antagonist passive release to reduce coupling on active motors.
            row = [0.0, 0.0, float(release_tor_by_motor[motor_id]), 0.0, 0.0]
        elif motor_id in same_finger_motors:
            # Same finger, non-active motor: pure current-mode tighten.
            row = [0.0, 0.0, tor_idle_d, 0.0, 0.0]
        else:
            # Other fingers also stay in pure current-mode tighten.
            row = [0.0, 0.0, tor_idle_d, 0.0, 0.0]
        matrix.append(row)
    return matrix


def _build_zero_matrix(defaults_hand):
    matrix = []
    for i in range(MOTOR_COUNT):
        matrix.append(
            [
                0.0,
                float(defaults_hand["vel"][i]),
                float(defaults_hand["tor"][i]),
                float(defaults_hand["kp"][i]),
                float(defaults_hand["kd"][i]),
            ]
        )
    return matrix


def _build_idle_tor_matrix(defaults_hand):
    matrix = []
    for i in range(MOTOR_COUNT):
        matrix.append(
            [
                0.0,
                0.0,
                float(defaults_hand["tor"][i]),
                0.0,
                0.0,
            ]
        )
    return matrix


def _build_hold_current_matrix(defaults_hand, current_pos: np.ndarray):
    matrix = []
    for i in range(MOTOR_COUNT):
        pos_i = float(current_pos[i]) if (current_pos is not None and i < len(current_pos)) else 0.0
        matrix.append(
            [
                pos_i,
                float(defaults_hand["vel"][i]),
                float(defaults_hand["tor"][i]),
                float(defaults_hand["kp"][i]),
                float(defaults_hand["kd"][i]),
            ]
        )
    return matrix


def _max_active_error(snapshot, target_by_motor, arrival_motor_ids: list[int] | None = None):
    errs = []
    if arrival_motor_ids is None:
        mids = [int(mid) for mid in target_by_motor.keys()]
    else:
        mids = [int(mid) for mid in arrival_motor_ids]
    for motor_id in mids:
        if motor_id not in target_by_motor:
            continue
        cur = float(snapshot["pos"][motor_id - 1])
        tgt = float(target_by_motor[motor_id])
        if not (math.isfinite(cur) and math.isfinite(tgt)):
            return float("inf")
        errs.append(abs(cur - tgt))
    return max(errs) if errs else 0.0


def _wait_finger_position_stable(node, motor_ids, runtime_tick=None, on_sample=None):
    """Wait for a full rolling window; never bypass stability on a timeout."""
    indices = np.asarray(motor_ids, dtype=int) - 1
    window_sec = float(POSITION_STABLE_WINDOW_SEC)
    max_range = float(POSITION_STABLE_MAX_RANGE_DEG)
    if not len(indices) or not math.isfinite(window_sec) or window_sec <= 0.0:
        raise ValueError("Position stability needs motor IDs and a positive finite window")
    if not math.isfinite(max_range) or max_range < 0.0:
        raise ValueError("Position stability range must be finite and non-negative")
    history = deque()
    started = time.monotonic()
    next_sample = started
    period = 1.0 / max(1.0, float(PROCESS_SAMPLE_HZ))
    last_valid_time = None
    while True:
        if callable(runtime_tick):
            interruption = runtime_tick()
            if interruption:
                reason = (str(interruption.get("kind", "runtime_interrupt"))
                          if isinstance(interruption, dict) else str(interruption))
                return {"position_stable_confirmed": False,
                        "position_stable_wait_sec": float(time.monotonic() - started),
                        "position_stable_max_span_deg": None,
                        "interrupted": True, "interrupt_reason": reason}
        now = time.monotonic()
        if now < next_sample:
            time.sleep(min(float(CONTROL_PERIOD_SEC), next_sample - now))
            continue
        next_sample = now + period
        snap = node.snapshot()
        if snap is not None and callable(on_sample):
            on_sample(now, snap)
        age = float(snap.get("feedback_age", float("inf"))) if snap is not None else float("inf")
        positions = (np.asarray(snap["pos"], dtype=float)[indices]
                     if snap is not None else None)
        valid = (math.isfinite(age) and 0.0 <= age <= FEEDBACK_STALE_SEC
                 and positions is not None and np.isfinite(positions).all())
        if not valid:
            history.clear()
            last_valid_time = None
            continue
        if last_valid_time is not None and now - last_valid_time > FEEDBACK_STALE_SEC:
            history.clear()
        last_valid_time = now
        history.append((now, positions.copy()))
        # Keep one sample at/before the window boundary so a short window
        # cannot pass merely because the oldest sample was pruned.
        cutoff = now - window_sec
        while len(history) > 1 and history[1][0] <= cutoff:
            history.popleft()
        if now - history[0][0] >= window_sec:
            span = np.ptp(np.stack([item[1] for item in history]), axis=0)
            if np.all(span <= max_range):
                return {"position_stable_confirmed": True,
                        "position_stable_wait_sec": float(time.monotonic() - started),
                        "position_stable_max_span_deg": float(np.max(span)),
                        "interrupted": False, "interrupt_reason": ""}


def _collect_window_samples(node: DataCollectionNode, duration_sec: float, sample_hz: float, t_ref: float):
    rows_glove = []
    rows_glove_skeleton = []
    rows_fb = []
    if duration_sec <= 0.0:
        return rows_glove, rows_glove_skeleton, rows_fb

    period = 1.0 / max(1.0, float(sample_hz))
    end_t = time.monotonic() + duration_sec
    next_t = time.monotonic()
    while time.monotonic() < end_t:
        now = time.monotonic()
        if now < next_t:
            time.sleep(min(0.005, next_t - now))
            continue
        next_t += period
        snap = node.snapshot()
        if snap is None:
            continue
        _ = float(time.monotonic() - t_ref)
        rows_glove.append(_extract_glove48(snap["glove48"]))
        rows_glove_skeleton.append(_extract_glove_skeleton176(snap["glove_skeleton176"]))
        rows_fb.append([float(time.monotonic() - t_ref)] + _flatten_feedback(snap))
    return rows_glove, rows_glove_skeleton, rows_fb


def _average_rows(rows):
    if not rows:
        return []
    arr = np.asarray(rows, dtype=np.float64)
    return [float(v) for v in np.nanmean(arr, axis=0).tolist()]


def _slice_rows_by_time_window(rows_glove, rows_glove_skeleton, rows_fb, t_start: float, t_end: float):
    out_glove = []
    out_glove_skeleton = []
    out_fb = []
    n = min(len(rows_glove), len(rows_glove_skeleton), len(rows_fb))
    for i in range(n):
        fb_row = rows_fb[i]
        if not fb_row:
            continue
        try:
            t_rel = float(fb_row[0])
        except Exception:
            continue
        if t_rel < float(t_start) or t_rel > float(t_end):
            continue
        out_glove.append(rows_glove[i])
        out_glove_skeleton.append(rows_glove_skeleton[i])
        out_fb.append(rows_fb[i])
    return out_glove, out_glove_skeleton, out_fb


def _execute_point(
    node: DataCollectionNode,
    defaults_hand,
    finger_motor_set,
    finger: str,
    combo: ComboSpec,
    target_by_motor: dict[int, float],
    active_motors: list[int],
    active_mode: str,
    point_tolerance: float,
    point_id: int,
    nominal_total: int,
    runtime_tick=None,
    process_include_post_stable: bool = False,
    preuniform: bool = False,
):
    snap_init = node.snapshot()
    hold_pos_by_motor = {}
    if snap_init is not None and "pos" in snap_init:
        hold_pos_by_motor = {
            mid: float(snap_init["pos"][mid - 1])
            for mid in range(1, MOTOR_COUNT + 1)
        }

    matrix_normal = _build_command_matrix_for_point(
        hand=node.hand,
        defaults_hand=defaults_hand,
        finger_motor_set=finger_motor_set,
        finger=finger,
        target_by_motor=target_by_motor,
        release_tor_by_motor={},
        hold_pos_by_motor=hold_pos_by_motor,
    )
    release_sec_plan = {}
    release_tor_plan = {}
    thumb_passive_release_currents = {}
    arrival_motor_ids = [int(mid) for mid in active_motors if int(mid) in target_by_motor]
    if not arrival_motor_ids:
        arrival_motor_ids = [int(mid) for mid in target_by_motor.keys()]
    muscles = HAND_MOTOR_MAP[node.hand][finger]
    sway_ids = sorted(int(mid) for role, mid in muscles.items()
                      if role in ("adduction", "abduction") and int(mid) in arrival_motor_ids)
    thumb_sway_first = bool(finger == "thumb" and not preuniform
                            and len(sway_ids) == 1 and len(arrival_motor_ids) > 1)
    thumb_combo_flexor_ids = [int(mid) for role, mid in muscles.items()
                             if thumb_sway_first and "flex" in role and int(mid) in arrival_motor_ids]
    thumb_combo_passive_extensors = [int(mid) for role, mid in muscles.items()
                                    if thumb_combo_flexor_ids and role.startswith("ext")
                                    and int(mid) not in target_by_motor]
    thumb_flexor_reached = False
    thumb_flexor_arrival_enter_t = None
    extensor_ids, extensor_release_ids = _extensor_first_motor_sets(
        node.hand, finger, target_by_motor, arrival_motor_ids)
    if thumb_sway_first:
        extensor_ids, extensor_release_ids = [], []
    extensor_first = bool(extensor_ids)
    thumb_preuniform = bool(preuniform and finger == "thumb" and len(target_by_motor) == 1)
    result_window_start = 0.0 if thumb_preuniform or thumb_sway_first else float(RESULT_WINDOW_START_SEC)
    result_window_end = (float(THUMB_COMBO_RECORD_SEC) if thumb_sway_first else
                         float(THUMB_PRE_RANDOM_RECORD_SEC) if thumb_preuniform else
                         float(RESULT_WINDOW_END_SEC))
    result_window_label = f"{result_window_start:.1f}-{result_window_end:.1f}"
    preuniform_currents = (_preuniform_passive_currents(node.hand, finger, target_by_motor)
                           if preuniform else {})
    continuous_assist = bool(preuniform_currents)
    preuniform_extensor = bool(continuous_assist and any(
        role.startswith("ext") and int(mid) in arrival_motor_ids
        for role, mid in HAND_MOTOR_MAP.get(node.hand, {}).get(finger, {}).items()))
    preuniform_release_sec = float(PRE_RANDOM_EXTENSOR_PASSIVE_RELEASE_SEC
                                  if preuniform_extensor else PRE_RANDOM_PASSIVE_RELEASE_SEC)
    if thumb_preuniform:
        preuniform_release_sec = float(THUMB_PRE_RANDOM_PASSIVE_RELEASE_SEC)
        if extensor_first:
            extensor_release_ids = sorted(preuniform_currents)
    preuniform_rest_currents = dict.fromkeys(preuniform_currents, 0.0)
    if preuniform_extensor and not thumb_preuniform:
        muscles = HAND_MOTOR_MAP[node.hand][finger]
        for role in ("abduction", "adduction"):
            preuniform_rest_currents[int(muscles[role])] = float(PRE_RANDOM_EXTENSOR_SWAY_TIGHTEN_TOR_A)
    extensor_stage_reached = False
    extensor_stage_elapsed = 0.0
    sway_stage_reached = False
    sway_stage_elapsed = 0.0
    if thumb_sway_first:
        release_tor_plan = {int(mid): float(THUMB_COMBO_RELEASE_TOR_A)
                            for mid in finger_motor_set[finger] if int(mid) not in sway_ids}
    elif extensor_first:
        release_tor_plan = (dict(preuniform_currents) if preuniform_extensor else
                            {mid: float(EXTENSOR_FIRST_RELEASE_CURRENT_A)
                             for mid in extensor_release_ids})
    elif continuous_assist:
        release_tor_plan = dict(preuniform_currents)
    elif ANTAGONIST_RELEASE_ENABLE:
        snap0 = snap_init if snap_init is not None else node.snapshot()
        cur_pos = (
            np.asarray(snap0["pos"], dtype=np.float64)
            if (snap0 is not None and "pos" in snap0)
            else np.zeros(MOTOR_COUNT, dtype=np.float64)
        )
        release_sec_plan, release_tor_plan = _build_combo_release_plan(
            hand=node.hand,
            current_pos=cur_pos,
            target_by_motor=target_by_motor,
            active_motor_ids=arrival_motor_ids,
        )

    if (finger == "thumb" and not preuniform and ANTAGONIST_RELEASE_ENABLE
            and not extensor_first and not thumb_sway_first):
        muscles = HAND_MOTOR_MAP[node.hand][finger]
        if any("flex" in role and int(mid) in arrival_motor_ids for role, mid in muscles.items()):
            thumb_passive_release_currents = {
                int(mid): float(THUMB_FLEXOR_EXTENSOR_RELEASE_TOR_A)
                for role, mid in muscles.items()
                if role.startswith("ext") and int(mid) not in target_by_motor
            }
            for active_role, opposite_role in (("adduction", "abduction"),
                                                ("abduction", "adduction")):
                if int(muscles[active_role]) in arrival_motor_ids:
                    opposite = int(muscles[opposite_role])
                    if opposite not in target_by_motor:
                        thumb_passive_release_currents[opposite] = float(THUMB_FLEXOR_EXTENSOR_RELEASE_TOR_A)
            # Use current thumb roles rather than legacy cross-finger pairs.
            thumb_motors = set(muscles.values())
            release_sec_plan = {mid: sec for mid, sec in release_sec_plan.items() if mid in thumb_motors}
            release_tor_plan = {mid: tor for mid, tor in release_tor_plan.items() if mid in thumb_motors}
            for mid, current in thumb_passive_release_currents.items():
                release_sec_plan.pop(mid, None)
                release_tor_plan[mid] = current
            print(f"[thumb-flexor-assist] passive motors={thumb_passive_release_currents}; "
                  f"kp/kd=0, release for {THUMB_FLEXOR_EXTENSOR_RELEASE_SEC:.3f}s, "
                  "then kp/kd/current=0 until point arrival or timeout.")

    release_active = (thumb_sway_first or extensor_first or continuous_assist or bool(release_sec_plan)
                      or bool(thumb_passive_release_currents))
    release_total_sec = max((float(v) for v in release_sec_plan.values()), default=0.0)
    current_cmd_matrix = None
    last_stage = None

    def _stage_label(phase: str, event: str) -> str:
        if phase == "move":
            if event == "preuniform_current0":
                return "current0"
            if event == "preuniform_passive_current":
                return "passive_current"
            if event in ("sway_first", "sway_current0", "extensor_first", "remaining_targets"):
                return event
            if "release" in event:
                return "release"
            if ("target" in event) or ("cleared" in event):
                return "target_hold"
            return "move"
        if phase == "arrived":
            return "arrived"
        if phase == "stable":
            return "stable"
        return phase

    def _apply_cmd_matrix(matrix, phase: str, event: str):
        nonlocal current_cmd_matrix, last_stage
        if finger == "thumb":
            matrix = _keep_four_fingers_at_default_current(node.hand, defaults_hand, matrix)
        node.set_command_matrix(matrix)
        current_cmd_matrix = [[float(v) for v in row[:5]] for row in matrix]
        stage = _stage_label(phase, event)
        if stage != last_stage:
            _print_point_stage(point_id=point_id, nominal_total=nominal_total, stage=stage)
            last_stage = stage

    def _thumb_combo_passive_currents():
        currents = {int(mid): float(THUMB_COMBO_PASSIVE_HOLD_TOR_A)
                    for mid in finger_motor_set[finger] if mid not in target_by_motor}
        if not thumb_flexor_reached:
            for mid in thumb_combo_passive_extensors:
                currents[mid] = float(THUMB_COMBO_FLEXOR_EXTENSOR_RELEASE_TOR_A)
        return currents

    def _override_tor_map(elapsed_sec: float):
        releasing = elapsed_sec < max(0.0, float(THUMB_FLEXOR_EXTENSOR_RELEASE_SEC))
        out = {mid: current if releasing else 0.0
               for mid, current in thumb_passive_release_currents.items()}
        for mid, rel_sec in release_sec_plan.items():
            motor_id = int(mid)
            rel_sec_v = max(0.0, float(rel_sec))
            if rel_sec_v > float(elapsed_sec):
                out[motor_id] = float(release_tor_plan.get(motor_id, DATA_COLLECTION_RELEASE_TOR_DEFAULT_A))
        return out

    override_tor_map_now = (dict(release_tor_plan) if thumb_sway_first else
                           dict(preuniform_currents) if continuous_assist else
                           _override_tor_map(0.0) if release_active and not extensor_first else {})
    if thumb_sway_first:
        first_matrix = _build_command_matrix_for_point(
            hand=node.hand, defaults_hand=defaults_hand,
            finger_motor_set=finger_motor_set, finger=finger,
            target_by_motor={mid: target_by_motor[mid] for mid in sway_ids},
            release_tor_by_motor=release_tor_plan,
        )
        print(f"[sway-first] finger={finger} sway={sway_ids}; other finger motors "
              f"release at {THUMB_COMBO_RELEASE_TOR_A:+.3f}A for {THUMB_COMBO_RELEASE_SEC:.3f}s, "
              "then kp/kd/current=0 until sway arrival.")
        _apply_cmd_matrix(first_matrix, phase="move", event="sway_first")
    elif extensor_first:
        first_targets = {mid: target_by_motor[mid] for mid in extensor_ids}
        # Delay other active motors (e.g. thumb opposition). Flexors and sway
        # motors must stay in pure current release, including the active sway.
        for mid in arrival_motor_ids:
            if mid not in first_targets and mid not in release_tor_plan:
                first_targets[mid] = hold_pos_by_motor.get(mid, 0.0)
        first_matrix = _build_command_matrix_for_point(
            hand=node.hand, defaults_hand=defaults_hand,
            finger_motor_set=finger_motor_set, finger=finger,
            target_by_motor=first_targets, release_tor_by_motor=release_tor_plan,
        )
        release_current = (THUMB_PRE_RANDOM_PASSIVE_RELEASE_TOR_A if thumb_preuniform else
                           PRE_RANDOM_EXTENSOR_PASSIVE_RELEASE_TOR_A if preuniform_extensor else
                           EXTENSOR_FIRST_RELEASE_CURRENT_A)
        print(f"[extensor-first] finger={finger} extensors={extensor_ids} "
              f"release_motors={extensor_release_ids} "
              f"release_current=+{release_current:.3f}A kp/kd=0; "
              "move extensors immediately, then remaining target motors.")
        if preuniform_extensor:
            print(f"[preuniform-assist] release for {preuniform_release_sec:.3f}s, "
                  f"then passive_currents={preuniform_rest_currents}, "
                  "kp/kd=0 until extensor arrival or timeout.")
        _apply_cmd_matrix(first_matrix, phase="move", event="extensor_first")
    elif override_tor_map_now:
        if continuous_assist:
            print(f"[preuniform-assist] finger={finger} passive_currents={preuniform_currents}; "
                  f"kp/kd=0, release for {preuniform_release_sec:.3f}s, "
                  f"then passive_currents={preuniform_rest_currents}, kp/kd=0 until arrival or timeout.")
        matrix_release = _build_command_matrix_for_point(
            hand=node.hand,
            defaults_hand=defaults_hand,
            finger_motor_set=finger_motor_set,
            finger=finger,
            target_by_motor=target_by_motor,
            release_tor_by_motor=override_tor_map_now,
            hold_pos_by_motor=hold_pos_by_motor,
        )
        _apply_cmd_matrix(matrix_release, phase="move", event="init_override")
    else:
        _apply_cmd_matrix(matrix_normal, phase="move", event="init_target")

    t0 = time.monotonic()
    motion_stage = "sway_first" if thumb_sway_first else "extensor_first" if extensor_first else "simultaneous"
    stage_start = t0
    last_override_signature = tuple(
        sorted((int(mid), float(tor)) for mid, tor in override_tor_map_now.items())
    )
    process_glove = []
    process_glove_skeleton = []
    process_fb = []
    next_sample_t = t0
    sample_period = 1.0 / max(1.0, float(PROCESS_SAMPLE_HZ))
    arrival_enter_t = None
    reached = False
    timeout = False
    max_err = float("inf")
    interrupted = False
    interrupt_reason = ""

    while True:
        if callable(runtime_tick):
            rt = None
            try:
                rt = runtime_tick()
            except Exception:
                rt = None
            if rt:
                interrupted = True
                if isinstance(rt, dict):
                    interrupt_reason = str(rt.get("kind", "runtime_interrupt"))
                else:
                    interrupt_reason = str(rt)
                break
        now = time.monotonic()
        override_ids_now = []
        if thumb_sway_first:
            if motion_stage == "sway_first":
                releasing = (now - t0) < max(0.0, float(THUMB_COMBO_RELEASE_SEC))
                override_tor_map_now = {mid: current if releasing else 0.0
                                       for mid, current in release_tor_plan.items()}
            else:
                override_tor_map_now = _thumb_combo_passive_currents()
            override_ids_now = sorted(override_tor_map_now)
        elif continuous_assist:
            releasing = (now - t0) < max(0.0, preuniform_release_sec)
            override_tor_map_now = {mid: current if releasing else preuniform_rest_currents[mid]
                                    for mid, current in preuniform_currents.items()}
            override_ids_now = sorted(override_tor_map_now)
        elif release_active and not extensor_first:
            override_tor_map_now = _override_tor_map(now - t0)
            override_ids_now = sorted(override_tor_map_now.keys())

        # Thumb combos: sway-only release -> zero; remaining targets -> passive hold.
        # Preuniform points keep their own release policy; ordinary points use
        # timed release -> default current (thumb manual flexor targets -> zero).
        desired_override_map = override_tor_map_now if override_ids_now else {}
        desired_override_signature = tuple(
            sorted((int(mid), float(tor)) for mid, tor in desired_override_map.items())
        )
        if (not extensor_first or continuous_assist) and desired_override_signature != last_override_signature:
            if desired_override_signature:
                matrix_override = _build_command_matrix_for_point(
                    hand=node.hand,
                    defaults_hand=defaults_hand,
                    finger_motor_set=finger_motor_set,
                    finger=finger,
                    target_by_motor=({mid: target_by_motor[mid] for mid in sway_ids}
                                     if thumb_sway_first and motion_stage == "sway_first"
                                     else target_by_motor),
                    release_tor_by_motor=desired_override_map,
                    hold_pos_by_motor=hold_pos_by_motor,
                )
                if override_ids_now:
                    if continuous_assist:
                        print(f"[preuniform-assist] finger={finger} 放线结束，"
                              f"passive_currents={preuniform_rest_currents}, kp/kd=0 "
                              "至主动电机到位或超时。")
                    _apply_cmd_matrix(
                        matrix_override,
                        phase="move",
                        event=("sway_current0" if thumb_sway_first else
                               "preuniform_passive_current" if preuniform_extensor else
                               "preuniform_current0" if continuous_assist else
                               "override_release_changed"),
                    )
            else:
                _apply_cmd_matrix(matrix_normal, phase="move", event="override_cleared_to_target")
            last_override_signature = desired_override_signature
        if now >= next_sample_t:
            next_sample_t += sample_period
            snap = node.snapshot()
            if snap is not None and snap["feedback_age"] <= FEEDBACK_STALE_SEC:
                t_rel = float(now - t0)
                process_glove.append(_extract_glove48(snap["glove48"]))
                process_glove_skeleton.append(_extract_glove_skeleton176(snap["glove_skeleton176"]))
                process_fb.append([t_rel] + _flatten_feedback(snap))
                max_err = _max_active_error(
                    snap,
                    target_by_motor,
                    arrival_motor_ids=arrival_motor_ids,
                )
                if (thumb_sway_first and motion_stage == "remaining_targets"
                        and thumb_combo_flexor_ids and not thumb_flexor_reached):
                    flexor_error = _max_active_error(snap, target_by_motor, thumb_combo_flexor_ids)
                    if flexor_error <= point_tolerance:
                        if thumb_flexor_arrival_enter_t is None:
                            thumb_flexor_arrival_enter_t = now
                        elif (now - thumb_flexor_arrival_enter_t) >= ARRIVAL_CONFIRM_SEC:
                            thumb_flexor_reached = True
                            print("[thumb-flexor-arrived] flexor arrival confirmed; "
                                  f"passive extensors switch to {THUMB_COMBO_PASSIVE_HOLD_TOR_A:+.3f}A "
                                  "while other targets finish.")
                    else:
                        thumb_flexor_arrival_enter_t = None
                stage_error = (_max_active_error(snap, target_by_motor, sway_ids)
                               if motion_stage == "sway_first" else
                               _max_active_error(snap, target_by_motor, extensor_ids)
                               if motion_stage == "extensor_first" else max_err)
                if stage_error <= point_tolerance:
                    if arrival_enter_t is None:
                        arrival_enter_t = now
                    elif ((now - arrival_enter_t) >= ARRIVAL_CONFIRM_SEC
                          and (motion_stage != "sway_first"
                               or (now - t0) >= float(THUMB_COMBO_RELEASE_SEC))):
                        if motion_stage == "sway_first":
                            sway_stage_reached = True
                            sway_stage_elapsed = float(now - stage_start)
                            passive_currents = _thumb_combo_passive_currents()
                            second_matrix = _build_command_matrix_for_point(
                                hand=node.hand, defaults_hand=defaults_hand,
                                finger_motor_set=finger_motor_set, finger=finger,
                                target_by_motor=target_by_motor,
                                release_tor_by_motor=passive_currents,
                            )
                            motion_stage = "remaining_targets"
                            stage_start = now
                            arrival_enter_t = None
                            last_override_signature = tuple(sorted(passive_currents.items()))
                            print(f"[sway-arrived] finger={finger}; hold sway target, "
                                  f"move remaining targets; passive currents={passive_currents}, kp/kd=0.")
                            _apply_cmd_matrix(second_matrix, phase="move", event="remaining_targets")
                            continue
                        if motion_stage == "extensor_first":
                            extensor_stage_reached = True
                            extensor_stage_elapsed = float(now - stage_start)
                            remaining = set(arrival_motor_ids) - set(extensor_ids)
                            if remaining:
                                # Move the remaining targets while every other
                                # motor on this finger is in zero-current mode.
                                passive_currents = {int(mid): 0.0
                                                    for mid in finger_motor_set[finger]
                                                    if int(mid) not in target_by_motor}
                                second_matrix = _build_command_matrix_for_point(
                                    hand=node.hand, defaults_hand=defaults_hand,
                                    finger_motor_set=finger_motor_set, finger=finger,
                                    target_by_motor=target_by_motor,
                                    release_tor_by_motor=passive_currents,
                                )
                                print(f"[extensor-arrived] finger={finger} "
                                      f"elapsed={extensor_stage_elapsed:.3f}s; "
                                      "hold extensor targets, move remaining target motors; "
                                      "other motors on this finger: kp/kd/current=0.")
                                for mid in extensor_release_ids:
                                    release_sec_plan[mid] = extensor_stage_elapsed
                                motion_stage = "remaining_targets"
                                stage_start = now
                                arrival_enter_t = None
                                _apply_cmd_matrix(second_matrix, phase="move",
                                                  event="remaining_targets")
                                continue
                        reached = True
                        _print_point_stage(
                            point_id=point_id,
                            nominal_total=nominal_total,
                            stage="arrived",
                        )
                        break
                else:
                    arrival_enter_t = None
            else:
                # Stale/missing feedback must not bridge the confirmation period.
                arrival_enter_t = None
                thumb_flexor_arrival_enter_t = None
        stage_timeout = EXTENSOR_FIRST_TIMEOUT_SEC if motion_stage == "extensor_first" else MOVE_TIMEOUT_SEC
        if (time.monotonic() - stage_start) >= stage_timeout:
            timeout = True
            if extensor_first:
                print(f"[extensor-point-timeout] finger={finger} stage={motion_stage}; "
                      "stop movement and tighten all motors before sampling.")
            break
        time.sleep(0.005)

    move_elapsed = float(time.monotonic() - t0)
    if thumb_sway_first:
        if not sway_stage_reached:
            sway_stage_elapsed = move_elapsed
        release_sec_plan = dict.fromkeys(
            release_tor_plan, min(sway_stage_elapsed, max(0.0, float(THUMB_COMBO_RELEASE_SEC))))
        release_total_sec = max(release_sec_plan.values(), default=0.0)
    if continuous_assist:
        release_sec_plan = {mid: min(move_elapsed, max(0.0, preuniform_release_sec))
                            if current != 0.0 else 0.0
                            for mid, current in preuniform_currents.items()}
        release_total_sec = max(release_sec_plan.values(), default=0.0)
    if extensor_first:
        if not extensor_stage_reached:
            extensor_stage_elapsed = move_elapsed
        for mid in extensor_release_ids:
            release_sec_plan.setdefault(mid, move_elapsed)
        release_total_sec = max(release_sec_plan.values(), default=0.0)

    if thumb_passive_release_currents:
        release_sec_plan.update(dict.fromkeys(
            thumb_passive_release_currents,
            min(move_elapsed, max(0.0, float(THUMB_FLEXOR_EXTENSOR_RELEASE_SEC)))))
        release_total_sec = max(release_sec_plan.values(), default=0.0)

    motion_metadata = {
        "motion_policy": ("sway_first" if thumb_sway_first else "extensor_first" if extensor_first else
                          "preuniform_assist" if continuous_assist else "simultaneous"),
        "sway_stage_reached": sway_stage_reached if thumb_sway_first else None,
        "sway_stage_elapsed_sec": sway_stage_elapsed,
        "extensor_stage_reached": extensor_stage_reached if extensor_first else None,
        "extensor_stage_elapsed_sec": extensor_stage_elapsed,
        "timeout_stage": motion_stage if timeout else "",
    }

    # Keep the same pure-current command through tightening, settling and sampling.
    matrix_tor_tighten = _build_uniform_current_matrix(POST_ARRIVAL_TIGHTEN_TOR_A)
    tighten_duration = max(0.0, float(POST_ARRIVAL_TIGHTEN_SEC))
    if thumb_preuniform or thumb_sway_first:
        matrix_tor_tighten = _build_idle_tor_matrix(defaults_hand)
        tighten_current = float(THUMB_COMBO_TIGHTEN_TOR_A if thumb_sway_first
                                else THUMB_PRE_RANDOM_TIGHTEN_TOR_A)
        for mid in finger_motor_set[finger]:
            matrix_tor_tighten[mid - 1] = [0.0, 0.0, tighten_current, 0.0, 0.0]
        tighten_duration = max(0.0, float(THUMB_COMBO_TIGHTEN_SEC if thumb_sway_first
                                         else THUMB_PRE_RANDOM_TIGHTEN_SEC))
        if any("flex" in role and int(mid) in arrival_motor_ids
               for role, mid in muscles.items()):
            tighten_duration = max(0.0, float(THUMB_FLEXOR_TIGHTEN_SEC))

    post_arrival_tighten_elapsed = 0.0
    if (reached or timeout) and not interrupted:
        _apply_cmd_matrix(
            matrix_tor_tighten,
            phase="tighten", event="post_motion_tighten",
        )
        tighten_t0 = time.monotonic()
        next_tighten_sample_t = tighten_t0
        while (time.monotonic() - tighten_t0) < tighten_duration:
            # Keep pause/stop/emergency commands responsive during tightening.
            if callable(runtime_tick):
                rt = runtime_tick()
                if rt:
                    interrupted = True
                    interrupt_reason = (str(rt.get("kind", "runtime_interrupt"))
                                        if isinstance(rt, dict) else str(rt))
                    break
            now = time.monotonic()
            if process_include_post_stable and now >= next_tighten_sample_t:
                next_tighten_sample_t += sample_period
                snap = node.snapshot()
                if snap is not None:
                    process_glove.append(_extract_glove48(snap["glove48"]))
                    process_glove_skeleton.append(_extract_glove_skeleton176(snap["glove_skeleton176"]))
                    process_fb.append([float(now - t0)] + _flatten_feedback(snap))
            remaining = tighten_duration - (time.monotonic() - tighten_t0)
            if remaining > 0.0:
                time.sleep(min(float(CONTROL_PERIOD_SEC), remaining))
        post_arrival_tighten_elapsed = float(time.monotonic() - tighten_t0)
    motion_metadata.update(
        post_arrival_tighten_applied=bool(post_arrival_tighten_elapsed > 0.0),
        post_arrival_tighten_elapsed_sec=post_arrival_tighten_elapsed,
        position_stable_confirmed=False,
        position_stable_wait_sec=0.0,
        position_stable_max_span_deg=None,
        position_stable_motor_ids=[int(mid) for mid in finger_motor_set[finger]],
        position_stable_window_sec=float(POSITION_STABLE_WINDOW_SEC),
        position_stable_range_limit_deg=float(POSITION_STABLE_MAX_RANGE_DEG),
    )

    if (reached or timeout) and not interrupted:
        _apply_cmd_matrix(matrix_tor_tighten, phase="position_settle",
                          event="post_motion_position_settle")
        tension_scope = ("thumb motors" if thumb_preuniform or thumb_sway_first else
                         "thumb/wrist motors" if finger == "thumb" else f"all {MOTOR_COUNT} motors")
        tension_current = (THUMB_COMBO_TIGHTEN_TOR_A if thumb_sway_first else
                           THUMB_PRE_RANDOM_TIGHTEN_TOR_A if thumb_preuniform else
                           POST_ARRIVAL_TIGHTEN_TOR_A)
        print(f"[position-settle] finger={finger} motors={finger_motor_set[finger]}; "
              f"require each position range <= {POSITION_STABLE_MAX_RANGE_DEG:.3f} deg "
              f"over {POSITION_STABLE_WINDOW_SEC:.3f}s before sampling; "
              f"{tension_scope} keep {tension_current:.3f}A through sampling.")

        def record_settle_sample(now, snap):
            if process_include_post_stable:
                process_glove.append(_extract_glove48(snap["glove48"]))
                process_glove_skeleton.append(_extract_glove_skeleton176(snap["glove_skeleton176"]))
                process_fb.append([float(now - t0)] + _flatten_feedback(snap))

        stability = _wait_finger_position_stable(
            node, finger_motor_set[finger], runtime_tick=runtime_tick,
            on_sample=record_settle_sample,
        )
        motion_metadata.update({key: value for key, value in stability.items()
                                if key.startswith("position_stable_")})
        interrupted = stability["interrupted"]
        interrupt_reason = stability["interrupt_reason"]

    if interrupted:
        return {
            **motion_metadata,
            "reached": False,
            "timeout": False,
            "interrupted": True,
            "interrupt_reason": str(interrupt_reason),
            "move_elapsed_sec": move_elapsed,
            "max_error": float(max_err if math.isfinite(max_err) else 1e9),
            "process_glove_rows": process_glove,
            "process_glove_skeleton_rows": process_glove_skeleton,
            "process_fb_rows": process_fb,
            "result_glove_rows": [],
            "result_glove_skeleton_rows": [],
            "result_fb_rows": [],
            "target_cmd_matrix": matrix_normal,
            "assist_release_applied": bool(release_active),
            "assist_release_targets": sorted(int(mid) for mid in release_sec_plan.keys()),
            "assist_release_sec": float(release_total_sec),
            "assist_release_plan_sec": {
                int(mid): float(sec) for mid, sec in sorted(release_sec_plan.items())
            },
            "assist_release_tor_plan": {
                int(mid): float(tor) for mid, tor in sorted(release_tor_plan.items())
            },
            "result_window_label": result_window_label,
            "result_sample_count": 0,
            "result_glove_avg": [],
            "result_glove_skeleton_avg": [],
            "result_fb_avg": [],
            "active_mode": str(active_mode),
            "active_motors": [int(mid) for mid in arrival_motor_ids],
        }
    _apply_cmd_matrix(matrix_tor_tighten, phase="stable", event="position_settled_keep_tension")

    # Start a separate result clock only after the position stability gate passes.
    stable_duration = (result_window_end if thumb_preuniform or thumb_sway_first
                       else max(float(STABLE_HOLD_SEC), result_window_end))
    stable_t0 = time.monotonic()
    stable_glove_rows, stable_glove_skeleton_rows, stable_fb_rows = _collect_window_samples(
        node=node,
        duration_sec=stable_duration,
        sample_hz=PROCESS_SAMPLE_HZ,
        t_ref=stable_t0,
    )
    result_glove_rows, result_glove_skeleton_rows, result_fb_rows = _slice_rows_by_time_window(
        stable_glove_rows,
        stable_glove_skeleton_rows,
        stable_fb_rows,
        t_start=result_window_start,
        t_end=result_window_end,
    )

    result_glove_avg = _average_rows(result_glove_rows)
    result_glove_skeleton_avg = _average_rows(result_glove_skeleton_rows)
    result_fb_avg = _average_rows(result_fb_rows)

    if bool(process_include_post_stable):
        # Optional CSV-mode behavior:
        # include post-reach/timeout stable phase in process logs so process spans
        # cmd publish -> result window end for each point.
        process_glove.extend(stable_glove_rows)
        process_glove_skeleton.extend(stable_glove_skeleton_rows)
        for row in stable_fb_rows:
            if row and len(row) >= 2:
                try:
                    t_rel = (float(row[0]) + float(move_elapsed) + post_arrival_tighten_elapsed
                             + motion_metadata["position_stable_wait_sec"])
                except Exception:
                    t_rel = float("nan")
                process_fb.append([t_rel] + [float(v) for v in row[1:]])
            else:
                process_fb.append(list(row) if isinstance(row, list) else [])

    return {
        **motion_metadata,
        "reached": bool(reached),
        "timeout": bool(timeout),
        "interrupted": False,
        "interrupt_reason": "",
        "move_elapsed_sec": move_elapsed,
        "max_error": float(max_err if math.isfinite(max_err) else 1e9),
        "process_glove_rows": process_glove,
        "process_glove_skeleton_rows": process_glove_skeleton,
        "process_fb_rows": process_fb,
        "result_glove_rows": result_glove_rows,
        "result_glove_skeleton_rows": result_glove_skeleton_rows,
        "result_fb_rows": result_fb_rows,
        "target_cmd_matrix": matrix_normal,
        "assist_release_applied": bool(release_active),
        "assist_release_targets": sorted(int(mid) for mid in release_sec_plan.keys()),
        "assist_release_sec": float(release_total_sec),
        "assist_release_plan_sec": {
            int(mid): float(sec) for mid, sec in sorted(release_sec_plan.items())
        },
        "assist_release_tor_plan": {
            int(mid): float(tor) for mid, tor in sorted(release_tor_plan.items())
        },
        "result_window_label": result_window_label,
        "result_sample_count": len(result_fb_rows),
        "result_glove_avg": result_glove_avg,
        "result_glove_skeleton_avg": result_glove_skeleton_avg,
        "result_fb_avg": result_fb_avg,
        "active_mode": str(active_mode),
        "active_motors": [int(mid) for mid in arrival_motor_ids],
    }


def _point_motion_metadata(outcome):
    keys = ("motion_policy", "extensor_stage_reached", "extensor_stage_elapsed_sec", "timeout_stage",
            "sway_stage_reached", "sway_stage_elapsed_sec",
            "post_arrival_tighten_applied", "post_arrival_tighten_elapsed_sec",
            "position_stable_confirmed", "position_stable_wait_sec", "position_stable_max_span_deg",
            "position_stable_motor_ids", "position_stable_window_sec", "position_stable_range_limit_deg")
    return {key: outcome[key] for key in keys if key in outcome}


def _wait_zero_positions(node: DataCollectionNode, motor_ids: list[int]):
    t0 = time.monotonic()
    timeout = float(RETURN_ZERO_TIMEOUT_SEC) + float(ZERO_CALIBRATE_STABLE_SEC)
    stable_enter_t = None
    last_max_abs_pos = None
    indices = np.asarray(motor_ids, dtype=int) - 1
    while (time.monotonic() - t0) < timeout:
        positions = _zero_feedback_positions(node)
        if positions is None:
            stable_enter_t = None
        else:
            last_max_abs_pos = float(np.max(np.abs(positions[indices])))
            if last_max_abs_pos <= POINT_ARRIVAL_TOL:
                if stable_enter_t is None:
                    stable_enter_t = time.monotonic()
                elif (time.monotonic() - stable_enter_t) >= float(ZERO_CALIBRATE_STABLE_SEC):
                    return True, last_max_abs_pos
            else:
                stable_enter_t = None
        time.sleep(CONTROL_PERIOD_SEC)
    return False, last_max_abs_pos


def _run_zero_calibration(node: DataCollectionNode, defaults_hand, finger=None):
    if finger == "thumb":
        # During thumb collection, never release or position-control the four fingers.
        thumb_ids = sorted(set(HAND_MOTOR_MAP[node.hand]["thumb"].values()))
        matrix = _build_idle_tor_matrix(defaults_hand)
        zero_matrix = _build_zero_matrix(defaults_hand)
        for mid in thumb_ids:
            matrix[mid - 1] = zero_matrix[mid - 1][:]
        opposition_id = int(HAND_MOTOR_MAP[node.hand]["thumb"]["opposition"])
        matrix[opposition_id - 1][2] = float(ZERO_THUMB_OPPOSITION_TOR_A)
        attempts = 0
        last_max_abs_pos = None
        print(f"[calibrate] 仅拇指6台电机位控回0，对掌M{opposition_id}前馈 "
              f"{ZERO_THUMB_OPPOSITION_TOR_A:+.3f}A，四指保持默认纯电流张紧。")
        while attempts < int(ZERO_CALIBRATE_MAX_ATTEMPTS):
            attempts += 1
            node.set_command_matrix(matrix)
            reached, last_max_abs_pos = _wait_zero_positions(node, thumb_ids)
            if reached:
                return {"success": True, "attempts": attempts,
                        "max_attempts": int(ZERO_CALIBRATE_MAX_ATTEMPTS),
                        "last_max_abs_pos": last_max_abs_pos}
        positions = _zero_feedback_positions(node)
        hold_matrix = _build_idle_tor_matrix(defaults_hand)
        if positions is not None:
            measured_hold = _build_hold_current_matrix(defaults_hand, positions)
            for mid in thumb_ids:
                hold_matrix[mid - 1] = measured_hold[mid - 1][:]
        else:
            for mid in thumb_ids:
                hold_matrix[mid - 1] = [0.0] * 5
        node.set_command_matrix(hold_matrix)
        return {"success": False, "attempts": attempts,
                "max_attempts": int(ZERO_CALIBRATE_MAX_ATTEMPTS),
                "last_max_abs_pos": last_max_abs_pos, "failed_stage": "thumb"}

    _run_zero_prerelease(node, defaults_hand)
    extensor_motor_ids = [int(HAND_MOTOR_MAP[node.hand][finger]["extensor"])
                          for finger in ("index", "middle", "ring", "little")]
    positions = _zero_feedback_positions(node)
    attempts = 0
    last_max_abs_pos = None
    stage = "extensor_first"
    matrix = _build_zero_matrix(defaults_hand)
    # Keep the same extensor feedforward while the remaining motors return.
    for mid in extensor_motor_ids:
        matrix[mid - 1][2] = float(ZERO_EXTENSOR_TOR_A)
    extensor_matrix = _build_hold_current_matrix(defaults_hand, positions) if positions is not None else None
    if extensor_matrix is not None:
        for mid in _zero_flexor_motor_ids(node.hand) + _zero_sway_motor_ids(node.hand):
            extensor_matrix[mid - 1] = [0.0, 0.0, float(ZERO_FLEX_SWAY_TIGHTEN_TOR_A), 0.0, 0.0]
        for mid in extensor_motor_ids:
            extensor_matrix[mid - 1] = matrix[mid - 1][:]
        print(f"[calibrate] 四指伸肌先位控回0: motors={extensor_motor_ids}, "
              f"tor={float(ZERO_EXTENSOR_TOR_A):.3f}A；四指屈肌/侧摆纯电流 "
              f"{float(ZERO_FLEX_SWAY_TIGHTEN_TOR_A):.3f}A 张紧 kp/kd=0；拇指和腕部保持位置。")
        while attempts < int(ZERO_CALIBRATE_MAX_ATTEMPTS):
            attempts += 1
            if stage == "extensor_first":
                node.set_command_matrix(extensor_matrix)
                reached, last_max_abs_pos = _wait_zero_positions(node, extensor_motor_ids)
                if not reached:
                    print(f"[WARN] 四指伸肌回0未到位，不启动其他电机回0 "
                          f"(attempt={attempts}/{int(ZERO_CALIBRATE_MAX_ATTEMPTS)})")
                    continue
                stage = "all_motors"
                print("[calibrate] 四指伸肌已到位，其他电机开始位控回0。")
            node.set_command_matrix(matrix)
            reached, last_max_abs_pos = _wait_zero_positions(node, list(range(1, MOTOR_COUNT + 1)))
            if reached:
                return {"success": True, "attempts": attempts,
                        "max_attempts": int(ZERO_CALIBRATE_MAX_ATTEMPTS),
                        "last_max_abs_pos": last_max_abs_pos}
            print(f"[WARN] 全手回0超时，继续重试 "
                  f"(attempt={attempts}/{int(ZERO_CALIBRATE_MAX_ATTEMPTS)})")
    else:
        print("[WARN] 放线结束后无有效反馈，取消伸肌及其他电机回0。")

    # Stop the failed return at measured positions; do not keep releasing lines.
    positions = _zero_feedback_positions(node)
    node.set_command_matrix(_build_hold_current_matrix(defaults_hand, positions)
                            if positions is not None else _build_uniform_current_matrix(0.0))
    return {"success": False, "attempts": attempts,
            "max_attempts": int(ZERO_CALIBRATE_MAX_ATTEMPTS),
            "last_max_abs_pos": last_max_abs_pos, "failed_stage": stage}


def _normalize_target_fingers(target_finger_raw: str) -> list[str]:
    v = (target_finger_raw or "all").strip().lower()
    if v in ("", "all", "*"):
        return list(FINGER_ORDER_FIXED)
    fingers = [finger.strip() for finger in v.split(",")]
    if any(finger not in FINGER_ORDER_FIXED for finger in fingers):
        raise ValueError(
            f"Unsupported target_finger={target_finger_raw!r}, "
            f"expected all or a comma-separated list of: {'/'.join(FINGER_ORDER_FIXED)}"
        )
    return list(dict.fromkeys(fingers))


def _prepare_output_dirs(base_dir: Path, hand: str, fingers: list[str]):
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = base_dir / f"{run_stamp}_{hand}_collection"
    hand_root = run_root / f"{hand}_hand"
    dirs = {}
    for finger in fingers:
        finger_root = hand_root / finger
        d = {
            "finger_root": finger_root,
            "process_glove": finger_root / "process" / "glove_data",
            "process_glove_skeleton": finger_root / "process" / "Glove Skeleton Data",
            "process_cmd": finger_root / "process" / "motor_cmd",
            "process_fb": finger_root / "process" / "motor_feedback",
            "result_glove": finger_root / "result" / "glove_data",
            "result_glove_skeleton": finger_root / "result" / "Glove Skeleton Data",
            "result_cmd": finger_root / "result" / "motor_cmd",
            "result_fb": finger_root / "result" / "motor_feedback",
        }
        for p in d.values():
            p.mkdir(parents=True, exist_ok=True)
        dirs[finger] = d
    return run_root, hand_root, dirs


def _save_point_files(
    finger_dirs: dict,
    point_id: int,
    point_meta: dict,
    process_glove_rows,
    process_glove_skeleton_rows,
    process_fb_rows,
    target_cmd_matrix,
    result_glove_avg,
    result_glove_skeleton_avg,
    result_fb_avg,
    result_window_label: str,
    result_sample_count: int,
):
    fname = f"{point_id:05d}.xlsx"
    process_glove_path = finger_dirs["process_glove"] / fname
    process_glove_skeleton_path = finger_dirs["process_glove_skeleton"] / fname
    process_cmd_path = finger_dirs["process_cmd"] / fname
    process_fb_path = finger_dirs["process_fb"] / fname
    result_glove_path = finger_dirs["result_glove"] / fname
    result_glove_skeleton_path = finger_dirs["result_glove_skeleton"] / fname
    result_cmd_path = finger_dirs["result_cmd"] / fname
    result_fb_path = finger_dirs["result_fb"] / fname

    # process/glove_data: one row = one sample = 48D glove, no headers/comments.
    process_glove_sheet = []
    for row in process_glove_rows:
        vals = [float(v) for v in row]
        vals += [float("nan")] * max(0, 48 - len(vals))
        process_glove_sheet.append(vals[:48])

    # process/Glove Skeleton Data:
    # each sample -> 25 rows (one row per node), then one blank row.
    # row format: [side_code, node_id, x, y, z, qx, qy, qz, qw]
    process_glove_skeleton_sheet = []
    for row in process_glove_skeleton_rows:
        node_rows = _skeleton176_to_node_rows(row)
        for nr in node_rows:
            process_glove_skeleton_sheet.append(nr)
        process_glove_skeleton_sheet.append([])

    # process/motor_cmd & result/motor_cmd: fixed 28x5 matrix for this point.
    cmd_sheet = []
    for i in range(MOTOR_COUNT):
        if i < len(target_cmd_matrix):
            row = [float(v) for v in target_cmd_matrix[i]]
        else:
            row = [float("nan")] * 5
        row += [float("nan")] * max(0, 5 - len(row))
        cmd_sheet.append(row[:5])

    # process/motor_feedback: each sample is 28x5, blank line between groups.
    process_fb_sheet = []
    for flat_row in process_fb_rows:
        _, mats = _split_motor_flat_row(flat_row)
        for mvals in mats:
            process_fb_sheet.append([float(v) for v in mvals[:5]])
        process_fb_sheet.append([])

    # result/glove_data: single 48D row (average over result window).
    result_glove_sheet = []
    avg_glove = [float(v) for v in result_glove_avg] if result_glove_avg else []
    avg_glove += [float("nan")] * max(0, 48 - len(avg_glove))
    result_glove_sheet.append(avg_glove[:48])

    # result/Glove Skeleton Data:
    # single averaged 176D row.
    result_glove_skeleton_sheet = []
    avg_glove_skeleton = (
        [float(v) for v in result_glove_skeleton_avg] if result_glove_skeleton_avg else []
    )
    avg_glove_skeleton += [float("nan")] * max(0, 176 - len(avg_glove_skeleton))
    result_glove_skeleton_sheet.append(avg_glove_skeleton[:176])

    # result/motor_feedback:
    # row1: two columns [window, sample_count], then 28x5 average matrix.
    result_fb_sheet = [[str(result_window_label), int(result_sample_count)]]
    _, avg_mats = _split_motor_flat_row(result_fb_avg if result_fb_avg else [])
    for mvals in avg_mats:
        result_fb_sheet.append([float(v) for v in mvals[:5]])

    _write_xlsx_rows(process_glove_path, process_glove_sheet)
    _write_xlsx_rows(process_glove_skeleton_path, process_glove_skeleton_sheet)
    _write_xlsx_rows(process_cmd_path, cmd_sheet)
    _write_xlsx_rows(process_fb_path, process_fb_sheet)
    _write_xlsx_rows(result_glove_path, result_glove_sheet)
    _write_xlsx_rows(result_glove_skeleton_path, result_glove_skeleton_sheet)
    _write_xlsx_rows(result_cmd_path, cmd_sheet)
    _write_xlsx_rows(result_fb_path, result_fb_sheet)


def _point_log_headers():
    return [
        "point_id",
        "global_point_id",
        "finger",
        "combo_id",
        "reached",
        "timeout",
        "move_elapsed_sec",
        "max_error",
        "active_mode",
        "active_motors",
        "zero_calib_state",
        "zero_calib_fail_streak",
    ]


def _zero_calibration_event_headers():
    return [
        "global_point_id",
        "finger",
        "success",
        "attempts",
        "max_attempts",
        "last_max_abs_pos",
        "calib_state_after_event",
        "calib_fail_streak_after_event",
    ]


def _choose_combo(
    combo_list,
    combo_counts,
    combo_min,
    combo_max,
    combo_cooldown,
    combo_coverage,
    rng: random.Random,
):
    candidates = []
    weights = []
    for combo in combo_list:
        if combo_cooldown.get(combo.combo_id, 0) > 0:
            continue
        c = combo_counts[combo.combo_id]
        if c >= combo_max[combo.combo_id]:
            continue
        floor = combo_min[combo.combo_id]
        gap = max(0, floor - c)
        remain = max(0, combo_max[combo.combo_id] - c)
        w_quota = 1.0
        w_quota += float(COMBO_WEIGHT_QUOTA_GAP_GAIN) * (float(gap) / max(1.0, float(floor)))
        w_quota += float(COMBO_WEIGHT_REMAIN_GAIN) * (
            float(remain) / max(1.0, float(combo_max[combo.combo_id]))
        )
        coverage = combo_coverage.get(combo.combo_id)
        cov_deficit = _coverage_deficit(coverage) if coverage is not None else 0.0
        w = w_quota * (1.0 + float(COMBO_WEIGHT_COVERAGE_GAIN) * cov_deficit)
        candidates.append(combo)
        weights.append(w)
    if not candidates:
        return None
    return rng.choices(candidates, weights=weights, k=1)[0]


def _quota_done(counts, min_map, max_map, nominal_total):
    total = sum(int(v) for v in counts.values())
    all_over_floor = all(int(counts[k]) >= int(min_map[k]) for k in counts)
    all_over_cap = all(int(counts[k]) >= int(max_map[k]) for k in counts)
    if all_over_cap:
        return True
    if all_over_floor and total >= nominal_total:
        return True
    return False


def _finger_done(
    combo_counts,
    combo_min,
    combo_max,
    nominal_total,
    mode_counts=None,
    mode_min=None,
    mode_max=None,
):
    combo_done = _quota_done(combo_counts, combo_min, combo_max, nominal_total)
    if mode_counts is None or mode_min is None or mode_max is None:
        return combo_done
    mode_done = _quota_done(mode_counts, mode_min, mode_max, nominal_total)
    return bool(combo_done and mode_done)


def _combo_base_quota(finger: str):
    return QUOTA_BASE_PER_COMBO if finger == "thumb" else FOUR_FINGER_QUOTA_BASE_PER_COMBO


def _build_finger_quota_plan(finger: str, combos: list[ComboSpec], points_per_combo: int = 0):
    if points_per_combo > 0:
        target = int(points_per_combo)
        fixed = {c.combo_id: target for c in combos}
        return len(combos) * target, fixed.copy(), fixed.copy(), {key: float(target) for key in fixed}
    base_quota = _combo_base_quota(finger)
    nominal_total = int(len(combos) * base_quota)
    ratio = max(0.0, float(QUOTA_FLOAT_RATIO))
    base_target = {c.combo_id: float(base_quota) for c in combos}

    if finger == "thumb":
        on_combos = [c for c in combos if str(c.opposition_mode).lower() == "on"]
        off_combos = [c for c in combos if str(c.opposition_mode).lower() != "on"]
        if on_combos and off_combos:
            on_ratio = min(1.0, max(0.0, float(THUMB_OPPOSITION_ON_RATIO)))
            total_target = float(nominal_total)
            on_total = total_target * on_ratio
            off_total = total_target - on_total
            per_on = on_total / float(len(on_combos))
            per_off = off_total / float(len(off_combos))
            base_target = {}
            for c in on_combos:
                base_target[c.combo_id] = float(per_on)
            for c in off_combos:
                base_target[c.combo_id] = float(per_off)

    combo_min = {}
    combo_max = {}
    for c in combos:
        tgt = max(1.0, float(base_target.get(c.combo_id, base_quota)))
        lo = max(1, int(round(tgt * (1.0 - ratio))))
        hi = max(lo, int(round(tgt * (1.0 + ratio))))
        combo_min[c.combo_id] = lo
        combo_max[c.combo_id] = hi

    return nominal_total, combo_min, combo_max, base_target


def _normalize_mode_ratio(raw_ratio: dict, mode_keys: list[str]):
    out = {}
    for k in mode_keys:
        try:
            v = float(raw_ratio.get(k, 0.0))
        except Exception:
            v = 0.0
        if not math.isfinite(v) or v < 0.0:
            v = 0.0
        out[k] = v
    s = sum(out.values())
    if s <= 1e-9:
        uni = 1.0 / max(1, len(mode_keys))
        return {k: float(uni) for k in mode_keys}
    return {k: float(out[k] / s) for k in mode_keys}


def _build_active_mode_quota_plan(finger: str, nominal_total: int):
    if finger == "thumb":
        mode_keys = ["full", "single", "double"]
        ratio_map = _normalize_mode_ratio(dict(THUMB_ACTIVE_MODE_RATIO), mode_keys)
    else:
        mode_keys = ["xy", "x_only", "y_only"]
        ratio_map = _normalize_mode_ratio(dict(FOUR_FINGER_ACTIVE_MODE_RATIO), mode_keys)

    disabled_modes = set()
    if bool(DISABLE_RANDOM_SINGLE_MOTOR_MODES):
        if finger == "thumb":
            # Current collection policy: thumb random stage keeps full-only.
            disabled_modes.update({"single", "double"})
        else:
            disabled_modes.update({"x_only", "y_only"})
        remain_modes = [m for m in mode_keys if m not in disabled_modes]
        remain_sum = sum(float(ratio_map.get(m, 0.0)) for m in remain_modes)
        if remain_sum <= 1e-9:
            uni = 1.0 / max(1, len(remain_modes))
            for m in remain_modes:
                ratio_map[m] = float(uni)
        else:
            for m in remain_modes:
                ratio_map[m] = float(ratio_map.get(m, 0.0)) / float(remain_sum)
        for m in disabled_modes:
            ratio_map[m] = 0.0

    ratio = max(0.0, float(QUOTA_FLOAT_RATIO))
    mode_target = {}
    mode_min = {}
    mode_max = {}
    for mode in mode_keys:
        if mode in disabled_modes:
            mode_target[mode] = 0.0
            mode_min[mode] = 0
            mode_max[mode] = 0
            continue
        tgt = max(1.0, float(nominal_total) * float(ratio_map.get(mode, 0.0)))
        lo = max(1, int(round(tgt * (1.0 - ratio))))
        hi = max(lo, int(round(tgt * (1.0 + ratio))))
        mode_target[mode] = float(tgt)
        mode_min[mode] = int(lo)
        mode_max[mode] = int(hi)
    return mode_keys, mode_target, mode_min, mode_max


def _choose_active_mode(mode_keys, mode_counts, mode_min, mode_max, rng: random.Random):
    candidates = []
    weights = []
    for mode in mode_keys:
        c = int(mode_counts.get(mode, 0))
        cap = int(mode_max.get(mode, 0))
        if c >= cap:
            continue
        floor = int(mode_min.get(mode, 0))
        gap = max(0, floor - c)
        remain = max(0, cap - c)
        w = 1.0
        w += float(COMBO_WEIGHT_QUOTA_GAP_GAIN) * (float(gap) / max(1.0, float(floor)))
        w += float(COMBO_WEIGHT_REMAIN_GAIN) * (float(remain) / max(1.0, float(cap)))
        candidates.append(mode)
        weights.append(w)
    if not candidates:
        return None
    return str(rng.choices(candidates, weights=weights, k=1)[0])


def _parse_bool_text(text: str, default: bool = True) -> bool:
    if text is None:
        return default
    value = str(text).strip().lower()
    if value in ("1", "true", "yes", "y", "on", "enable", "enabled"):
        return True
    if value in ("0", "false", "no", "n", "off", "disable", "disabled"):
        return False
    return default


def _write_finger_summary_from_state(finger_state: dict, zero_calib_state: str, zero_calib_fail_streak: int):
    finger = str(finger_state["finger"])
    point_id = int(finger_state["point_id"])
    unreached = list(finger_state["unreached"])
    summary = {
        "finger": finger,
        "updated_at": datetime.now().isoformat(),
        "point_count": point_id,
        "unreached_count": len(unreached),
        "unreached_ratio": (len(unreached) / point_id) if point_id > 0 else 0.0,
        "unreached_point_ids": unreached,
        "combo_counts": finger_state["combo_counts"],
        "combo_cooldown": finger_state["combo_cooldown"],
        "combo_target_none_consecutive": finger_state["combo_target_none_consec"],
        "combo_stall_zero_retry_count": finger_state["combo_stall_zero_retry_count"],
        "combo_stall_over_ratio_hit_count": finger_state["combo_stall_over_ratio_hit_count"],
        "combo_stall_finish_reason": finger_state["combo_stall_finish_reason"],
        "finger_reopen_round": int(finger_state.get("finger_reopen_round", 0)),
        "finger_reopen_no_progress_round": int(finger_state.get("finger_reopen_no_progress_round", 0)),
        "edge_enhancement_done": bool(finger_state.get("edge_enhancement_done", False)),
        "combo_coverage": {
            cid: _coverage_stats(finger_state["combo_coverage"][cid])
            for cid in finger_state["combo_counts"]
        },
        "combo_min": finger_state["combo_min"],
        "combo_max": finger_state["combo_max"],
        "active_mode_counts": finger_state["mode_counts"],
        "active_mode_target": finger_state["mode_target"],
        "active_mode_min": finger_state["mode_min"],
        "active_mode_max": finger_state["mode_max"],
        "nominal_total": int(finger_state["nominal_total"]),
        "zero_calibration_state": str(zero_calib_state),
        "zero_calibration_fail_streak": int(zero_calib_fail_streak),
        "zero_calibration_events": int(finger_state["finger_zero_calib_events"]),
        "zero_calibration_failed_events": int(finger_state["finger_zero_calib_failed_events"]),
    }
    (finger_state["finger_dirs"]["finger_root"] / "finger_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _choose_manual_combo(finger_state: dict, rng: random.Random):
    pending = []
    for combo in finger_state["combos"]:
        cid = str(combo.combo_id)
        cur = int(finger_state["combo_counts"].get(cid, 0))
        floor = int(finger_state["combo_min"].get(cid, 0))
        deficit = max(0, floor - cur)
        if deficit > 0:
            pending.append((combo, deficit))

    if pending:
        max_deficit = max(d for _, d in pending)
        candidates = [combo for combo, d in pending if d == max_deficit]
        return rng.choice(candidates), f"min_deficit={max_deficit}"

    best_combo = None
    best_cov = -1.0
    for combo in finger_state["combos"]:
        cid = str(combo.combo_id)
        cov = _coverage_deficit(finger_state["combo_coverage"][cid])
        if cov > best_cov:
            best_cov = cov
            best_combo = combo
    if best_combo is None:
        return None, "no_combo"
    return best_combo, f"coverage_deficit={best_cov:.3f}"


def _find_combo_by_id(finger_state: dict, combo_id: str):
    for combo in finger_state["combos"]:
        if str(combo.combo_id) == str(combo_id):
            return combo
    return None


def _parse_manual_motor_targets(tokens: list[str]):
    target_by_motor = {}
    for raw in tokens:
        tok = str(raw).strip()
        if not tok:
            continue
        tok = tok.replace("=", ":")
        if ":" not in tok:
            raise ValueError(f"invalid token '{raw}', expected mID:pos")
        left, right = tok.split(":", 1)
        left = left.strip().lower()
        right = right.strip()
        if left.startswith("m"):
            left = left[1:]
        mid = int(left)
        pos = float(right)
        if mid < 1 or mid > MOTOR_COUNT:
            raise ValueError(f"motor id out of range: {mid}, expected 1..{MOTOR_COUNT}")
        target_by_motor[int(mid)] = float(pos)
    if not target_by_motor:
        raise ValueError("empty targets, expected at least one motor target, e.g. m17:-120")
    return target_by_motor


def _manual_active_motor_limit(finger: str) -> int:
    return 3 if str(finger).strip().lower() == "thumb" else 2


def _pick_manual_combo_by_targets(finger_state: dict, target_by_motor: dict[int, float], prefer_combo_id: str | None):
    if prefer_combo_id:
        combo = _find_combo_by_id(finger_state, prefer_combo_id)
        if combo is not None:
            return combo, "manual_combo"

    combos = list(finger_state["combos"])
    active_set = set(int(mid) for mid in target_by_motor.keys())
    if not combos:
        return None, "no_combo"

    exact = []
    subset = []
    scored = []
    for combo in combos:
        combo_set = set(int(mid) for mid in combo.active_motors)
        if active_set == combo_set:
            exact.append(combo)
        if active_set.issubset(combo_set):
            subset.append(combo)
        score = len(active_set & combo_set)
        scored.append((score, combo))

    if exact:
        return exact[0], "exact_active_set"
    if subset:
        return subset[0], "subset_active_set"
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], f"max_overlap={scored[0][0]}"


def _infer_manual_active_mode(finger: str, combo: ComboSpec, target_by_motor: dict[int, float]):
    active_ids = sorted(int(mid) for mid in target_by_motor.keys())
    n = len(active_ids)
    if finger == "thumb":
        if n >= 3:
            return "full"
        if n == 2:
            return "double"
        return "single"

    if n >= 2:
        return "xy"
    if n == 1:
        mid = active_ids[0]
        if int(mid) == int(combo.y_motor):
            return "y_only"
        if int(mid) == int(combo.x_motor):
            return "x_only"
    return "xy"


def _infer_manual_coverage_key(combo: ComboSpec, target_by_motor: dict[int, float]):
    try:
        x_edges = np.asarray(combo.boundary.x, dtype=np.float64)
        if x_edges.size < 2:
            return None
        x_mid = int(combo.x_motor)
        if x_mid not in target_by_motor:
            return None
        xq = float(target_by_motor[x_mid])
        ix = int(np.clip(np.searchsorted(x_edges, xq, side="right") - 1, 0, len(x_edges) - 2))
        if combo.dims == 3 and combo.opposition_mode == "on" and combo.boundary.z is not None:
            z_mid = int(combo.z_motor) if combo.z_motor is not None else None
            if z_mid is None or z_mid not in target_by_motor:
                return None
            z_edges = np.asarray(combo.boundary.z, dtype=np.float64)
            if z_edges.size < 2:
                return None
            zq = float(target_by_motor[z_mid])
            iz = int(np.clip(np.searchsorted(z_edges, zq, side="right") - 1, 0, len(z_edges) - 2))
            return (ix, iz)
        return (ix,)
    except Exception:
        return None


def _print_finger_motor_status(
    node: DataCollectionNode,
    finger: str,
    finger_motor_set: dict[str, list[int]],
    target_by_motor: dict[int, float] | None = None,
    title: str = "finger-motors",
):
    snap = node.snapshot()
    motor_ids = list(finger_motor_set.get(str(finger), []))
    if snap is None:
        print(f"[{title}] finger={finger} snapshot unavailable.")
        return
    pos = np.asarray(snap["pos"], dtype=np.float64)
    parts = []
    for mid in motor_ids:
        cur = float(pos[mid - 1]) if 1 <= int(mid) <= len(pos) else float("nan")
        if target_by_motor is not None and int(mid) in target_by_motor:
            tgt = float(target_by_motor[int(mid)])
            err = cur - tgt
            parts.append(f"m{mid}:cur={cur:.3f},tgt={tgt:.3f},err={err:+.3f}")
        else:
            parts.append(f"m{mid}:cur={cur:.3f}")
    joined = " | ".join(parts) if parts else "(no motors)"
    print(f"[{title}] finger={finger} {joined}")


def _print_manual_supplement_status(finger_states: dict):
    print("\n[manual-supplement-status]")
    for finger, state in finger_states.items():
        combo_counts = state["combo_counts"]
        combo_min = state["combo_min"]
        pending_num = sum(
            1
            for cid in combo_counts
            if int(combo_counts[cid]) < int(combo_min.get(cid, 0))
        )
        total_num = len(combo_counts)
        point_count = int(state["point_id"])
        unreached_num = len(state["unreached"])
        print(
            f"  finger={finger} point_count={point_count} "
            f"unreached={unreached_num} pending_combo={pending_num}/{total_num}"
        )
        top_missing = sorted(
            [
                (
                    str(cid),
                    int(combo_min.get(cid, 0)) - int(combo_counts.get(cid, 0)),
                    int(combo_counts.get(cid, 0)),
                    int(combo_min.get(cid, 0)),
                )
                for cid in combo_counts
                if int(combo_counts.get(cid, 0)) < int(combo_min.get(cid, 0))
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        for cid, deficit, cur, floor in top_missing:
            print(f"    - combo={cid} deficit={deficit} count={cur}/{floor}")


def _motor_bounds_from_combos(combos: list[ComboSpec], motor_id: int):
    lo_vals = []
    hi_vals = []
    mid = int(motor_id)
    for combo in combos:
        b = combo.boundary
        if int(combo.x_motor) == mid:
            lo_vals.append(float(np.min(np.asarray(b.x, dtype=np.float64))))
            hi_vals.append(float(np.max(np.asarray(b.x, dtype=np.float64))))
        if int(combo.y_motor) == mid:
            lo_vals.append(float(np.min(np.asarray(b.y_min, dtype=np.float64))))
            hi_vals.append(float(np.max(np.asarray(b.y_max, dtype=np.float64))))
        if combo.z_motor is not None and int(combo.z_motor) == mid and b.z is not None:
            lo_vals.append(float(np.min(np.asarray(b.z, dtype=np.float64))))
            hi_vals.append(float(np.max(np.asarray(b.z, dtype=np.float64))))
    if not lo_vals or not hi_vals:
        return None
    return float(min(lo_vals)), float(max(hi_vals))


def _uniform_targets_from_zero_to_tight_boundary(tight_boundary: float, step: float):
    end = float(tight_boundary)
    step_v = max(1e-6, float(step))
    if not math.isfinite(end):
        return []
    if end >= -float(BOUNDARY_EPS):
        return []
    out = []
    k = 1
    while True:
        v = -step_v * float(k)
        if v <= (end + float(BOUNDARY_EPS)):
            break
        out.append(float(v))
        k += 1
        if k > 100000:
            break
    if (not out) or abs(float(out[-1]) - end) > max(1e-6, step_v * 0.25):
        out.append(float(end))
    else:
        out[-1] = float(end)
    return out


def _uniform_targets_from_current_to_zero(cur_value: float, step: float):
    cur = float(cur_value)
    step_v = max(1e-6, float(step))
    if not math.isfinite(cur):
        return []
    if cur <= float(TARGET_DIRECTION_EPS):
        return []
    out = []
    v = cur - step_v
    while v > float(TARGET_DIRECTION_EPS):
        out.append(float(v))
        v -= step_v
        if len(out) > 100000:
            break
    out.append(0.0)
    return out


def _pick_combo_for_targets_preferring_deficit(finger_state: dict, target_by_motor: dict[int, float]):
    active_set = {int(mid) for mid in target_by_motor.keys()}
    best_combo = None
    best_score = None
    for combo in finger_state["combos"]:
        combo_set = {int(mid) for mid in combo.active_motors}
        if not active_set.issubset(combo_set):
            continue
        cid = str(combo.combo_id)
        cur = int(finger_state["combo_counts"].get(cid, 0))
        floor = int(finger_state["combo_min"].get(cid, 0))
        cap = int(finger_state["combo_max"].get(cid, 0))
        deficit = max(0, floor - cur)
        remain = max(0, cap - cur)
        score = (int(deficit), int(remain), -int(cur))
        if best_score is None or score > best_score:
            best_score = score
            best_combo = combo
    if best_combo is not None:
        return best_combo
    combo, _ = _pick_manual_combo_by_targets(
        finger_state=finger_state,
        target_by_motor=target_by_motor,
        prefer_combo_id=None,
    )
    return combo


def _run_pre_random_uniform_sampling(
    node: DataCollectionNode,
    defaults_hand,
    finger_motor_set,
    finger_state: dict,
    run_counters: dict,
    calib_state: dict,
    hand_root: Path,
    runtime_tick=None,
):
    if not bool(PRE_RANDOM_UNIFORM_ENABLE):
        return True

    finger = str(finger_state["finger"]).strip().lower()
    combos = list(finger_state["combos"])
    if not combos:
        return True

    hand_map = HAND_MOTOR_MAP.get(node.hand, {})
    muscle_map = hand_map.get(finger, {})
    if not isinstance(muscle_map, dict) or not muscle_map:
        return True

    def _can_continue():
        if runtime_tick is None:
            return True
        try:
            return bool(runtime_tick())
        except Exception:
            return True

    ext_ids = []
    flex_ids = []
    sway_ids = []
    other_ids = []
    for name, mid_raw in muscle_map.items():
        try:
            mid = int(mid_raw)
        except Exception:
            continue
        if not (1 <= mid <= MOTOR_COUNT):
            continue
        key = str(name).strip().lower()
        if "ext" in key:
            ext_ids.append(mid)
        elif "flex" in key:
            flex_ids.append(mid)
        elif ("abduction" in key) or ("adduction" in key):
            sway_ids.append(mid)
        else:
            other_ids.append(mid)

    ext_ids = sorted(set(ext_ids))
    flex_ids = sorted(set(flex_ids))
    sway_ids = sorted(set(sway_ids))
    other_ids = sorted(set(other_ids))

    print(
        f"[preuniform-start] finger={finger} "
        f"flex={flex_ids} sway={sway_ids} ext={ext_ids} other={other_ids}"
    )

    # Start the planned per-motor sweeps without an extra whole-hand zero calibration.
    def _collect_target(target_by_motor: dict[int, float]):
        if not _can_continue():
            return False
        combo = _pick_combo_for_targets_preferring_deficit(finger_state, target_by_motor)
        if combo is None:
            print(
                f"[preuniform-warn] finger={finger} no combo for targets="
                f"{sorted((int(k), float(v)) for k, v in target_by_motor.items())}"
            )
            return True
        active_mode = _infer_manual_active_mode(finger, combo, target_by_motor)
        coverage_key = _infer_manual_coverage_key(combo, target_by_motor)
        _collect_one_manual_target_point(
            node=node,
            defaults_hand=defaults_hand,
            finger_motor_set=finger_motor_set,
            finger_state=finger_state,
            combo=combo,
            target_by_motor=target_by_motor,
            active_mode=active_mode,
            coverage_key=coverage_key,
            run_counters=run_counters,
            calib_state=calib_state,
            log_prefix="preuniform",
        )
        return True

    # Non-extensor motors: from zero to tighten-side boundary.
    non_ext_order = []
    for mid in flex_ids + sway_ids + other_ids:
        if mid not in non_ext_order:
            non_ext_order.append(mid)
    for mid in non_ext_order:
        if not _can_continue():
            print(f"[preuniform-interrupt] finger={finger} interrupted before motor sweep.")
            return False
        bounds = _motor_bounds_from_combos(combos, int(mid))
        if bounds is None:
            continue
        lo, _ = bounds
        tighten_boundary = min(0.0, float(lo))
        if int(mid) in sway_ids:
            step = float(PRE_RANDOM_SWAY_STEP)
        elif int(mid) in flex_ids:
            step = float(PRE_RANDOM_FLEX_STEP)
        else:
            step = float(PRE_RANDOM_OTHER_STEP)
        targets = _uniform_targets_from_zero_to_tight_boundary(tighten_boundary, step)
        if not targets:
            continue
        print(
            f"[preuniform-motor] finger={finger} motor=m{int(mid)} "
            f"targets={len(targets)} step={step:.3f} tight_boundary={tighten_boundary:.3f}"
        )
        for tgt in targets:
            if not _collect_target({int(mid): float(tgt)}):
                print(f"[preuniform-interrupt] finger={finger} interrupted during motor sweep.")
                return False

    # Extensor motors: for each flex partner, push flexor to max-tight then sample extensor toward zero.
    for ext_mid in ext_ids:
        if not _can_continue():
            print(f"[preuniform-interrupt] finger={finger} interrupted before extensor sweep.")
            return False
        partners = list(flex_ids) if flex_ids else [None]
        for flex_mid in partners:
            if not _can_continue():
                print(f"[preuniform-interrupt] finger={finger} interrupted in extensor sweep.")
                return False
            if flex_mid is not None:
                bounds_flex = _motor_bounds_from_combos(combos, int(flex_mid))
                if bounds_flex is not None:
                    flex_tight = min(0.0, float(bounds_flex[0]))
                    flex_targets = _uniform_targets_from_zero_to_tight_boundary(
                        flex_tight,
                        float(PRE_RANDOM_FLEX_STEP),
                    )
                    if flex_targets:
                        if not _collect_target({int(flex_mid): float(flex_targets[-1])}):
                            print(f"[preuniform-interrupt] finger={finger} interrupted in extensor partner setup.")
                            return False

            snap = node.snapshot()
            if snap is None:
                continue
            ext_cur = float(snap["pos"][int(ext_mid) - 1])
            ext_targets = _uniform_targets_from_current_to_zero(ext_cur, float(PRE_RANDOM_EXTENSOR_STEP))
            if not ext_targets:
                continue
            print(
                f"[preuniform-ext] finger={finger} ext=m{int(ext_mid)} "
                f"partner={('none' if flex_mid is None else f'm{int(flex_mid)}')} "
                f"cur={ext_cur:.3f} targets={len(ext_targets)}"
            )
            for tgt in ext_targets:
                if not _collect_target({int(ext_mid): float(tgt)}):
                    print(f"[preuniform-interrupt] finger={finger} interrupted in extensor targets.")
                    return False

    print(f"[preuniform-done] finger={finger} total_points_now={int(finger_state['point_id'])}")
    return True


def _run_post_random_edge_enhancement(
    node: DataCollectionNode,
    defaults_hand,
    finger_motor_set,
    finger_state: dict,
    run_counters: dict,
    calib_state: dict,
    rng: random.Random,
):
    finger = str(finger_state["finger"]).strip().lower()
    combos = list(finger_state.get("combos", []))
    if not combos:
        return

    tasks = []
    if finger == "thumb":
        out_combos = [c for c in combos if str(c.direction).strip().lower() == "out"]
        flex_combos = [c for c in combos if _is_thumb_flex_combo(c)]
        tasks.append(("thumb_abduction", out_combos, int(THUMB_ABDUCTION_EXTRA_POINTS)))
        tasks.append(("thumb_flexion", flex_combos, int(THUMB_FLEXION_EXTRA_POINTS)))
    else:
        ff_flex_combos = [c for c in combos if _is_four_finger_flex_combo(c)]
        tasks.append(("four_finger_flexion", ff_flex_combos, int(FOUR_FINGER_FLEXION_EXTRA_POINTS_PER_FINGER)))

    for task_name, task_combos, extra_points in tasks:
        target_points = max(0, int(extra_points))
        if target_points <= 0:
            continue
        if not task_combos:
            print(
                f"[edge-enhance-skip] finger={finger} task={task_name} "
                f"reason=no_eligible_combo target={target_points}"
            )
            continue

        collected = 0
        attempts = 0
        max_attempts = max(target_points * 20, target_points + 50)
        print(
            f"[edge-enhance-start] finger={finger} task={task_name} "
            f"target_points={target_points} combos={len(task_combos)}"
        )

        while collected < target_points and attempts < max_attempts:
            attempts += 1
            combo = rng.choice(task_combos)

            snap = node.snapshot()
            if snap is None:
                time.sleep(0.01)
                continue

            mode_keys = list(finger_state.get("mode_keys", []))
            mode_counts = dict(finger_state.get("mode_counts", {}))
            mode_min = dict(finger_state.get("mode_min", {}))
            mode_max = dict(finger_state.get("mode_max", {}))
            selected_mode = _choose_active_mode(
                mode_keys=mode_keys,
                mode_counts=mode_counts,
                mode_min=mode_min,
                mode_max=mode_max,
                rng=rng,
            )
            if selected_mode is None:
                selected_mode = "full" if finger == "thumb" else "xy"

            target_plan = _generate_target(
                hand=node.hand,
                combo=combo,
                current_pos=snap["pos"],
                coverage=finger_state["combo_coverage"][combo.combo_id],
                rng=rng,
                active_mode=selected_mode,
                edge_bias=True,
            )
            if target_plan is None:
                fallback_mode = "full" if finger == "thumb" else "xy"
                if selected_mode != fallback_mode:
                    target_plan = _generate_target(
                        hand=node.hand,
                        combo=combo,
                        current_pos=snap["pos"],
                        coverage=finger_state["combo_coverage"][combo.combo_id],
                        rng=rng,
                        active_mode=fallback_mode,
                        edge_bias=True,
                    )
            if target_plan is None:
                continue

            ok = _collect_one_manual_target_point(
                node=node,
                defaults_hand=defaults_hand,
                finger_motor_set=finger_motor_set,
                finger_state=finger_state,
                combo=combo,
                target_by_motor=target_plan.target_by_motor,
                active_mode=str(target_plan.active_mode),
                coverage_key=target_plan.coverage_key,
                run_counters=run_counters,
                calib_state=calib_state,
                log_prefix="edge",
            )
            if ok:
                collected += 1

        if collected < target_points:
            print(
                f"[edge-enhance-warn] finger={finger} task={task_name} "
                f"collected={collected}/{target_points} attempts={attempts}/{max_attempts}"
            )
        else:
            print(
                f"[edge-enhance-done] finger={finger} task={task_name} "
                f"collected={collected}/{target_points} attempts={attempts}"
            )


def _collect_one_manual_point(
    node: DataCollectionNode,
    defaults_hand,
    finger_motor_set,
    finger_state: dict,
    combo: ComboSpec,
    rng: random.Random,
    run_counters: dict,
    calib_state: dict,
    forced_mode: str | None = None,
    log_prefix: str = "manual",
):
    snap = node.snapshot()
    if snap is None:
        print("[manual-supplement] snapshot unavailable, skip.")
        return False

    mode_keys = finger_state["mode_keys"]
    selected_mode = forced_mode
    if selected_mode is None:
        selected_mode = _choose_active_mode(
            mode_keys=mode_keys,
            mode_counts=finger_state["mode_counts"],
            mode_min=finger_state["mode_min"],
            mode_max=finger_state["mode_max"],
            rng=rng,
        )
    if selected_mode is None:
        selected_mode = "full" if finger_state["finger"] == "thumb" else "xy"

    target_plan = _generate_target(
        hand=node.hand,
        combo=combo,
        current_pos=snap["pos"],
        coverage=finger_state["combo_coverage"][combo.combo_id],
        rng=rng,
        active_mode=selected_mode,
    )
    if target_plan is None:
        fallback_mode = "full" if finger_state["finger"] == "thumb" else "xy"
        if selected_mode != fallback_mode:
            target_plan = _generate_target(
                hand=node.hand,
                combo=combo,
                current_pos=snap["pos"],
                coverage=finger_state["combo_coverage"][combo.combo_id],
                rng=rng,
                active_mode=fallback_mode,
            )
    if target_plan is None:
        print(
            "[manual-supplement] target generation failed: "
            f"finger={finger_state['finger']} combo={combo.combo_id}"
        )
        return False

    used_mode = str(target_plan.active_mode).strip().lower() if target_plan.active_mode else selected_mode
    if used_mode not in finger_state["mode_counts"]:
        finger_state["mode_counts"][used_mode] = 0
        finger_state["mode_target"][used_mode] = 0.0
        finger_state["mode_min"][used_mode] = 0
        finger_state["mode_max"][used_mode] = max(1, int(finger_state["nominal_total"]))

    finger_state["point_id"] += 1
    run_counters["global_point"] += 1
    finger_state["mode_counts"][used_mode] += 1

    point_id = int(finger_state["point_id"])
    nominal_total = int(finger_state["nominal_total"])
    print(
        f"[{str(log_prefix)}-point-start] {point_id}/{nominal_total} "
        f"finger={finger_state['finger']} combo={combo.combo_id} mode={used_mode}"
    )

    outcome = _execute_point(
        node=node,
        defaults_hand=defaults_hand,
        finger_motor_set=finger_motor_set,
        finger=str(finger_state["finger"]),
        combo=combo,
        target_by_motor=target_plan.target_by_motor,
        active_motors=target_plan.active_motors,
        active_mode=used_mode,
        point_tolerance=POINT_ARRIVAL_TOL,
        point_id=point_id,
        nominal_total=nominal_total,
    )

    finger_state["combo_counts"][combo.combo_id] += 1
    _mark_coverage_hit(finger_state["combo_coverage"][combo.combo_id], target_plan.coverage_key)
    if not outcome["reached"]:
        run_counters["global_timeout"] += 1
        finger_state["unreached"].append(point_id)

    point_meta = {
        "point_id": point_id,
        "global_point_id": int(run_counters["global_point"]),
        "finger": str(finger_state["finger"]),
        "combo_id": str(combo.combo_id),
        "reached": outcome["reached"],
        "timeout": outcome["timeout"],
        "move_elapsed_sec": outcome["move_elapsed_sec"],
        "max_error": outcome["max_error"],
        **_point_motion_metadata(outcome),
        "active_mode": used_mode,
        "active_motors": [int(mid) for mid in target_plan.active_motors],
        "zero_calib_state": str(calib_state["zero_calib_state"]),
        "zero_calib_fail_streak": int(calib_state["zero_calib_fail_streak"]),
    }

    _save_point_files(
        finger_dirs=finger_state["finger_dirs"],
        point_id=point_id,
        point_meta=point_meta,
        process_glove_rows=outcome["process_glove_rows"],
        process_glove_skeleton_rows=outcome["process_glove_skeleton_rows"],
        process_fb_rows=outcome["process_fb_rows"],
        target_cmd_matrix=outcome["target_cmd_matrix"],
        result_glove_avg=outcome["result_glove_avg"],
        result_glove_skeleton_avg=outcome["result_glove_skeleton_avg"],
        result_fb_avg=outcome["result_fb_avg"],
        result_window_label=outcome["result_window_label"],
        result_sample_count=outcome["result_sample_count"],
    )

    _append_csv_row(
        finger_state["finger_dirs"]["finger_root"] / "point_log.csv",
        _point_log_headers(),
        [
            point_meta["point_id"],
            point_meta["global_point_id"],
            point_meta["finger"],
            point_meta["combo_id"],
            int(point_meta["reached"]),
            int(point_meta["timeout"]),
            point_meta["move_elapsed_sec"],
            point_meta["max_error"],
            point_meta["active_mode"],
            "|".join(str(v) for v in point_meta["active_motors"]),
            point_meta["zero_calib_state"],
            point_meta["zero_calib_fail_streak"],
        ],
    )

    _write_finger_summary_from_state(
        finger_state=finger_state,
        zero_calib_state=str(calib_state["zero_calib_state"]),
        zero_calib_fail_streak=int(calib_state["zero_calib_fail_streak"]),
    )
    print(
        f"[{str(log_prefix)}-point-done] "
        f"{point_id}/{nominal_total} finger={finger_state['finger']} combo={combo.combo_id} "
        f"reached={outcome['reached']} max_err={outcome['max_error']:.3f}"
    )
    return True


def _collect_one_manual_target_point(
    node: DataCollectionNode,
    defaults_hand,
    finger_motor_set,
    finger_state: dict,
    combo: ComboSpec,
    target_by_motor: dict[int, float],
    active_mode: str,
    coverage_key: tuple[int, ...] | None,
    run_counters: dict,
    calib_state: dict,
    log_prefix: str = "manual",
):
    finger_name = str(finger_state["finger"]).strip().lower()
    max_active = _manual_active_motor_limit(finger_name)
    active_count = len(target_by_motor)
    if active_count > max_active:
        print(
            "[manual-supplement] active motor limit exceeded: "
            f"finger={finger_name}, active={active_count}, limit={max_active}. "
            "该点已跳过。"
        )
        return False

    used_mode = str(active_mode).strip().lower() if active_mode else _infer_manual_active_mode(
        str(finger_state["finger"]),
        combo,
        target_by_motor,
    )
    if used_mode not in finger_state["mode_counts"]:
        finger_state["mode_counts"][used_mode] = 0
        finger_state["mode_target"][used_mode] = 0.0
        finger_state["mode_min"][used_mode] = 0
        finger_state["mode_max"][used_mode] = max(1, int(finger_state["nominal_total"]))

    active_motors = sorted(int(mid) for mid in target_by_motor.keys())
    finger_state["point_id"] += 1
    run_counters["global_point"] += 1
    finger_state["mode_counts"][used_mode] += 1

    point_id = int(finger_state["point_id"])
    nominal_total = int(finger_state["nominal_total"])
    print(
        f"[{str(log_prefix)}-point-start] {point_id}/{nominal_total} "
        f"finger={finger_state['finger']} combo={combo.combo_id} mode={used_mode} "
        f"targets={{{', '.join([f'm{k}:{v:.3f}' for k, v in sorted(target_by_motor.items())])}}}"
    )

    outcome = _execute_point(
        node=node,
        defaults_hand=defaults_hand,
        finger_motor_set=finger_motor_set,
        finger=str(finger_state["finger"]),
        combo=combo,
        target_by_motor=target_by_motor,
        active_motors=active_motors,
        active_mode=used_mode,
        point_tolerance=POINT_ARRIVAL_TOL,
        point_id=point_id,
        nominal_total=nominal_total,
        preuniform=(log_prefix == "preuniform"),
    )

    finger_state["combo_counts"][combo.combo_id] += 1
    _mark_coverage_hit(finger_state["combo_coverage"][combo.combo_id], coverage_key)
    if not outcome["reached"]:
        run_counters["global_timeout"] += 1
        finger_state["unreached"].append(point_id)

    point_meta = {
        "point_id": point_id,
        "global_point_id": int(run_counters["global_point"]),
        "finger": str(finger_state["finger"]),
        "combo_id": str(combo.combo_id),
        "reached": outcome["reached"],
        "timeout": outcome["timeout"],
        "move_elapsed_sec": outcome["move_elapsed_sec"],
        "max_error": outcome["max_error"],
        **_point_motion_metadata(outcome),
        "active_mode": used_mode,
        "active_motors": active_motors,
        "zero_calib_state": str(calib_state["zero_calib_state"]),
        "zero_calib_fail_streak": int(calib_state["zero_calib_fail_streak"]),
    }

    _save_point_files(
        finger_dirs=finger_state["finger_dirs"],
        point_id=point_id,
        point_meta=point_meta,
        process_glove_rows=outcome["process_glove_rows"],
        process_glove_skeleton_rows=outcome["process_glove_skeleton_rows"],
        process_fb_rows=outcome["process_fb_rows"],
        target_cmd_matrix=outcome["target_cmd_matrix"],
        result_glove_avg=outcome["result_glove_avg"],
        result_glove_skeleton_avg=outcome["result_glove_skeleton_avg"],
        result_fb_avg=outcome["result_fb_avg"],
        result_window_label=outcome["result_window_label"],
        result_sample_count=outcome["result_sample_count"],
    )

    _append_csv_row(
        finger_state["finger_dirs"]["finger_root"] / "point_log.csv",
        _point_log_headers(),
        [
            point_meta["point_id"],
            point_meta["global_point_id"],
            point_meta["finger"],
            point_meta["combo_id"],
            int(point_meta["reached"]),
            int(point_meta["timeout"]),
            point_meta["move_elapsed_sec"],
            point_meta["max_error"],
            point_meta["active_mode"],
            "|".join(str(v) for v in point_meta["active_motors"]),
            point_meta["zero_calib_state"],
            point_meta["zero_calib_fail_streak"],
        ],
    )

    _write_finger_summary_from_state(
        finger_state=finger_state,
        zero_calib_state=str(calib_state["zero_calib_state"]),
        zero_calib_fail_streak=int(calib_state["zero_calib_fail_streak"]),
    )
    print(
        f"[{str(log_prefix)}-point-done] "
        f"{point_id}/{nominal_total} finger={finger_state['finger']} combo={combo.combo_id} "
        f"reached={outcome['reached']} max_err={outcome['max_error']:.3f}"
    )
    _print_finger_motor_status(
        node=node,
        finger=str(finger_state["finger"]),
        finger_motor_set=finger_motor_set,
        target_by_motor=target_by_motor,
        title=f"{str(log_prefix)}-point-finger-status",
    )
    return True


def _run_manual_supplement_loop(
    enable: bool,
    node: DataCollectionNode,
    defaults_hand,
    finger_motor_set,
    finger_states: dict,
    rng: random.Random,
    run_counters: dict,
    calib_state: dict,
    hand_root: Path,
    confirm: bool = True,
):
    if not enable:
        return
    if not sys.stdin.isatty():
        print("[manual-supplement] stdin is not interactive, skip manual supplement.")
        return
    if not finger_states:
        print("[manual-supplement] no finger state available, skip.")
        return

    if confirm:
        answer = input("\n[data_collection] 自动采集已结束，进入手动补采模式？(y/N): ").strip().lower()
        if answer not in ("y", "yes", "1", "true", "on"):
            print("[manual-supplement] skip.")
            return

    current_finger = next(iter(finger_states.keys()))
    current_combo_id = None
    forced_mode = None
    print(
        "[manual-supplement] 已进入手动补采。命令: "
        "直接输入 mID:pos [mID:pos ...] 即补1点；"
        "n <k> mID:pos [mID:pos ...] 连补k点；"
        "s=状态, f <finger>, c <combo_id>, m <mode>, z=回0校准, q=退出, h=帮助"
    )
    _print_manual_supplement_status(finger_states)
    _print_finger_motor_status(
        node=node,
        finger=current_finger,
        finger_motor_set=finger_motor_set,
        target_by_motor=None,
        title="manual-select-finger-status",
    )

    while True:
        raw = input(
            f"[manual-supplement] finger={current_finger} combo={current_combo_id or 'auto'} "
            f"mode={forced_mode or 'auto'} > "
        ).strip()
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()

        if cmd in ("q", "quit", "exit"):
            print("[manual-supplement] exit.")
            break
        if cmd in ("h", "help", "?"):
            print(
                "命令说明: "
                "m17:-120 m22:5 = 按你给的电机目标补1点（默认当前手指）; "
                "n 3 m17:-120 m22:5 = 连补3点; "
                "s=显示状态; "
                "f <finger>=切换手指; c <combo_id>=指定combo(可空为auto); "
                "m <mode>=指定active_mode(thumb:full/single/double; 其他:xy/x_only/y_only, 可空为auto); "
                "z=执行回0校准; q=退出"
            )
            continue
        if cmd in ("s", "status"):
            _print_manual_supplement_status(finger_states)
            continue
        if cmd in ("f", "finger"):
            if len(parts) < 2:
                print(f"[manual-supplement] 当前手指: {current_finger}")
                continue
            nxt = parts[1].strip().lower()
            if nxt not in finger_states:
                print(f"[manual-supplement] unknown finger={nxt}, available={list(finger_states.keys())}")
                continue
            current_finger = nxt
            current_combo_id = None
            forced_mode = None
            _print_finger_motor_status(
                node=node,
                finger=current_finger,
                finger_motor_set=finger_motor_set,
                target_by_motor=None,
                title="manual-select-finger-status",
            )
            continue
        if cmd in ("c", "combo"):
            if len(parts) < 2:
                current_combo_id = None
                print("[manual-supplement] combo selection cleared -> auto.")
                continue
            cid = parts[1].strip()
            combo = _find_combo_by_id(finger_states[current_finger], cid)
            if combo is None:
                available = [str(c.combo_id) for c in finger_states[current_finger]["combos"]]
                print(
                    f"[manual-supplement] combo not found: {cid}. "
                    f"finger={current_finger}, available={available}"
                )
                continue
            current_combo_id = str(cid)
            continue
        if cmd in ("m", "mode"):
            if len(parts) < 2:
                forced_mode = None
                print("[manual-supplement] mode selection cleared -> auto.")
                continue
            mode = str(parts[1]).strip().lower()
            valid_modes = set(finger_states[current_finger]["mode_keys"])
            if mode not in valid_modes:
                print(
                    f"[manual-supplement] invalid mode={mode}. "
                    f"finger={current_finger}, valid={sorted(valid_modes)}"
                )
                continue
            forced_mode = mode
            continue
        if cmd in ("z", "zero"):
            calib_result = _run_zero_calibration(node, defaults_hand, finger=current_finger)
            calib_state["zero_calib_total_events"] += 1
            finger_states[current_finger]["finger_zero_calib_events"] += 1
            if calib_result["success"]:
                calib_state["zero_calib_state"] = "ok"
                calib_state["zero_calib_fail_streak"] = 0
                print(
                    "[manual-supplement] zero calibration done. "
                    f"attempts={calib_result['attempts']}/{calib_result['max_attempts']}"
                )
            else:
                calib_state["zero_calib_failed_events"] += 1
                finger_states[current_finger]["finger_zero_calib_failed_events"] += 1
                calib_state["zero_calib_fail_streak"] += 1
                calib_state["zero_calib_state"] = f"failed_after_{calib_result['max_attempts']}_attempts"
                print(
                    "[manual-supplement] zero calibration failed. "
                    f"attempts={calib_result['attempts']}/{calib_result['max_attempts']}, "
                    f"last_max_abs_pos={calib_result['last_max_abs_pos']}"
                )
            _append_csv_row(
                hand_root / "zero_calibration_events.csv",
                _zero_calibration_event_headers(),
                [
                    int(run_counters["global_point"]),
                    current_finger,
                    int(calib_result["success"]),
                    calib_result["attempts"],
                    calib_result["max_attempts"],
                    calib_result["last_max_abs_pos"],
                    calib_state["zero_calib_state"],
                    calib_state["zero_calib_fail_streak"],
                ],
            )
            _write_finger_summary_from_state(
                finger_state=finger_states[current_finger],
                zero_calib_state=str(calib_state["zero_calib_state"]),
                zero_calib_fail_streak=int(calib_state["zero_calib_fail_streak"]),
            )
            continue

        repeat = 1
        target_tokens = []
        if cmd in ("n", "run"):
            if len(parts) < 3:
                print("[manual-supplement] usage: n <k> m17:-120 m22:5")
                continue
            try:
                repeat = max(1, int(parts[1]))
            except Exception:
                print("[manual-supplement] invalid k, usage: n <k> m17:-120 m22:5")
                continue
            target_tokens = parts[2:]
        elif cmd in ("u", "use", "t", "target"):
            if len(parts) < 2:
                print("[manual-supplement] usage: m17:-120 m22:5")
                continue
            target_tokens = parts[1:]
        else:
            # Default manual supplement mode:
            # Any non-control input is treated as one manual point target on current finger.
            target_tokens = parts

        try:
            target_by_motor = _parse_manual_motor_targets(target_tokens)
        except Exception as exc:
            print(f"[manual-supplement] unknown command or invalid targets: {exc}. 输入 h 查看帮助。")
            continue

        state = finger_states[current_finger]
        finger_motor_ids = set(int(mid) for mid in finger_motor_set.get(current_finger, []))
        invalid = [mid for mid in target_by_motor.keys() if int(mid) not in finger_motor_ids]
        if invalid:
            print(
                f"[manual-supplement] target motors not in finger={current_finger}: {sorted(invalid)}; "
                f"allowed={sorted(finger_motor_ids)}"
            )
            continue
        max_active = _manual_active_motor_limit(current_finger)
        active_count = len(target_by_motor)
        if active_count > max_active:
            print(
                "[manual-supplement] active motor limit exceeded: "
                f"finger={current_finger}, active={active_count}, limit={max_active}. "
                "请减少目标电机数量后重试。"
            )
            continue

        combo, reason = _pick_manual_combo_by_targets(
            finger_state=state,
            target_by_motor=target_by_motor,
            prefer_combo_id=current_combo_id,
        )
        if combo is None:
            print(f"[manual-supplement] no combo available for finger={current_finger}")
            continue
        coverage_key = _infer_manual_coverage_key(combo, target_by_motor)
        active_mode = (
            forced_mode
            if forced_mode is not None
            else _infer_manual_active_mode(current_finger, combo, target_by_motor)
        )
        if active_mode not in set(state["mode_keys"]):
            active_mode = _infer_manual_active_mode(current_finger, combo, target_by_motor)

        for i in range(repeat):
            if repeat > 1:
                print(f"[manual-supplement] repeat {i + 1}/{repeat}")
            print(
                f"[manual-supplement] collect finger={current_finger} "
                f"combo={combo.combo_id} reason={reason} mode={active_mode}"
            )
            ok = _collect_one_manual_target_point(
                node=node,
                defaults_hand=defaults_hand,
                finger_motor_set=finger_motor_set,
                finger_state=state,
                combo=combo,
                target_by_motor=target_by_motor,
                active_mode=active_mode,
                coverage_key=coverage_key,
                run_counters=run_counters,
                calib_state=calib_state,
            )
            if not ok:
                break
        continue


def _write_json_atomic(path: Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _prepare_output_dirs_from_run_root(run_root: Path, hand: str, fingers: list[str]):
    hand_root = run_root / f"{hand}_hand"
    dirs = {}
    for finger in fingers:
        finger_root = hand_root / finger
        d = {
            "finger_root": finger_root,
            "process_glove": finger_root / "process" / "glove_data",
            "process_glove_skeleton": finger_root / "process" / "Glove Skeleton Data",
            "process_cmd": finger_root / "process" / "motor_cmd",
            "process_fb": finger_root / "process" / "motor_feedback",
            "result_glove": finger_root / "result" / "glove_data",
            "result_glove_skeleton": finger_root / "result" / "Glove Skeleton Data",
            "result_cmd": finger_root / "result" / "motor_cmd",
            "result_fb": finger_root / "result" / "motor_feedback",
        }
        for p in d.values():
            p.mkdir(parents=True, exist_ok=True)
        dirs[finger] = d
    return run_root, hand_root, dirs


def _finger_state_to_resume_payload(finger_state: dict):
    cov_counts = {}
    for cid, coverage in finger_state.get("combo_coverage", {}).items():
        try:
            cov_counts[str(cid)] = np.asarray(coverage.counts, dtype=np.int64).tolist()
        except Exception:
            continue
    return {
        "finger": str(finger_state.get("finger", "")),
        "quota_base_per_combo": int(_combo_base_quota(finger_state.get("finger", ""))),
        "point_id": int(finger_state.get("point_id", 0)),
        "unreached": [int(v) for v in list(finger_state.get("unreached", []))],
        "combo_counts": {str(k): int(v) for k, v in dict(finger_state.get("combo_counts", {})).items()},
        "combo_min": {str(k): int(v) for k, v in dict(finger_state.get("combo_min", {})).items()},
        "combo_max": {str(k): int(v) for k, v in dict(finger_state.get("combo_max", {})).items()},
        "combo_target_none_consec": {
            str(k): int(v) for k, v in dict(finger_state.get("combo_target_none_consec", {})).items()
        },
        "combo_cooldown": {str(k): int(v) for k, v in dict(finger_state.get("combo_cooldown", {})).items()},
        "combo_stall_zero_retry_count": {
            str(k): int(v) for k, v in dict(finger_state.get("combo_stall_zero_retry_count", {})).items()
        },
        "combo_stall_over_ratio_hit_count": {
            str(k): int(v) for k, v in dict(finger_state.get("combo_stall_over_ratio_hit_count", {})).items()
        },
        "combo_stall_finish_reason": {
            str(k): str(v) for k, v in dict(finger_state.get("combo_stall_finish_reason", {})).items()
        },
        "mode_counts": {str(k): int(v) for k, v in dict(finger_state.get("mode_counts", {})).items()},
        "mode_min": {str(k): int(v) for k, v in dict(finger_state.get("mode_min", {})).items()},
        "mode_max": {str(k): int(v) for k, v in dict(finger_state.get("mode_max", {})).items()},
        "finger_reopen_round": int(finger_state.get("finger_reopen_round", 0)),
        "finger_reopen_no_progress_round": int(finger_state.get("finger_reopen_no_progress_round", 0)),
        "finger_reopen_last_point_id": int(finger_state.get("finger_reopen_last_point_id", 0)),
        "finger_zero_calib_events": int(finger_state.get("finger_zero_calib_events", 0)),
        "finger_zero_calib_failed_events": int(finger_state.get("finger_zero_calib_failed_events", 0)),
        "preuniform_done": bool(finger_state.get("preuniform_done", False)),
        "edge_enhancement_done": bool(finger_state.get("edge_enhancement_done", False)),
        "done": bool(finger_state.get("done", False)),
        "combo_coverage_counts": cov_counts,
    }


def _apply_finger_resume_payload(finger_state: dict, payload: dict):
    if not isinstance(payload, dict):
        return

    # Older checkpoints used 2000 for every finger and did not store this field.
    previous_base = int(payload.get("quota_base_per_combo", 2000))
    base_changed = previous_base != int(_combo_base_quota(finger_state.get("finger", "")))
    saved_counts = payload.get("combo_counts")
    combo_set_changed = (isinstance(saved_counts, dict)
                         and set(saved_counts) != set(finger_state.get("combo_counts", {})))
    quota_changed = base_changed or combo_set_changed
    if base_changed:
        print(f"[resume] finger={finger_state.get('finger', '')} quota/combo "
              f"{previous_base} -> {_combo_base_quota(finger_state.get('finger', ''))}; keeping collected counts.")
    if combo_set_changed:
        print(f"[resume] finger={finger_state.get('finger', '')} combo policy changed; "
              "keeping historical point IDs and retained-combo counts, rebuilding quota limits.")

    def _merge_int_map(dst_name: str, src_name: str):
        dst = finger_state.get(dst_name, {})
        src = payload.get(src_name, {})
        if not isinstance(dst, dict) or not isinstance(src, dict):
            return
        for k in list(dst.keys()):
            if k in src:
                try:
                    dst[k] = int(src[k])
                except Exception:
                    continue

    def _merge_str_map(dst_name: str, src_name: str):
        dst = finger_state.get(dst_name, {})
        src = payload.get(src_name, {})
        if not isinstance(dst, dict) or not isinstance(src, dict):
            return
        for k in list(dst.keys()):
            if k in src:
                dst[k] = str(src[k])

    _merge_int_map("combo_counts", "combo_counts")
    _merge_int_map("mode_counts", "mode_counts")
    if not quota_changed:
        for name in ("combo_min", "combo_max", "combo_target_none_consec", "combo_cooldown",
                     "combo_stall_zero_retry_count", "combo_stall_over_ratio_hit_count",
                     "mode_min", "mode_max"):
            _merge_int_map(name, name)
        _merge_str_map("combo_stall_finish_reason", "combo_stall_finish_reason")

    try:
        finger_state["point_id"] = int(payload.get("point_id", finger_state.get("point_id", 0)))
    except Exception:
        pass
    try:
        finger_state["unreached"] = [int(v) for v in list(payload.get("unreached", finger_state.get("unreached", [])))]
    except Exception:
        pass
    finger_state["finger_reopen_round"] = int(payload.get("finger_reopen_round", finger_state.get("finger_reopen_round", 0)))
    finger_state["finger_reopen_no_progress_round"] = int(
        payload.get("finger_reopen_no_progress_round", finger_state.get("finger_reopen_no_progress_round", 0))
    )
    finger_state["finger_reopen_last_point_id"] = int(
        payload.get("finger_reopen_last_point_id", finger_state.get("finger_reopen_last_point_id", 0))
    )
    finger_state["finger_zero_calib_events"] = int(
        payload.get("finger_zero_calib_events", finger_state.get("finger_zero_calib_events", 0))
    )
    finger_state["finger_zero_calib_failed_events"] = int(
        payload.get("finger_zero_calib_failed_events", finger_state.get("finger_zero_calib_failed_events", 0))
    )
    finger_state["preuniform_done"] = bool(payload.get("preuniform_done", finger_state.get("preuniform_done", False)))
    finger_state["edge_enhancement_done"] = bool(
        payload.get("edge_enhancement_done", finger_state.get("edge_enhancement_done", False))
    )
    finger_state["done"] = bool(payload.get("done", finger_state.get("done", False)))
    if quota_changed:
        # Reassess completion and stall recovery against the freshly built quotas.
        finger_state["done"] = False
        finger_state["finger_reopen_round"] = 0
        finger_state["finger_reopen_no_progress_round"] = 0
        finger_state["finger_reopen_last_point_id"] = finger_state.get("point_id", 0)

    cov_src = payload.get("combo_coverage_counts", {})
    if isinstance(cov_src, dict):
        for cid, coverage in finger_state.get("combo_coverage", {}).items():
            arr = cov_src.get(str(cid))
            if arr is None:
                continue
            try:
                arr_np = np.asarray(arr, dtype=np.int64)
            except Exception:
                continue
            if arr_np.shape == coverage.counts.shape:
                coverage.counts[:] = arr_np


def _build_runtime_health_payload(
    hand: str,
    phase: str,
    reason: str,
    current_finger: str,
    current_combo: str,
    current_stage: str,
    global_point: int,
    global_timeout: int,
    zero_calib_state: str,
    zero_calib_fail_streak: int,
    finger_states: dict,
):
    fingers = {}
    for finger, state in finger_states.items():
        point_id = int(state.get("point_id", 0))
        nominal_total = int(state.get("nominal_total", 0))
        combo_none = dict(state.get("combo_target_none_consec", {}))
        combo_none_max = max([int(v) for v in combo_none.values()], default=0)
        fingers[str(finger)] = {
            "point_id": point_id,
            "nominal_total": nominal_total,
            "progress_ratio": (point_id / max(1.0, float(nominal_total))),
            "unreached_count": len(list(state.get("unreached", []))),
            "combo_pending_count": sum(
                1
                for cid, cur in dict(state.get("combo_counts", {})).items()
                if int(cur) < int(dict(state.get("combo_min", {})).get(cid, 0))
            ),
            "combo_target_none_max": combo_none_max,
            "reopen_round": int(state.get("finger_reopen_round", 0)),
            "reopen_no_progress_round": int(state.get("finger_reopen_no_progress_round", 0)),
            "done": bool(state.get("done", False)),
        }
    return {
        "updated_at": datetime.now().isoformat(),
        "hand": hand,
        "phase": str(phase),
        "reason": str(reason),
        "current": {
            "finger": str(current_finger),
            "combo": str(current_combo),
            "stage": str(current_stage),
        },
        "global": {
            "point": int(global_point),
            "timeout": int(global_timeout),
            "timeout_ratio": float(global_timeout) / max(1.0, float(global_point)),
        },
        "zero_calibration": {
            "state": str(zero_calib_state),
            "fail_streak": int(zero_calib_fail_streak),
        },
        "fingers": fingers,
    }


def _load_resume_state_if_available(resume_file: Path):
    resume_file = Path(resume_file)
    if not resume_file.exists():
        return None
    try:
        obj = json.loads(resume_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _parse_args():
    parser = argparse.ArgumentParser(description="Y-hand random workspace data collection")
    parser.add_argument(
        "--run-mode",
        default="default",
        choices=["default", "controller", "worker"],
        help="Run mode: default(legacy), controller(menu only), worker(collection only).",
    )
    parser.add_argument("--hand", default=DEFAULT_HAND, choices=list(HAND_CHOICES))
    parser.add_argument("--target-finger", default="all")
    parser.add_argument("--cmd-topic", default="")
    parser.add_argument("--feedback-topic", default="")
    parser.add_argument("--glove-topic", default=DEFAULT_GLOVE_TOPIC)
    parser.add_argument("--glove-raw-skeleton-topic", default=DEFAULT_GLOVE_RAW_SKELETON_TOPIC)
    parser.add_argument("--defaults-path", default="")
    parser.add_argument("--boundary-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument(
        "--points-per-combo", type=int, default=0,
        help="Override each combo's point quota. 0 uses the configured quotas; other collection stages are unchanged.",
    )
    parser.add_argument(
        "--resume",
        default="false",
        help="Resume from checkpoint file (true/false). Default: false",
    )
    parser.add_argument(
        "--resume-file",
        default="",
        help="Path to resume checkpoint json. Default: <output_dir>/<hand>_resume_latest.json",
    )
    parser.add_argument(
        "--live-status-file",
        default="",
        help="Path to live runtime health json. Default: <output_dir>/<hand>_live_status.json",
    )
    parser.add_argument(
        "--control-file",
        default="",
        help="Path to controller->worker command queue file.",
    )
    parser.add_argument("--disable-glove-required", action="store_true")
    parser.add_argument(
        "--manual-supplement-after-run",
        default="true",
        help="Enter manual supplement loop after auto collection (true/false). Default: true",
    )
    args = parser.parse_args()
    if args.points_per_combo < 0:
        parser.error("--points-per-combo must be non-negative")
    return args


def _spawn_worker_terminal(cmd: list[str], title: str) -> subprocess.Popen:
    display_available = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    gnome_terminal = None
    try:
        import shutil

        gnome_terminal = shutil.which("gnome-terminal")
    except Exception:
        gnome_terminal = None

    if display_available and gnome_terminal:
        return subprocess.Popen(
            [gnome_terminal, "--wait", "--title", str(title), "--", *cmd],
            start_new_session=True,
        )
    return subprocess.Popen(cmd, start_new_session=True)


def _build_worker_cmd_from_args(args, manual_after_run: bool) -> list[str]:
    script_path = Path(__file__).resolve()
    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--run-mode",
        "worker",
        "--hand",
        str(args.hand),
        "--target-finger",
        str(args.target_finger),
        "--cmd-topic",
        str(args.cmd_topic or ""),
        "--feedback-topic",
        str(args.feedback_topic or ""),
        "--glove-topic",
        str(args.glove_topic or DEFAULT_GLOVE_TOPIC),
        "--glove-raw-skeleton-topic",
        str(args.glove_raw_skeleton_topic or DEFAULT_GLOVE_RAW_SKELETON_TOPIC),
        "--defaults-path",
        str(args.defaults_path or ""),
        "--boundary-path",
        str(args.boundary_path or ""),
        "--output-dir",
        str(args.output_dir or ""),
        "--seed",
        str(args.seed or ""),
        "--resume",
        str(args.resume or "false"),
        "--resume-file",
        str(args.resume_file or ""),
        "--live-status-file",
        str(args.live_status_file or ""),
        "--control-file",
        str(args.control_file or ""),
        "--manual-supplement-after-run",
        "true" if bool(manual_after_run) else "false",
        "--points-per-combo",
        str(args.points_per_combo),
    ]
    if bool(args.disable_glove_required):
        cmd.append("--disable-glove-required")
    return cmd


def _run_controller_menu(args):
    script_dir = Path(__file__).resolve().parent
    hand = str(args.hand).strip().lower()
    manual_after_run_enabled = _parse_bool_text(
        args.manual_supplement_after_run,
        default=True,
    )
    output_root = Path(args.output_dir) if args.output_dir else default_output_root(script_dir)
    control_file = _resolve_control_file_path(args.control_file, output_root, hand)
    args.control_file = str(control_file)
    try:
        control_file.parent.mkdir(parents=True, exist_ok=True)
        control_file.write_text("", encoding="utf-8")
    except Exception:
        pass

    worker_proc = None

    def _worker_running():
        return (worker_proc is not None) and (worker_proc.poll() is None)

    def _send_to_worker(cmd_text: str):
        if not _worker_running():
            print("[data-collect-controller] no running worker. use 'a' first.")
            return False
        try:
            _append_control_command(control_file, cmd_text)
            print(f"[data-collect-controller] sent -> worker: {cmd_text}")
            return True
        except Exception as exc:
            print(f"[data-collect-controller] send command failed: {exc}")
            return False

    def _start_worker(manual_after_run: bool, first_cmd: str = ""):
        nonlocal worker_proc
        if _worker_running():
            print("[data-collect-controller] worker already running.")
            return
        try:
            control_file.parent.mkdir(parents=True, exist_ok=True)
            control_file.write_text("", encoding="utf-8")
        except Exception:
            pass
        cmd = _build_worker_cmd_from_args(args, manual_after_run=manual_after_run)
        title = f"data_collection_{hand}_worker"
        try:
            worker_proc = _spawn_worker_terminal(cmd, title=title)
            print(
                "[data-collect-controller] worker started: "
                f"pid={worker_proc.pid}, manual_after_run={manual_after_run}"
            )
            if first_cmd:
                time.sleep(0.2)
                _append_control_command(control_file, first_cmd)
                print(f"[data-collect-controller] sent -> worker: {first_cmd}")
        except Exception as exc:
            print(f"[data-collect-controller] failed to start worker: {exc}")
            worker_proc = None

    print(
        "\n[data-collect-controller] commands: "
        "a=start worker, p=pause+zero, f <finger>=switch finger, r=resume interrupted, "
        "z=zero, t=tighten(-0.1A), i=current0, m=manual supplement, e=emergency stop, "
        "s=status, k=stop worker, q=quit, h=help"
    )
    while True:
        if worker_proc is not None:
            rc = worker_proc.poll()
            if rc is not None:
                print(f"[data-collect-controller] worker exited with code={rc}")
                worker_proc = None
        try:
            raw = input("[data-collect-controller] > ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = "q"
        if not raw:
            continue
        parts = str(raw).split()
        if not parts:
            continue
        cmd = str(parts[0]).strip().lower()
        if cmd in ("h", "help", "?"):
            print(
                "[data-collect-controller] "
                "a start worker | p pause+zero | f <finger> switch | r resume interrupted | "
                "z zero | t tighten(-0.1A) | i current0 | m manual supplement | e emergency stop | "
                "s status | k stop worker | q quit"
            )
            continue
        if cmd == "s":
            if worker_proc is None:
                print("[data-collect-controller] worker: idle")
            else:
                state = "running" if worker_proc.poll() is None else f"exited({worker_proc.poll()})"
                print(f"[data-collect-controller] worker: {state}, pid={worker_proc.pid}")
            print(f"[data-collect-controller] control_file={control_file}")
            continue
        if cmd == "a":
            if _worker_running():
                print("[data-collect-controller] worker already running. use p/r/f/z/t/i/m/e.")
            else:
                _start_worker(manual_after_run=manual_after_run_enabled)
            continue
        if cmd == "m":
            if _worker_running():
                _send_to_worker("m")
            else:
                _start_worker(
                    manual_after_run=manual_after_run_enabled,
                    first_cmd="m",
                )
            continue
        if cmd == "f":
            if len(parts) < 2:
                print("[data-collect-controller] usage: f <finger>")
                continue
            finger_pick = str(parts[1]).strip().lower()
            _send_to_worker(f"f {finger_pick}")
            continue
        if cmd in ("p", "r", "z", "t", "i", "e"):
            _send_to_worker(cmd)
            continue
        if cmd == "k":
            if not _worker_running():
                print("[data-collect-controller] no running worker.")
                worker_proc = None
                continue
            try:
                _append_control_command(control_file, "q")
                print("[data-collect-controller] sent -> worker: q")
            except Exception:
                pass
            waited = 0.0
            while _worker_running() and waited < 3.0:
                time.sleep(0.1)
                waited += 0.1
            if _worker_running():
                try:
                    os.killpg(worker_proc.pid, signal.SIGTERM)
                    print(f"[data-collect-controller] SIGTERM sent to process-group pgid={worker_proc.pid}")
                except Exception as exc:
                    try:
                        worker_proc.terminate()
                        print(f"[data-collect-controller] terminate sent to pid={worker_proc.pid}")
                    except Exception:
                        print(f"[data-collect-controller] terminate failed: {exc}")
            else:
                print("[data-collect-controller] worker exited.")
                worker_proc = None
            continue
        if cmd in ("q", "quit", "exit"):
            if _worker_running():
                try:
                    _append_control_command(control_file, "q")
                    print("[data-collect-controller] sent -> worker: q")
                except Exception:
                    pass
                waited = 0.0
                while _worker_running() and waited < 3.0:
                    time.sleep(0.1)
                    waited += 0.1
                if _worker_running():
                    try:
                        os.killpg(worker_proc.pid, signal.SIGTERM)
                    except Exception:
                        try:
                            worker_proc.terminate()
                        except Exception:
                            pass
            print("[data-collect-controller] exit.")
            return
        print(f"[data-collect-controller] unknown command: {raw}")


def main():
    args = _parse_args()
    run_mode = str(args.run_mode).strip().lower()
    if run_mode == "controller" and (not sys.stdin.isatty()):
        print("[WARN] controller mode without interactive stdin, fallback to worker mode.")
        run_mode = "worker"
    if run_mode == "controller":
        _run_controller_menu(args)
        return

    hand = args.hand.strip().lower()
    target_fingers = _normalize_target_fingers(args.target_finger)
    manual_supplement_after_run = _parse_bool_text(args.manual_supplement_after_run, default=True)
    resume_enabled = _parse_bool_text(args.resume, default=False)
    script_dir = Path(__file__).resolve().parent

    cmd_topic, feedback_topic = get_hand_topics(hand)
    if args.cmd_topic:
        cmd_topic = args.cmd_topic
    if args.feedback_topic:
        feedback_topic = args.feedback_topic
    glove_topic = args.glove_topic.strip() if args.glove_topic else DEFAULT_GLOVE_TOPIC
    glove_raw_skeleton_topic = (
        args.glove_raw_skeleton_topic.strip()
        if args.glove_raw_skeleton_topic
        else DEFAULT_GLOVE_RAW_SKELETON_TOPIC
    )

    defaults_path = Path(args.defaults_path) if args.defaults_path else (script_dir / "muscle_adjustment_defaults.py")
    boundary_path = (
        Path(args.boundary_path)
        if args.boundary_path
        else (script_dir / "motor_test_logs" / "workspace_boundaries" / hand / "boundary_models.json")
    )
    output_root = Path(args.output_dir) if args.output_dir else default_output_root(script_dir)
    resume_file = (
        Path(args.resume_file).expanduser()
        if args.resume_file
        else (output_root / f"{hand}_resume_latest.json")
    )
    live_status_file = (
        Path(args.live_status_file).expanduser()
        if args.live_status_file
        else (output_root / f"{hand}_live_status.json")
    )
    control_file = _resolve_control_file_path(args.control_file, output_root, hand)

    seed = RNG_SEED
    if args.seed:
        seed = int(args.seed)
    rng = random.Random(seed)

    defaults_hand = _load_defaults_from_py(defaults_path, hand)
    _, combos_json = _load_boundary_json(boundary_path, hand)
    combos_by_finger = _build_combo_specs(combos_json)
    if "thumb" in target_fingers:
        excluded = sorted(set(combos_json) & THUMB_EXCLUDED_COMBO_IDS)
        if excluded:
            print(f"[combo-policy] thumb excluded={excluded}; "
                  f"enabled={len(combos_by_finger['thumb'])}")
    finger_motor_set = _finger_motor_sets(hand)

    for finger in target_fingers:
        if not combos_by_finger.get(finger):
            raise RuntimeError(f"No combos found for finger={finger} in boundary file: {boundary_path}")

    resume_state = _load_resume_state_if_available(resume_file) if resume_enabled else None
    resuming = False
    if resume_state is not None:
        if str(resume_state.get("hand", "")).strip().lower() != hand:
            print(
                f"[resume] hand mismatch, ignore checkpoint: "
                f"checkpoint_hand={resume_state.get('hand')} runtime_hand={hand}"
            )
            resume_state = None
        else:
            ckpt_fingers = [str(v).strip().lower() for v in list(resume_state.get("target_fingers", []))]
            if ckpt_fingers != list(target_fingers):
                print(
                    "[resume] target_fingers mismatch, ignore checkpoint: "
                    f"checkpoint={ckpt_fingers} runtime={target_fingers}"
                )
                resume_state = None

    if resume_state is not None:
        run_root = Path(str(resume_state.get("run_root", ""))).expanduser()
        if str(run_root) == "." or (not str(run_root)):
            run_root = output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hand}_collection"
        run_root, hand_root, finger_dirs = _prepare_output_dirs_from_run_root(run_root, hand, target_fingers)
        resuming = True
    else:
        run_root, hand_root, finger_dirs = _prepare_output_dirs(output_root, hand, target_fingers)

    print(f"[data_collection] run_root={run_root}")
    print(f"[data_collection] defaults={defaults_path}")
    print(f"[data_collection] boundary={boundary_path}")
    print(
        f"[data_collection] hand={hand}, cmd={cmd_topic}, feedback={feedback_topic}, "
        f"glove={glove_topic}, glove_raw={glove_raw_skeleton_topic}"
    )
    print(f"[data_collection] target_finger={','.join(target_fingers)}")
    print(
        "[data_collection] antagonist_release="
        f"{'on' if ANTAGONIST_RELEASE_ENABLE else 'off'}, "
        f"policy=sec:data_collection_config(delta*per_deg, default={DATA_COLLECTION_RELEASE_SEC_PER_DEG_DEFAULT:.6f}s/deg, per-pair overridable), "
        f"tor:data_collection_config(default={DATA_COLLECTION_RELEASE_TOR_DEFAULT_A:.3f}A, per-pair overridable), "
        "multi-motor=pairwise_opposite_only(decoupled_by_active_delta)+active_excluded"
    )
    print(
        f"[data_collection] resume={'on' if resume_enabled else 'off'}, "
        f"resuming={'yes' if resuming else 'no'}, checkpoint={resume_file}"
    )
    print(f"[data_collection] control_file={control_file}")

    rclpy.init()
    node = DataCollectionNode(
        hand=hand,
        cmd_topic=cmd_topic,
        feedback_topic=feedback_topic,
        glove_topic=glove_topic,
        glove_raw_skeleton_topic=glove_raw_skeleton_topic,
        defaults_hand=defaults_hand,
    )
    spin_thread = threading.Thread(target=_spin_node, args=(node,), daemon=True)
    spin_thread.start()

    global_point = int(resume_state.get("global_point", 0)) if resume_state else 0
    global_timeout = int(resume_state.get("global_timeout", 0)) if resume_state else 0
    global_start = time.monotonic()
    stop_requested = False
    zero_calib_state = str(resume_state.get("zero_calib_state", "ok")) if resume_state else "ok"
    zero_calib_fail_streak = int(resume_state.get("zero_calib_fail_streak", 0)) if resume_state else 0
    zero_calib_total_events = int(resume_state.get("zero_calib_total_events", 0)) if resume_state else 0
    zero_calib_failed_events = int(resume_state.get("zero_calib_failed_events", 0)) if resume_state else 0
    finger_states = {}
    startup_first_frame_files = resume_state.get("startup_first_frame_files") if resume_state else None
    runtime_phase = "startup"
    runtime_reason = "init"
    runtime_current_finger = ""
    runtime_current_combo = ""
    runtime_current_stage = ""
    runtime_interactive = bool(sys.stdin.isatty()) and (run_mode != "worker")
    runtime_control_reader = None
    if (run_mode == "worker") or bool(str(args.control_file or "").strip()):
        runtime_control_reader = _init_control_reader(control_file)
    runtime_command_enabled = bool(runtime_interactive or (runtime_control_reader is not None))
    runtime_pending_cmds = []
    runtime_paused = False
    runtime_switch_finger = ""
    runtime_interrupted_finger = ""
    runtime_help_printed = False

    def _poll_runtime_command():
        nonlocal runtime_pending_cmds
        if runtime_pending_cmds:
            return runtime_pending_cmds.pop(0)
        if runtime_interactive:
            cmd = _poll_stdin_line_nonblocking()
            if cmd:
                return cmd
        if runtime_control_reader is not None:
            new_cmds = _poll_control_commands(runtime_control_reader)
            if new_cmds:
                runtime_pending_cmds.extend(new_cmds)
                return runtime_pending_cmds.pop(0)
        return None

    def _persist_runtime_state(
        phase: str | None = None,
        reason: str | None = None,
        current_finger: str | None = None,
        current_combo: str | None = None,
        current_stage: str | None = None,
    ):
        nonlocal runtime_phase, runtime_reason, runtime_current_finger, runtime_current_combo, runtime_current_stage
        if phase is not None:
            runtime_phase = str(phase)
        if reason is not None:
            runtime_reason = str(reason)
        if current_finger is not None:
            runtime_current_finger = str(current_finger)
        if current_combo is not None:
            runtime_current_combo = str(current_combo)
        if current_stage is not None:
            runtime_current_stage = str(current_stage)

        resume_payload = {
            "updated_at": datetime.now().isoformat(),
            "hand": hand,
            "target_fingers": list(target_fingers),
            "run_root": str(run_root),
            "hand_root": str(hand_root),
            "output_root": str(output_root),
            "global_point": int(global_point),
            "global_timeout": int(global_timeout),
            "zero_calib_state": str(zero_calib_state),
            "zero_calib_fail_streak": int(zero_calib_fail_streak),
            "zero_calib_total_events": int(zero_calib_total_events),
            "zero_calib_failed_events": int(zero_calib_failed_events),
            "phase": str(runtime_phase),
            "reason": str(runtime_reason),
            "current_finger": str(runtime_current_finger),
            "current_combo": str(runtime_current_combo),
            "current_stage": str(runtime_current_stage),
            "startup_first_frame_files": startup_first_frame_files,
            "finger_states": {
                str(f): _finger_state_to_resume_payload(st)
                for f, st in finger_states.items()
            },
        }
        try:
            _write_json_atomic(resume_file, resume_payload)
        except Exception:
            pass
        health_payload = _build_runtime_health_payload(
            hand=hand,
            phase=runtime_phase,
            reason=runtime_reason,
            current_finger=runtime_current_finger,
            current_combo=runtime_current_combo,
            current_stage=runtime_current_stage,
            global_point=global_point,
            global_timeout=global_timeout,
            zero_calib_state=zero_calib_state,
            zero_calib_fail_streak=zero_calib_fail_streak,
            finger_states=finger_states,
        )
        try:
            _write_json_atomic(hand_root / "runtime_health.json", health_payload)
        except Exception:
            pass
        try:
            if live_status_file:
                _write_json_atomic(live_status_file, health_payload)
        except Exception:
            pass

    if runtime_interactive and (not runtime_help_printed):
        _print_runtime_auto_help()
        runtime_help_printed = True
    elif (runtime_control_reader is not None) and (not runtime_help_printed):
        print(
            "[auto-control] worker command channel ready. "
            "accepted via controller: p/f/r/z/t/i/m/e/q"
        )
        runtime_help_printed = True

    try:
        ready_ok, ready_msg = node.wait_ready(timeout_sec=READY_WAIT_TIMEOUT_SEC)
        if not ready_ok:
            if args.disable_glove_required:
                print(f"[WARN] startup not fully ready: {ready_msg}; continue due to --disable-glove-required")
            else:
                raise RuntimeError(f"Startup not ready: {ready_msg}")

        if startup_first_frame_files is None:
            startup_first_frame_files = _save_startup_first_frame(node=node, hand_root=hand_root)
            if startup_first_frame_files is not None:
                print(
                    "[data_collection] startup first frame saved: "
                    f"{startup_first_frame_files.get('json', '')}"
                )
        else:
            print(
                "[resume] reuse startup first frame: "
                f"{startup_first_frame_files.get('json', '') if isinstance(startup_first_frame_files, dict) else ''}"
            )
        _persist_runtime_state(
            phase="auto",
            reason="ready",
            current_finger="",
            current_combo="",
            current_stage="ready",
        )

        if runtime_interactive:
            print(
                "\n[auto-menu] 已就绪。输入命令后按回车："
                " a=开始自动采集, "
                "f <finger>=设置优先手指, "
                "m=人工补点, z=回0, t=全手-0.1A, i=全手0A, e=急停(0A), q=安全退出, h=帮助"
            )
            while (not stop_requested):
                try:
                    raw_menu = input("[auto-menu] > ").strip()
                except EOFError:
                    break
                if not raw_menu:
                    continue
                parts_menu = raw_menu.split()
                cmd_menu = parts_menu[0].strip().lower()
                if cmd_menu in ("h", "help", "?"):
                    print(
                        "[auto-menu] commands: "
                        "a start auto | f <finger> priority | m manual supplement | z zero | "
                        "t tighten(-0.1A) | i current0 | e emergency stop(0A) | q quit"
                    )
                    continue
                if cmd_menu == "a":
                    print("[auto-menu] start auto collection.")
                    break
                if cmd_menu == "q":
                    stop_requested = True
                    print("[auto-menu] safe exit requested before auto start.")
                    break
                if cmd_menu == "f":
                    if len(parts_menu) < 2:
                        print(f"[auto-menu] usage: f <finger>, available={target_fingers}")
                        continue
                    finger_pick = str(parts_menu[1]).strip().lower()
                    if finger_pick not in target_fingers:
                        print(f"[auto-menu] unknown finger={finger_pick}, available={target_fingers}")
                        continue
                    ordered = [finger_pick] + [v for v in target_fingers if v != finger_pick]
                    target_fingers[:] = ordered
                    print(f"[auto-menu] finger priority updated: {target_fingers}")
                    continue
                if cmd_menu == "m":
                    _persist_runtime_state(
                        phase="manual",
                        reason="startup_manual_enter",
                        current_finger="",
                        current_combo="",
                        current_stage="manual_wait",
                    )
                    run_counters = {
                        "global_point": int(global_point),
                        "global_timeout": int(global_timeout),
                    }
                    calib_state = {
                        "zero_calib_state": str(zero_calib_state),
                        "zero_calib_fail_streak": int(zero_calib_fail_streak),
                        "zero_calib_total_events": int(zero_calib_total_events),
                        "zero_calib_failed_events": int(zero_calib_failed_events),
                    }
                    _run_manual_supplement_loop(
                        enable=True,
                        node=node,
                        defaults_hand=defaults_hand,
                        finger_motor_set=finger_motor_set,
                        finger_states=finger_states,
                        rng=rng,
                        run_counters=run_counters,
                        calib_state=calib_state,
                        hand_root=hand_root,
                        confirm=False,
                    )
                    global_point = int(run_counters["global_point"])
                    global_timeout = int(run_counters["global_timeout"])
                    zero_calib_state = str(calib_state["zero_calib_state"])
                    zero_calib_fail_streak = int(calib_state["zero_calib_fail_streak"])
                    zero_calib_total_events = int(calib_state["zero_calib_total_events"])
                    zero_calib_failed_events = int(calib_state["zero_calib_failed_events"])
                    _persist_runtime_state(
                        phase="auto",
                        reason="startup_manual_exit",
                        current_finger="",
                        current_combo="",
                        current_stage="ready",
                    )
                    continue
                if cmd_menu == "z":
                    calib_result = _run_zero_calibration(
                        node, defaults_hand, finger="thumb" if target_fingers == ["thumb"] else None)
                    zero_calib_total_events += 1
                    if calib_result["success"]:
                        zero_calib_state = "ok"
                        zero_calib_fail_streak = 0
                        print(
                            "[auto-menu] zero calibration done. "
                            f"attempts={calib_result['attempts']}/{calib_result['max_attempts']}"
                        )
                    else:
                        zero_calib_failed_events += 1
                        zero_calib_fail_streak += 1
                        zero_calib_state = f"failed_after_{calib_result['max_attempts']}_attempts"
                        print(
                            "[auto-menu] zero calibration failed. "
                            f"attempts={calib_result['attempts']}/{calib_result['max_attempts']}, "
                            f"last_max_abs_pos={calib_result['last_max_abs_pos']}"
                        )
                    _append_csv_row(
                        hand_root / "zero_calibration_events.csv",
                        _zero_calibration_event_headers(),
                        [
                            global_point,
                            "menu",
                            int(calib_result["success"]),
                            calib_result["attempts"],
                            calib_result["max_attempts"],
                            calib_result["last_max_abs_pos"],
                            zero_calib_state,
                            zero_calib_fail_streak,
                        ],
                    )
                    continue
                if cmd_menu == "t":
                    node.set_command_matrix(_build_uniform_current_matrix(-0.1))
                    print("[auto-menu] all motors tighten current set to -0.1A.")
                    continue
                if cmd_menu in ("i", "e"):
                    node.set_command_matrix(_build_uniform_current_matrix(0.0))
                    if cmd_menu == "e":
                        print("[auto-menu] emergency stop applied (all currents=0).")
                    else:
                        print("[auto-menu] all motors current set to 0A.")
                    continue
                print(f"[auto-menu] unknown command: {raw_menu!r}")

        for finger_idx, finger in enumerate(target_fingers):
            if stop_requested:
                break
            existing_finger_state = finger_states.get(finger)
            if existing_finger_state is None:
                combos = combos_by_finger[finger]
                nominal_total, combo_min, combo_max, combo_target = _build_finger_quota_plan(
                    finger, combos, points_per_combo=args.points_per_combo)
                combo_min_base = {str(k): int(v) for k, v in combo_min.items()}
                combo_max_base = {str(k): int(v) for k, v in combo_max.items()}
                combo_counts = {c.combo_id: 0 for c in combos}
                combo_coverage = {c.combo_id: _build_combo_coverage_state(c) for c in combos}
                combo_target_none_consec = {c.combo_id: 0 for c in combos}
                combo_cooldown = {c.combo_id: 0 for c in combos}
                combo_stall_zero_retry_count = {c.combo_id: 0 for c in combos}
                combo_stall_over_ratio_hit_count = {c.combo_id: 0 for c in combos}
                combo_stall_finish_reason = {c.combo_id: "" for c in combos}
                finger_reopen_round = 0
                finger_reopen_no_progress_round = 0
                finger_reopen_last_point_id = 0
                mode_keys, mode_target, mode_min, mode_max = _build_active_mode_quota_plan(
                    finger=finger,
                    nominal_total=nominal_total,
                )
                mode_counts = {mode: 0 for mode in mode_keys}
                point_id = 0
                unreached = []
                finger_zero_calib_events = 0
                finger_zero_calib_failed_events = 0
                finger_state = {
                    "finger": finger,
                    "combos": combos,
                    "nominal_total": nominal_total,
                    "combo_counts": combo_counts,
                    "combo_min": combo_min,
                    "combo_max": combo_max,
                    "combo_target": combo_target,
                    "combo_min_base": combo_min_base,
                    "combo_max_base": combo_max_base,
                    "combo_coverage": combo_coverage,
                    "combo_cooldown": combo_cooldown,
                    "combo_target_none_consec": combo_target_none_consec,
                    "combo_stall_zero_retry_count": combo_stall_zero_retry_count,
                    "combo_stall_over_ratio_hit_count": combo_stall_over_ratio_hit_count,
                    "combo_stall_finish_reason": combo_stall_finish_reason,
                    "finger_reopen_round": finger_reopen_round,
                    "finger_reopen_no_progress_round": finger_reopen_no_progress_round,
                    "mode_keys": mode_keys,
                    "mode_counts": mode_counts,
                    "mode_target": mode_target,
                    "mode_min": mode_min,
                    "mode_max": mode_max,
                    "point_id": point_id,
                    "unreached": unreached,
                    "finger_dirs": finger_dirs[finger],
                    "finger_zero_calib_events": finger_zero_calib_events,
                    "finger_zero_calib_failed_events": finger_zero_calib_failed_events,
                    "finger_reopen_last_point_id": finger_reopen_last_point_id,
                    "preuniform_done": False,
                    "edge_enhancement_done": False,
                    "done": False,
                }
                if resume_state is not None:
                    resume_finger_states = resume_state.get("finger_states", {})
                    resume_payload = resume_finger_states.get(str(finger))
                    if isinstance(resume_payload, dict):
                        _apply_finger_resume_payload(finger_state, resume_payload)
                        print(
                            f"[resume] finger={finger} point={int(finger_state.get('point_id', 0))} "
                            f"done={bool(finger_state.get('done', False))} "
                            f"preuniform_done={bool(finger_state.get('preuniform_done', False))}"
                        )
                finger_states[finger] = finger_state
            else:
                finger_state = existing_finger_state
                print(
                    f"[resume-runtime] finger={finger} point={int(finger_state.get('point_id', 0))} "
                    f"done={bool(finger_state.get('done', False))}"
                )

            # Sync local mirrors from current finger_state
            combos = list(finger_state["combos"])
            nominal_total = int(finger_state["nominal_total"])
            combo_counts = finger_state["combo_counts"]
            combo_min = finger_state["combo_min"]
            combo_max = finger_state["combo_max"]
            combo_target = finger_state["combo_target"]
            combo_min_base = finger_state.get("combo_min_base", combo_min)
            combo_max_base = finger_state.get("combo_max_base", combo_max)
            combo_coverage = finger_state["combo_coverage"]
            combo_target_none_consec = finger_state["combo_target_none_consec"]
            combo_cooldown = finger_state["combo_cooldown"]
            combo_stall_zero_retry_count = finger_state.get("combo_stall_zero_retry_count", {})
            combo_stall_over_ratio_hit_count = finger_state.get("combo_stall_over_ratio_hit_count", {})
            combo_stall_finish_reason = finger_state.get("combo_stall_finish_reason", {})
            mode_keys = list(finger_state.get("mode_keys", []))
            mode_counts = finger_state.get("mode_counts", {})
            mode_target = finger_state.get("mode_target", {})
            mode_min = finger_state.get("mode_min", {})
            mode_max = finger_state.get("mode_max", {})
            point_id = int(finger_state["point_id"])
            unreached = list(finger_state["unreached"])
            finger_reopen_round = int(finger_state.get("finger_reopen_round", 0))
            finger_reopen_no_progress_round = int(finger_state.get("finger_reopen_no_progress_round", 0))
            finger_reopen_last_point_id = int(finger_state.get("finger_reopen_last_point_id", point_id))
            finger_zero_calib_events = int(finger_state.get("finger_zero_calib_events", 0))
            finger_zero_calib_failed_events = int(finger_state.get("finger_zero_calib_failed_events", 0))

            range_min = min(combo_min.values()) if combo_min else 0
            range_max = max(combo_max.values()) if combo_max else 0
            print(
                f"\n[finger-start] {finger} combos={len(combos)} nominal_total={nominal_total} "
                f"range_per_combo=[{range_min},{range_max}]"
            )
            if finger == "thumb":
                on_target_total = sum(
                    float(combo_target.get(c.combo_id, 0.0))
                    for c in combos
                    if str(c.opposition_mode).lower() == "on"
                )
                all_target_total = sum(float(combo_target.get(c.combo_id, 0.0)) for c in combos)
                on_ratio_now = (
                    on_target_total / all_target_total
                    if all_target_total > 1e-9
                    else 0.0
                )
                print(
                    f"[finger-start] thumb opposition target ratio: on={on_ratio_now:.3f}, "
                    f"off={1.0 - on_ratio_now:.3f}"
                )
            if finger == "thumb":
                print(
                    "[finger-start] thumb active mode target ratio: "
                    f"full={float(mode_target.get('full', 0.0))/max(1.0, float(nominal_total)):.3f}, "
                    f"single={float(mode_target.get('single', 0.0))/max(1.0, float(nominal_total)):.3f}, "
                    f"double={float(mode_target.get('double', 0.0))/max(1.0, float(nominal_total)):.3f}"
                )
                if int(THUMB_ABDUCTION_EXTRA_POINTS) > 0:
                    print(
                        "[finger-start] thumb abduction enhancement: "
                        f"extra_points={int(THUMB_ABDUCTION_EXTRA_POINTS)}, "
                        f"edge_region_ratio={float(THUMB_ABDUCTION_EDGE_REGION_RATIO):.3f}, "
                        f"edge_bias_prob={float(THUMB_ABDUCTION_EDGE_BIAS_PROB):.3f}"
                    )
                if int(THUMB_FLEXION_EXTRA_POINTS) > 0:
                    print(
                        "[finger-start] thumb flexion enhancement: "
                        f"extra_points={int(THUMB_FLEXION_EXTRA_POINTS)}, "
                        f"edge_region_ratio={float(FLEXION_EDGE_REGION_RATIO):.3f}, "
                        f"edge_bias_prob={float(FLEXION_EDGE_BIAS_PROB):.3f}"
                    )
            else:
                print(
                    "[finger-start] active mode target ratio: "
                    f"xy={float(mode_target.get('xy', 0.0))/max(1.0, float(nominal_total)):.3f}, "
                    f"x_only={float(mode_target.get('x_only', 0.0))/max(1.0, float(nominal_total)):.3f}, "
                    f"y_only={float(mode_target.get('y_only', 0.0))/max(1.0, float(nominal_total)):.3f}"
                )
                if int(FOUR_FINGER_FLEXION_EXTRA_POINTS_PER_FINGER) > 0:
                    print(
                        "[finger-start] four-finger flexion enhancement: "
                        f"extra_points_per_finger={int(FOUR_FINGER_FLEXION_EXTRA_POINTS_PER_FINGER)}, "
                        f"edge_region_ratio={float(FLEXION_EDGE_REGION_RATIO):.3f}, "
                        f"edge_bias_prob={float(FLEXION_EDGE_BIAS_PROB):.3f}"
                    )

            _persist_runtime_state(
                phase="auto",
                reason="finger_start",
                current_finger=finger,
                current_combo="",
                current_stage="finger_start",
            )

            if bool(finger_state.get("done", False)):
                print(f"[resume] finger={finger} already done, skip.")
                _persist_runtime_state(
                    phase="auto",
                    reason="finger_skipped_done",
                    current_finger=finger,
                    current_combo="",
                    current_stage="finger_done",
                )
                continue

            finger_interrupted = False
            pause_notice_printed = False

            def _run_zero_calibration_for_runtime(reason_tag: str):
                nonlocal zero_calib_state, zero_calib_fail_streak
                nonlocal zero_calib_total_events, zero_calib_failed_events
                nonlocal finger_zero_calib_events, finger_zero_calib_failed_events
                calib_result = _run_zero_calibration(node, defaults_hand, finger=finger)
                zero_calib_total_events += 1
                finger_zero_calib_events += 1
                finger_state["finger_zero_calib_events"] = finger_zero_calib_events
                if calib_result["success"]:
                    zero_calib_state = "ok"
                    zero_calib_fail_streak = 0
                    print(
                        f"[runtime] zero calibration done ({reason_tag}). "
                        f"attempts={calib_result['attempts']}/{calib_result['max_attempts']}"
                    )
                else:
                    zero_calib_failed_events += 1
                    finger_zero_calib_failed_events += 1
                    finger_state["finger_zero_calib_failed_events"] = finger_zero_calib_failed_events
                    zero_calib_fail_streak += 1
                    zero_calib_state = f"failed_after_{calib_result['max_attempts']}_attempts"
                    print(
                        f"[runtime] zero calibration failed ({reason_tag}). "
                        f"attempts={calib_result['attempts']}/{calib_result['max_attempts']}, "
                        f"last_max_abs_pos={calib_result['last_max_abs_pos']}"
                    )
                _append_csv_row(
                    hand_root / "zero_calibration_events.csv",
                    _zero_calibration_event_headers(),
                    [
                        global_point,
                        finger,
                        int(calib_result["success"]),
                        calib_result["attempts"],
                        calib_result["max_attempts"],
                        calib_result["last_max_abs_pos"],
                        zero_calib_state,
                        zero_calib_fail_streak,
                    ],
                )

            def _pause_hold_current_for_runtime(reason_tag: str):
                snap_now = node.snapshot()
                if snap_now is not None and "pos" in snap_now:
                    hold_matrix = _build_hold_current_matrix(
                        defaults_hand=defaults_hand,
                        current_pos=np.asarray(snap_now["pos"], dtype=np.float64),
                    )
                    if finger == "thumb":
                        hold_matrix = _keep_four_fingers_at_default_current(node.hand, defaults_hand, hold_matrix)
                    node.set_command_matrix(hold_matrix)
                    print(f"[auto-control] paused-hold ({reason_tag}) at current positions.")
                else:
                    node.set_command_matrix(_build_idle_tor_matrix(defaults_hand))
                    print(f"[auto-control] paused-hold ({reason_tag}) fallback idle-tor (no snapshot).")

            def _preuniform_runtime_tick():
                nonlocal stop_requested, finger_interrupted, runtime_paused
                nonlocal runtime_switch_finger, runtime_interrupted_finger
                nonlocal global_point, global_timeout
                nonlocal zero_calib_state, zero_calib_fail_streak
                nonlocal zero_calib_total_events, zero_calib_failed_events
                nonlocal finger_zero_calib_events, finger_zero_calib_failed_events
                while runtime_command_enabled:
                    raw_cmd = _poll_runtime_command()
                    if raw_cmd is None:
                        break
                    parts = str(raw_cmd).split()
                    if not parts:
                        continue
                    cmd = parts[0].strip().lower()
                    if cmd in ("h", "help", "?"):
                        _print_runtime_auto_help()
                        continue
                    if cmd == "p":
                        runtime_paused = True
                        runtime_interrupted_finger = finger
                        _run_zero_calibration_for_runtime("pause_command_preuniform")
                        _pause_hold_current_for_runtime("pause_after_zero_preuniform")
                        return False
                    if cmd == "f":
                        if len(parts) < 2:
                            print(f"[auto-control] usage: f <finger>, available={target_fingers}")
                            continue
                        nxt = str(parts[1]).strip().lower()
                        if nxt not in target_fingers:
                            print(f"[auto-control] unknown finger={nxt}, available={target_fingers}")
                            continue
                        runtime_switch_finger = nxt
                        runtime_interrupted_finger = finger
                        runtime_paused = False
                        finger_interrupted = True
                        return False
                    if cmd == "r":
                        if runtime_interrupted_finger:
                            runtime_switch_finger = str(runtime_interrupted_finger)
                            runtime_paused = False
                            finger_interrupted = True
                            print(f"[auto-control] resume interrupted finger={runtime_interrupted_finger}")
                            return False
                        print("[auto-control] no interrupted finger to resume.")
                        continue
                    if cmd == "z":
                        _run_zero_calibration_for_runtime("runtime_command_preuniform")
                        continue
                    if cmd == "t":
                        node.set_command_matrix(_build_uniform_current_matrix(-0.1))
                        print("[auto-control] all motors tighten current set to -0.1A.")
                        continue
                    if cmd == "i":
                        node.set_command_matrix(_build_uniform_current_matrix(0.0))
                        print("[auto-control] all motors current set to 0A.")
                        continue
                    if cmd == "e":
                        node.set_command_matrix(_build_uniform_current_matrix(0.0))
                        runtime_paused = True
                        runtime_interrupted_finger = finger
                        print("[auto-control] emergency stop applied (all currents=0), now paused.")
                        return False
                    if cmd == "m":
                        _persist_runtime_state(
                            phase="manual",
                            reason="runtime_manual_enter_preuniform",
                            current_finger=finger,
                            current_combo="",
                            current_stage="manual_wait",
                        )
                        run_counters = {
                            "global_point": int(global_point),
                            "global_timeout": int(global_timeout),
                        }
                        calib_state = {
                            "zero_calib_state": str(zero_calib_state),
                            "zero_calib_fail_streak": int(zero_calib_fail_streak),
                            "zero_calib_total_events": int(zero_calib_total_events),
                            "zero_calib_failed_events": int(zero_calib_failed_events),
                        }
                        _run_manual_supplement_loop(
                            enable=True,
                            node=node,
                            defaults_hand=defaults_hand,
                            finger_motor_set=finger_motor_set,
                            finger_states=finger_states,
                            rng=rng,
                            run_counters=run_counters,
                            calib_state=calib_state,
                            hand_root=hand_root,
                            confirm=False,
                        )
                        global_point = int(run_counters["global_point"])
                        global_timeout = int(run_counters["global_timeout"])
                        zero_calib_state = str(calib_state["zero_calib_state"])
                        zero_calib_fail_streak = int(calib_state["zero_calib_fail_streak"])
                        zero_calib_total_events = int(calib_state["zero_calib_total_events"])
                        zero_calib_failed_events = int(calib_state["zero_calib_failed_events"])
                        _persist_runtime_state(
                            phase="auto",
                            reason="runtime_manual_exit_preuniform",
                            current_finger=finger,
                            current_combo="",
                            current_stage="resume_after_manual",
                        )
                        continue
                    if cmd == "q":
                        stop_requested = True
                        finger_interrupted = True
                        print("[auto-control] safe exit requested.")
                        return False
                    print(f"[auto-control] unknown command: {raw_cmd!r}")

                if stop_requested or runtime_paused:
                    finger_interrupted = True
                    return False
                return True

            if bool(PRE_RANDOM_UNIFORM_ENABLE) and (not bool(finger_state.get("preuniform_done", False))):
                pre_run_counters = {
                    "global_point": int(global_point),
                    "global_timeout": int(global_timeout),
                }
                pre_calib_state = {
                    "zero_calib_state": str(zero_calib_state),
                    "zero_calib_fail_streak": int(zero_calib_fail_streak),
                    "zero_calib_total_events": int(zero_calib_total_events),
                    "zero_calib_failed_events": int(zero_calib_failed_events),
                }
                pre_completed = _run_pre_random_uniform_sampling(
                    node=node,
                    defaults_hand=defaults_hand,
                    finger_motor_set=finger_motor_set,
                    finger_state=finger_state,
                    run_counters=pre_run_counters,
                    calib_state=pre_calib_state,
                    hand_root=hand_root,
                    runtime_tick=_preuniform_runtime_tick,
                )
                global_point = int(pre_run_counters["global_point"])
                global_timeout = int(pre_run_counters["global_timeout"])
                zero_calib_state = str(pre_calib_state["zero_calib_state"])
                zero_calib_fail_streak = int(pre_calib_state["zero_calib_fail_streak"])
                zero_calib_total_events = int(pre_calib_state["zero_calib_total_events"])
                zero_calib_failed_events = int(pre_calib_state["zero_calib_failed_events"])
                point_id = int(finger_state["point_id"])
                finger_zero_calib_events = int(finger_state["finger_zero_calib_events"])
                finger_zero_calib_failed_events = int(finger_state["finger_zero_calib_failed_events"])
                finger_reopen_last_point_id = int(point_id)
                if pre_completed:
                    finger_state["preuniform_done"] = True
                    print(
                        f"[finger-preuniform] finger={finger} points_after_preuniform={point_id}/{nominal_total}"
                    )
                    _persist_runtime_state(
                        phase="auto",
                        reason="finger_preuniform_done",
                        current_finger=finger,
                        current_combo="",
                        current_stage="preuniform_done",
                    )
                else:
                    print(f"[finger-preuniform] finger={finger} interrupted before completion.")
            elif bool(PRE_RANDOM_UNIFORM_ENABLE):
                print(f"[resume] finger={finger} preuniform already done, skip preuniform stage.")

            if runtime_switch_finger and runtime_switch_finger != finger:
                if finger not in target_fingers[finger_idx + 1 :]:
                    target_fingers.append(finger)
                for _idx in range(finger_idx + 1, len(target_fingers)):
                    if str(target_fingers[_idx]) == runtime_switch_finger:
                        del target_fingers[_idx]
                        break
                target_fingers.insert(finger_idx + 1, runtime_switch_finger)
                print(f"[auto-control] switch finger: {finger} -> {runtime_switch_finger}")
                runtime_switch_finger = ""
                finger_interrupted = True

            while (not finger_interrupted) and (not _finger_done(
                combo_counts,
                combo_min,
                combo_max,
                nominal_total,
                mode_counts=mode_counts,
                mode_min=mode_min,
                mode_max=mode_max,
            )):
                requested_switch_finger = ""
                while runtime_command_enabled:
                    raw_cmd = _poll_runtime_command()
                    if raw_cmd is None:
                        break
                    parts = str(raw_cmd).split()
                    if not parts:
                        continue
                    cmd = parts[0].strip().lower()
                    if cmd in ("h", "help", "?"):
                        _print_runtime_auto_help()
                        continue
                    if cmd == "p":
                        runtime_paused = True
                        runtime_interrupted_finger = finger
                        _run_zero_calibration_for_runtime("pause_command")
                        _pause_hold_current_for_runtime("pause_after_zero")
                        continue
                    if cmd == "f":
                        if len(parts) < 2:
                            print(f"[auto-control] usage: f <finger>, available={target_fingers}")
                            continue
                        nxt = str(parts[1]).strip().lower()
                        if nxt not in target_fingers:
                            print(f"[auto-control] unknown finger={nxt}, available={target_fingers}")
                            continue
                        requested_switch_finger = nxt
                        runtime_paused = False
                        pause_notice_printed = False
                        continue
                    if cmd == "r":
                        if runtime_interrupted_finger:
                            requested_switch_finger = str(runtime_interrupted_finger)
                            runtime_paused = False
                            pause_notice_printed = False
                            print(f"[auto-control] resume interrupted finger={runtime_interrupted_finger}")
                        else:
                            print("[auto-control] no interrupted finger to resume.")
                        continue
                    if cmd == "z":
                        _run_zero_calibration_for_runtime("runtime_command")
                        continue
                    if cmd == "t":
                        node.set_command_matrix(_build_uniform_current_matrix(-0.1))
                        print("[auto-control] all motors tighten current set to -0.1A.")
                        continue
                    if cmd == "i":
                        node.set_command_matrix(_build_uniform_current_matrix(0.0))
                        print("[auto-control] all motors current set to 0A.")
                        continue
                    if cmd == "e":
                        node.set_command_matrix(_build_uniform_current_matrix(0.0))
                        runtime_paused = True
                        runtime_interrupted_finger = finger
                        print("[auto-control] emergency stop applied (all currents=0), now paused.")
                        continue
                    if cmd == "m":
                        _persist_runtime_state(
                            phase="manual",
                            reason="runtime_manual_enter",
                            current_finger=finger,
                            current_combo="",
                            current_stage="manual_wait",
                        )
                        run_counters = {
                            "global_point": int(global_point),
                            "global_timeout": int(global_timeout),
                        }
                        calib_state = {
                            "zero_calib_state": str(zero_calib_state),
                            "zero_calib_fail_streak": int(zero_calib_fail_streak),
                            "zero_calib_total_events": int(zero_calib_total_events),
                            "zero_calib_failed_events": int(zero_calib_failed_events),
                        }
                        _run_manual_supplement_loop(
                            enable=True,
                            node=node,
                            defaults_hand=defaults_hand,
                            finger_motor_set=finger_motor_set,
                            finger_states=finger_states,
                            rng=rng,
                            run_counters=run_counters,
                            calib_state=calib_state,
                            hand_root=hand_root,
                            confirm=False,
                        )
                        global_point = int(run_counters["global_point"])
                        global_timeout = int(run_counters["global_timeout"])
                        zero_calib_state = str(calib_state["zero_calib_state"])
                        zero_calib_fail_streak = int(calib_state["zero_calib_fail_streak"])
                        zero_calib_total_events = int(calib_state["zero_calib_total_events"])
                        zero_calib_failed_events = int(calib_state["zero_calib_failed_events"])
                        _persist_runtime_state(
                            phase="auto",
                            reason="runtime_manual_exit",
                            current_finger=finger,
                            current_combo="",
                            current_stage="resume_after_manual",
                        )
                        continue
                    if cmd == "q":
                        stop_requested = True
                        print("[auto-control] safe exit requested.")
                        break
                    print(f"[auto-control] unknown command: {raw_cmd!r}")

                if stop_requested:
                    finger_interrupted = True
                    break

                if requested_switch_finger and requested_switch_finger != finger:
                    runtime_switch_finger = str(requested_switch_finger)
                    runtime_interrupted_finger = finger
                    runtime_paused = False
                    pause_notice_printed = False
                    if finger not in target_fingers[finger_idx + 1 :]:
                        target_fingers.append(finger)
                    for _idx in range(finger_idx + 1, len(target_fingers)):
                        if str(target_fingers[_idx]) == runtime_switch_finger:
                            del target_fingers[_idx]
                            break
                    target_fingers.insert(finger_idx + 1, runtime_switch_finger)
                    finger_interrupted = True
                    print(f"[auto-control] switch finger: {finger} -> {runtime_switch_finger}")
                    break

                if runtime_paused:
                    if not pause_notice_printed:
                        print(f"[auto-control] paused on finger={finger}. input r/f/m/z/t/i/e/q")
                        pause_notice_printed = True
                    time.sleep(0.05)
                    continue
                pause_notice_printed = False

                for cid in combo_cooldown:
                    if combo_cooldown[cid] > 0:
                        combo_cooldown[cid] -= 1

                combo = _choose_combo(
                    combos,
                    combo_counts,
                    combo_min,
                    combo_max,
                    combo_cooldown,
                    combo_coverage,
                    rng,
                )
                if combo is None:
                    if any(v > 0 for v in combo_cooldown.values()):
                        time.sleep(0.01)
                        continue
                    finger_ratio = float(point_id) / max(1.0, float(nominal_total))
                    ratio_need = float(FINGER_COMPLETE_MIN_RATIO)
                    if finger_ratio >= ratio_need:
                        print(
                            f"[finger-complete-by-ratio] finger={finger} "
                            f"ratio={finger_ratio:.3f}>={ratio_need:.3f}, stop finger."
                        )
                        break

                    if finger_reopen_round >= int(FINGER_UNFREEZE_MAX_ROUNDS):
                        print(
                            f"[WARN] no available combo for finger={finger}, ratio={finger_ratio:.3f}<{ratio_need:.3f}, "
                            f"unfreeze_round_exhausted={finger_reopen_round}/{int(FINGER_UNFREEZE_MAX_ROUNDS)}; stop this finger."
                        )
                        break

                    gained_since_last_reopen = int(point_id) - int(finger_reopen_last_point_id)
                    if gained_since_last_reopen <= 0:
                        finger_reopen_no_progress_round += 1
                    else:
                        finger_reopen_no_progress_round = 0
                    finger_reopen_last_point_id = int(point_id)

                    if finger_reopen_no_progress_round >= int(FINGER_UNFREEZE_MAX_NO_PROGRESS_ROUNDS):
                        print(
                            f"[WARN] finger={finger} no progress after repeated unfreeze rounds "
                            f"({finger_reopen_no_progress_round}/{int(FINGER_UNFREEZE_MAX_NO_PROGRESS_ROUNDS)}), "
                            f"ratio={finger_ratio:.3f}; stop to avoid dead-loop."
                        )
                        break

                    finger_reopen_round += 1
                    finger_state["finger_reopen_round"] = int(finger_reopen_round)
                    finger_state["finger_reopen_no_progress_round"] = int(finger_reopen_no_progress_round)
                    print(
                        f"[finger-unfreeze] finger={finger} ratio={finger_ratio:.3f}<{ratio_need:.3f}, "
                        f"round={finger_reopen_round}/{int(FINGER_UNFREEZE_MAX_ROUNDS)}; "
                        "run zero calibration and reopen combo caps."
                    )
                    calib_result = _run_zero_calibration(node, defaults_hand, finger=finger)
                    zero_calib_total_events += 1
                    finger_zero_calib_events += 1
                    finger_state["finger_zero_calib_events"] = finger_zero_calib_events
                    if calib_result["success"]:
                        zero_calib_state = "ok"
                        zero_calib_fail_streak = 0
                    else:
                        zero_calib_failed_events += 1
                        finger_zero_calib_failed_events += 1
                        finger_state["finger_zero_calib_failed_events"] = (
                            finger_zero_calib_failed_events
                        )
                        zero_calib_fail_streak += 1
                        zero_calib_state = f"failed_after_{calib_result['max_attempts']}_attempts"
                    _append_csv_row(
                        hand_root / "zero_calibration_events.csv",
                        _zero_calibration_event_headers(),
                        [
                            global_point,
                            finger,
                            int(calib_result["success"]),
                            calib_result["attempts"],
                            calib_result["max_attempts"],
                            calib_result["last_max_abs_pos"],
                            zero_calib_state,
                            zero_calib_fail_streak,
                        ],
                    )
                    for cid in combo_counts.keys():
                        combo_min[cid] = int(combo_min_base.get(cid, combo_min[cid]))
                        combo_max[cid] = int(combo_max_base.get(cid, combo_max[cid]))
                        combo_cooldown[cid] = 0
                        combo_target_none_consec[cid] = 0
                        combo_stall_over_ratio_hit_count[cid] = 0
                        combo_stall_zero_retry_count[cid] = 0
                    continue

                snap = node.snapshot()
                if snap is None:
                    time.sleep(0.05)
                    continue
                selected_mode = _choose_active_mode(
                    mode_keys=mode_keys,
                    mode_counts=mode_counts,
                    mode_min=mode_min,
                    mode_max=mode_max,
                    rng=rng,
                )
                if selected_mode is None:
                    selected_mode = "full" if finger == "thumb" else "xy"
                target_plan = _generate_target(
                    hand=hand,
                    combo=combo,
                    current_pos=snap["pos"],
                    coverage=combo_coverage[combo.combo_id],
                    rng=rng,
                    active_mode=selected_mode,
                )
                if target_plan is None:
                    fallback_mode = "full" if finger == "thumb" else "xy"
                    if selected_mode != fallback_mode:
                        target_plan = _generate_target(
                            hand=hand,
                            combo=combo,
                            current_pos=snap["pos"],
                            coverage=combo_coverage[combo.combo_id],
                            rng=rng,
                            active_mode=fallback_mode,
                        )
                if target_plan is None:
                    cid = combo.combo_id
                    combo_target_none_consec[cid] += 1
                    n_fail = combo_target_none_consec[cid]
                    stall_trigger = max(1, int(COMBO_STALL_FAIL_TRIGGER))
                    stall_ratio = max(0.0, float(COMBO_STALL_COMPLETE_RATIO))
                    stall_complete_hit_windows = max(1, int(COMBO_STALL_COMPLETE_HIT_WINDOWS))
                    stall_max_zero_retry = max(0, int(COMBO_STALL_MAX_ZERO_RETRY))
                    combo_target_now = max(1.0, float(combo_target.get(cid, _combo_base_quota(finger))))
                    combo_ratio_now = float(combo_counts[cid]) / combo_target_now

                    if n_fail >= stall_trigger:
                        if combo_ratio_now >= stall_ratio:
                            combo_stall_over_ratio_hit_count[cid] = int(
                                combo_stall_over_ratio_hit_count[cid]
                            ) + 1
                            hits_now = int(combo_stall_over_ratio_hit_count[cid])
                            combo_target_none_consec[cid] = 0
                            combo_cooldown[cid] = int(TARGET_NONE_COOLDOWN_CYCLES)
                            if hits_now >= stall_complete_hit_windows:
                                combo_min[cid] = int(combo_counts[cid])
                                combo_max[cid] = int(combo_counts[cid])
                                combo_cooldown[cid] = 0
                                reason = (
                                    "complete_by_stall_after_ratio_windows: "
                                    f"ratio={combo_ratio_now:.3f}>=threshold({stall_ratio:.3f}), "
                                    f"count={combo_counts[cid]}, target={combo_target_now:.1f}, "
                                    f"stall_windows={hits_now}/{stall_complete_hit_windows}, "
                                    f"window_consecutive_none={n_fail}"
                                )
                                combo_stall_finish_reason[cid] = reason
                                print(f"[stall-complete] finger={finger} combo={cid} {reason}")
                            else:
                                print(
                                    "[stall-window] "
                                    f"finger={finger} combo={cid} ratio={combo_ratio_now:.3f}>={stall_ratio:.3f}, "
                                    f"stall_windows={hits_now}/{stall_complete_hit_windows}, "
                                    f"window_consecutive_none={n_fail}; continue random sampling."
                                )
                            continue

                        combo_stall_over_ratio_hit_count[cid] = 0
                        used_retry = int(combo_stall_zero_retry_count[cid])
                        if used_retry < stall_max_zero_retry:
                            combo_stall_zero_retry_count[cid] = used_retry + 1
                            combo_target_none_consec[cid] = 0
                            combo_cooldown[cid] = int(TARGET_NONE_COOLDOWN_CYCLES)
                            print(
                                "[stall-recover] "
                                f"finger={finger} combo={cid} ratio={combo_ratio_now:.3f}<{stall_ratio:.3f}, "
                                f"consecutive_none={n_fail}, "
                                f"zero_retry={combo_stall_zero_retry_count[cid]}/{stall_max_zero_retry}; "
                                "run zero calibration and retry."
                            )

                            calib_result = _run_zero_calibration(node, defaults_hand, finger=finger)
                            zero_calib_total_events += 1
                            finger_zero_calib_events += 1
                            finger_state["finger_zero_calib_events"] = finger_zero_calib_events
                            if calib_result["success"]:
                                zero_calib_state = "ok"
                                zero_calib_fail_streak = 0
                                print(
                                    "[calibrate] zero calibration done. "
                                    f"attempts={calib_result['attempts']}/{calib_result['max_attempts']}"
                                )
                            else:
                                zero_calib_failed_events += 1
                                finger_zero_calib_failed_events += 1
                                finger_state["finger_zero_calib_failed_events"] = (
                                    finger_zero_calib_failed_events
                                )
                                zero_calib_fail_streak += 1
                                zero_calib_state = f"failed_after_{calib_result['max_attempts']}_attempts"
                                print(
                                    "[ERROR] zero calibration failed after max attempts. "
                                    f"attempts={calib_result['attempts']}/{calib_result['max_attempts']}, "
                                    f"last_max_abs_pos={calib_result['last_max_abs_pos']}, "
                                    "continue with label-tagged records."
                                )
                            _append_csv_row(
                                hand_root / "zero_calibration_events.csv",
                                _zero_calibration_event_headers(),
                                [
                                    global_point,
                                    finger,
                                    int(calib_result["success"]),
                                    calib_result["attempts"],
                                    calib_result["max_attempts"],
                                    calib_result["last_max_abs_pos"],
                                    zero_calib_state,
                                    zero_calib_fail_streak,
                                ],
                            )
                            continue

                        combo_min[cid] = int(combo_counts[cid])
                        combo_max[cid] = int(combo_counts[cid])
                        combo_target_none_consec[cid] = 0
                        combo_cooldown[cid] = 0
                        reason = (
                            "complete_by_stall_after_zero_retry_exhausted: "
                            f"ratio={combo_ratio_now:.3f}<{stall_ratio:.3f}, "
                            f"count={combo_counts[cid]}, target={combo_target_now:.1f}, "
                            f"consecutive_none={n_fail}, "
                            f"zero_retry={combo_stall_zero_retry_count[cid]}/{stall_max_zero_retry}"
                        )
                        combo_stall_finish_reason[cid] = reason
                        print(f"[stall-complete] finger={finger} combo={cid} {reason}")
                        continue

                    if (n_fail % max(1, int(TARGET_NONE_WARN_EVERY))) == 0:
                        print(
                            f"[warn-target] finger={finger} combo={cid} consecutive_target_none={n_fail} "
                            f"(current_count={combo_counts[cid]}, min={combo_min[cid]}, max={combo_max[cid]})"
                        )
                    if n_fail >= int(TARGET_NONE_MAX_CONSEC_PER_COMBO):
                        # Avoid dead-loop near local saturated/degenerate states:
                        # temporarily cooldown this combo and retry it later.
                        combo_cooldown[cid] = int(TARGET_NONE_COOLDOWN_CYCLES)
                        combo_target_none_consec[cid] = 0
                        print(
                            f"[cooldown-combo] finger={finger} combo={cid} "
                            f"hit {TARGET_NONE_MAX_CONSEC_PER_COMBO} target-none retries; "
                            f"cooldown={TARGET_NONE_COOLDOWN_CYCLES} cycles then retry."
                        )
                    continue
                combo_target_none_consec[combo.combo_id] = 0
                combo_stall_over_ratio_hit_count[combo.combo_id] = 0
                target = target_plan.target_by_motor
                used_mode = str(target_plan.active_mode).strip().lower() if target_plan.active_mode else selected_mode
                if used_mode not in mode_counts:
                    mode_counts[used_mode] = 0
                    mode_target[used_mode] = 0.0
                    mode_min[used_mode] = 0
                    mode_max[used_mode] = max(1, nominal_total)

                point_id += 1
                global_point += 1
                mode_counts[used_mode] += 1
                finger_state["point_id"] = point_id
                print(f"[point] {point_id}/{nominal_total} start")
                _persist_runtime_state(
                    phase="auto",
                    reason="point_start",
                    current_finger=finger,
                    current_combo=str(combo.combo_id),
                    current_stage="move",
                )
                point_interrupt_manual = False

                def _runtime_tick_during_point():
                    nonlocal runtime_paused, runtime_interrupted_finger
                    nonlocal requested_switch_finger, stop_requested
                    nonlocal point_interrupt_manual
                    nonlocal global_point, global_timeout
                    nonlocal zero_calib_state, zero_calib_fail_streak
                    nonlocal zero_calib_total_events, zero_calib_failed_events
                    nonlocal finger_zero_calib_events, finger_zero_calib_failed_events
                    while runtime_command_enabled:
                        raw_cmd = _poll_runtime_command()
                        if raw_cmd is None:
                            break
                        parts = str(raw_cmd).split()
                        if not parts:
                            continue
                        cmd = parts[0].strip().lower()
                        if cmd in ("h", "help", "?"):
                            _print_runtime_auto_help()
                            continue
                        if cmd == "p":
                            runtime_paused = True
                            runtime_interrupted_finger = finger
                            _run_zero_calibration_for_runtime("pause_command_point")
                            _pause_hold_current_for_runtime("pause_after_zero_point")
                            return {"kind": "pause_zero"}
                        if cmd == "f":
                            if len(parts) < 2:
                                print(f"[auto-control] usage: f <finger>, available={target_fingers}")
                                continue
                            nxt = str(parts[1]).strip().lower()
                            if nxt not in target_fingers:
                                print(f"[auto-control] unknown finger={nxt}, available={target_fingers}")
                                continue
                            requested_switch_finger = nxt
                            runtime_paused = False
                            return {"kind": "switch"}
                        if cmd == "r":
                            if runtime_interrupted_finger:
                                requested_switch_finger = str(runtime_interrupted_finger)
                                runtime_paused = False
                                print(f"[auto-control] resume interrupted finger={runtime_interrupted_finger}")
                                return {"kind": "switch"}
                            print("[auto-control] no interrupted finger to resume.")
                            continue
                        if cmd == "z":
                            _run_zero_calibration_for_runtime("runtime_command_point")
                            continue
                        if cmd == "t":
                            node.set_command_matrix(_build_uniform_current_matrix(-0.1))
                            runtime_paused = True
                            runtime_interrupted_finger = finger
                            print("[auto-control] all motors tighten current set to -0.1A (paused).")
                            return {"kind": "pause_hold"}
                        if cmd == "i":
                            node.set_command_matrix(_build_uniform_current_matrix(0.0))
                            runtime_paused = True
                            runtime_interrupted_finger = finger
                            print("[auto-control] all motors current set to 0A (paused).")
                            return {"kind": "pause_hold"}
                        if cmd == "e":
                            node.set_command_matrix(_build_uniform_current_matrix(0.0))
                            runtime_paused = True
                            runtime_interrupted_finger = finger
                            print("[auto-control] emergency stop applied (all currents=0), now paused.")
                            return {"kind": "emergency"}
                        if cmd == "m":
                            runtime_paused = True
                            runtime_interrupted_finger = finger
                            point_interrupt_manual = True
                            _pause_hold_current_for_runtime("manual_enter_point")
                            return {"kind": "manual"}
                        if cmd == "q":
                            stop_requested = True
                            print("[auto-control] safe exit requested.")
                            return {"kind": "stop"}
                        print(f"[auto-control] unknown command: {raw_cmd!r}")
                    return None

                outcome = _execute_point(
                    node=node,
                    defaults_hand=defaults_hand,
                    finger_motor_set=finger_motor_set,
                    finger=finger,
                    combo=combo,
                    target_by_motor=target,
                    active_motors=target_plan.active_motors,
                    active_mode=used_mode,
                    point_tolerance=POINT_ARRIVAL_TOL,
                    point_id=point_id,
                    nominal_total=nominal_total,
                    runtime_tick=_runtime_tick_during_point,
                )

                if bool(outcome.get("interrupted", False)):
                    point_id = max(0, point_id - 1)
                    global_point = max(0, global_point - 1)
                    mode_counts[used_mode] = max(0, int(mode_counts.get(used_mode, 0)) - 1)
                    finger_state["point_id"] = point_id
                    print(
                        f"[point-interrupt] finger={finger} "
                        f"reason={str(outcome.get('interrupt_reason', 'runtime_interrupt'))} "
                        f"point_rollback={point_id}/{nominal_total}"
                    )

                    if point_interrupt_manual:
                        _persist_runtime_state(
                            phase="manual",
                            reason="runtime_manual_enter",
                            current_finger=finger,
                            current_combo="",
                            current_stage="manual_wait",
                        )
                        run_counters = {
                            "global_point": int(global_point),
                            "global_timeout": int(global_timeout),
                        }
                        calib_state = {
                            "zero_calib_state": str(zero_calib_state),
                            "zero_calib_fail_streak": int(zero_calib_fail_streak),
                            "zero_calib_total_events": int(zero_calib_total_events),
                            "zero_calib_failed_events": int(zero_calib_failed_events),
                        }
                        _run_manual_supplement_loop(
                            enable=True,
                            node=node,
                            defaults_hand=defaults_hand,
                            finger_motor_set=finger_motor_set,
                            finger_states=finger_states,
                            rng=rng,
                            run_counters=run_counters,
                            calib_state=calib_state,
                            hand_root=hand_root,
                            confirm=False,
                        )
                        global_point = int(run_counters["global_point"])
                        global_timeout = int(run_counters["global_timeout"])
                        zero_calib_state = str(calib_state["zero_calib_state"])
                        zero_calib_fail_streak = int(calib_state["zero_calib_fail_streak"])
                        zero_calib_total_events = int(calib_state["zero_calib_total_events"])
                        zero_calib_failed_events = int(calib_state["zero_calib_failed_events"])
                        _persist_runtime_state(
                            phase="auto",
                            reason="runtime_manual_exit",
                            current_finger=finger,
                            current_combo="",
                            current_stage="resume_after_manual",
                        )
                    if stop_requested:
                        finger_interrupted = True
                        break
                    if requested_switch_finger and requested_switch_finger != finger:
                        runtime_switch_finger = str(requested_switch_finger)
                        runtime_interrupted_finger = finger
                        runtime_paused = False
                        pause_notice_printed = False
                        if finger not in target_fingers[finger_idx + 1 :]:
                            target_fingers.append(finger)
                        for _idx in range(finger_idx + 1, len(target_fingers)):
                            if str(target_fingers[_idx]) == runtime_switch_finger:
                                del target_fingers[_idx]
                                break
                        target_fingers.insert(finger_idx + 1, runtime_switch_finger)
                        finger_interrupted = True
                        print(f"[auto-control] switch finger: {finger} -> {runtime_switch_finger}")
                        break
                    continue

                combo_counts[combo.combo_id] += 1
                _mark_coverage_hit(combo_coverage[combo.combo_id], target_plan.coverage_key)
                if not outcome["reached"]:
                    global_timeout += 1
                    unreached.append(point_id)

                point_meta = {
                    "point_id": point_id,
                    "global_point_id": global_point,
                    "finger": finger,
                    "combo_id": combo.combo_id,
                    "reached": outcome["reached"],
                    "timeout": outcome["timeout"],
                    "move_elapsed_sec": outcome["move_elapsed_sec"],
                    "max_error": outcome["max_error"],
                    **_point_motion_metadata(outcome),
                    "active_mode": used_mode,
                    "active_motors": [int(mid) for mid in target_plan.active_motors],
                    "zero_calib_state": zero_calib_state,
                    "zero_calib_fail_streak": zero_calib_fail_streak,
                }

                _save_point_files(
                    finger_dirs=finger_dirs[finger],
                    point_id=point_id,
                    point_meta=point_meta,
                    process_glove_rows=outcome["process_glove_rows"],
                    process_glove_skeleton_rows=outcome["process_glove_skeleton_rows"],
                    process_fb_rows=outcome["process_fb_rows"],
                    target_cmd_matrix=outcome["target_cmd_matrix"],
                    result_glove_avg=outcome["result_glove_avg"],
                    result_glove_skeleton_avg=outcome["result_glove_skeleton_avg"],
                    result_fb_avg=outcome["result_fb_avg"],
                    result_window_label=outcome["result_window_label"],
                    result_sample_count=outcome["result_sample_count"],
                )

                _append_csv_row(
                    finger_dirs[finger]["finger_root"] / "point_log.csv",
                    _point_log_headers(),
                    [
                        point_meta["point_id"],
                        point_meta["global_point_id"],
                        point_meta["finger"],
                        point_meta["combo_id"],
                        int(point_meta["reached"]),
                        int(point_meta["timeout"]),
                        point_meta["move_elapsed_sec"],
                        point_meta["max_error"],
                        point_meta["active_mode"],
                        "|".join(str(v) for v in point_meta["active_motors"]),
                        point_meta["zero_calib_state"],
                        point_meta["zero_calib_fail_streak"],
                    ],
                )

                finger_summary = {
                    "finger": finger,
                    "updated_at": datetime.now().isoformat(),
                    "point_count": point_id,
                    "unreached_count": len(unreached),
                    "unreached_ratio": (len(unreached) / point_id) if point_id > 0 else 0.0,
                    "unreached_point_ids": unreached,
                    "combo_counts": combo_counts,
                    "combo_cooldown": combo_cooldown,
                    "combo_target_none_consecutive": combo_target_none_consec,
                    "combo_stall_zero_retry_count": combo_stall_zero_retry_count,
                    "combo_stall_over_ratio_hit_count": combo_stall_over_ratio_hit_count,
                    "combo_stall_finish_reason": combo_stall_finish_reason,
                    "finger_reopen_round": int(finger_reopen_round),
                    "finger_reopen_no_progress_round": int(finger_reopen_no_progress_round),
                    "edge_enhancement_done": bool(finger_state.get("edge_enhancement_done", False)),
                    "combo_coverage": {cid: _coverage_stats(combo_coverage[cid]) for cid in combo_counts},
                    "combo_min": combo_min,
                    "combo_max": combo_max,
                    "active_mode_counts": mode_counts,
                    "active_mode_target": mode_target,
                    "active_mode_min": mode_min,
                    "active_mode_max": mode_max,
                    "nominal_total": nominal_total,
                    "zero_calibration_state": zero_calib_state,
                    "zero_calibration_fail_streak": zero_calib_fail_streak,
                    "zero_calibration_events": finger_zero_calib_events,
                    "zero_calibration_failed_events": finger_zero_calib_failed_events,
                }
                (finger_dirs[finger]["finger_root"] / "finger_summary.json").write_text(
                    json.dumps(finger_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                print(
                    f"[point-done] {point_id}/{nominal_total} "
                    f"reached={outcome['reached']} max_err={outcome['max_error']:.3f}"
                )
                _persist_runtime_state(
                    phase="auto",
                    reason="point_done",
                    current_finger=finger,
                    current_combo=str(combo.combo_id),
                    current_stage="stable",
                )

                if (global_point % ZERO_CALIBRATE_EVERY_POINTS) == 0:
                    print(f"[calibrate] global_point={global_point}, running zero calibration...")
                    calib_result = _run_zero_calibration(node, defaults_hand, finger=finger)
                    zero_calib_total_events += 1
                    finger_zero_calib_events += 1
                    finger_state["finger_zero_calib_events"] = finger_zero_calib_events
                    if calib_result["success"]:
                        zero_calib_state = "ok"
                        zero_calib_fail_streak = 0
                        print(
                            "[calibrate] zero calibration done. "
                            f"attempts={calib_result['attempts']}/{calib_result['max_attempts']}"
                        )
                    else:
                        zero_calib_failed_events += 1
                        finger_zero_calib_failed_events += 1
                        finger_state["finger_zero_calib_failed_events"] = (
                            finger_zero_calib_failed_events
                        )
                        zero_calib_fail_streak += 1
                        zero_calib_state = f"failed_after_{calib_result['max_attempts']}_attempts"
                        print(
                            "[ERROR] zero calibration failed after max attempts. "
                            f"attempts={calib_result['attempts']}/{calib_result['max_attempts']}, "
                            f"last_max_abs_pos={calib_result['last_max_abs_pos']}, "
                            "continue with label-tagged records."
                        )
                    _append_csv_row(
                        hand_root / "zero_calibration_events.csv",
                        _zero_calibration_event_headers(),
                        [
                            global_point,
                            finger,
                            int(calib_result["success"]),
                            calib_result["attempts"],
                            calib_result["max_attempts"],
                            calib_result["last_max_abs_pos"],
                            zero_calib_state,
                            zero_calib_fail_streak,
                        ],
                    )
                    # Flush finger summary immediately so calibration labels are visible
                    # even if this finger ends right after calibration.
                    finger_summary_after_calib = {
                        "finger": finger,
                        "updated_at": datetime.now().isoformat(),
                        "point_count": point_id,
                        "unreached_count": len(unreached),
                        "unreached_ratio": (len(unreached) / point_id) if point_id > 0 else 0.0,
                        "unreached_point_ids": unreached,
                        "combo_counts": combo_counts,
                        "combo_cooldown": combo_cooldown,
                        "combo_target_none_consecutive": combo_target_none_consec,
                        "combo_stall_zero_retry_count": combo_stall_zero_retry_count,
                        "combo_stall_over_ratio_hit_count": combo_stall_over_ratio_hit_count,
                        "combo_stall_finish_reason": combo_stall_finish_reason,
                        "finger_reopen_round": int(finger_reopen_round),
                        "finger_reopen_no_progress_round": int(finger_reopen_no_progress_round),
                        "edge_enhancement_done": bool(finger_state.get("edge_enhancement_done", False)),
                        "combo_coverage": {cid: _coverage_stats(combo_coverage[cid]) for cid in combo_counts},
                        "combo_min": combo_min,
                        "combo_max": combo_max,
                        "active_mode_counts": mode_counts,
                        "active_mode_target": mode_target,
                        "active_mode_min": mode_min,
                        "active_mode_max": mode_max,
                        "nominal_total": nominal_total,
                        "zero_calibration_state": zero_calib_state,
                        "zero_calibration_fail_streak": zero_calib_fail_streak,
                        "zero_calibration_events": finger_zero_calib_events,
                        "zero_calibration_failed_events": finger_zero_calib_failed_events,
                    }
                    (finger_dirs[finger]["finger_root"] / "finger_summary.json").write_text(
                        json.dumps(finger_summary_after_calib, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

            if (not finger_interrupted) and (not bool(finger_state.get("edge_enhancement_done", False))):
                edge_run_counters = {
                    "global_point": int(global_point),
                    "global_timeout": int(global_timeout),
                }
                edge_calib_state = {
                    "zero_calib_state": str(zero_calib_state),
                    "zero_calib_fail_streak": int(zero_calib_fail_streak),
                    "zero_calib_total_events": int(zero_calib_total_events),
                    "zero_calib_failed_events": int(zero_calib_failed_events),
                }
                _persist_runtime_state(
                    phase="auto",
                    reason="edge_enhancement_start",
                    current_finger=finger,
                    current_combo="",
                    current_stage="edge_enhance",
                )
                _run_post_random_edge_enhancement(
                    node=node,
                    defaults_hand=defaults_hand,
                    finger_motor_set=finger_motor_set,
                    finger_state=finger_state,
                    run_counters=edge_run_counters,
                    calib_state=edge_calib_state,
                    rng=rng,
                )
                global_point = int(edge_run_counters["global_point"])
                global_timeout = int(edge_run_counters["global_timeout"])
                zero_calib_state = str(edge_calib_state["zero_calib_state"])
                zero_calib_fail_streak = int(edge_calib_state["zero_calib_fail_streak"])
                zero_calib_total_events = int(edge_calib_state["zero_calib_total_events"])
                zero_calib_failed_events = int(edge_calib_state["zero_calib_failed_events"])
                point_id = int(finger_state["point_id"])
                finger_zero_calib_events = int(finger_state["finger_zero_calib_events"])
                finger_zero_calib_failed_events = int(finger_state["finger_zero_calib_failed_events"])
                finger_reopen_last_point_id = int(point_id)
                finger_state["edge_enhancement_done"] = True
                _persist_runtime_state(
                    phase="auto",
                    reason="edge_enhancement_done",
                    current_finger=finger,
                    current_combo="",
                    current_stage="edge_enhance_done",
                )

            finger_summary_final = {
                "finger": finger,
                "updated_at": datetime.now().isoformat(),
                "point_count": point_id,
                "unreached_count": len(unreached),
                "unreached_ratio": (len(unreached) / point_id) if point_id > 0 else 0.0,
                "unreached_point_ids": unreached,
                "combo_counts": combo_counts,
                "combo_cooldown": combo_cooldown,
                "combo_target_none_consecutive": combo_target_none_consec,
                "combo_stall_zero_retry_count": combo_stall_zero_retry_count,
                "combo_stall_over_ratio_hit_count": combo_stall_over_ratio_hit_count,
                "combo_stall_finish_reason": combo_stall_finish_reason,
                "finger_reopen_round": int(finger_reopen_round),
                "finger_reopen_no_progress_round": int(finger_reopen_no_progress_round),
                "edge_enhancement_done": bool(finger_state.get("edge_enhancement_done", False)),
                "combo_coverage": {cid: _coverage_stats(combo_coverage[cid]) for cid in combo_counts},
                "combo_min": combo_min,
                "combo_max": combo_max,
                "active_mode_counts": mode_counts,
                "active_mode_target": mode_target,
                "active_mode_min": mode_min,
                "active_mode_max": mode_max,
                "nominal_total": nominal_total,
                "zero_calibration_state": zero_calib_state,
                "zero_calibration_fail_streak": zero_calib_fail_streak,
                "zero_calibration_events": finger_zero_calib_events,
                "zero_calibration_failed_events": finger_zero_calib_failed_events,
            }
            finger_state["point_id"] = point_id
            finger_state["finger_zero_calib_events"] = finger_zero_calib_events
            finger_state["finger_zero_calib_failed_events"] = finger_zero_calib_failed_events
            finger_state["finger_reopen_last_point_id"] = int(finger_reopen_last_point_id)
            finger_state["done"] = (not finger_interrupted)
            (finger_dirs[finger]["finger_root"] / "finger_summary.json").write_text(
                json.dumps(finger_summary_final, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            if finger_interrupted:
                print(
                    f"[finger-paused] finger={finger} point={point_id}/{nominal_total} "
                    f"unreached={len(unreached)}"
                )
                _persist_runtime_state(
                    phase="auto",
                    reason="finger_paused",
                    current_finger=finger,
                    current_combo="",
                    current_stage="finger_paused",
                )
                if stop_requested:
                    break
            else:
                print(
                    f"[finger-done] finger={finger} total={point_id} "
                    f"unreached={len(unreached)} ratio={len(unreached)/max(1,point_id):.4f}"
                )
                _persist_runtime_state(
                    phase="auto",
                    reason="finger_done",
                    current_finger=finger,
                    current_combo="",
                    current_stage="finger_done",
                )

            if stop_requested:
                break

        auto_global_point = int(global_point)
        auto_global_timeout = int(global_timeout)
        if not stop_requested:
            run_counters = {
                "global_point": int(global_point),
                "global_timeout": int(global_timeout),
            }
            calib_state = {
                "zero_calib_state": str(zero_calib_state),
                "zero_calib_fail_streak": int(zero_calib_fail_streak),
                "zero_calib_total_events": int(zero_calib_total_events),
                "zero_calib_failed_events": int(zero_calib_failed_events),
            }
            _persist_runtime_state(
                phase="manual",
                reason="manual_supplement_enter",
                current_finger="",
                current_combo="",
                current_stage="manual_wait",
            )
            _run_manual_supplement_loop(
                enable=manual_supplement_after_run,
                node=node,
                defaults_hand=defaults_hand,
                finger_motor_set=finger_motor_set,
                finger_states=finger_states,
                rng=rng,
                run_counters=run_counters,
                calib_state=calib_state,
                hand_root=hand_root,
            )
            global_point = int(run_counters["global_point"])
            global_timeout = int(run_counters["global_timeout"])
            zero_calib_state = str(calib_state["zero_calib_state"])
            zero_calib_fail_streak = int(calib_state["zero_calib_fail_streak"])
            zero_calib_total_events = int(calib_state["zero_calib_total_events"])
            zero_calib_failed_events = int(calib_state["zero_calib_failed_events"])
            _persist_runtime_state(
                phase="auto",
                reason="manual_supplement_exit",
                current_finger="",
                current_combo="",
                current_stage="post_manual",
            )

        total_elapsed = time.monotonic() - global_start
        global_summary = {
            "hand": hand,
            "target_finger": args.target_finger,
            "executed_fingers": target_fingers,
            "created_at": datetime.now().isoformat(),
            "run_root": str(run_root),
            "resume_enabled": bool(resume_enabled),
            "resumed_from_checkpoint": bool(resuming),
            "resume_file": str(resume_file),
            "live_status_file": str(live_status_file),
            "global_points": global_point,
            "global_timeouts": global_timeout,
            "global_timeout_ratio": global_timeout / max(1, global_point),
            "elapsed_sec": total_elapsed,
            "avg_point_per_sec": global_point / max(1e-6, total_elapsed),
            "zero_calibration_state": zero_calib_state,
            "zero_calibration_fail_streak": zero_calib_fail_streak,
            "zero_calibration_events": zero_calib_total_events,
            "zero_calibration_failed_events": zero_calib_failed_events,
            "zero_calibration_events_file": str(hand_root / "zero_calibration_events.csv"),
            "manual_supplement_enabled": bool(manual_supplement_after_run),
            "manual_supplement_points": max(0, int(global_point) - int(auto_global_point)),
            "manual_supplement_timeouts": max(0, int(global_timeout) - int(auto_global_timeout)),
            "startup_first_frame_files": startup_first_frame_files,
        }
        (hand_root / "run_summary.json").write_text(
            json.dumps(global_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[data_collection] finished. summary={hand_root / 'run_summary.json'}")
        _persist_runtime_state(
            phase="finished",
            reason="run_finished",
            current_finger="",
            current_combo="",
            current_stage="finished",
        )

    except KeyboardInterrupt:
        stop_requested = True
        print("\n[data_collection] KeyboardInterrupt received. 已保存的数据不会丢失，安全退出中...")
        _persist_runtime_state(
            phase="interrupted",
            reason="keyboard_interrupt",
            current_stage="interrupt",
        )
    finally:
        # On exit, fall back to pure tor tighten (no position control).
        try:
            node.set_command_matrix(_build_idle_tor_matrix(defaults_hand))
            time.sleep(0.2)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        if stop_requested:
            print("[data_collection] exited by user interrupt.")
        _persist_runtime_state(
            phase="stopped",
            reason="process_exit",
            current_stage="stopped",
        )


if __name__ == "__main__":
    if DEVICES_MSG_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Cannot import devices_pkg.msg. Please source ROS2 workspace before running."
        ) from DEVICES_MSG_IMPORT_ERROR
    main()
