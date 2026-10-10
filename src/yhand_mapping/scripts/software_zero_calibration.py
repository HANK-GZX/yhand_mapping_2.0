#!/usr/bin/env python3
"""Back-delta software-zero reference collection and automatic zeroing."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
except ImportError as exc:  # pragma: no cover
    rclpy = None
    ExternalShutdownException = Exception
    Node = object
    RCLPY_IMPORT_ERROR = exc
else:
    RCLPY_IMPORT_ERROR = None

try:
    from devices_pkg.msg import Motor, YHandMsg
except ImportError as exc:  # pragma: no cover
    Motor = None
    YHandMsg = None
    DEVICES_MSG_IMPORT_ERROR = exc
else:
    DEVICES_MSG_IMPORT_ERROR = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from software_zero_common import (  # noqa: E402
    BACK_DELTA_REFERENCE_KIND,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DEFAULTS_PATH,
    MOTOR_COUNT,
    build_state_payload,
    current_boot_id,
    default_reference_path,
    default_state_path,
    hand_groups,
    load_muscle_defaults,
    load_state,
    load_zero_config,
    normalize_hand,
    resolve_pkg_path,
    save_json_atomic,
    zero_offset_from_state,
)


MODE_CHOICES = ("collect-reference", "calibrate-auto", "show-reference", "show-state")
HAND_ARG_CHOICES = ("left", "right", "both")
_UNSET = object()

CURRENT_DEFAULTS = {
    "max_current_abs_a": 0.5,
    "release_current_a": 0.1,
    "tension_current_a": -0.1,
}

CONTROL_DEFAULTS = {
    "control_period_sec": 0.02,
    "position_torque_key": "tor",
    "position_hard_default_abs_deg": 2000.0,
    "ready_timeout_sec": 8.0,
    "restore_release_sec": 0.0,
    "restore_arrival_tolerance_deg": 7.0,
    "restore_arrival_confirm_frames": 30,
    "restore_move_timeout_sec": 6.0,
    "wrist_back_auto_front_release_sec": 8.0,
    "wrist_back_auto_tension_current_a": -0.2,
    "wrist_back_auto_side_tension_current_a": -0.1,
    "wrist_back_auto_other_tension_current_a": -0.1,
    "wrist_back_auto_contract_window_sec": 5.0,
    "wrist_back_auto_contract_threshold_deg": 3.0,
    "wrist_back_auto_contract_timeout_sec": 10.0,
    "wrist_back_auto_all_tension_sec": 5.0,
    "wrist_back_auto_all_tension_stable_window_sec": 3.0,
    "wrist_back_auto_all_tension_stable_threshold_deg": 3.0,
    "wrist_back_auto_all_tension_stable_timeout_sec": 15.0,
    "final_zero_tension_sec": 3.0,
    "final_zero_tension_stable_window_sec": 5.0,
    "final_zero_tension_stable_threshold_deg": 3.0,
    "final_zero_tension_stable_timeout_sec": 10.0,
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _config_section(config: dict, section_name: str) -> dict:
    section = config.get(section_name, {})
    return section if isinstance(section, dict) else {}


def _param_value(values: dict, defaults: dict, key: str, *, fallback_key: str | None = None, default=_UNSET):
    if key in values:
        return values[key]
    if fallback_key is not None and fallback_key in values:
        return values[fallback_key]
    if default is not _UNSET:
        return default
    if key in defaults:
        return defaults[key]
    if fallback_key is not None and fallback_key in defaults:
        return defaults[fallback_key]
    raise KeyError(f"missing parameter default for {key!r}")


def _control_value(control: dict, key: str, *, fallback_key: str | None = None, default=_UNSET):
    return _param_value(control, CONTROL_DEFAULTS, key, fallback_key=fallback_key, default=default)


def _control_float(control: dict, key: str, *, fallback_key: str | None = None, default=_UNSET) -> float:
    return float(_control_value(control, key, fallback_key=fallback_key, default=default))


def _control_int(control: dict, key: str, *, fallback_key: str | None = None, default=_UNSET) -> int:
    return int(_control_value(control, key, fallback_key=fallback_key, default=default))


def _control_bool(control: dict, key: str, *, fallback_key: str | None = None, default=_UNSET) -> bool:
    return bool(_control_value(control, key, fallback_key=fallback_key, default=default))


def _current_float(currents: dict, key: str, *, default=_UNSET) -> float:
    return float(_param_value(currents, CURRENT_DEFAULTS, key, default=default))


def _json_clone(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def _now() -> float:
    return time.monotonic()


def _uniq(values: list[int]) -> list[int]:
    return sorted({int(v) for v in values})


def _spin_node(node: Node) -> None:
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        return


# ---------------------------------------------------------------------------
# ROS I/O node
# ---------------------------------------------------------------------------

class SoftwareZeroControlNode(Node):
    def __init__(self, *, hand: str, cmd_topic: str, feedback_topic: str, defaults: dict[str, list[float]], config: dict):
        super().__init__(f"software_zero_calibration_{hand}")
        if DEVICES_MSG_IMPORT_ERROR is not None:
            raise RuntimeError(f"Cannot import devices_pkg messages: {DEVICES_MSG_IMPORT_ERROR}")

        self.hand = normalize_hand(hand)
        self.cmd_topic = str(cmd_topic)
        self.feedback_topic = str(feedback_topic)
        self.defaults = defaults
        self.config = config
        control = _config_section(config, "control")
        currents = _config_section(config, "currents")
        self.control_period_sec = _control_float(control, "control_period_sec")
        self.max_current_abs = _current_float(currents, "max_current_abs_a")
        pos_tor_key = str(_control_value(control, "position_torque_key")).strip()
        self.position_torque_key = pos_tor_key if pos_tor_key in ("tor", "tor_active") else "tor"

        self._lock = threading.Lock()
        self.latest_pos = [0.0] * MOTOR_COUNT
        self.feedback_received = False
        self.last_feedback_ts = 0.0
        self.command_matrix = self.build_current_matrix({}, default_current=0.0)

        self.publisher = self.create_publisher(YHandMsg, self.cmd_topic, 10)
        self.feedback_sub = self.create_subscription(YHandMsg, self.feedback_topic, self._feedback_cb, 10)
        self.timer = self.create_timer(self.control_period_sec, self._publish_cb)
        self.get_logger().info(
            f"[software_zero_calibration] hand={self.hand}, cmd={self.cmd_topic}, feedback={self.feedback_topic}"
        )

    def _feedback_cb(self, msg) -> None:
        motors = list(getattr(msg, "motors", []) or [])
        now = _now()
        with self._lock:
            for i in range(min(MOTOR_COUNT, len(motors))):
                self.latest_pos[i] = float(motors[i].pos)
            self.feedback_received = True
            self.last_feedback_ts = now

    def _publish_cb(self) -> None:
        with self._lock:
            matrix = copy.deepcopy(self.command_matrix)
        msg = YHandMsg()
        msg.motors = [
            Motor(pos=float(row[0]), vel=float(row[1]), tor=float(row[2]), kp=float(row[3]), kd=float(row[4]))
            for row in matrix
        ]
        msg.pretightenforce.flag = False
        msg.pretightenforce.torque = 0.0
        msg.pretightenforce.ms = 0
        self.publisher.publish(msg)

    def _clamp_current(self, value: float) -> float:
        value = float(value)
        return max(-self.max_current_abs, min(self.max_current_abs, value))

    def build_current_matrix(self, current_by_motor: dict[int, float], *, default_current: float = 0.0) -> list[list[float]]:
        matrix = []
        for mid in range(1, MOTOR_COUNT + 1):
            current = current_by_motor.get(mid, default_current)
            matrix.append([0.0, 0.0, self._clamp_current(current), 0.0, 0.0])
        return matrix

    def build_mixed_matrix(
        self,
        *,
        targets_by_motor: dict[int, float] | None = None,
        current_by_motor: dict[int, float] | None = None,
        default_current: float = 0.0,
        position_current: float | None = None,
    ) -> list[list[float]]:
        targets_by_motor = dict(targets_by_motor or {})
        current_by_motor = dict(current_by_motor or {})
        matrix = []
        for mid in range(1, MOTOR_COUNT + 1):
            idx = mid - 1
            if mid in targets_by_motor:
                if position_current is None:
                    tor_values = self.defaults.get(
                        self.position_torque_key,
                        self.defaults.get("tor", [0.0] * MOTOR_COUNT),
                    )
                    tor_cmd = float(tor_values[idx])
                else:
                    tor_cmd = float(position_current)
                matrix.append([
                    float(targets_by_motor[mid]),
                    float(self.defaults["vel"][idx]),
                    self._clamp_current(tor_cmd),
                    float(self.defaults["kp"][idx]),
                    float(self.defaults["kd"][idx]),
                ])
            else:
                current = current_by_motor.get(mid, default_current)
                matrix.append([0.0, 0.0, self._clamp_current(current), 0.0, 0.0])
        return matrix

    def set_command_matrix(self, matrix: list[list[float]]) -> None:
        with self._lock:
            self.command_matrix = copy.deepcopy(matrix)

    def wait_feedback(self, timeout_sec: float) -> bool:
        start = _now()
        while _now() - start < float(timeout_sec):
            with self._lock:
                if self.feedback_received:
                    return True
            time.sleep(0.05)
        return False

    def positions(self) -> list[float]:
        with self._lock:
            return [float(v) for v in self.latest_pos]

    def feedback_age(self) -> float:
        with self._lock:
            if not self.feedback_received:
                return float("inf")
            return _now() - self.last_feedback_ts


# ---------------------------------------------------------------------------
# Back-delta runner
# ---------------------------------------------------------------------------

class BackDeltaRunner:
    def __init__(self, *, hand: str, node: SoftwareZeroControlNode, config: dict, config_path: Path, defaults_path: Path):
        self.hand = normalize_hand(hand)
        self.node = node
        self.config = config
        self.config_path = config_path
        self.defaults_path = defaults_path
        self.currents = _config_section(config, "currents")
        self.control = _config_section(config, "control")
        self.position_hard_default_abs_deg = abs(self.control_float("position_hard_default_abs_deg"))
        self.position_hard_limits = self._load_position_hard_limits()
        self.motor_groups = self._load_back_delta_motor_groups()
        print(
            "[software-zero] back-delta profile: "
            f"hand={self.hand}, hard_limit_motors={len(self.position_hard_limits)}, "
            f"fallback_abs={self.position_hard_default_abs_deg:.1f}deg"
        )

    def control_value(self, key: str, *, fallback_key: str | None = None, default=_UNSET):
        return _control_value(self.control, key, fallback_key=fallback_key, default=default)

    def control_float(self, key: str, *, fallback_key: str | None = None, default=_UNSET) -> float:
        return _control_float(self.control, key, fallback_key=fallback_key, default=default)

    def control_int(self, key: str, *, fallback_key: str | None = None, default=_UNSET) -> int:
        return _control_int(self.control, key, fallback_key=fallback_key, default=default)

    def control_bool(self, key: str, *, fallback_key: str | None = None, default=_UNSET) -> bool:
        return _control_bool(self.control, key, fallback_key=fallback_key, default=default)

    def current_float(self, key: str, *, default=_UNSET) -> float:
        return _current_float(self.currents, key, default=default)

    @property
    def release_current(self) -> float:
        return self.current_float("release_current_a")

    @property
    def tension_current(self) -> float:
        return self.current_float("tension_current_a")

    def _load_position_hard_limits(self) -> dict[int, tuple[float, float]]:
        root = self.config.get("position_hard_limits_deg", {})
        if not isinstance(root, dict):
            return {}
        hand_obj = root.get(self.hand, root)
        if not isinstance(hand_obj, dict):
            return {}
        out: dict[int, tuple[float, float]] = {}
        for mid in range(1, MOTOR_COUNT + 1):
            raw = hand_obj.get(str(mid), hand_obj.get(mid))
            if not isinstance(raw, list) or len(raw) != 2:
                continue
            lo, hi = float(raw[0]), float(raw[1])
            if lo < hi:
                out[mid] = (lo, hi)
        return out

    def _load_back_delta_motor_groups(self) -> dict[str, list[int]]:
        root = self.config.get("back_delta_motor_groups", {})
        hand_cfg = root.get(self.hand, {}) if isinstance(root, dict) else {}
        return {
            "front_release": _uniq(list(hand_cfg.get("front_release", []))),
            "rear_contract": _uniq(list(hand_cfg.get("rear_contract", []))),
            "rear_side": _uniq(list(hand_cfg.get("rear_side", []))),
        }

    def clip_position_target(self, value: float, motor_id: int) -> float:
        mid = int(motor_id)
        v = float(value)
        bounds = self.position_hard_limits.get(mid)
        if bounds is not None:
            lo, hi = bounds
            return max(lo, min(hi, v))
        hard = float(self.position_hard_default_abs_deg)
        if hard <= 0.0:
            return v
        return max(-hard, min(hard, v))

    def clip_targets(self, targets: dict[int, float]) -> dict[int, float]:
        return {int(mid): self.clip_position_target(value, int(mid)) for mid, value in targets.items()}

    def prompt(self, message: str) -> None:
        try:
            input(f"\n{message}\n确认后按 Enter 继续。")
        except EOFError as exc:
            raise RuntimeError("当前运行环境不可读取标准输入（stdin）。") from exc

    def snapshot(self, label: str) -> list[float]:
        pos = self.node.positions()
        ts = datetime.now().isoformat(timespec="milliseconds")
        print(f"[software-zero] recorded {label}; time={ts}; feedback_age={self.node.feedback_age():.3f}s")
        return pos

    def set_mixed(
        self,
        targets: dict[int, float] | None = None,
        currents: dict[int, float] | None = None,
        *,
        default_current: float = 0.0,
        position_current: float | None = None,
    ) -> None:
        self.node.set_command_matrix(
            self.node.build_mixed_matrix(
                targets_by_motor=targets or {},
                current_by_motor=currents or {},
                default_current=default_current,
                position_current=position_current,
            )
        )

    def _run_timed_phase(
        self,
        *,
        targets: dict[int, float],
        currents: dict[int, float],
        duration_sec: float,
        default_current: float,
        label: str,
    ) -> None:
        print(
            "[software-zero] timed phase: "
            f"label={label}, duration={duration_sec:.1f}s, targets={len(targets)}, currents={currents}"
        )
        self.set_mixed(targets=targets, currents=currents, default_current=default_current)
        end = _now() + max(0.0, float(duration_sec))
        while _now() < end:
            time.sleep(0.05)

    def _wait_stable_current(
        self,
        *,
        watch_motor_ids: list[int],
        currents: dict[int, float],
        default_current: float,
        window_sec: float,
        threshold_deg: float,
        timeout_sec: float,
        label: str,
    ) -> bool:
        watch_ids = _uniq(watch_motor_ids)
        self.set_mixed(targets={}, currents=currents, default_current=default_current)
        samples: list[tuple[float, list[float]]] = []
        start = _now()
        while _now() - start < float(timeout_sec):
            now = _now()
            pos = self.node.positions()
            samples.append((now, pos))
            cutoff = now - float(window_sec)
            samples = [(ts, values) for ts, values in samples if ts >= cutoff]
            if samples and now - samples[0][0] >= float(window_sec):
                ready = True
                for mid in watch_ids:
                    values = [values[mid - 1] for _ts, values in samples]
                    if max(values) - min(values) > float(threshold_deg):
                        ready = False
                        break
                if ready:
                    print(f"[software-zero] stable ok: {label}")
                    return True
            time.sleep(0.05)
        print(f"[software-zero] stable timeout: {label}; continuing with current feedback")
        return False

    def current_tension_with_stable_check(
        self,
        *,
        watch_motor_ids: list[int],
        currents: dict[int, float],
        label: str,
        hold_sec: float,
        window_sec: float,
        threshold_deg: float,
        timeout_sec: float,
    ) -> bool:
        watch_ids = _uniq(watch_motor_ids)
        current_map = {int(mid): float(value) for mid, value in dict(currents).items()}
        print(
            "[software-zero] current tension stable-check: "
            f"label={label}, watch={watch_ids}, hold={hold_sec:.1f}s, "
            f"window={window_sec:.1f}s, threshold={threshold_deg:.1f}deg, timeout={timeout_sec:.1f}s"
        )
        if float(hold_sec) > 1e-9:
            self._run_timed_phase(
                targets={},
                currents=current_map,
                duration_sec=float(hold_sec),
                default_current=0.0,
                label=f"{label}_hold",
            )
        return self._wait_stable_current(
            watch_motor_ids=watch_ids,
            currents=current_map,
            default_current=0.0,
            window_sec=float(window_sec),
            threshold_deg=float(threshold_deg),
            timeout_sec=float(timeout_sec),
            label=label,
        )

    def wait_arrival(
        self,
        targets: dict[int, float],
        *,
        timeout_sec: float,
        tolerance_deg: float,
        confirm_sec: float,
        label: str,
    ) -> bool:
        wait_targets = {int(mid): float(value) for mid, value in targets.items()}
        start = _now()
        confirm_start: float | None = None
        while _now() - start < float(timeout_sec):
            pos = self.node.positions()
            max_err = 0.0
            for mid, target in wait_targets.items():
                max_err = max(max_err, abs(float(pos[mid - 1]) - target))
            if max_err <= float(tolerance_deg):
                if confirm_start is None:
                    confirm_start = _now()
                if _now() - confirm_start >= float(confirm_sec):
                    print(f"[software-zero] arrival ok: {label}, max_err={max_err:.3f}deg")
                    return True
            else:
                confirm_start = None
            time.sleep(0.05)
        print(f"[software-zero] arrival timeout: {label}; continuing with current feedback")
        return False

    def move_targets(
        self,
        targets: dict[int, float],
        *,
        default_current: float = 0.0,
        arrival_tolerance_deg: float,
        arrival_confirm_sec: float,
        arrival_timeout_sec: float,
        label: str,
    ) -> None:
        clipped = self.clip_targets(targets)
        self.set_mixed(targets=clipped, currents={}, default_current=default_current)
        self.wait_arrival(
            clipped,
            timeout_sec=arrival_timeout_sec,
            tolerance_deg=arrival_tolerance_deg,
            confirm_sec=arrival_confirm_sec,
            label=label,
        )

    def restore_to(self, target_positions: list[float], *, label: str) -> None:
        release_sec = self.control_float("restore_release_sec")
        tol = self.control_float("restore_arrival_tolerance_deg")
        timeout = self.control_float("restore_move_timeout_sec")
        confirm_frames = max(1, self.control_int("restore_arrival_confirm_frames"))
        confirm_sec = confirm_frames * max(0.001, float(self.node.control_period_sec))
        print(
            f"[software-zero] restore whole hand to {label}: "
            f"pre_release={self.release_current:+.3f}A {release_sec:.1f}s + "
            f"all-motor position control (tol={tol:.2f}, confirm_frames={confirm_frames}, timeout={timeout:.2f}s)"
        )
        if release_sec > 1e-9:
            self._run_timed_phase(
                targets={},
                currents={mid: self.release_current for mid in range(1, MOTOR_COUNT + 1)},
                duration_sec=release_sec,
                default_current=0.0,
                label=f"restore_release_{label}",
            )
        targets = {mid: target_positions[mid - 1] for mid in range(1, MOTOR_COUNT + 1)}
        self.move_targets(
            targets,
            default_current=0.0,
            arrival_tolerance_deg=tol,
            arrival_timeout_sec=timeout,
            arrival_confirm_sec=confirm_sec,
            label=f"restore_{label}",
        )

    def restore_to_zero_offset(self, zero_offset: list[float], *, label: str) -> list[float]:
        if len(zero_offset) != MOTOR_COUNT:
            raise ValueError(f"zero_offset length must be {MOTOR_COUNT}, got {len(zero_offset)}")
        target_positions = [self.clip_position_target(float(zero_offset[i]), i + 1) for i in range(MOTOR_COUNT)]
        print(f"[software-zero] restore to computed zero pose: {label}")
        self.restore_to(target_positions, label=label)
        return target_positions

    def final_zero_tension(self, *, label: str) -> list[float]:
        sec = self.control_float("final_zero_tension_sec")
        currents = {mid: self.tension_current for mid in range(1, MOTOR_COUNT + 1)}
        print(f"[software-zero] final zero all-motor tension: {self.tension_current:+.3f}A, {sec:.1f}s")
        self.current_tension_with_stable_check(
            watch_motor_ids=list(range(1, MOTOR_COUNT + 1)),
            currents=currents,
            label=label,
            hold_sec=sec,
            window_sec=self.control_float("final_zero_tension_stable_window_sec"),
            threshold_deg=self.control_float("final_zero_tension_stable_threshold_deg"),
            timeout_sec=self.control_float("final_zero_tension_stable_timeout_sec"),
        )
        return self.snapshot(label)

    def run_wrist_back_auto(self, *, label: str) -> list[float]:
        front_release_ids = _uniq(self.motor_groups["front_release"])
        rear_contract_ids = _uniq(self.motor_groups["rear_contract"])
        rear_side_ids = _uniq(self.motor_groups["rear_side"])
        if not front_release_ids or not rear_contract_ids:
            raise RuntimeError(f"back-delta motor group is empty for hand={self.hand}")

        release_sec = self.control_float("wrist_back_auto_front_release_sec")
        tension_current = self.control_float("wrist_back_auto_tension_current_a", default=self.tension_current)
        side_current = self.control_float("wrist_back_auto_side_tension_current_a")
        other_tension_current = float(self.control_value("wrist_back_auto_other_tension_current_a", default=self.tension_current))
        rear_contract_set = set(rear_contract_ids)
        rear_side_set = set(rear_side_ids)
        rear_currents = {
            mid: (side_current if mid in rear_side_set else tension_current)
            for mid in rear_contract_ids
        }
        sync_currents = {mid: self.release_current for mid in front_release_ids}
        sync_currents.update(rear_currents)
        print(
            "[software-zero] wrist-back sync phase: "
            f"front_release={front_release_ids}, release_current={self.release_current:+.3f}A, "
            f"rear_contract={rear_contract_ids}, side_motors={rear_side_ids}, "
            f"side_current={side_current:+.3f}A, default_current={tension_current:+.3f}A, sec={release_sec:.1f}"
        )
        self._run_timed_phase(
            targets={},
            currents=sync_currents,
            duration_sec=release_sec,
            default_current=0.0,
            label=f"{label}_front_release_rear_contract_sync",
        )
        print("[software-zero] wrist-back rear contraction stable gate after front release")
        self._wait_stable_current(
            watch_motor_ids=rear_contract_ids,
            currents=rear_currents,
            default_current=0.0,
            window_sec=self.control_float("wrist_back_auto_contract_window_sec"),
            threshold_deg=self.control_float("wrist_back_auto_contract_threshold_deg"),
            timeout_sec=self.control_float("wrist_back_auto_contract_timeout_sec"),
            label=f"{label}_rear_contract",
        )
        all_tension_ids = _uniq(front_release_ids + rear_contract_ids)
        all_tension_currents = {}
        for mid in all_tension_ids:
            if mid in rear_side_set:
                all_tension_currents[mid] = side_current
            elif mid in rear_contract_set:
                all_tension_currents[mid] = tension_current
            else:
                all_tension_currents[mid] = other_tension_current
        post_sec = self.control_float("wrist_back_auto_all_tension_sec")
        print(
            "[software-zero] wrist-back final all-motor tension: "
            f"motors={all_tension_ids}, side_current={side_current:+.3f}A, "
            f"default_current={tension_current:+.3f}A, other_current={other_tension_current:+.3f}A, sec={post_sec:.1f}"
        )
        self.current_tension_with_stable_check(
            watch_motor_ids=all_tension_ids,
            currents=all_tension_currents,
            label=f"{label}_all_tension",
            hold_sec=post_sec,
            window_sec=self.control_float("wrist_back_auto_all_tension_stable_window_sec"),
            threshold_deg=self.control_float("wrist_back_auto_all_tension_stable_threshold_deg"),
            timeout_sec=self.control_float("wrist_back_auto_all_tension_stable_timeout_sec"),
        )
        return self.snapshot(label)


# ---------------------------------------------------------------------------
# Back-delta data model and workflows
# ---------------------------------------------------------------------------

def _hands_from_arg(hand_arg: str) -> list[str]:
    value = str(hand_arg or "").strip().lower()
    if value == "both":
        return ["left", "right"]
    return [normalize_hand(value)]


def _per_hand_path(raw_path: str, *, config: dict, hand: str, both: bool, default_fn, label: str) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        return default_fn(config, hand)
    if both:
        if "{hand}" not in raw:
            raise ValueError(f"{label} must contain '{{hand}}' when --hand both is used")
        raw = raw.format(hand=hand)
    return resolve_pkg_path(raw)


def _require_pose(raw, *, label: str) -> list[float]:
    if not isinstance(raw, list) or len(raw) != MOTOR_COUNT:
        raise ValueError(f"{label} must be list(len={MOTOR_COUNT}), got {type(raw).__name__}")
    return [float(v) for v in raw]


def _pose_delta(back_pose: list[float], initial_pose: list[float]) -> list[float]:
    return [float(back_pose[i]) - float(initial_pose[i]) for i in range(MOTOR_COUNT)]


def _pose_subtract_delta(back_pose: list[float], delta: list[float]) -> list[float]:
    return [float(back_pose[i]) - float(delta[i]) for i in range(MOTOR_COUNT)]


def _print_pose_summary(label: str, values: list[float]) -> None:
    vals = [float(v) for v in values]
    mean = sum(vals) / len(vals)
    print(f"[back-delta] {label}: min={min(vals):.3f}, max={max(vals):.3f}, mean={mean:.3f}")
    top = sorted(((idx + 1, abs(v), v) for idx, v in enumerate(vals)), key=lambda row: row[1], reverse=True)[:10]
    print(f"[back-delta] {label} top abs motors:")
    for mid, _abs_v, value in top:
        print(f"  m{mid:02d}: {value:+.3f}deg")


def print_offset_summary(zero_offset: list[float], hand: str) -> None:
    groups = hand_groups(hand)
    print(
        "[software-zero] zero_offset summary: "
        f"min={min(zero_offset):.3f}, max={max(zero_offset):.3f}, "
        f"mean={sum(zero_offset) / len(zero_offset):.3f}"
    )
    for name in ("wrist", "four_finger", "thumb"):
        mids = groups[name]
        values = [zero_offset[mid - 1] for mid in mids]
        print(f"  {name}: motors={mids}, range=[{min(values):.3f}, {max(values):.3f}]")


def _build_reference_payload(
    *,
    hand: str,
    initial_pose: list[float],
    back_pose: list[float],
    delta: list[float],
    exit_zero_target_pose: list[float],
    exit_true_zero_pose: list[float],
    metadata: dict,
) -> dict:
    return {
        "version": "1.0",
        "kind": BACK_DELTA_REFERENCE_KIND,
        "hand": normalize_hand(hand),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "boot_id": current_boot_id(),
        "back_delta_data": {
            "initial_power_on": [float(v) for v in initial_pose],
            "back_max": [float(v) for v in back_pose],
            "delta_back_minus_initial": [float(v) for v in delta],
            "exit_zero_target_pose": [float(v) for v in exit_zero_target_pose],
            "exit_true_zero_pose": [float(v) for v in exit_true_zero_pose],
        },
        "metadata": dict(metadata),
    }


def _load_back_delta_reference(path: Path, hand: str) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if str(obj.get("kind", "")).strip() != BACK_DELTA_REFERENCE_KIND:
        raise ValueError(f"not a {BACK_DELTA_REFERENCE_KIND} file: {path}")
    if normalize_hand(obj.get("hand", "")) != normalize_hand(hand):
        raise ValueError(f"reference hand mismatch: expected {hand}, got {obj.get('hand')}")
    data = obj.get("back_delta_data", {})
    initial = _require_pose(data.get("initial_power_on"), label="back_delta_data.initial_power_on")
    back = _require_pose(data.get("back_max"), label="back_delta_data.back_max")
    delta = _require_pose(data.get("delta_back_minus_initial"), label="back_delta_data.delta_back_minus_initial")
    return {"payload": obj, "initial": initial, "back": back, "delta": delta}


def _runtime_metadata(
    *,
    mode: str,
    hand: str,
    cmd_topic: str,
    feedback_topic: str,
    config: dict,
    config_path: Path,
    defaults_path: Path,
    reference_path: Path,
    state_path: Path,
) -> dict:
    return {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "mode": str(mode),
        "hand": str(hand),
        "topics": {"cmd_topic": str(cmd_topic), "feedback_topic": str(feedback_topic)},
        "paths": {
            "config_path": str(config_path),
            "defaults_path": str(defaults_path),
            "reference_path": str(reference_path),
            "state_path": str(state_path),
        },
        "back_delta_motor_groups": _json_clone(config.get("back_delta_motor_groups", {})).get(hand, {}),
        "control": _json_clone(config.get("control", {})),
        "currents": _json_clone(config.get("currents", {})),
        "groups": hand_groups(hand),
    }


def collect_reference(runner: BackDeltaRunner, *, reference_path: Path, runtime_meta: dict) -> None:
    runner.prompt("请确认当前是上电后的初始/近零位姿态；该位置会作为 back-delta 参考零位")
    initial_pose = runner.snapshot("initial_power_on")

    print("[back-delta] reference back-bend action")
    back_pose = runner.run_wrist_back_auto(label="back_max_reference")
    delta = _pose_delta(back_pose, initial_pose)

    _print_pose_summary("reference initial_power_on", initial_pose)
    _print_pose_summary("reference back_max", back_pose)
    _print_pose_summary("reference delta=back-initial", delta)

    print("[back-delta] reference exit: restore 28 motors to initial_power_on, then tension all 28 motors")
    exit_zero_target_pose = runner.restore_to_zero_offset(initial_pose, label="back_delta_reference_initial_power_on_exit")
    exit_true_zero_pose = runner.final_zero_tension(label="back_delta_reference_ZERO_final_all_tension")
    _print_pose_summary("reference exit true zero pose", exit_true_zero_pose)

    payload = _build_reference_payload(
        hand=runner.hand,
        initial_pose=initial_pose,
        back_pose=back_pose,
        delta=delta,
        exit_zero_target_pose=exit_zero_target_pose,
        exit_true_zero_pose=exit_true_zero_pose,
        metadata={
            "config_path": str(runner.config_path),
            "defaults_path": str(runner.defaults_path),
            "runtime": runtime_meta,
            "zero_formula": "zero_estimate = auto_back_max - reference_delta_back_minus_initial",
        },
    )
    save_json_atomic(reference_path, payload)
    print(f"[back-delta] saved reference: {reference_path}")


def calibrate_auto(runner: BackDeltaRunner, *, reference_path: Path, state_path: Path, runtime_meta: dict) -> None:
    ref = _load_back_delta_reference(reference_path, runner.hand)
    ref_delta = ref["delta"]
    _print_pose_summary("loaded reference delta=back-initial", ref_delta)

    runner.prompt("即将执行 back-delta 自动找零；确认手腕后弯和回零路径安全")
    print("[back-delta] auto back-bend action")
    back_now = runner.run_wrist_back_auto(label="back_max_auto")
    zero_estimate = _pose_subtract_delta(back_now, ref_delta)

    _print_pose_summary("auto back_max", back_now)
    _print_pose_summary("computed zero_estimate=back_now-reference_delta", zero_estimate)
    print_offset_summary(zero_estimate, runner.hand)

    print("[back-delta] auto exit: restore 28 motors to computed zero, then tension all 28 motors")
    zero_target_pose = runner.restore_to_zero_offset(zero_estimate, label="back_delta_ZERO_calc_pose")
    true_zero_pose = runner.final_zero_tension(label="back_delta_ZERO_final_all_tension")

    runtime_data = {
        "reference_initial_power_on": ref["initial"],
        "reference_back_max": ref["back"],
        "reference_delta_back_minus_initial": ref_delta,
        "auto_back_max": back_now,
        "zero_estimate": zero_estimate,
        "zero_target_pose": zero_target_pose,
        "true_zero_pose": true_zero_pose,
    }
    payload = build_state_payload(
        runner.hand,
        "back-delta-auto",
        list(true_zero_pose),
        runtime_data,
        reference_path,
        runner.config_path,
        metadata={
            "note": "Back-delta software-zero state; regenerate after every reboot before using proxy.",
            "runtime": runtime_meta,
            "zero_formula": "zero_estimate = auto_back_max - reference_delta_back_minus_initial",
        },
    )
    save_json_atomic(state_path, payload)
    print(f"[back-delta] saved state: {state_path}")


def show_reference(reference_path: Path, hand: str) -> int:
    try:
        ref = _load_back_delta_reference(reference_path, hand)
    except FileNotFoundError:
        print(f"[back-delta] reference file not found: {reference_path}")
        return 2
    print(f"[back-delta] reference_path={reference_path}")
    _print_pose_summary("initial_power_on", ref["initial"])
    _print_pose_summary("back_max", ref["back"])
    _print_pose_summary("delta=back-initial", ref["delta"])
    return 0


def show_state(state_path: Path, hand: str) -> int:
    try:
        state = load_state(state_path, hand)
    except FileNotFoundError:
        print(f"[back-delta] state file not found: {state_path}")
        return 2
    offset = zero_offset_from_state(state)
    print(f"[back-delta] state_path={state_path}")
    print(f"[back-delta] mode={state.get('mode')}, created_at={state.get('created_at')}")
    print_offset_summary(offset, hand)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODE_CHOICES, required=True)
    parser.add_argument("--hand", choices=HAND_ARG_CHOICES, required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--defaults-path", default=str(DEFAULT_DEFAULTS_PATH))
    parser.add_argument("--cmd-topic", default="")
    parser.add_argument("--feedback-topic", default="")
    parser.add_argument("--reference-path", default="")
    parser.add_argument("--state-path", default="")
    return parser.parse_args()


def _run_motion_for_hand(
    args: argparse.Namespace,
    *,
    hand: str,
    config: dict,
    config_path: Path,
    defaults_path: Path,
    reference_path: Path,
    state_path: Path,
) -> int:
    cmd_topic = args.cmd_topic.strip() or f"/yhand_{hand}_raw_cmd"
    feedback_topic = args.feedback_topic.strip() or f"/yhand_{hand}_raw_feedback"
    defaults = load_muscle_defaults(defaults_path, hand)

    node = None
    spin_thread = None
    try:
        node = SoftwareZeroControlNode(
            hand=hand,
            cmd_topic=cmd_topic,
            feedback_topic=feedback_topic,
            defaults=defaults,
            config=config,
        )
        spin_thread = threading.Thread(target=_spin_node, args=(node,), daemon=True)
        spin_thread.start()

        ready_timeout = _control_float(_config_section(config, "control"), "ready_timeout_sec")
        if not node.wait_feedback(timeout_sec=ready_timeout):
            print(f"错误: {ready_timeout:.1f}s 内没有收到反馈。请确认 driver 发布 {feedback_topic}，并订阅 {cmd_topic}。")
            return 2

        runner = BackDeltaRunner(
            hand=hand,
            node=node,
            config=config,
            config_path=config_path,
            defaults_path=defaults_path,
        )
        runtime_meta = _runtime_metadata(
            mode=f"back-delta-{args.mode}",
            hand=hand,
            cmd_topic=cmd_topic,
            feedback_topic=feedback_topic,
            config=config,
            config_path=config_path,
            defaults_path=defaults_path,
            reference_path=reference_path,
            state_path=state_path,
        )
        print(
            "[back-delta] run profile: "
            f"mode={args.mode}, hand={hand}, cmd={cmd_topic}, feedback={feedback_topic}, "
            f"reference={reference_path}, state={state_path}"
        )

        if args.mode == "collect-reference":
            collect_reference(runner, reference_path=reference_path, runtime_meta=runtime_meta)
        elif args.mode == "calibrate-auto":
            calibrate_auto(runner, reference_path=reference_path, state_path=state_path, runtime_meta=runtime_meta)
        else:
            raise RuntimeError(f"unsupported mode: {args.mode}")
        return 0
    finally:
        if node is not None:
            node.set_command_matrix(node.build_current_matrix({}, default_current=0.0))
            time.sleep(0.1)
            node.destroy_node()
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)


def main() -> int:
    args = parse_args()
    hands = _hands_from_arg(args.hand)
    both = len(hands) > 1
    config_path = resolve_pkg_path(args.config)
    defaults_path = resolve_pkg_path(args.defaults_path)
    config = load_zero_config(config_path)

    if both and (args.cmd_topic.strip() or args.feedback_topic.strip()):
        print("错误: --hand both 时不要传单一 cmd/feedback topic；请使用默认 /yhand_<hand>_raw_* 话题。")
        return 2

    try:
        paths_by_hand = {
            hand: {
                "reference": _per_hand_path(
                    args.reference_path,
                    config=config,
                    hand=hand,
                    both=both,
                    default_fn=default_reference_path,
                    label="reference_path",
                ),
                "state": _per_hand_path(
                    args.state_path,
                    config=config,
                    hand=hand,
                    both=both,
                    default_fn=default_state_path,
                    label="state_path",
                ),
            }
            for hand in hands
        }
    except ValueError as exc:
        print(f"错误: {exc}")
        return 2

    if args.mode == "show-reference":
        rc = 0
        for hand in hands:
            rc = max(rc, show_reference(paths_by_hand[hand]["reference"], hand))
        return rc
    if args.mode == "show-state":
        rc = 0
        for hand in hands:
            rc = max(rc, show_state(paths_by_hand[hand]["state"], hand))
        return rc

    if RCLPY_IMPORT_ERROR is not None:
        print(f"错误: 无法导入 rclpy，请先 source ROS2 环境: {RCLPY_IMPORT_ERROR}")
        return 2

    rclpy.init()
    try:
        for hand in hands:
            print(f"[back-delta] ===== start hand={hand} =====")
            rc = _run_motion_for_hand(
                args,
                hand=hand,
                config=config,
                config_path=config_path,
                defaults_path=defaults_path,
                reference_path=paths_by_hand[hand]["reference"],
                state_path=paths_by_hand[hand]["state"],
            )
            if rc != 0:
                return rc
            print(f"[back-delta] ===== finished hand={hand} =====")
        return 0
    except KeyboardInterrupt:
        print("\n[back-delta] interrupted")
        return 130
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
