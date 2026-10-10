#!/usr/bin/env python3
import argparse
import importlib.util
import select
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

try:
    from muscle_adjustment_defaults import DEFAULTS as MUSCLE_ADJUSTMENT_DEFAULTS
except ImportError as exc:
    MUSCLE_ADJUSTMENT_DEFAULTS = None
    MUSCLE_DEFAULTS_IMPORT_ERROR = exc
else:
    MUSCLE_DEFAULTS_IMPORT_ERROR = None

try:
    from devices_pkg.msg import Motor, YHandMsg
except ImportError as exc:
    Motor = None
    YHandMsg = None
    DEVICES_MSG_IMPORT_ERROR = exc
else:
    DEVICES_MSG_IMPORT_ERROR = None

# 发布频率 50 Hz.
from hand_motor_map import (
    named_motors,
)

PUBLISH_PERIOD_SEC = 0.02
DEFAULT_KD = 0.0
DEFAULT_KP = 0.0
# 非主动电机的保底张紧电流
DEFAULT_HOLD_CURRENT = -0.01
# 初始阶段默认电流（使用张紧电流）
INIT_HOLD_CURRENT = DEFAULT_HOLD_CURRENT
# Pretighten is disabled
PRETIGHTEN_FLAG = False
PRETIGHTEN_TORQUE = 0.0
PRETIGHTEN_MS = 0
MOTOR_COUNT = 28

# 初始状态保持时间，单位秒
INIT_DURATION_SEC = 0.5
# Action 1 中握拳的持续时间
FIST_DURATION_SEC = 5.0
# Action 1: 四指深屈/屈肌收线，其余电机放线。
ACTION1_FIST_CURRENT = -0.20
ACTION1_RELEASE_CURRENT = 0.05
# 动作结束后的保持电流沿用原配置，不随握拳阶段电流改变。
POST_ACTION_FIST_PRIMARY_SOURCE_CURRENT = -0.45
POST_ACTION_FIST_INDEX_ASSIST_SOURCE_CURRENT = -0.30
POST_ACTION_FIST_RING_SOURCE_CURRENT = -0.62
ACTION1_OPEN_RING_EXT_CURRENT = -0.56
# Action 2: 先回零后直接进入位控“张开-闭合”循环（持续循环，手动停止）
ACTION2_TOGGLE_INTERVAL_SEC = 2
ACTION2_CYCLE_SEC = 2.0 * ACTION2_TOGGLE_INTERVAL_SEC
ACTION2_OPEN_EXT_CURRENT = -0.20
ACTION2_OPEN_FLEX_CURRENT = 0.20
ACTION2_CLOSE_EXT_CURRENT = 0.20
ACTION2_CLOSE_FLEX_CURRENT = -0.15

# Action 3: 往复运动（先对掌，再回初始位姿，持续循环，手动停止）
ACTION3_OPPOSITION_CURRENT = -0.45
ACTION3_ABDUCTION_CURRENT = -0.5
# 长屈电流保持较小，避免弯曲过猛
ACTION3_LONG_FLEX_CURRENT = -0.45
# 左手新对掌电流；右手沿用原参数。
ACTION3_LEFT_OPPOSITION_CURRENT = -0.25
ACTION3_LEFT_ABDUCTION_CURRENT = -0.20
ACTION3_LEFT_LONG_FLEX_CURRENT = -0.25
ACTION3_LEFT_RELEASE_CURRENT = 0.10
ACTION3_OPPOSITION_HOLD_SEC = 2.4
ACTION3_RETURN_HOLD_SEC = 1.0
ACTION3_CYCLE_SEC = ACTION3_OPPOSITION_HOLD_SEC + ACTION3_RETURN_HOLD_SEC

WAIT_LOG_INTERVAL_SEC = 2.0

# 动作执行完成后的保持电流参数（防止“松掉”）
POST_ACTION_HOLD_SCALE = 0.20
POST_ACTION_HOLD_MIN_ABS_CURRENT = 0.05
POST_ACTION_HOLD_MAX_ABS_CURRENT = 0.12

# Action 0: 位置控制回零参数
ACTION0_TARGET_POS = 0.0
ACTION0_POS_TOL = 14.0
ACTION0_CONFIRM_FRAMES = 5
ACTION0_TIMEOUT_SEC = 6.0

# 位控阶段无名指参数放大系数（基于 muscle_adjustment_defaults 默认值）
RING_POS_CTRL_VEL_SCALE = 1.15
RING_POS_CTRL_KP_SCALE = 1.45
RING_POS_CTRL_KD_SCALE = 1.30
RING_POS_CTRL_TOR_SCALE = 1.45

# Action 2 额外无名指位控增强（在位控基准增强后再次放大）
ACTION2_RING_CLOSE_VEL_EXTRA_SCALE = 1.25
ACTION2_RING_CLOSE_KP_EXTRA_SCALE = 1.20
ACTION2_RING_CLOSE_KD_EXTRA_SCALE = 1.15
ACTION2_RING_CLOSE_TOR_EXTRA_SCALE = 1.20

# 小拇指位控轻微降档（避免运动过快/过硬）
PINKY_POS_CTRL_VEL_SCALE = 0.92
PINKY_POS_CTRL_KP_SCALE = 0.90
PINKY_POS_CTRL_KD_SCALE = 0.92
PINKY_POS_CTRL_TOR_SCALE = 0.90


def _topics_for_hand(hand: str) -> Tuple[str, str]:
    if hand == "left":
        return "/yhand_left_cmd", "/yhand_left_feedback"
    if hand == "right":
        return "/yhand_right_cmd", "/yhand_right_feedback"
    raise ValueError(f"Unsupported hand={hand}, expected 'left' or 'right'.")


# 左手电机编号
LEFT_HAND_MOTORS = named_motors('left')

# 右手电机编号（与 right.py 的 CONTROL_MAP 对齐）
RIGHT_HAND_MOTORS = named_motors('right')

HAND_MOTORS_BY_SIDE = {
    "left": LEFT_HAND_MOTORS,
    "right": RIGHT_HAND_MOTORS,
}


def _build_action_currents(hand_motors: Dict[str, int]):
    active_flexion = {
        "index_deep": hand_motors["index_flex_deep"],
        "middle_deep": hand_motors["middle_flex_deep"],
        "ring": hand_motors["ring_flex"],
        "pinky": hand_motors["pinky_flex"],
    }
    active_extension = {
        "index": hand_motors["index_ext"],
        "middle": hand_motors["middle_ext"],
        "ring": hand_motors["ring_ext"],
        "pinky": hand_motors["pinky_ext"],
    }

    fist_current = {motor_id: ACTION1_FIST_CURRENT for motor_id in active_flexion.values()}
    open_current: Dict[int, float] = {
        active_extension["index"]: -0.45,
        active_extension["middle"]: -0.45,
        active_extension["ring"]: ACTION1_OPEN_RING_EXT_CURRENT,
        active_extension["pinky"]: -0.45,
    }
    return fist_current, open_current, active_flexion, active_extension


def _build_small_hold_currents(source_current: Dict[int, float]) -> Dict[int, float]:
    hold_current: Dict[int, float] = {}
    for motor_id, current in source_current.items():
        sign = 1.0 if float(current) >= 0.0 else -1.0
        abs_scaled = abs(float(current)) * POST_ACTION_HOLD_SCALE
        abs_scaled = max(abs_scaled, POST_ACTION_HOLD_MIN_ABS_CURRENT)
        abs_scaled = min(abs_scaled, POST_ACTION_HOLD_MAX_ABS_CURRENT)
        hold_current[motor_id] = sign * abs_scaled
    return hold_current


def _build_fist_hold_currents(hand_motors: Dict[str, int]) -> Dict[int, float]:
    return _build_small_hold_currents({
        hand_motors["index_flex_deep"]: POST_ACTION_FIST_PRIMARY_SOURCE_CURRENT,
        hand_motors["index_flex_superficial"]: POST_ACTION_FIST_INDEX_ASSIST_SOURCE_CURRENT,
        hand_motors["middle_flex_deep"]: POST_ACTION_FIST_PRIMARY_SOURCE_CURRENT,
        hand_motors["ring_flex"]: POST_ACTION_FIST_RING_SOURCE_CURRENT,
        hand_motors["pinky_flex"]: POST_ACTION_FIST_PRIMARY_SOURCE_CURRENT,
    })


def _load_position_control_defaults(hand: str) -> Dict[str, list[float]]:
    defaults_map = MUSCLE_ADJUSTMENT_DEFAULTS
    if defaults_map is None:
        defaults_path = Path(__file__).resolve().with_name("muscle_adjustment_defaults.py")
        if defaults_path.exists():
            spec = importlib.util.spec_from_file_location(
                "muscle_adjustment_defaults_dynamic",
                str(defaults_path),
            )
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                defaults_map = getattr(mod, "DEFAULTS", None)

    if defaults_map is None:
        raise RuntimeError(
            "Cannot import muscle_adjustment_defaults.DEFAULTS. "
            "Please check scripts/muscle_adjustment_defaults.py."
        ) from MUSCLE_DEFAULTS_IMPORT_ERROR

    hand_defaults = defaults_map.get(hand)
    if not isinstance(hand_defaults, dict):
        raise RuntimeError(
            f"Cannot find hand defaults in muscle_adjustment_defaults.py: hand={hand}"
        )

    out: Dict[str, list[float]] = {}
    for key, fallback in (
        ("vel", 0.0),
        ("kp", 500.0),
        ("kd", 10.0),
        ("tor", DEFAULT_HOLD_CURRENT),
    ):
        values = list(hand_defaults.get(key, []))
        if len(values) < MOTOR_COUNT:
            values += [fallback] * (MOTOR_COUNT - len(values))
        elif len(values) > MOTOR_COUNT:
            values = values[:MOTOR_COUNT]

        normalized: list[float] = []
        for value in values:
            try:
                normalized.append(float(value))
            except (TypeError, ValueError):
                normalized.append(float(fallback))
        out[key] = normalized
    return out


class ManusCalibrationNode(Node):
    def __init__(
        self,
        hand: str,
        command_topic: str,
        feedback_topic: str,
        feedback_timeout_sec: float,
    ):
        super().__init__(f"manus_calibration_{hand}")
        if YHandMsg is None or Motor is None:
            raise RuntimeError(
                "Cannot import devices_pkg.msg (YHandMsg/Motor). "
                "Please source the ROS2 workspace before running this script."
            )

        self.hand = hand
        self.command_topic = command_topic
        self.feedback_topic = feedback_topic
        self.feedback_timeout_sec = float(feedback_timeout_sec)

        hand_motors = HAND_MOTORS_BY_SIDE[self.hand]
        self.thumb_opposition_motor = int(hand_motors["thumb_opposition"])
        self.thumb_abduction_motor = int(hand_motors["thumb_abd"])
        self.thumb_long_flex_motor = int(hand_motors["thumb_long_flex"])
        self.action3_lock_motors = {
            int(hand_motors[name])
            for name in (
                "index_ext",
                "index_flex_superficial",
                "index_flex_deep",
                "index_abd",
                "index_add",
                "middle_ext",
                "middle_flex_superficial",
                "middle_flex_deep",
                "middle_abd",
                "middle_add",
                "ring_ext",
                "ring_flex",
                "ring_abd",
                "ring_add",
                "pinky_ext",
                "pinky_flex",
                "pinky_abd",
                "pinky_add",
            )
        }
        self.fist_current, self.open_current, self.active_flexion, self.active_extension = _build_action_currents(
            hand_motors
        )
        self.fist_hold_current = _build_fist_hold_currents(hand_motors)
        self.open_hold_current = _build_small_hold_currents(self.open_current)
        self.action3_thumb_currents = {
            self.thumb_opposition_motor: ACTION3_OPPOSITION_CURRENT,
            self.thumb_abduction_motor: ACTION3_ABDUCTION_CURRENT,
            self.thumb_long_flex_motor: ACTION3_LONG_FLEX_CURRENT,
        }
        if self.hand == "left":
            self.action3_thumb_currents = {
                self.thumb_opposition_motor: ACTION3_LEFT_OPPOSITION_CURRENT,
                self.thumb_abduction_motor: ACTION3_LEFT_ABDUCTION_CURRENT,
                self.thumb_long_flex_motor: ACTION3_LEFT_LONG_FLEX_CURRENT,
                int(hand_motors["thumb_short_ext"]): ACTION3_LEFT_RELEASE_CURRENT,
                int(hand_motors["thumb_long_ext"]): ACTION3_LEFT_RELEASE_CURRENT,
                int(hand_motors["thumb_add"]): ACTION3_LEFT_RELEASE_CURRENT,
            }
        self.action3_thumb_motors = set(self.action3_thumb_currents)
        self.position_ctrl_defaults = _load_position_control_defaults(self.hand)
        self.position_ctrl_vel = [float(v) for v in self.position_ctrl_defaults["vel"]]
        self.position_ctrl_kp = [float(v) for v in self.position_ctrl_defaults["kp"]]
        self.position_ctrl_kd = [float(v) for v in self.position_ctrl_defaults["kd"]]
        self.position_ctrl_tor = [float(v) for v in self.position_ctrl_defaults["tor"]]
        self.ring_position_ctrl_motors = {
            int(hand_motors["ring_ext"]),
            int(hand_motors["ring_flex"]),
            int(hand_motors["ring_abd"]),
            int(hand_motors["ring_add"]),
        }
        self.pinky_position_ctrl_motors = {
            int(hand_motors["pinky_ext"]),
            int(hand_motors["pinky_flex"]),
            int(hand_motors["pinky_abd"]),
            int(hand_motors["pinky_add"]),
        }
        self._boost_ring_position_control_params()
        self._attenuate_pinky_position_control_params()
        self.action2_position_ctrl_vel = [float(v) for v in self.position_ctrl_vel]
        self.action2_position_ctrl_kp = [float(v) for v in self.position_ctrl_kp]
        self.action2_position_ctrl_kd = [float(v) for v in self.position_ctrl_kd]
        self.action2_position_ctrl_tor = [float(v) for v in self.position_ctrl_tor]
        self._boost_action2_ring_close_params()
        self.action2_open_position_ctrl_tor = list(self.position_ctrl_tor)
        for motor_id in self.active_extension.values():
            self.action2_open_position_ctrl_tor[motor_id - 1] = ACTION2_OPEN_EXT_CURRENT
            self.action2_position_ctrl_tor[motor_id - 1] = ACTION2_CLOSE_EXT_CURRENT
        for motor_id in (
            *self.active_flexion.values(),
            hand_motors["index_flex_superficial"],
            hand_motors["middle_flex_superficial"],
        ):
            self.action2_open_position_ctrl_tor[motor_id - 1] = ACTION2_OPEN_FLEX_CURRENT
        for motor_id in self.active_flexion.values():
            self.action2_position_ctrl_tor[motor_id - 1] = ACTION2_CLOSE_FLEX_CURRENT
        self.action0_near_zero_frames = 0
        self.action1_recorded_positions = [0.0] * MOTOR_COUNT
        self.has_action1_recorded_positions = False
        self.action3_hold_positions = [0.0] * MOTOR_COUNT
        self.has_action3_hold_positions = False

        self.current_positions = [0.0] * MOTOR_COUNT
        self.current_velocities = [0.0] * MOTOR_COUNT
        self.has_feedback = False
        self._state_lock = threading.Lock()
        self.current_action = None
        self.idle_mode = "idle"
        self.action_done_event = threading.Event()
        self.action_done_message = ""
        self.phase = "init"
        self.phase_start_time = time.monotonic()
        self.last_logged_phase = ""
        self.stop_requested = False
        self.feedback_wait_start_ts = time.monotonic()
        self.last_wait_log_ts = 0.0
        self.bootstrap_send_count = 0

        self.cmd_publisher = self.create_publisher(YHandMsg, self.command_topic, 10)
        self.create_subscription(YHandMsg, self.feedback_topic, self.feedback_callback, 10)
        self.timer = self.create_timer(PUBLISH_PERIOD_SEC, self.timer_callback)

        self.get_logger().info(f"hand={self.hand}")
        self.get_logger().info(f"Command topic: {self.command_topic}")
        self.get_logger().info(f"Feedback topic: {self.feedback_topic}")
        self.get_logger().info(
            "Strategy: action-dependent control. "
            "Current-loop actions use kp/kd=0; position-control actions use per-motor defaults."
        )
        self.get_logger().info(
            f"Active flexion motors: {self.active_flexion}, active extension motors: {self.active_extension}"
        )
        self.get_logger().info(
            f"Action 3 lock motors (4 fingers): {sorted(self.action3_lock_motors)}, "
            f"thumb_opposition={self.thumb_opposition_motor}, "
            f"thumb_abduction={self.thumb_abduction_motor}, thumb_long_flex={self.thumb_long_flex_motor}"
        )
        self.get_logger().info(f"Action 3 current commands (A): {self.action3_thumb_currents}")
        self.get_logger().info(
            "Position-control params loaded from muscle_adjustment_defaults.py "
            "(per-motor vel/kp/kd/tor)."
        )
        self.get_logger().info(
            f"Ring-finger boost enabled: motors={sorted(self.ring_position_ctrl_motors)}, "
            f"action1_flex={ACTION1_FIST_CURRENT:.3f}, "
            f"action1_ext={ACTION1_OPEN_RING_EXT_CURRENT:.3f}, "
            f"vel_scale={RING_POS_CTRL_VEL_SCALE:.2f}, "
            f"kp_scale={RING_POS_CTRL_KP_SCALE:.2f}, kd_scale={RING_POS_CTRL_KD_SCALE:.2f}, "
            f"tor_scale={RING_POS_CTRL_TOR_SCALE:.2f}"
        )
        self.get_logger().info(
            f"Action2 ring extra boost: vel={ACTION2_RING_CLOSE_VEL_EXTRA_SCALE:.2f}, "
            f"kp={ACTION2_RING_CLOSE_KP_EXTRA_SCALE:.2f}, "
            f"kd={ACTION2_RING_CLOSE_KD_EXTRA_SCALE:.2f}, "
            f"tor={ACTION2_RING_CLOSE_TOR_EXTRA_SCALE:.2f}"
        )
        self.get_logger().info(
            f"Pinky-finger attenuation enabled: motors={sorted(self.pinky_position_ctrl_motors)}, "
            f"vel_scale={PINKY_POS_CTRL_VEL_SCALE:.2f}, "
            f"kp_scale={PINKY_POS_CTRL_KP_SCALE:.2f}, kd_scale={PINKY_POS_CTRL_KD_SCALE:.2f}, "
            f"tor_scale={PINKY_POS_CTRL_TOR_SCALE:.2f}"
        )

    def _boost_ring_position_control_params(self):
        for motor_id in self.ring_position_ctrl_motors:
            idx = motor_id - 1
            self.position_ctrl_vel[idx] = float(self.position_ctrl_vel[idx]) * RING_POS_CTRL_VEL_SCALE
            self.position_ctrl_kp[idx] = float(self.position_ctrl_kp[idx]) * RING_POS_CTRL_KP_SCALE
            self.position_ctrl_kd[idx] = float(self.position_ctrl_kd[idx]) * RING_POS_CTRL_KD_SCALE
            self.position_ctrl_tor[idx] = float(self.position_ctrl_tor[idx]) * RING_POS_CTRL_TOR_SCALE

    def _attenuate_pinky_position_control_params(self):
        for motor_id in self.pinky_position_ctrl_motors:
            idx = motor_id - 1
            self.position_ctrl_vel[idx] = float(self.position_ctrl_vel[idx]) * PINKY_POS_CTRL_VEL_SCALE
            self.position_ctrl_kp[idx] = float(self.position_ctrl_kp[idx]) * PINKY_POS_CTRL_KP_SCALE
            self.position_ctrl_kd[idx] = float(self.position_ctrl_kd[idx]) * PINKY_POS_CTRL_KD_SCALE
            self.position_ctrl_tor[idx] = float(self.position_ctrl_tor[idx]) * PINKY_POS_CTRL_TOR_SCALE

    def _boost_action2_ring_close_params(self):
        for motor_id in self.ring_position_ctrl_motors:
            idx = motor_id - 1
            self.action2_position_ctrl_vel[idx] = (
                float(self.action2_position_ctrl_vel[idx]) * ACTION2_RING_CLOSE_VEL_EXTRA_SCALE
            )
            self.action2_position_ctrl_kp[idx] = (
                float(self.action2_position_ctrl_kp[idx]) * ACTION2_RING_CLOSE_KP_EXTRA_SCALE
            )
            self.action2_position_ctrl_kd[idx] = (
                float(self.action2_position_ctrl_kd[idx]) * ACTION2_RING_CLOSE_KD_EXTRA_SCALE
            )
            self.action2_position_ctrl_tor[idx] = (
                float(self.action2_position_ctrl_tor[idx]) * ACTION2_RING_CLOSE_TOR_EXTRA_SCALE
            )

    def start_action(self, action_index: str):
        if action_index not in {"0", "1", "2", "3"}:
            raise ValueError(f"Unsupported action index: {action_index}")
        with self._state_lock:
            if self.current_action is not None:
                raise RuntimeError("Previous action is still running.")
            self.current_action = action_index
            self.phase = "init"
            self.phase_start_time = time.monotonic()
            self.last_logged_phase = ""
            self.action0_near_zero_frames = 0
            self.stop_requested = False
            self.action_done_message = ""
            self.action_done_event.clear()
        self.get_logger().info(f"Action {action_index} started.")

    def _finish_action(self, idle_mode: str, message: str):
        with self._state_lock:
            self.current_action = None
            self.stop_requested = False
            self.idle_mode = idle_mode
            self.action_done_message = message
            self.action_done_event.set()
        self.get_logger().info(message)

    def request_stop_for_loop_actions(self) -> bool:
        with self._state_lock:
            if self.current_action not in {"2", "3"}:
                return False
            self.stop_requested = True
            return True

    def _is_stop_requested_for(self, action_index: str) -> bool:
        with self._state_lock:
            return self.current_action == action_index and self.stop_requested

    def wait_action_done(self, timeout_sec: float | None = None) -> bool:
        return self.action_done_event.wait(timeout=timeout_sec)

    def consume_action_done_message(self) -> str:
        with self._state_lock:
            msg = self.action_done_message
            self.action_done_message = ""
            return msg

    def _record_action1_positions(self):
        with self._state_lock:
            self.action1_recorded_positions = [float(p) for p in self.current_positions]
            self.has_action1_recorded_positions = True
        self.get_logger().info(
            "Action 1 final motor positions recorded for Action 2 position control."
        )

    def _record_action3_hold_positions(self):
        with self._state_lock:
            self.action3_hold_positions = [float(p) for p in self.current_positions]
            self.has_action3_hold_positions = True
        self.get_logger().info(
            "Action 3 hold positions recorded. "
            f"Thumb motors: opposition={self.thumb_opposition_motor}, "
            f"abduction={self.thumb_abduction_motor}, long_flex={self.thumb_long_flex_motor}."
        )

    def feedback_callback(self, msg: YHandMsg):
        if len(msg.motors) < MOTOR_COUNT:
            return
        self.current_positions = [float(msg.motors[i].pos) for i in range(MOTOR_COUNT)]
        self.current_velocities = [float(msg.motors[i].vel) for i in range(MOTOR_COUNT)]
        if not self.has_feedback:
            # Start action timing from the first valid feedback frame.
            self.phase = "init"
            self.phase_start_time = time.monotonic()
            self.last_logged_phase = ""
            self.get_logger().info("Received first feedback frame, current commands enabled.")
        self.has_feedback = True

    def _log_waiting_state(self):
        cmd_sub_count = self.count_subscribers(self.command_topic)
        fb_pub_count = self.count_publishers(self.feedback_topic)
        self.get_logger().info(
            "Waiting for feedback before sending current commands. "
            f"cmd_subscribers={cmd_sub_count}, feedback_publishers={fb_pub_count}"
        )

    def _publish_idle(self):
        if self.idle_mode == "hold_fist":
            self.publish_currents(self.fist_hold_current, default_current=DEFAULT_HOLD_CURRENT)
        elif self.idle_mode == "hold_open":
            self.publish_currents(self.open_hold_current, default_current=DEFAULT_HOLD_CURRENT)
        elif self.idle_mode == "hold_action1_pose":
            self.publish_action1_recorded_position()
        elif self.idle_mode == "hold_zero_pos":
            self.publish_all_zero_position()
        elif self.idle_mode == "hold_action2_zero_pos":
            self.publish_action2_zero_position()
        elif self.idle_mode == "hold_action3":
            self.publish_action3_return_initial_pose()
        else:
            self.publish_currents({}, default_current=INIT_HOLD_CURRENT)

    def _all_positions_near_zero(self) -> bool:
        for pos in self.current_positions:
            if abs(float(pos) - ACTION0_TARGET_POS) > ACTION0_POS_TOL:
                return False
        return True

    def timer_callback(self):
        now = time.monotonic()
        if not self.has_feedback:
            # Important: Y_Hand_Node only publishes feedback after receiving a command.
            # So we must send a safe bootstrap command to break startup deadlock.
            self.publish_currents({}, default_current=INIT_HOLD_CURRENT)
            self.bootstrap_send_count += 1
            if (now - self.last_wait_log_ts) >= WAIT_LOG_INTERVAL_SEC:
                self.last_wait_log_ts = now
                self._log_waiting_state()
                self.get_logger().info(
                    f"Bootstrap command sent while waiting for first feedback, "
                    f"count={self.bootstrap_send_count}"
                )
            if self.feedback_timeout_sec > 0.0 and (now - self.feedback_wait_start_ts) >= self.feedback_timeout_sec:
                self.get_logger().error(
                    "Feedback wait timeout. Please check hand selection/topics/driver state. "
                    f"hand={self.hand}, cmd_topic={self.command_topic}, feedback_topic={self.feedback_topic}"
                )
                rclpy.shutdown()
            return

        with self._state_lock:
            action = self.current_action

        if action is None:
            self._publish_idle()
            return

        elapsed = now - self.phase_start_time
        if action == "0":
            self._run_action_zero(now, elapsed)
        elif action == "1":
            self._run_action_one(now, elapsed)
        elif action == "2":
            self._run_action_two(now, elapsed)
        elif action == "3":
            self._run_action_three(now, elapsed)
        else:
            self._publish_idle()

    def _run_action_zero(self, now: float, elapsed: float):
        if self.phase == "init":
            self._log_phase_once("Action 0: position control, return all motors to zero.")
            self.phase = "return_zero"
            self.phase_start_time = now
            self.action0_near_zero_frames = 0
            self.publish_all_zero_position()
            return

        if self.phase == "return_zero":
            self.publish_all_zero_position()
            if self._all_positions_near_zero():
                self.action0_near_zero_frames += 1
            else:
                self.action0_near_zero_frames = 0

            if self.action0_near_zero_frames >= ACTION0_CONFIRM_FRAMES:
                self._finish_action("hold_zero_pos", "Action 0 finished. All motors reached zero and are held.")
                return
            if elapsed >= ACTION0_TIMEOUT_SEC:
                self._finish_action("hold_zero_pos", "Action 0 timeout. Switched to zero-position hold.")
                return

        self.publish_all_zero_position()

    def _run_action_one(self, now: float, elapsed: float):
        if self.phase == "init":
            self._log_phase_once("Action 1: pretension at -0.01A for 0.5s.")
            self.publish_currents({}, default_current=INIT_HOLD_CURRENT)
            if elapsed >= INIT_DURATION_SEC:
                self.phase = "fist"
                self.phase_start_time = now
        elif self.phase == "fist":
            self._log_phase_once("Action 1: four-finger deep flexors/flexors -0.2A; other motors +0.05A.")
            self.publish_currents(self.fist_current, default_current=ACTION1_RELEASE_CURRENT)
            if elapsed >= FIST_DURATION_SEC:
                self._record_action1_positions()
                self._finish_action(
                    "hold_fist",
                    "Action 1 finished. Final motor state recorded for Action 2; holding fist with small current.",
                )
        else:
            self._finish_action("hold_fist", "Action 1 finished. Holding fist with small current.")

    def _run_action_two(self, now: float, elapsed: float):
        if self.phase == "init":
            with self._state_lock:
                has_recorded = self.has_action1_recorded_positions
            if not has_recorded:
                self.publish_all_zero_position()
                self._finish_action(
                    "hold_zero_pos",
                    "Action 2 aborted: Action 1 positions not recorded. Please run Action 1 first.",
                )
                return
            # No startup waiting: immediately enter open/close oscillation.
            self.phase = "oscillate_open_close"
            self.phase_start_time = now
            self._log_phase_once("Action 2: start immediate open/close loop (position control).")
            return

        if self.phase == "oscillate_open_close":
            if self._is_stop_requested_for("2"):
                self.publish_action2_zero_position()
                self._finish_action(
                    "hold_action2_zero_pos",
                    "Action 2 stopped by user. Returned to initial position (zero).",
                )
                return

            cycle_elapsed = elapsed % ACTION2_CYCLE_SEC
            if cycle_elapsed < ACTION2_TOGGLE_INTERVAL_SEC:
                self._log_phase_once("Action 2: open/close loop - open (position control).")
                self.publish_action2_zero_position()
            else:
                self._log_phase_once("Action 2: open/close loop - close (position control).")
                self.publish_action1_recorded_position()
            return

        self.publish_all_zero_position()
        self._finish_action(
            "hold_zero_pos",
            "Action 2 finished. Holding initial position (zero) in position control.",
        )

    def _run_action_three(self, now: float, elapsed: float):
        if self.phase == "init":
            self._record_action3_hold_positions()
            self.phase = "oscillate_opposition_return"
            self.phase_start_time = now
            self._log_phase_once(
                "Action 3: opposition -> return-to-initial oscillation started."
            )
            self.publish_action3_thumb_currents(self.action3_thumb_currents)
            return

        if self.phase == "oscillate_opposition_return":
            if self._is_stop_requested_for("3"):
                self.publish_action3_return_initial_pose()
                self._finish_action(
                    "hold_action3",
                    "Action 3 stopped by user. Returned to initial pose.",
                )
                return

            cycle_elapsed = elapsed % ACTION3_CYCLE_SEC
            if cycle_elapsed < ACTION3_OPPOSITION_HOLD_SEC:
                self.publish_action3_thumb_currents(self.action3_thumb_currents)
            else:
                self.publish_action3_return_initial_pose()
            return

        self.publish_action3_return_initial_pose()
        self._finish_action("hold_action3", "Action 3 finished. Holding initial pose.")

    def _log_phase_once(self, message: str):
        if self.phase != self.last_logged_phase:
            self.get_logger().info(message)
            self.last_logged_phase = self.phase

    def publish_currents(self, motor_current: Dict[int, float], default_current: float):
        msg = YHandMsg()
        msg.motors = []
        for motor_id in range(1, MOTOR_COUNT + 1):
            current = motor_current.get(motor_id, default_current)
            msg.motors.append(
                Motor(
                    pos=float(self.current_positions[motor_id - 1]),
                    vel=float(self.current_velocities[motor_id - 1]),
                    tor=float(current),
                    kp=float(DEFAULT_KP),
                    kd=float(DEFAULT_KD),
                )
            )
        msg.pretightenforce.flag = PRETIGHTEN_FLAG
        msg.pretightenforce.torque = PRETIGHTEN_TORQUE
        msg.pretightenforce.ms = PRETIGHTEN_MS
        self.cmd_publisher.publish(msg)

    def _normalize_control_vector(
        self,
        values: float | Sequence[float],
        fallback_values: Sequence[float],
    ) -> list[float]:
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            normalized = list(values)
            if len(normalized) < MOTOR_COUNT:
                normalized += list(fallback_values[len(normalized) : MOTOR_COUNT])
            elif len(normalized) > MOTOR_COUNT:
                normalized = normalized[:MOTOR_COUNT]

            out: list[float] = []
            for idx, value in enumerate(normalized):
                try:
                    out.append(float(value))
                except (TypeError, ValueError):
                    out.append(float(fallback_values[idx]))
            return out

        try:
            scalar = float(values)
        except (TypeError, ValueError):
            scalar = float(fallback_values[0]) if fallback_values else 0.0
        return [scalar] * MOTOR_COUNT

    def publish_position_targets(
        self,
        target_positions: list[float],
        kp: float | Sequence[float],
        kd: float | Sequence[float],
        tor: float | Sequence[float],
        target_vel: float | Sequence[float],
    ):
        if len(target_positions) < MOTOR_COUNT:
            target_positions = target_positions + [ACTION0_TARGET_POS] * (MOTOR_COUNT - len(target_positions))
        elif len(target_positions) > MOTOR_COUNT:
            target_positions = target_positions[:MOTOR_COUNT]

        kp_values = self._normalize_control_vector(kp, self.position_ctrl_kp)
        kd_values = self._normalize_control_vector(kd, self.position_ctrl_kd)
        tor_values = self._normalize_control_vector(tor, self.position_ctrl_tor)
        vel_values = self._normalize_control_vector(target_vel, self.position_ctrl_vel)

        msg = YHandMsg()
        msg.motors = []
        for idx in range(MOTOR_COUNT):
            msg.motors.append(
                Motor(
                    pos=float(target_positions[idx]),
                    vel=float(vel_values[idx]),
                    tor=float(tor_values[idx]),
                    kp=float(kp_values[idx]),
                    kd=float(kd_values[idx]),
                )
            )
        msg.pretightenforce.flag = PRETIGHTEN_FLAG
        msg.pretightenforce.torque = PRETIGHTEN_TORQUE
        msg.pretightenforce.ms = PRETIGHTEN_MS
        self.cmd_publisher.publish(msg)

    def publish_all_zero_position(self):
        self.publish_position_targets(
            target_positions=[ACTION0_TARGET_POS] * MOTOR_COUNT,
            kp=self.position_ctrl_kp,
            kd=self.position_ctrl_kd,
            tor=self.position_ctrl_tor,
            target_vel=self.position_ctrl_vel,
        )

    def publish_action1_recorded_position(self):
        with self._state_lock:
            has_recorded = self.has_action1_recorded_positions
            recorded = [float(v) for v in self.action1_recorded_positions]
        if not has_recorded:
            self.publish_all_zero_position()
            return
        self.publish_position_targets(
            target_positions=recorded,
            kp=self.action2_position_ctrl_kp,
            kd=self.action2_position_ctrl_kd,
            tor=self.action2_position_ctrl_tor,
            target_vel=self.action2_position_ctrl_vel,
        )

    def publish_action2_zero_position(self):
        self.publish_position_targets(
            target_positions=[ACTION0_TARGET_POS] * MOTOR_COUNT,
            kp=self.position_ctrl_kp,
            kd=self.position_ctrl_kd,
            tor=self.action2_open_position_ctrl_tor,
            target_vel=self.position_ctrl_vel,
        )

    def publish_action3_thumb_currents(self, thumb_currents: Dict[int, float]):
        with self._state_lock:
            has_hold = self.has_action3_hold_positions
            hold_positions = [float(v) for v in self.action3_hold_positions]

        if not has_hold:
            hold_positions = [float(v) for v in self.current_positions]

        msg = YHandMsg()
        msg.motors = []
        for motor_id in range(1, MOTOR_COUNT + 1):
            idx = motor_id - 1
            if motor_id in thumb_currents:
                msg.motors.append(
                    Motor(
                        pos=float(self.current_positions[idx]),
                        vel=float(self.current_velocities[idx]),
                        tor=float(thumb_currents[motor_id]),
                        kp=float(DEFAULT_KP),
                        kd=float(DEFAULT_KD),
                    )
                )
            elif motor_id in self.action3_lock_motors:
                msg.motors.append(
                    Motor(
                        pos=float(hold_positions[idx]),
                        vel=float(self.position_ctrl_vel[idx]),
                        tor=float(self.position_ctrl_tor[idx]),
                        kp=float(self.position_ctrl_kp[idx]),
                        kd=float(self.position_ctrl_kd[idx]),
                    )
                )
            else:
                msg.motors.append(
                    Motor(
                        pos=float(self.current_positions[idx]),
                        vel=float(self.current_velocities[idx]),
                        tor=float(DEFAULT_HOLD_CURRENT),
                        kp=float(DEFAULT_KP),
                        kd=float(DEFAULT_KD),
                    )
                )
        msg.pretightenforce.flag = PRETIGHTEN_FLAG
        msg.pretightenforce.torque = PRETIGHTEN_TORQUE
        msg.pretightenforce.ms = PRETIGHTEN_MS
        self.cmd_publisher.publish(msg)

    def publish_action3_return_initial_pose(self):
        with self._state_lock:
            has_hold = self.has_action3_hold_positions
            hold_positions = [float(v) for v in self.action3_hold_positions]

        if not has_hold:
            hold_positions = [float(v) for v in self.current_positions]
        self.publish_position_targets(
            target_positions=hold_positions,
            kp=self.position_ctrl_kp,
            kd=self.position_ctrl_kd,
            tor=self.position_ctrl_tor,
            target_vel=self.position_ctrl_vel,
        )


def select_action() -> str:
    print("Select action:")
    print("  0: 全电机回零（位置控制）")
    print(f"  1: 四指握拳（电流控制，持续{FIST_DURATION_SEC:.1f}s，结束后记录最终28路电机位置）")
    print(f"  2: 位控循环张开-闭合（开合间隔{ACTION2_TOGGLE_INTERVAL_SEC:.1f}s，运行中输入o停止并回初始位）")
    print(
        f"  3: 对掌往复循环（对掌{ACTION3_OPPOSITION_HOLD_SEC:.1f}s -> 回初始位姿{ACTION3_RETURN_HOLD_SEC:.1f}s，运行中输入o停止）"
    )
    print("  x: 退出")
    while True:
        choice = input("Input action index (0/1/2/3/x): ").strip().lower()
        if choice in {"0", "1", "2", "3", "x"}:
            return choice
        print("Invalid input. Please enter 0, 1, 2, 3 or x.")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Current-loop calibration actions for Y-Hand.")
    parser.add_argument("--hand", choices=["left", "right"], default="left", help="Target hand.")
    parser.add_argument(
        "--action",
        choices=["0", "1", "2", "3"],
        default="",
        help="Calibration action index. If omitted, interactive selection is used.",
    )
    parser.add_argument(
        "--cmd-topic",
        default="",
        help="Command topic override. Defaults by hand: left=/yhand_left_cmd, right=/yhand_right_cmd.",
    )
    parser.add_argument(
        "--feedback-topic",
        default="",
        help="Feedback topic override. Defaults by hand: left=/yhand_left_feedback, right=/yhand_right_feedback.",
    )
    parser.add_argument(
        "--feedback-timeout-sec",
        type=float,
        default=12.0,
        help="Wait timeout for first feedback. <=0 means wait forever.",
    )
    return parser.parse_known_args(argv)


def _spin_node(node: Node):
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        return
    except KeyboardInterrupt:
        return


def _read_terminal_command_nonblocking() -> str | None:
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    except (OSError, ValueError):
        return None
    if not ready:
        return None
    line = sys.stdin.readline()
    if not line:
        return None
    return line.strip().lower()


def main(args=None):
    cli_args, ros_args = _parse_args(args)

    command_topic, feedback_topic = _topics_for_hand(cli_args.hand)
    if cli_args.cmd_topic:
        command_topic = cli_args.cmd_topic
    if cli_args.feedback_topic:
        feedback_topic = cli_args.feedback_topic

    rclpy.init(args=ros_args)
    node = ManusCalibrationNode(
        hand=cli_args.hand,
        command_topic=command_topic,
        feedback_topic=feedback_topic,
        feedback_timeout_sec=cli_args.feedback_timeout_sec,
    )
    spin_thread = threading.Thread(target=_spin_node, args=(node,), daemon=True)
    spin_thread.start()

    try:
        next_action = cli_args.action if cli_args.action else select_action()
        while rclpy.ok():
            if next_action == "x":
                print("Exit requested by user.")
                break

            try:
                node.start_action(next_action)
            except RuntimeError as exc:
                print(f"Action start skipped: {exc}")
                time.sleep(0.2)
                continue

            if next_action in {"2", "3"}:
                print("Action running. Input 'o' then Enter to stop and return to initial position.")

            while rclpy.ok():
                if next_action in {"2", "3"}:
                    cmd = _read_terminal_command_nonblocking()
                    if cmd == "o":
                        if node.request_stop_for_loop_actions():
                            print("Stop requested. Returning to initial position...")
                    elif cmd:
                        print(f"Unknown runtime command: {cmd}. Use 'o' to stop current loop action.")
                if node.wait_action_done(timeout_sec=0.2):
                    break

            if not rclpy.ok():
                break

            done_msg = node.consume_action_done_message()
            if done_msg:
                print(done_msg)

            next_action = select_action()
    except KeyboardInterrupt:
        pass
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
