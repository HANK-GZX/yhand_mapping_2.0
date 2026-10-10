#!/usr/bin/env python3
"""Zero-offset proxy for Y-hand command and feedback topics."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

try:
    import rclpy
    from rclpy.node import Node
except ImportError as exc:  # pragma: no cover
    rclpy = None
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

# ---------------------------------------------------------------------------
# Shared software-zero helpers
# ---------------------------------------------------------------------------

from software_zero_common import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    MOTOR_COUNT,
    current_boot_id,
    default_state_path,
    load_state,
    load_zero_config,
    normalize_hand,
    resolve_pkg_path,
    zero_offset_from_state,
)

RAW_CMD_TOPIC_TEMPLATE = "/yhand_{hand}_raw_cmd"
RAW_FEEDBACK_TOPIC_TEMPLATE = "/yhand_{hand}_raw_feedback"
CMD_TOPIC_TEMPLATE = "/yhand_{hand}_cmd"
FEEDBACK_TOPIC_TEMPLATE = "/yhand_{hand}_feedback"


# ---------------------------------------------------------------------------
# Message copying helpers
# ---------------------------------------------------------------------------


def _copy_pretighten(dst, src) -> None:
    try:
        dst.pretightenforce.flag = bool(src.pretightenforce.flag)
        dst.pretightenforce.torque = float(src.pretightenforce.torque)
        dst.pretightenforce.ms = int(src.pretightenforce.ms)
    except Exception:
        dst.pretightenforce.flag = False
        dst.pretightenforce.torque = 0.0
        dst.pretightenforce.ms = 0


def _copy_motor_with_pos(motor, pos: float):
    return Motor(
        pos=float(pos),
        vel=float(motor.vel),
        tor=float(motor.tor),
        kp=float(motor.kp),
        kd=float(motor.kd),
    )


# ---------------------------------------------------------------------------
# ROS proxy node
# ---------------------------------------------------------------------------

class SoftwareZeroProxy(Node):
    def __init__(
        self,
        *,
        hand: str,
        state_path: Path,
        raw_cmd_topic: str,
        raw_feedback_topic: str,
        cmd_topic: str,
        feedback_topic: str,
        reload_period_sec: float,
    ):
        super().__init__(f"software_zero_proxy_{hand}")
        if DEVICES_MSG_IMPORT_ERROR is not None:
            raise RuntimeError(f"Cannot import devices_pkg messages: {DEVICES_MSG_IMPORT_ERROR}")

        self.hand = normalize_hand(hand)
        self.state_path = Path(state_path)
        self.raw_cmd_topic = str(raw_cmd_topic)
        self.raw_feedback_topic = str(raw_feedback_topic)
        self.cmd_topic = str(cmd_topic)
        self.feedback_topic = str(feedback_topic)
        self._lock = threading.Lock()
        self.zero_offset = self._load_zero_offset()

        self.raw_cmd_pub = self.create_publisher(YHandMsg, self.raw_cmd_topic, 10)
        self.feedback_pub = self.create_publisher(YHandMsg, self.feedback_topic, 10)
        self.cmd_sub = self.create_subscription(YHandMsg, self.cmd_topic, self._cmd_cb, 10)
        self.raw_feedback_sub = self.create_subscription(
            YHandMsg,
            self.raw_feedback_topic,
            self._raw_feedback_cb,
            10,
        )
        self.reload_timer = None
        if float(reload_period_sec) > 0.0:
            self.reload_timer = self.create_timer(float(reload_period_sec), self._reload_cb)

        self.get_logger().info(
            "[software_zero_proxy] "
            f"hand={self.hand}, state={self.state_path}, "
            f"cmd {self.cmd_topic} -> {self.raw_cmd_topic}, "
            f"feedback {self.raw_feedback_topic} -> {self.feedback_topic}"
        )

    def _load_zero_offset(self) -> list[float]:
        if not self.state_path.exists():
            raise FileNotFoundError(
                f"software-zero state not found: {self.state_path}. "
                "Run back-delta software_zero_calibration.py first; no default zero_offset is used."
            )
        state = load_state(self.state_path, self.hand)
        runtime_boot_id = current_boot_id()
        state_boot_id = state.get("boot_id")
        if runtime_boot_id is not None and state_boot_id != runtime_boot_id:
            raise RuntimeError(
                "software-zero state was not generated in this boot. "
                "Run back-delta software_zero_calibration.py again before starting the proxy."
            )
        return zero_offset_from_state(state)

    def _reload_cb(self) -> None:
        try:
            new_offset = self._load_zero_offset()
        except Exception as exc:
            self.get_logger().error(f"failed to reload zero state: {exc}")
            return
        with self._lock:
            self.zero_offset = new_offset

    def _offset_snapshot(self) -> list[float]:
        with self._lock:
            return list(self.zero_offset)

    def _cmd_cb(self, msg) -> None:
        offset = self._offset_snapshot()
        motors = list(getattr(msg, "motors", []) or [])
        out = YHandMsg()
        out.motors = []
        for i, motor in enumerate(motors[:MOTOR_COUNT]):
            out.motors.append(_copy_motor_with_pos(motor, float(motor.pos) + offset[i]))
        for motor in motors[MOTOR_COUNT:]:
            out.motors.append(_copy_motor_with_pos(motor, float(motor.pos)))
        _copy_pretighten(out, msg)
        self.raw_cmd_pub.publish(out)

    def _raw_feedback_cb(self, msg) -> None:
        offset = self._offset_snapshot()
        motors = list(getattr(msg, "motors", []) or [])
        out = YHandMsg()
        out.motors = []
        for i, motor in enumerate(motors[:MOTOR_COUNT]):
            out.motors.append(_copy_motor_with_pos(motor, float(motor.pos) - offset[i]))
        for motor in motors[MOTOR_COUNT:]:
            out.motors.append(_copy_motor_with_pos(motor, float(motor.pos)))
        _copy_pretighten(out, msg)
        self.feedback_pub.publish(out)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=("left", "right"), required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--state-path", default="")
    parser.add_argument("--raw-cmd-topic", default="")
    parser.add_argument("--raw-feedback-topic", default="")
    parser.add_argument("--cmd-topic", default="")
    parser.add_argument("--feedback-topic", default="")
    parser.add_argument(
        "--reload-state-period-sec",
        type=float,
        default=0.0,
        help="0 disables reload. Positive values reload the state file periodically.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hand = normalize_hand(args.hand)
    config = load_zero_config(resolve_pkg_path(args.config))
    state_path = resolve_pkg_path(args.state_path) if args.state_path else default_state_path(config, hand)
    raw_cmd_topic = args.raw_cmd_topic.strip() or RAW_CMD_TOPIC_TEMPLATE.format(hand=hand)
    raw_feedback_topic = args.raw_feedback_topic.strip() or RAW_FEEDBACK_TOPIC_TEMPLATE.format(hand=hand)
    cmd_topic = args.cmd_topic.strip() or CMD_TOPIC_TEMPLATE.format(hand=hand)
    feedback_topic = args.feedback_topic.strip() or FEEDBACK_TOPIC_TEMPLATE.format(hand=hand)

    if RCLPY_IMPORT_ERROR is not None:
        print(f"[software-zero-proxy] error: cannot import rclpy. Source ROS2 first: {RCLPY_IMPORT_ERROR}", file=sys.stderr)
        return 2

    rclpy.init()
    node = None
    try:
        node = SoftwareZeroProxy(
            hand=hand,
            state_path=state_path,
            raw_cmd_topic=raw_cmd_topic,
            raw_feedback_topic=raw_feedback_topic,
            cmd_topic=cmd_topic,
            feedback_topic=feedback_topic,
            reload_period_sec=float(args.reload_state_period_sec),
        )
        rclpy.spin(node)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[software-zero-proxy] error: {exc}", file=sys.stderr)
        return 2
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
