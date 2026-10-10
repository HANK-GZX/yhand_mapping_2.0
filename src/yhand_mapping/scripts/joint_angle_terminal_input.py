#!/usr/bin/env python3
import argparse
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class JointAngleTerminalPublisher(Node):
    def __init__(self, topic: str, publish_hz: float, burst_count: int):
        super().__init__("joint_angle_terminal_input")
        self.publisher = self.create_publisher(Float64MultiArray, topic, 10)
        self._lock = threading.Lock()
        self._latest_angles = None
        self._burst_count = int(burst_count)
        self._remaining_publishes = 0
        self._timer = self.create_timer(1.0 / publish_hz, self._publish_latest)

    def set_angles(self, angles):
        with self._lock:
            self._latest_angles = [float(value) for value in angles]
            self._remaining_publishes = self._burst_count

    def _publish_latest(self):
        with self._lock:
            if self._latest_angles is None or self._remaining_publishes <= 0:
                return
            values = list(self._latest_angles)
            self._remaining_publishes -= 1

        msg = Float64MultiArray()
        msg.data = values
        self.publisher.publish(msg)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive publisher for one finger's four joint angles."
    )
    parser.add_argument("--hand", required=True, choices=["left", "right"])
    parser.add_argument(
        "--finger",
        required=True,
        choices=["thumb", "index", "middle", "ring", "little"],
    )
    parser.add_argument("--topic", default="/target_joint_angles")
    parser.add_argument("--publish-hz", type=float, default=50.0)
    parser.add_argument("--burst-count", type=int, default=5)
    return parser.parse_args()


def _parse_angles(raw: str):
    tokens = raw.replace(",", " ").split()
    if len(tokens) != 4:
        raise ValueError("需要输入 4 个关节角")
    return [float(token) for token in tokens]


def main():
    args = _parse_args()
    if args.publish_hz <= 0.0:
        raise ValueError("--publish-hz must be greater than zero")
    if args.burst_count <= 0:
        raise ValueError("--burst-count must be greater than zero")

    rclpy.init()
    node = JointAngleTerminalPublisher(
        topic=args.topic,
        publish_hz=float(args.publish_hz),
        burst_count=int(args.burst_count),
    )
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("=" * 64)
    print(f"关节角输入终端: hand={args.hand}, finger={args.finger}")
    print(
        f"发布话题: {args.topic}, 每次输入以 {args.publish_hz:g} Hz "
        f"连续发布 {args.burst_count} 帧后停止"
    )
    print("输入 4 个角度，空格或逗号分隔，例如: 10 20 30 40")
    print("输入 zero 发送全零，输入 q 退出并关闭整个 launch。")
    print("=" * 64)

    try:
        while rclpy.ok():
            raw = input(f"{args.finger}> ").strip()
            command = raw.lower()
            if command in ("q", "quit", "exit", "x"):
                break
            if command in ("zero", "z"):
                angles = [0.0, 0.0, 0.0, 0.0]
            else:
                try:
                    angles = _parse_angles(raw)
                except ValueError as exc:
                    print(f"输入无效: {exc}")
                    continue

            node.set_angles(angles)
            print(
                f"已发送 {args.finger} 目标角度: {angles} "
                f"(burst={args.burst_count})"
            )
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        node.destroy_node()


if __name__ == "__main__":
    main()
