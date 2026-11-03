#!/usr/bin/env python3
import argparse
import math
import threading

from joint_coupling import DEFAULT_REFERENCE, JointCouplingReference

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
    parser.add_argument("--manual-coupling", choices=["off", "reject"], default="off")
    parser.add_argument("--coupling-reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--relative", choices=["true", "false"], default="true")
    return parser.parse_args()


def _parse_angles(raw: str):
    tokens = raw.replace(",", " ").split()
    if len(tokens) != 4:
        raise ValueError("需要输入 4 个关节角")
    values = [float(token) for token in tokens]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("关节角必须为有限数")
    return values


def input_target(raw, finger, *, coupling=None, relative=True):
    """Return a full target or a read-only conditional hint; never project q."""
    tokens = raw.replace(",", " ").split()
    if coupling is not None and relative and len(tokens) in (2, 3):
        values = [float(token) for token in tokens]
        hint = coupling.condition(finger, values[:2], values[2] if len(values) == 3 else None)
        return None, coupling.format_result(hint) + "；仅查询，未发送目标。"
    angles = [0.] * 4 if raw.lower() in ("zero", "z") else _parse_angles(raw)
    if coupling is not None and relative:
        result = coupling.check(finger, angles)
        if not result['accepted']:
            raise ValueError(coupling.format_result(result) + "；当前输出保持不变。")
    return angles, None


def main():
    args = _parse_args()
    if args.publish_hz <= 0.0:
        raise ValueError("--publish-hz must be greater than zero")
    if args.burst_count <= 0:
        raise ValueError("--burst-count must be greater than zero")
    if args.hand != "left" and args.manual_coupling != "off":
        raise ValueError("当前日志耦合参考仅适用于左手")
    coupling = JointCouplingReference(args.coupling_reference) if args.manual_coupling == "reject" else None

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
    print("输入顺序: " + ("Δq0 Δq1 Δq2 Δq3（°，相对角度）" if args.relative == "true"
                          else "q0 q1 q2 q3（°，绝对glove角度）"))
    print("输入 4 个角度，空格或逗号分隔。")
    if coupling is not None:
        print("已启用日志耦合限制：q0/q1作为条件，q2/q3必须成对匹配；缺少记录则拒绝本次目标。")
        print("参考来自人手输入分布，不是灵巧手机械可达边界。")
        if args.relative == "true":
            print("输入 q0 q1 查询配对；输入 q0 q1 q2 查询对应q3区间；查询不发送目标。")
            print("容差：" + "、".join(f"q{i}±{v:g}°" for i, v in enumerate(coupling.tolerance))
                  + "；区间不可独立任意组合。")
        else:
            print("绝对角度由主控制器按所选模型减参考角，再检查Δq配对；执行结果见主控制终端。")
    print("输入 zero 发送全零，输入 q 退出并关闭整个 launch。")
    print("=" * 64)

    try:
        while rclpy.ok():
            raw = input(f"{args.finger}> ").strip()
            command = raw.lower()
            if command in ("q", "quit", "exit", "x"):
                break
            try:
                angles, hint = input_target(raw, args.finger, coupling=coupling, relative=args.relative == "true")
            except ValueError as exc:
                print(f"输入无效: {exc}")
                continue
            if angles is None:
                print(hint)
                continue

            node.set_angles(angles)
            print(
                f"已发布 {args.finger} 目标角度: {angles}（执行结果见主控制终端） "
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
