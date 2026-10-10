#!/usr/bin/env python3
"""Stop current, set SDK motor zeros on Ctrl+C, verify feedback, then exit."""

import argparse
import math
import os
import signal
import subprocess
import threading
import time

import rclpy
from devices_pkg.msg import YHandMsg
from rclpy.context import Context
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions


# SDK Set_Zero waits up to 2 s per motor and sleeps 1 s after a missing reply.
# Allow all 28 calls to complete even when individual devices do not reply.
ZERO_TIMEOUT = 90.0
ZERO_TOLERANCE = 1.0  # Motor position feedback is in degrees.
ZERO_CONFIRMATIONS = 3


class MotorFeedback:
    def __init__(self):
        self.sequence = 0
        self.message = None

    def receive(self, message):
        self.sequence += 1
        self.message = message


def motor_command(torque):
    message = YHandMsg()
    for motor in message.motors:
        motor.pos = 0.0
        motor.vel = 0.0
        motor.tor = torque
        motor.kp = 0.0
        motor.kd = 0.0
    message.pretightenforce.flag = False
    message.pretightenforce.torque = 0.0
    message.pretightenforce.ms = 0
    return message


def send_stop(node, publisher, executor, driver):
    node.get_logger().info('停止收线：连续发送 0.5 秒全部 28 路零电流指令。')
    command = motor_command(0.0)
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline and driver.poll() is None:
        publisher.publish(command)
        executor.spin_once(timeout_sec=0.01)
    if not publisher.wait_for_all_acked(Duration(seconds=0.2)):
        node.get_logger().warning('零电流消息未在等待时间内获得 DDS 确认，请检查设备状态。')


def invalid_zero_motors(message, tolerance):
    if message is None or len(message.motors) != 28:
        return list(range(1, 29))
    return [i for i, motor in enumerate(message.motors, 1)
            if not math.isfinite(motor.pos) or abs(motor.pos) > tolerance
            or motor.error != 0]


def set_zero_and_check(node, publisher, executor, driver, feedback, args):
    if driver.poll() is not None or publisher.get_subscription_count() == 0:
        node.get_logger().error('无法设置零点：SDK 驱动未连接。')
        return False

    # Flush already received stop feedback before sending the one-shot request.
    # Feedback has no device timestamp or request ID; do not reuse the cached
    # pre-request message as a successful zero check.
    drain_deadline = time.monotonic() + 0.2
    while time.monotonic() < drain_deadline:
        executor.spin_once(timeout_sec=0.01)
    feedback.message = None
    checked_sequence = feedback.sequence
    request = motor_command(0.0)
    request.pretightenforce.flag = True
    # The official hand_pretighten(0, 0) sends zero current, then calls Set_Zero
    # once on every motor. Never repeat this request during feedback polling.
    node.get_logger().info('设置零点：请求 SDK 对全部 28 路调用 Set_Zero，电流为 0。')
    publisher.publish(request)
    started = time.monotonic()
    deadline = started + args.zero_timeout
    probe = motor_command(0.0)
    next_probe = started + 0.2
    confirmations = 0
    while time.monotonic() < deadline and driver.poll() is None:
        executor.spin_once(timeout_sec=0.01)
        now = time.monotonic()
        # Normal zero-current messages make the official driver read and publish
        # feedback after its blocking Set_Zero callback finishes. Limit probes
        # to 5 Hz so a long SDK callback does not create a large pending queue.
        if now >= next_probe:
            publisher.publish(probe)
            next_probe = now + 0.2
        if feedback.sequence == checked_sequence:
            continue
        checked_sequence = feedback.sequence
        if now < started + 0.2:
            continue
        if invalid_zero_motors(feedback.message, args.zero_tolerance):
            confirmations = 0
        else:
            confirmations += 1
        if confirmations >= ZERO_CONFIRMATIONS:
            positions = ', '.join(f'{i}:{motor.pos:.3f}'
                                  for i, motor in enumerate(feedback.message.motors, 1))
            node.get_logger().info(
                f'零点反馈检查通过：全部 28 路连续 {ZERO_CONFIRMATIONS} 帧 '
                f'|pos|≤{args.zero_tolerance:g}°，error=0；位置 [电机号:度] {positions}')
            return True

    if feedback.message is None:
        detail = '没有收到归零请求之后的反馈'
    else:
        bad = invalid_zero_motors(feedback.message, args.zero_tolerance)
        detail = ', '.join(f'{i}:pos={feedback.message.motors[i - 1].pos:.3f},'
                           f'error={feedback.message.motors[i - 1].error}' for i in bad)
        if not detail:
            detail = '反馈帧数不足，未完成连续确认'
    node.get_logger().error(f'零点反馈检查未通过：{detail}；仍将关闭驱动。')
    return False


def stop_driver(driver):
    if driver is None or driver.poll() is not None:
        return
    # The driver owns this session, so only its processes receive these signals.
    for signum, timeout in ((signal.SIGINT, 2.0), (signal.SIGTERM, 1.0),
                            (signal.SIGKILL, 1.0)):
        try:
            os.killpg(driver.pid, signum)
        except ProcessLookupError:
            break
        try:
            driver.wait(timeout=timeout)
            break
        except subprocess.TimeoutExpired:
            continue


def run(args, stop_requested, zero_requested):
    context = Context()
    node = None
    executor = None
    driver = None
    result = 0
    try:
        # SIGINT/SIGTERM must not destroy DDS before we publish the stop command.
        rclpy.init(args=[], context=context, signal_handler_options=SignalHandlerOptions.NO)
        node = Node(f'{args.driver_node_name}_pretighten', context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        publisher = node.create_publisher(YHandMsg, args.cmd_topic, 10)
        feedback = MotorFeedback()
        subscription = node.create_subscription(YHandMsg, args.feedback_topic, feedback.receive, 1)
        command = motor_command(args.torque)
        if stop_requested.is_set():
            return 0
        # Terminal Ctrl+C reaches the controller, but not the driver. The controller
        # sends zeros first and explicitly shuts down the driver afterwards.
        driver = subprocess.Popen(
            [args.driver_executable, args.driver_node_name, args.feedback_topic,
             args.cmd_topic, args.dev_config],
            start_new_session=True,
        )
        node.get_logger().info(
            f'预紧电流 {args.torque}；Ctrl+C 将停止收线、设置零点、检查反馈后关闭驱动。')
        while not stop_requested.is_set():
            if driver.poll() is not None:
                node.get_logger().error(f'设备驱动已退出，退出码 {driver.returncode}。')
                result = 1
                break
            executor.spin_once(timeout_sec=0.01)
            # Wait for the driver's subscription before applying current.
            if not stop_requested.is_set() and publisher.get_subscription_count() > 0:
                publisher.publish(command)
    finally:
        try:
            if node is not None and driver is not None and driver.poll() is None:
                send_stop(node, publisher, executor, driver)
                if zero_requested.is_set():
                    if not set_zero_and_check(node, publisher, executor, driver, feedback, args):
                        result = 1
                else:
                    node.get_logger().info('未收到 Ctrl+C 归零请求，保持原零点。')
        finally:
            if node is not None:
                node.get_logger().info('关闭 SDK 驱动。')
            stop_driver(driver)
            if executor is not None:
                executor.shutdown()
            if node is not None:
                node.destroy_node()
            if context.ok():
                context.shutdown()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver-executable', required=True)
    parser.add_argument('--driver-node-name', required=True)
    parser.add_argument('--cmd-topic', required=True)
    parser.add_argument('--feedback-topic', required=True)
    parser.add_argument('--dev-config', required=True)
    parser.add_argument('--torque', type=float, default=-0.05)
    parser.add_argument('--zero-timeout', type=float, default=ZERO_TIMEOUT)
    parser.add_argument('--zero-tolerance', type=float, default=ZERO_TOLERANCE)
    args = parser.parse_args()
    if not math.isfinite(args.zero_timeout) or args.zero_timeout <= 0:
        parser.error('--zero-timeout must be finite and positive')
    if not math.isfinite(args.zero_tolerance) or args.zero_tolerance <= 0:
        parser.error('--zero-tolerance must be finite and positive')
    stop_requested = threading.Event()
    zero_requested = threading.Event()

    def on_signal(signum, _frame):
        if signum == signal.SIGINT:
            zero_requested.set()
        stop_requested.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, on_signal)
    return run(args, stop_requested, zero_requested)


if __name__ == '__main__':
    raise SystemExit(main())
