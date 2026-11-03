#!/usr/bin/env python3
"""Record left-hand motor feedback snapshots with interactive finger selection."""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime
import math
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty

sys.path.insert(0, str(Path(__file__).resolve().parent / 'q_2_m linear mapping'))
from left_little_q_to_motor import FINGER_SPECS


FINGERS = dict(zip('12345', ('thumb', 'index', 'middle', 'ring', 'little')))
FIELDS = ('snapshot_id', 'capture_time', 'feedback_received_time', 'feedback_age_ms',
          'finger', 'motor_id', 'position_deg', 'delta_from_previous_deg', 'error_code')


def timestamp():
    return datetime.now().astimezone().isoformat(timespec='milliseconds')


class FeedbackRecorder:
    def __init__(self, stream, max_age):
        self.writer = csv.DictWriter(stream, fieldnames=FIELDS)
        self.writer.writeheader()
        stream.flush()
        self.stream = stream
        self.max_age = max_age
        self.latest = None
        self.sequence = 0
        self.previous = {}

    def receive(self, message):
        if len(message.motors) != 28:
            self.latest = None
            return
        values = tuple((float(m.pos), int(m.error)) for m in message.motors)
        if not all(math.isfinite(pos) for pos, _ in values):
            self.latest = None
            return
        self.latest = (values, timestamp(), time.monotonic())

    def capture(self, finger):
        if self.latest is None:
            print('尚未收到有效的28路电机反馈，本次未记录。', flush=True)
            return
        values, received_time, received_at = self.latest
        age = time.monotonic() - received_at
        if age > self.max_age:
            print(f'反馈已过期（{age:.3f}秒），本次未记录。', flush=True)
            return
        captured = timestamp()
        self.sequence += 1
        spec = FINGER_SPECS[finger]
        positions = {mid: values[mid - 1][0] for mid in spec['motor_inputs']}
        previous = self.previous.get(finger, {})
        print(f'\n#{self.sequence} [{captured}] {spec["label"]} '
              f'| 反馈接收时间={received_time} | 距接收={age * 1000:.1f} ms', flush=True)
        for mid, pos in positions.items():
            delta = pos - previous[mid] if mid in previous else None
            error = values[mid - 1][1]
            change = f'{delta:+.3f}°' if delta is not None else '首次记录'
            print(f'  M{mid:<2} = {pos:>10.3f}°  Δ={change}'
                  + (f'  error={error}' if error else ''), flush=True)
            self.writer.writerow(dict(
                snapshot_id=self.sequence, capture_time=captured,
                feedback_received_time=received_time, feedback_age_ms=f'{age * 1000:.3f}',
                finger=finger, motor_id=mid, position_deg=pos,
                delta_from_previous_deg=delta if delta is not None else '', error_code=error))
        self.stream.flush()
        self.previous[finger] = positions


@contextmanager
def terminal_keys():
    """Restore terminal settings even after Ctrl+C or an exception."""
    saved = termios.tcgetattr(sys.stdin.fileno()) if sys.stdin.isatty() else None
    try:
        if saved is not None:
            tty.setcbreak(sys.stdin.fileno())
        yield
    finally:
        if saved is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)


def main():
    parser = argparse.ArgumentParser(description='按1–5选择左手手指，按s记录电机反馈位置。')
    parser.add_argument('--topic', default='/yhand_left_feedback')
    parser.add_argument('--max-age', type=float, default=0.5, help='反馈有效期（秒）')
    parser.add_argument('--output', type=Path, help='CSV路径，默认写入motor_test_logs/test_motor_feed_back/')
    args = parser.parse_args()
    if not math.isfinite(args.max_age) or args.max_age <= 0:
        parser.error('--max-age必须为有限正数')
    if args.output is None:
        args.output = (Path(__file__).resolve().parent / 'motor_test_logs' / 'test_motor_feed_back'
                       / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.csv'))
    args.output = args.output.expanduser().resolve()

    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data
    from devices_pkg.msg import YHandMsg

    rclpy.init(args=[])
    node = None
    try:
        node = rclpy.create_node('test_motor_feed_back')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', newline='', encoding='utf-8-sig') as stream:
            recorder = FeedbackRecorder(stream, args.max_age)
            node.create_subscription(YHandMsg, args.topic, recorder.receive, qos_profile_sensor_data)
            print(f'订阅: {args.topic}\nCSV: {args.output}', flush=True)
            print('1=拇指  2=食指  3=中指  4=无名指  5=小指\n'
                  '按s记录；Δ为同一手指相对上一次记录的电机位置差；按q或Ctrl+C退出。\n'
                  '时间戳使用本机时间；反馈时间是本机接收时间。', flush=True)
            if not sys.stdin.isatty():
                print('当前非交互终端，请逐行输入命令并回车。', flush=True)
            finger = None
            with terminal_keys():
                while rclpy.ok():
                    rclpy.spin_once(node, timeout_sec=0.05)
                    if not select.select([sys.stdin], [], [], 0)[0]:
                        continue
                    raw = os.read(sys.stdin.fileno(), 1).decode() if sys.stdin.isatty() else sys.stdin.readline()
                    if not raw:
                        break
                    key = raw.strip().lower()
                    if key in FINGERS:
                        finger = FINGERS[key]
                        spec = FINGER_SPECS[finger]
                        print(f'已选择{spec["label"]}：'
                              + ' / '.join(f'M{mid}' for mid in spec['motor_inputs']), flush=True)
                    elif key == 's':
                        if finger is None:
                            print('请先按1–5选择手指。', flush=True)
                        else:
                            recorder.capture(finger)
                    elif key in ('q', 'quit', 'exit'):
                        break
                    elif key:
                        print('使用1–5选择手指，s记录，q退出。', flush=True)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
