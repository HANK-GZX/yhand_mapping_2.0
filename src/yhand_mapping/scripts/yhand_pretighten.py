#!/usr/bin/env python3
"""Run the SDK Y_Hand_M2_Demo and verify its reported absolute zero positions."""

import argparse
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time


MOTOR_COUNT = 28
ZERO_CONFIRMATIONS = 3
ZERO_TOLERANCE_DEG = 1.0
ZERO_TIMEOUT_SEC = 10.0
POSITION_FRAME = 'M2_POSITION_FRAME '


class ZeroFeedback:
    def __init__(self, tolerance=ZERO_TOLERANCE_DEG):
        self.tolerance = tolerance
        self.motors = ()
        self.consecutive = 0

    def at_zero(self, position, error):
        return position is not None and math.isfinite(position) and abs(position) <= self.tolerance and error == 0

    def receive(self, line):
        values = line.removeprefix(POSITION_FRAME).split()
        if len(values) != 2 * MOTOR_COUNT:
            self.consecutive = 0
            return
        try:
            motors = tuple((float(values[i]), int(values[i + 1])) for i in range(0, len(values), 2))
        except ValueError:
            self.consecutive = 0
            return
        self.record(motors)

    def record(self, motors):
        self.motors = tuple(motors)
        if len(self.motors) == MOTOR_COUNT and all(self.at_zero(*motor) for motor in self.motors):
            self.consecutive += 1
        else:
            self.consecutive = 0

    def report(self):
        rows = ['电机   反馈位置(°)   绝对误差(°)   错误码   状态']
        for index in range(MOTOR_COUNT):
            label = f'M{index + 1:02d}'
            if index >= len(self.motors):
                rows.append(f'{label}            -            -        -   无反馈')
                continue
            position, error = self.motors[index]
            status = '到位' if self.at_zero(position, error) else '未到位'
            pos_text = '-' if position is None else f'{position:+.4f}'
            distance_text = '-' if position is None else f'{abs(position):.4f}'
            error_text = '-' if error is None else str(error)
            rows.append(f'{label}  {pos_text:>11}  {distance_text:>11}  {error_text:>7}   {status}')
        return '\n'.join(rows)


def stop_demo(demo):
    for signum, timeout in ((signal.SIGINT, 2.0), (signal.SIGTERM, 1.0),
                            (signal.SIGKILL, 1.0)):
        if demo.poll() is not None:
            break
        try:
            os.killpg(demo.pid, signum)
        except ProcessLookupError:
            break
        try:
            demo.communicate(timeout=timeout)
            break
        except subprocess.TimeoutExpired:
            continue


def run(args, stop_requested):
    if stop_requested():
        return 0
    command = [str(args.demo_executable), str(args.sdk_config), args.dev_config,
               *map(str, args.max_current_limit)]
    print(f'启动 {args.hand} 手 SDK Y_Hand_M2_Demo，目标绝对位置 0°。', flush=True)
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          start_new_session=True) as demo:
        feedback = ZeroFeedback(args.zero_tolerance)
        deadline = time.monotonic() + args.zero_timeout
        confirmed = False
        pending = b''
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(demo.stdout, selectors.EVENT_READ)
                while not stop_requested():
                    if not confirmed and time.monotonic() >= deadline:
                        print(f'绝对零位确认超时，{args.zero_timeout:g} 秒内未达到全路条件。\n'
                              f'{feedback.report()}', flush=True)
                        return 1
                    if not selector.select(timeout=0.05):
                        continue
                    data = os.read(demo.stdout.fileno(), 8192)
                    if not data:
                        print(f'SDK Y_Hand_M2_Demo 已退出。\n{feedback.report()}', flush=True)
                        return 1
                    *lines, pending = (pending + data).split(b'\n')
                    for raw_line in lines:
                        line = raw_line.decode('utf-8', errors='replace')
                        if not line.startswith(POSITION_FRAME):
                            print(line, flush=True)
                            continue
                        feedback.receive(line)
                        if not confirmed and feedback.consecutive >= ZERO_CONFIRMATIONS:
                            confirmed = True
                            print(f'绝对零位确认通过，28 路连续 {ZERO_CONFIRMATIONS} 帧满足 '
                                  f'|pos|≤{feedback.tolerance:g}° 且 error=0，继续保持 0°。\n'
                                  f'{feedback.report()}', flush=True)
            if not confirmed:
                print(f'已停止位置控制，绝对零位尚未确认。\n{feedback.report()}', flush=True)
            return 0
        finally:
            stop_demo(demo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo-executable', type=Path, required=True)
    parser.add_argument('--sdk-config', type=Path, required=True)
    parser.add_argument('--dev-config', required=True)
    parser.add_argument('--hand', choices=('left', 'right'), required=True)
    parser.add_argument('--max-current-limit', type=float, nargs=MOTOR_COUNT,
                        default=[0.5] * MOTOR_COUNT,
                        help='28 motor current limits in A (default: 0.5 each).')
    parser.add_argument('--zero-tolerance', type=float, default=ZERO_TOLERANCE_DEG,
                        help='Maximum absolute position error in degrees (default: 1).')
    parser.add_argument('--zero-timeout', type=float, default=ZERO_TIMEOUT_SEC,
                        help='Seconds from SDK startup to confirm all motors at zero (default: 10).')
    args = parser.parse_args()
    if not args.demo_executable.is_file() or not os.access(args.demo_executable, os.X_OK):
        parser.error(f'SDK demo executable is missing: {args.demo_executable}. Build yhand_mapping with the SDK checkout.')
    if not args.sdk_config.is_file():
        parser.error(f'SDK M2 configuration was not found: {args.sdk_config}')
    if any(not math.isfinite(value) or value <= 0 for value in args.max_current_limit):
        parser.error('--max-current-limit must contain 28 finite positive values')
    if not math.isfinite(args.zero_tolerance) or args.zero_tolerance <= 0:
        parser.error('--zero-tolerance must be finite and positive')
    if not math.isfinite(args.zero_timeout) or args.zero_timeout <= 0:
        parser.error('--zero-timeout must be finite and positive')
    stopping = False

    def on_signal(_signum, _frame):
        nonlocal stopping
        stopping = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, on_signal)
    return run(args, lambda: stopping)


if __name__ == '__main__':
    raise SystemExit(main())
