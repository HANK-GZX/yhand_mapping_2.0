#!/usr/bin/env python3
"""Read-only capture of operator thumb q and motor feedback; offline profile build.

The operator prepares and verifies each robot pose/tendon pretension using one
separate command source. This process never publishes motor commands or zeros.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time

from thumb_combo_policy import COMBOS, MOTORS, ThumbComboPolicy, positive, vector

PACKAGE = Path(__file__).resolve().parents[2]


def average(records, key):
    return [sum(record[key][j] for record in records)/len(records) for j in range(len(records[0][key]))]


def accepted_frame(records, combo, label, *, now, window_sec, max_age=.2,
                   max_q_span=3., max_motor_span=3.):
    if combo not in ('neutral', *COMBOS):
        raise ValueError('未知combo')
    recent = [r for r in records if now-window_sec <= r['time'] <= now]
    if len(recent) < 10 or recent[-1]['time'] < now-max_age or recent[0]['time'] > now-window_sec+.15:
        raise ValueError('需要至少一个完整窗口的新鲜反馈和左手套数据')
    if any(b['time']-a['time'] > max_age for a,b in zip(recent[:-1],recent[1:])):
        raise ValueError('采样窗口内发生数据断流')
    # Averages refer to separate streams joined only while both are fresh.
    for key, limit in [('operator_q_deg', max_q_span), ('motor_positions_deg', max_motor_span)]:
        size = 4 if key == 'operator_q_deg' else 6
        for j in range(size):
            values = [vector(r[key], size, key)[j] for r in recent]
            if max(values)-min(values) > limit:
                raise ValueError(f'{key}未稳定，通道{j}跨度超过{limit:g}°')
    currents = [vector(r['motor_current_A'], 6, '电流') for r in recent]
    return {'label': label, 'combo_id': combo,
            'operator_q_deg': average(recent, 'operator_q_deg'),
            'motor_positions_deg': average(recent, 'motor_positions_deg'),
            'motor_current_A': [sum(v[j] for v in currents)/len(currents) for j in range(6)],
            'samples': len(recent), 'window_sec': window_sec,
            'accepted_taut_pose': True}


def build_profile(recording, settings):
    result = copy.deepcopy(settings)
    if recording.get('hand') != 'left' or recording.get('motor_order') != list(MOTORS):
        raise ValueError('需要左拇指六电机标定记录')
    frames = recording['frames']
    if len(frames) < 3 or frames[0]['combo_id'] != 'neutral':
        raise ValueError('先采集中立，再采集外展和内收对掌关键点')
    neutral = frames[0]
    q0 = vector(neutral['operator_q_deg'], 4, '中立操作手角度')
    m0 = vector(neutral['motor_positions_deg'], 6, '中立电机位置')
    tolerance = positive(result['neutral_motor_tolerance_deg'], 'neutral_motor_tolerance_deg')
    if max(abs(x) for x in m0) > tolerance:
        raise ValueError('中立电机不在原零位附近；不能用减掉反馈位置来偷偷改机械零位')
    result.update(calibration_status='measured', motor_zero_id=recording['motor_zero_id'],
                  operator_zero_q_deg=q0, keyframes=[])
    for i, frame in enumerate(frames):
        if frame.get('accepted_taut_pose') is not True or frame.get('samples', 0) < 10:
            raise ValueError('关键点需要操作者确认动作和预紧且包含稳定采样')
        q = vector(frame['operator_q_deg'], 4, '操作手角度')
        p = vector(frame['motor_positions_deg'], 6, '电机位置')
        # Store actual motor encoder coordinates. Only the neutral frame snaps
        # to the existing zero (within tolerance); no per-tendon re-zero occurs.
        result['keyframes'].append({'label': frame['label'], 'combo_id': frame['combo_id'],
            'q_relative_deg': [v-z for v, z in zip(q, q0)],
            'motor_positions_deg': p if i else [0.]*6,
            'measured_current_A': vector(frame['motor_current_A'], 6, '关键点电流')})
    # Use the runtime validator before handing a file to the controller.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'validate.json'
        path.write_text(json.dumps(result, allow_nan=False))
        ThumbComboPolicy(path)
    return result


def capture(args):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from devices_pkg.msg import YHandMsg
    from std_msgs.msg import Float64MultiArray
    rclpy.init(args=[])
    node = rclpy.create_node('left_thumb_combo_calibration_recorder')
    data = {'q': None, 'q_at': None, 'feedback': None, 'feedback_at': None, 'left_at': None}
    records, frames = [], []
    commands = queue.Queue()

    def glove_cb(msg):
        try:
            q = vector(list(msg.data)[8:12], 4, '左拇指手套角')
        except ValueError:
            return
        data.update(q=q, q_at=time.monotonic())

    def raw_cb(msg):
        values = list(msg.data)
        # Same left-hand detection as the shared teleop controller.
        if values and len(values) % 176 == 0:
            for start in range(0, len(values), 176):
                chunk = values[start:start+176]
                if all(math.isfinite(v) for v in chunk) and abs(chunk[0]) <= 1e-6:
                    data['left_at'] = time.monotonic()
                    break

    def feedback_cb(msg):
        motors = list(msg.motors)
        if len(motors) != 28:
            return
        try:
            rows = [vector([motors[mid-1].pos, motors[mid-1].vel, motors[mid-1].tor,
                            motors[mid-1].error], 4, f'M{mid}反馈') for mid in MOTORS]
        except (ValueError, AttributeError):
            return
        if any(row[3] != 0 for row in rows):
            data['feedback_at'] = None
            return
        data.update(feedback=rows, feedback_at=time.monotonic())

    node.create_subscription(Float64MultiArray, args.glove_topic, glove_cb, qos_profile_sensor_data)
    node.create_subscription(Float64MultiArray, args.raw_topic, raw_cb, qos_profile_sensor_data)
    node.create_subscription(YHandMsg, args.feedback_topic, feedback_cb, 1)

    def read_input():
        for line in sys.stdin:
            commands.put(line.strip())
        commands.put('quit')

    threading.Thread(target=read_input, daemon=True).start()
    print('只记录，不发电机指令。先由唯一控制源调好机器人姿态和预紧，同时摆好对应操作手姿态。', flush=True)
    print('确认后输入：neutral / out [标签] / in [标签]；可重复out/in增加关键点；undo撤销；save保存；quit退出。', flush=True)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.02)
            now = time.monotonic()
            if all(data[key] is not None and 0 <= now-data[key] <= .2
                   for key in ('q_at', 'feedback_at', 'left_at')):
                fb = data['feedback']
                records.append({'time': now, 'operator_q_deg': list(data['q']),
                                'motor_positions_deg': [row[0] for row in fb],
                                'motor_current_A': [row[2] for row in fb]})
            records[:] = [r for r in records if r['time'] >= now-args.window_sec-.5]
            while not commands.empty():
                command = commands.get().split(maxsplit=1)
                if not command:
                    continue
                action = command[0]
                if action == 'quit':
                    return
                if action == 'undo':
                    if frames:
                        frames.pop()
                    print(f'保留{len(frames)}个关键点', flush=True)
                elif action == 'save':
                    payload = {'schema_version': 1, 'hand': 'left', 'motor_order': list(MOTORS),
                               'motor_zero_id': args.motor_zero_id, 'frames': frames,
                               'captured_utc': datetime.now(timezone.utc).isoformat()}
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
                    print(f'已保存{len(frames)}个关键点到{args.output}', flush=True)
                elif action in ('neutral', 'out', 'in'):
                    if (action == 'neutral' and frames) or (action != 'neutral' and not frames):
                        print('首点必须是neutral；需要重新采集中立时先undo', flush=True)
                        continue
                    combo = 'neutral' if action == 'neutral' else f'thumb_flex_long_{action}_opp_on'
                    try:
                        if not all(data[key] is not None and 0 <= now-data[key] <= .2
                                   for key in ('q_at', 'feedback_at', 'left_at')):
                            raise ValueError('当前电机反馈或左手套数据无效/过期')
                        frame = accepted_frame(records, combo, command[1] if len(command) > 1 else action,
                                               now=now, window_sec=args.window_sec)
                        frames.append(frame)
                        print('已记录：'+json.dumps(frame, ensure_ascii=False), flush=True)
                    except ValueError as exc:
                        print(f'未记录：{exc}', flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    cap = modes.add_parser('capture', help='只读订阅，人工确认稳定关键点')
    cap.add_argument('--output', type=Path, required=True)
    cap.add_argument('--motor-zero-id', required=True, help='本次预紧归零标识；不能跨零位复用')
    cap.add_argument('--glove-topic', default='/glove_joint_data')
    cap.add_argument('--raw-topic', default='/glove_joint_data_raw_skeleton')
    cap.add_argument('--feedback-topic', default='/yhand_left_feedback')
    cap.add_argument('--window-sec', type=float, default=1.)
    build = modes.add_parser('build', help='离线从采集记录和实测约束生成控制文件')
    build.add_argument('--recording', type=Path, required=True)
    build.add_argument('--settings', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.mode == 'capture':
            positive(args.window_sec, 'window_sec')
            if args.window_sec < .5:
                raise ValueError('采样窗口不能短于0.5秒')
            if not args.motor_zero_id.strip():
                raise ValueError('motor-zero-id不能为空')
            capture(args)
        else:
            result = build_profile(json.loads(args.recording.read_text()), json.loads(args.settings.read_text()))
            result['recording_sha256'] = hashlib.sha256(args.recording.read_bytes()).hexdigest()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
            print(args.output)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
