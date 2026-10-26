#!/usr/bin/env python3
"""Independent five-finger motor-feedback state estimation. No motor commands."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from left_thumb_motor_to_q_estimator import (
    FeedbackState, MotorToQEstimator, PACKAGE, create_estimator_node, finite, fit_model)
from left_little_q_to_motor import FINGER_SPECS

FINGERS = ('thumb', 'index', 'middle', 'ring', 'little')
DATA_ROOT = PACKAGE/'scripts/motor_test_logs/data_collection'
COLLECTIONS = {'thumb': '20261019_170304_left_collection', 'index': 'left_hand',
               'middle': 'middle_data', 'ring': 'ring_data', 'little': 'little_data'}
CACHES = {'index': 'analysis/index_linearity/data_cache.npz',
          'middle': 'analysis/middle_linearity/data_cache.npz',
          'ring': 'analysis/ring_distribution/result_arrays.npz'}


def load_finger_data(collection, finger, refresh=False):
    """Read paired result means; never mix feedback axes across different runs."""
    import numpy as np

    folder = next((p for p in [collection/'left_hand'/finger, collection/finger, collection]
                   if (p/'point_log.csv').is_file()), None)
    if folder is None:
        raise ValueError(f'{collection}下未找到{finger}/point_log.csv')
    point_log = folder/'point_log.csv'
    with point_log.open(encoding='utf-8-sig') as stream:
        metadata = list(csv.DictReader(stream))
    if not metadata or any(row['finger'] != finger for row in metadata):
        raise ValueError('采集日志为空或手指标记错误')
    point_ids = np.array([int(row['point_id']) for row in metadata])
    if len(np.unique(point_ids)) != len(point_ids):
        raise ValueError('采集日志point_id重复')
    own_cache = collection/f'analysis/motor_to_q_estimator/{finger}/data_cache.npz'
    candidates = [own_cache]
    if finger in CACHES:
        candidates += [collection/CACHES[finger]]
    cache = next((p for p in candidates if p.is_file()), None) if not refresh else None
    if cache is not None:
        snapshot = cache.parent/'point_log_snapshot.csv'
        if snapshot.is_file():
            with snapshot.open(encoding='utf-8-sig') as stream:
                if list(csv.DictReader(stream)) != metadata:
                    raise ValueError('缓存采集日志已变化；使用--refresh-cache重新读取')
        with np.load(cache, allow_pickle=False) as data:
            ids = data['point_ids'] if 'point_ids' in data else data['ids']
            if not np.array_equal(ids, point_ids):
                raise ValueError('缓存point_id与当前日志不一致；使用--refresh-cache重新读取')
            glove = data['glove'].copy() if 'glove' in data else data['glove48'].copy()
            feedback = data['feedback'].copy()
    else:
        os.environ.setdefault('MPLCONFIGDIR', '/tmp/yhand_inverse_fit_mpl')
        from plot_collection_combo_distribution import load_points
        points, report = load_points(folder/'result', point_log, 'left', finger, [], False)
        if any(report[key] for key in ['invalid_points', 'missing_glove_point_ids',
                                      'missing_motor_point_ids', 'unlabelled_point_ids']):
            raise ValueError('采集结果存在无效或缺失配对')
        if not np.array_equal([p['point_id'] for p in points], point_ids):
            raise ValueError('结果与日志point_id顺序不一致')
        glove = np.stack([p['glove'] for p in points])
        feedback = np.stack([p['motor'] for p in points])
        own_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(own_cache, point_ids=point_ids, glove=glove, feedback=feedback)
        (own_cache.parent/'point_log_snapshot.csv').write_bytes(point_log.read_bytes())
        cache = own_cache
    if glove.shape != (len(metadata), 48) or feedback.shape != (len(metadata), 28, 5):
        raise ValueError('缓存glove/feedback形状错误')
    start, end = FINGER_SPECS[finger]['glove_slice']
    ids = tuple(FINGER_SPECS[finger]['motor_inputs'])
    q, m = glove[:, start:end], feedback[:, np.array(ids)-1, 0]
    if not np.isfinite(q).all() or not np.isfinite(m).all():
        raise ValueError('采集q或电机反馈包含非有限值')
    keep = np.array([row['active_mode'] == 'xy' for row in metadata])
    if keep.sum() < 30:
        raise ValueError('正式xy采样点不足30个')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return q[keep], m[keep], [r for r, k in zip(metadata, keep) if k], {
        'cache_sha256': digest(cache), 'point_log_sha256': digest(point_log),
        'point_log': str(point_log.relative_to(PACKAGE)) if point_log.is_relative_to(PACKAGE) else str(point_log),
        'active_mode': 'xy', 'mixed_session_training': False}


def load_models(model_dir, selected=FINGERS, model=None):
    return {finger: MotorToQEstimator(model_dir/f'left_{finger}_motor_to_q_estimator.json',
                                     model, finger=finger) for finger in selected}


class FiveFingerState:
    """Each finger fails independently; a complete vector requires all five."""

    def __init__(self, models, *, timeout=.5, q_zero=None, motor_offset=None,
                 allow_extrapolation=False):
        if tuple(models) != FINGERS:
            raise ValueError('五指模型必须按thumb/index/middle/ring/little排列')
        self.models = models
        self.motor_ids = tuple(mid for model in models.values() for mid in model.motor_ids)
        zero = ([v for model in models.values() for v in model.q_zero] if q_zero is None
                else finite(q_zero, 20, '五指q参考'))
        offsets = [0.0]*24 if motor_offset is None else finite(motor_offset, 24, '五指电机偏移')
        self.states = {}
        cursor = 0
        for i, (finger, model) in enumerate(models.items()):
            n = len(model.motor_ids)
            self.states[finger] = FeedbackState(model, timeout=timeout, q_zero=zero[i*4:i*4+4],
                motor_offset=offsets[cursor:cursor+n], allow_extrapolation=allow_extrapolation)
            cursor += n

    def update(self, message, now):
        for state in self.states.values():
            state.update(message, now)

    def snapshot(self, now):
        fingers = {f: s.snapshot(now) for f, s in self.states.items()}
        valid = all(s['valid'] for s in fingers.values())
        publishable = all(s['publishable'] for s in fingers.values())
        result = {'state': 'ok' if valid else 'incomplete_or_invalid', 'valid': valid,
                  'publishable': publishable, 'fingers': fingers,
                  'sequence': next(iter(fingers.values()))['sequence'],
                  'finger_order': list(FINGERS), 'feedback_age_sec': next(iter(fingers.values()))['feedback_age_sec']}
        if all('q_absolute_deg' in s for s in fingers.values()):
            for key in ['q_absolute_deg', 'q_relative_deg']:
                result[key] = [value for s in fingers.values() for value in s[key]]
        return result


def create_five_node(args, models):
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from devices_pkg.msg import YHandMsg
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension, String

    class FiveNode(Node):
        def __init__(self):
            super().__init__('left_hand_motor_to_q_estimator')
            self.state = FiveFingerState(models, timeout=args.feedback_timeout, q_zero=args.q_zero,
                motor_offset=args.motor_offset, allow_extrapolation=args.allow_extrapolation)
            self.outputs = {}
            for finger in ['five', *FINGERS]:
                prefix = 'left_hand' if finger == 'five' else f'left_{finger}'
                topics = ([args.q_topic, args.relative_q_topic, args.joint_state_topic, args.valid_topic]
                          if finger == 'five' else [f'/{prefix}_estimated_q', f'/{prefix}_estimated_q_relative',
                               f'/{prefix}_estimated_joint_states', f'/{prefix}_estimate_valid'])
                self.outputs[finger] = [self.create_publisher(typ, topic, 10)
                    for typ, topic in zip([Float64MultiArray, Float64MultiArray, JointState, Bool], topics)]
            self.status_pub = self.create_publisher(String, args.status_topic, 10)
            self.create_subscription(YHandMsg, args.feedback_topic, self.feedback_cb, qos_profile_sensor_data)
            self.create_timer(1.0/args.publish_hz, self.tick)
            self.last_published = dict.fromkeys(self.outputs, 0)
            self.last_status = None
            self.last_status_at = -math.inf
            self.receipt_stamp = None
            from motor_to_q_logging import attach_log
            self.diagnostic_log = attach_log(self, args, models)
            self.get_logger().info(f'订阅{args.feedback_topic}，五指24电机→20个绝对角（度）；不发送电机命令')

        def feedback_cb(self, msg):
            self.receipt_stamp = self.get_clock().now().to_msg()
            now = time.monotonic()
            self.state.update(msg, now)
            if self.diagnostic_log is not None:
                self.diagnostic_log.feedback(msg, self.state.snapshot(now), now)

        def destroy_node(self):
            try:
                if self.diagnostic_log is not None:
                    self.diagnostic_log.close()
            except OSError as exc:
                self.get_logger().error(f'日志关闭失败：{exc}')
            return super().destroy_node()

        def tick(self):
            now = time.monotonic()
            status = self.state.snapshot(now)
            for finger, s in [('five', status), *status['fingers'].items()]:
                q_pub, rel_pub, joint_pub, valid_pub = self.outputs[finger]
                valid_pub.publish(Bool(data=s['valid']))
                if not s['publishable'] or self.last_published[finger] == s['sequence']:
                    continue
                size = 20 if finger == 'five' else 4
                for pub, key in [(q_pub, 'q_absolute_deg'), (rel_pub, 'q_relative_deg')]:
                    msg = Float64MultiArray(data=s[key])
                    msg.layout.dim = [MultiArrayDimension(label='thumb_index_middle_ring_little' if finger == 'five' else finger,
                                                          size=size, stride=size)]
                    pub.publish(msg)
                joint = JointState()
                joint.header.stamp = self.receipt_stamp
                joint.name = ([name for model in models.values() for name in model.q_names]
                              if finger == 'five' else list(models[finger].q_names))
                joint.position = [math.radians(v) for v in s['q_absolute_deg']]
                joint_pub.publish(joint)
                self.last_published[finger] = s['sequence']
            signature = tuple(s['state'] for s in status['fingers'].values())
            if signature != self.last_status or now-self.last_status_at >= 1:
                self.status_pub.publish(String(data=json.dumps(status, allow_nan=False)))
                self.get_logger().info('; '.join(f"{f}: {s['state']}; q={s.get('q_absolute_deg')}"
                                               for f, s in status['fingers'].items()))
                self.last_status, self.last_status_at = signature, now

    return FiveNode()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    fit = subs.add_parser('fit', help='从各指真实配对数据独立拟合，不反解q→m模型')
    fit.add_argument('--finger', choices=['five', *FINGERS], default='five')
    fit.add_argument('--data-root', type=Path, default=DATA_ROOT)
    fit.add_argument('--collection-dir', type=Path, help='单指采集根目录；五指模式用--data-root')
    fit.add_argument('--model-dir', type=Path, default=PACKAGE/'config')
    fit.add_argument('--refresh-cache', action='store_true')
    predict = subs.add_parser('predict', help='不启动ROS的离线预览')
    run = subs.add_parser('run', help='订阅实际电机反馈，五指或单指状态估计')
    for sub in [predict, run]:
        sub.add_argument('--finger', choices=['five', *FINGERS], default='five')
        sub.add_argument('--model-dir', type=Path, default=PACKAGE/'config')
        sub.add_argument('--model', choices=['auto', 'linear', 'quadratic'], default='auto')
        sub.add_argument('--q-zero', type=float, nargs='+', help='单指4角或五指20角；默认读取各模型参考')
        sub.add_argument('--motor-offset', type=float, nargs='+', help='按模型motor_ids顺序；默认全0')
    predict.add_argument('--positions', type=float, nargs='+', required=True)
    run.add_argument('--feedback-topic', '--ifeedback-topic', default='/yhand_left_feedback')
    for option in ['q-topic', 'relative-q-topic', 'joint-state-topic', 'valid-topic', 'status-topic']:
        run.add_argument('--'+option, default=None)
    run.add_argument('--publish-hz', type=float, default=50)
    run.add_argument('--feedback-timeout', type=float, default=.5)
    run.add_argument('--allow-extrapolation', action='store_true')
    run.add_argument('--no-log', dest='log_enabled', action='store_false', help='关闭记录；默认开启')
    run.add_argument('--log-dir', type=Path, help='日志目录；默认按北京时间创建新目录，拒绝覆盖已有文件')
    run.add_argument('--q-cmd-topic', help='默认/left_<finger>_q_command_state，记录控制器实际采用的目标')
    run.add_argument('--glove-topic', default='/glove_joint_data', help='操作员手套48维输入')
    run.add_argument('--glove-raw-topic', default='/glove_joint_data_raw_skeleton')
    run.add_argument('--robot-glove-topic', default='', help='可选独立机器手测量；必须与操作员手套话题不同')
    run.add_argument('--log-max-age', type=float, default=.1, help='侧输入与电机反馈接收时刻的最大对齐年龄，秒')
    args = parser.parse_args(argv)
    selected = FINGERS if args.finger == 'five' else (args.finger,)
    try:
        if args.command == 'fit':
            if args.finger == 'five' and args.collection_dir is not None:
                raise ValueError('--collection-dir仅用于单指')
            for finger in selected:
                collection = args.collection_dir or args.data_root/COLLECTIONS[finger]
                fit_model(argparse.Namespace(finger=finger, collection_dir=collection,
                    refresh_cache=args.refresh_cache, output=args.model_dir/f'left_{finger}_motor_to_q_estimator.json'))
            return 0
        models = load_models(args.model_dir, selected, None if args.model == 'auto' else args.model)
        count = sum(len(m.motor_ids) for m in models.values())
        if args.q_zero is not None:
            args.q_zero = finite(args.q_zero, 4*len(selected), 'q参考')
        if args.motor_offset is not None:
            args.motor_offset = finite(args.motor_offset, count, '电机偏移')
        if args.command == 'predict':
            m = finite(args.positions, count, '电机反馈位置')
            msg = type('Feedback', (), {})()
            msg.motors = [type('Motor', (), {'pos': 0.0, 'error': 0})() for _ in range(28)]
            for mid, value in zip([mid for model in models.values() for mid in model.motor_ids], m):
                msg.motors[mid-1].pos = value
            state = (FiveFingerState(models, q_zero=args.q_zero, motor_offset=args.motor_offset,
                                    allow_extrapolation=True) if args.finger == 'five' else
                     FeedbackState(models[args.finger], q_zero=args.q_zero, motor_offset=args.motor_offset,
                                   allow_extrapolation=True))
            state.update(msg, 0)
            print(json.dumps(state.snapshot(0), allow_nan=False))
            return 0
        if not math.isfinite(args.publish_hz) or args.publish_hz <= 0:
            raise ValueError('publish_hz必须为正有限数')
        if not math.isfinite(args.log_max_age) or args.log_max_age <= 0:
            raise ValueError('log_max_age必须为正有限数')
        if args.robot_glove_topic and args.robot_glove_topic == args.glove_topic:
            raise ValueError('机器手实测手套话题不能与操作员输入话题相同')
        if args.q_cmd_topic is None:
            args.q_cmd_topic = f'/left_{args.finger}_q_command_state'
        FeedbackState(next(iter(models.values())), timeout=args.feedback_timeout)
        prefix = 'left_hand' if args.finger == 'five' else f'left_{args.finger}'
        for key, suffix in [('q_topic', 'estimated_q'), ('relative_q_topic', 'estimated_q_relative'),
                            ('joint_state_topic', 'estimated_joint_states'), ('valid_topic', 'estimate_valid'),
                            ('status_topic', 'estimate_status')]:
            if getattr(args, key) is None:
                setattr(args, key, f'/{prefix}_{suffix}')
        topics = [args.feedback_topic, args.q_topic, args.relative_q_topic, args.joint_state_topic,
                  args.valid_topic, args.status_topic]
        if args.log_enabled:
            topics += [args.q_cmd_topic, args.glove_topic]
            topics += [t for t in [args.glove_raw_topic, args.robot_glove_topic] if t]
        if args.finger == 'five':
            topics += [f'/left_{finger}_{suffix}' for finger in FINGERS for suffix in
                       ['estimated_q', 'estimated_q_relative', 'estimated_joint_states', 'estimate_valid']]
        if len(set(topics)) != len(topics) or any(not t.strip() for t in topics):
            raise ValueError('输入输出话题必须非空且互不相同')
        import rclpy
        from rclpy.executors import ExternalShutdownException
        rclpy.init(args=[])
        node = None
        try:
            node = create_five_node(args, models) if args.finger == 'five' else create_estimator_node(args, models[args.finger])
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f'五指状态估计错误：{exc}\n')


if __name__ == '__main__':
    sys.exit(main())
