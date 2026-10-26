#!/usr/bin/env python3
"""Independent left-thumb feedback-position -> glove-angle estimator.

fit: offline supervised regression; run: subscribe to feedback, publish estimates;
predict: inspect one six-position input without importing ROS. Never sends commands.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

# q->m definitions live beside the estimator folder; collection loaders stay in scripts.
SCRIPTS = Path(__file__).resolve().parents[1]
for directory in (SCRIPTS, SCRIPTS/'q_2_m linear mapping'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from left_little_q_to_motor import FINGER_SPECS


PACKAGE = SCRIPTS.parent
MODEL_PATH = PACKAGE / 'config/left_thumb_motor_to_q_estimator.json'
MOTOR_IDS = (4, 6, 7, 8, 14, 27)
Q_NAMES = tuple(f'left_thumb_q{i}' for i in range(4))
Q_ZERO = (-3.0, 42.0, 40.0, -36.0)
SEED = 20261023


def finite(values, size, label):
    values = [float(v) for v in values]
    if len(values) != size or not all(math.isfinite(v) for v in values):
        raise ValueError(f'{label}需要{size}个有限数值')
    return values


def feature_row(x, degree):
    result = [1.0, *x]
    if degree == 2:
        result += [x[i]*x[j] for i in range(len(x)) for j in range(i, len(x))]
    return result


class MotorToQEstimator:
    """Pure-Python inference in recorded motor-feedback coordinates, degrees."""

    def __init__(self, path=MODEL_PATH, model=None, *, finger='thumb'):
        self.path = Path(path)
        self.config = json.loads(self.path.read_text(encoding='utf-8'))
        c = self.config
        if finger not in FINGER_SPECS:
            raise ValueError('未知手指')
        self.finger = finger
        self.motor_ids = tuple(FINGER_SPECS[finger]['motor_inputs'])
        self.q_names = tuple(f'left_{finger}_q{i}' for i in range(4))
        if (c.get('schema_version') != 1 or c.get('mapping') != 'motor_feedback_to_q'
                or c.get('hand') != 'left' or c.get('finger') != finger
                or c.get('motor_ids') != list(self.motor_ids) or c.get('q_names') != list(self.q_names)
                or c.get('motor_units') != 'degrees' or c.get('q_units') != 'degrees'
                or c.get('output_coordinates') != 'absolute_glove_angles'):
            raise ValueError(f'需要左手{finger}电机反馈到四个绝对手套角的版本1模型')
        self.q_zero = finite(c['relative_output_reference_deg'], 4, '相对角参考')
        self.model = model or c['default_model']
        if self.model not in ('linear', 'quadratic'):
            raise ValueError('model必须是linear或quadratic')
        fit = c['fits'][self.model]
        self.degree = 1 if self.model == 'linear' else 2
        if fit['degree'] != self.degree:
            raise ValueError('模型名称与多项式阶数不一致')
        self.center = finite(fit['input_center'], len(self.motor_ids), '输入中心')
        self.scale = finite(fit['input_scale'], len(self.motor_ids), '输入尺度')
        if min(self.scale) <= 0:
            raise ValueError('输入尺度必须大于0')
        terms = len(feature_row([0.0]*len(self.motor_ids), self.degree))
        self.coefficients = [finite(row, 4, '输出系数') for row in fit['coefficients']]
        if len(self.coefficients) != terms:
            raise ValueError('特征数与系数行数不一致')
        self.bounds = [finite(c['motor_training_bounds_deg'][str(mid)], 2, '训练范围')
                       for mid in self.motor_ids]
        if any(lo > hi for lo, hi in self.bounds):
            raise ValueError('训练范围上下界颠倒')
        self.q_bounds = [finite(c['q_training_bounds_deg'][f'q{i}'], 2, '关节训练范围')
                         for i in range(4)]
        if any(lo > hi for lo, hi in self.q_bounds):
            raise ValueError('关节训练范围上下界颠倒')

    def predict(self, positions):
        m = finite(positions, len(self.motor_ids), '电机位置')
        try:
            x = [(v-c)/s for v, c, s in zip(m, self.center, self.scale)]
            phi = feature_row(x, self.degree)
            return finite([math.fsum(v*row[j] for v, row in zip(phi, self.coefficients))
                           for j in range(4)], 4, '估计关节角')
        except OverflowError as exc:
            raise ValueError('电机输入过大，多项式计算溢出') from exc

    def outside_motors(self, positions):
        m = finite(positions, len(self.motor_ids), '电机位置')
        # Float32 transport and CSV-decimal round trips can straddle endpoints.
        return [mid for mid, v, (lo, hi) in zip(self.motor_ids, m, self.bounds)
                if v < lo-1e-5 or v > hi+1e-5]

    def outside_q(self, q):
        values = finite(q, 4, '估计关节角')
        return [i for i, (v, (lo, hi)) in enumerate(zip(values, self.q_bounds))
                if v < lo-1e-5 or v > hi+1e-5]


class FeedbackState:
    """Freshness and input-quality state independent of ROS and wall clocks."""

    def __init__(self, estimator, *, timeout=0.5, q_zero=None,
                 motor_offset=None, allow_extrapolation=False):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('feedback_timeout必须为正有限数')
        self.estimator = estimator
        self.timeout = timeout
        self.q_zero = finite(estimator.q_zero if q_zero is None else q_zero, 4, 'q零位')
        self.motor_offset = finite([0.0]*len(estimator.motor_ids) if motor_offset is None else motor_offset,
                                   len(estimator.motor_ids), '电机坐标偏移')
        self.allow_extrapolation = allow_extrapolation
        self.sequence = 0
        self.received_at = None
        self.last = {'state': 'waiting_for_feedback', 'valid': False, 'publishable': False}

    def update(self, message, now):
        self.sequence += 1
        self.received_at = now
        try:
            if len(message.motors) != 28:
                raise ValueError('YHandMsg需要28台电机')
            selected = [message.motors[mid-1] for mid in self.estimator.motor_ids]
            raw = finite([m.pos for m in selected], len(selected), '反馈位置')
            error_motors = [mid for mid, m in zip(self.estimator.motor_ids, selected) if int(m.error) != 0]
            if error_motors:
                self.last = {'state': 'motor_error', 'valid': False, 'publishable': False,
                             'error_motors': error_motors}
                return
            positions = finite([v-o for v, o in zip(raw, self.motor_offset)], len(selected), '对齐后的反馈位置')
            q = self.estimator.predict(positions)
            relative_q = finite([v-z for v, z in zip(q, self.q_zero)], 4, '估计相对角')
            outside = self.estimator.outside_motors(positions)
            outside_q = self.estimator.outside_q(q)
            in_bounds = not outside and not outside_q
            self.last = {'state': 'ok' if in_bounds else 'outside_training_bounds',
                         'valid': in_bounds, 'publishable': in_bounds or self.allow_extrapolation,
                         'outside_motor_ids': outside, 'outside_q_indices': outside_q,
                         'motor_positions_deg': raw,
                         'model_motor_positions_deg': positions, 'q_absolute_deg': q,
                         'q_relative_deg': relative_q}
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            self.last = {'state': 'invalid_feedback', 'valid': False, 'publishable': False,
                         'reason': str(exc)}

    def snapshot(self, now):
        age = None if self.received_at is None else max(0.0, now-self.received_at)
        if age is not None and age > self.timeout:
            return {'state': 'stale_feedback', 'valid': False, 'publishable': False,
                    'feedback_age_sec': age, 'sequence': self.sequence}
        return {**self.last, 'feedback_age_sec': age, 'sequence': self.sequence}


def fit_model(args):
    import numpy as np
    from fit_left_thumb_q_to_motor import load

    collection = args.collection_dir.resolve()
    finger = getattr(args, 'finger', 'thumb')
    if finger == 'thumb':
        q, m, metadata, provenance = load(collection, args.refresh_cache)
        q_zero, reference_source = list(Q_ZERO), 'User supplied motor-zero angles; relative subtraction only, not a fit constraint.'
    else:
        from left_hand_motor_to_q_estimator import load_finger_data
        q, m, metadata, provenance = load_finger_data(collection, finger, args.refresh_cache)
        forward = json.loads((PACKAGE/f'config/left_{finger}_q_to_motor_models.json').read_text(encoding='utf-8'))
        q_zero, reference_source = forward['q_reference'], forward['q_reference_source']
    motor_ids = tuple(FINGER_SPECS[finger]['motor_inputs'])
    dimension = len(motor_ids)
    groups = np.array([r['combo_id'] for r in metadata])
    if len(np.unique(groups)) < 2:
        raise ValueError('至少需要两个动作组合才能进行留组合验证')

    def design(x, degree):
        return np.array([feature_row(row, degree) for row in x])

    def solve(x, y, degree):
        center, scale = x.mean(0), x.std(0)
        scale[scale < 1e-12] = 1
        phi = design((x-center)/scale, degree)
        beta, _, rank, _ = np.linalg.lstsq(phi, y, rcond=None)
        return center, scale, beta, int(rank)

    def predict(x, fitted, degree):
        center, scale, beta, _ = fitted
        return design((x-center)/scale, degree) @ beta

    def metrics(target, predicted):
        err = predicted-target
        return {'n': len(target), 'mae_deg': abs(err).mean(0).tolist(),
                'rmse_deg': np.sqrt((err*err).mean(0)).tolist(),
                'p95_abs_error_deg': np.percentile(abs(err), 95, axis=0).tolist()}

    folds = np.empty(len(q), int)
    rng = np.random.default_rng(SEED)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        folds[rng.permutation(indices)] = np.arange(len(indices)) % 5
    result = {
        'schema_version': 1, 'mapping': 'motor_feedback_to_q', 'hand': 'left', 'finger': finger,
        'motor_ids': list(motor_ids), 'q_names': [f'left_{finger}_q{i}' for i in range(4)],
        'glove_q_slice': list(FINGER_SPECS[finger]['glove_slice']),
        'motor_units': 'degrees', 'q_units': 'degrees',
        'output_coordinates': 'absolute_glove_angles', 'default_model': 'quadratic',
        'training_points': len(q), 'training_timeouts': sum(r['timeout'] == '1' for r in metadata),
        'training_scope': f"One mean point per formal active_mode={'full' if finger == 'thumb' else 'xy'} result pair, including timeouts; actual motor feedback positions, never command targets.",
        'source_collection': str(collection.relative_to(PACKAGE)) if collection.is_relative_to(PACKAGE) else str(collection),
        'source_sha256': provenance, 'validation_seed': SEED,
        'motor_training_bounds_deg': {str(mid): [float(m[:, i].min()), float(m[:, i].max())]
                                      for i, mid in enumerate(motor_ids)},
        'q_training_bounds_deg': {f'q{i}': [float(q[:, i].min()), float(q[:, i].max())] for i in range(4)},
        'relative_output_reference_deg': q_zero,
        'relative_reference_source': reference_source+'; relative subtraction only; no fit constraint or automatic encoder re-zeroing.',
        'limitations': 'Validation is within one recorded session against mean glove labels, not dynamic/contact or cross-session accuracy. Marginal motor bounds do not establish joint coverage. Motor offsets must match training encoder coordinates; no automatic boot-time zero alignment.',
        'fits': {}}
    for name, degree in [('linear', 1), ('quadratic', 2)]:
        fitted = solve(m, q, degree)
        validation = {}
        for scheme, labels in [('stratified_5fold', folds), ('leave_one_combo_out', groups)]:
            estimate = np.empty_like(q)
            for label in np.unique(labels):
                test = labels == label
                estimate[test] = predict(m[test], solve(m[~test], q[~test], degree), degree)
            validation[scheme] = metrics(q, estimate)
        center, scale, beta, rank = fitted
        result['fits'][name] = {
            'degree': degree, 'input_center': center.tolist(), 'input_scale': scale.tolist(),
            'feature_order': ['1', *[f'x{i}' for i in range(dimension)]] + (
                [f'x{i}*x{j}' for i in range(dimension) for j in range(i, dimension)] if degree == 2 else []),
            'coefficients': beta.tolist(), 'design_rank': rank,
            'training_metrics': metrics(q, predict(m, fitted, degree)), 'validation': validation}
    result['default_model'] = min(result['fits'], key=lambda name:
        sum(result['fits'][name]['validation']['leave_one_combo_out']['mae_deg']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'model_file': str(args.output), 'training_points': len(q),
                      'default_model': result['default_model'],
                      'selected_leave_combo_mae_deg': result['fits'][result['default_model']]['validation']['leave_one_combo_out']['mae_deg'],
                      'quadratic_leave_combo_mae_deg': result['fits']['quadratic']['validation']['leave_one_combo_out']['mae_deg']}, ensure_ascii=False))


def create_estimator_node(args, estimator):
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from devices_pkg.msg import YHandMsg
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension, String

    class EstimatorNode(Node):
        def __init__(self):
            super().__init__(f'left_{estimator.finger}_motor_to_q_estimator')
            self.state = FeedbackState(estimator, timeout=args.feedback_timeout,
                                       q_zero=args.q_zero, motor_offset=args.motor_offset,
                                       allow_extrapolation=args.allow_extrapolation)
            self.q_pub = self.create_publisher(Float64MultiArray, args.q_topic, 10)
            self.relative_pub = self.create_publisher(Float64MultiArray, args.relative_q_topic, 10)
            self.joint_pub = self.create_publisher(JointState, args.joint_state_topic, 10)
            self.valid_pub = self.create_publisher(Bool, args.valid_topic, 10)
            self.status_pub = self.create_publisher(String, args.status_topic, 10)
            self.create_subscription(YHandMsg, args.feedback_topic, self.feedback_cb, qos_profile_sensor_data)
            self.create_timer(1.0/args.publish_hz, self.tick)
            self.last_published = 0
            self.last_status = None
            self.last_status_at = -math.inf
            self.receipt_stamp = None
            from motor_to_q_logging import attach_log
            self.diagnostic_log = attach_log(self, args, {estimator.finger: estimator})
            self.get_logger().info(f'订阅 {args.feedback_topic}，M{estimator.motor_ids} → {args.q_topic}（绝对角，度）；不启动驱动或发送电机命令')

        def feedback_cb(self, message):
            self.receipt_stamp = self.get_clock().now().to_msg()
            now = time.monotonic()
            self.state.update(message, now)
            if self.diagnostic_log is not None:
                self.diagnostic_log.feedback(message, self.state.snapshot(now), now)

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
            self.valid_pub.publish(Bool(data=status['valid']))
            if status['publishable'] and status['sequence'] != self.last_published:
                for pub, key in [(self.q_pub, 'q_absolute_deg'), (self.relative_pub, 'q_relative_deg')]:
                    msg = Float64MultiArray(data=status[key])
                    msg.layout.dim = [MultiArrayDimension(label=f'left_{estimator.finger}_q0_q1_q2_q3_degrees', size=4, stride=4)]
                    pub.publish(msg)
                joint = JointState()
                joint.header.stamp = self.receipt_stamp
                joint.name = list(estimator.q_names)
                joint.position = [math.radians(v) for v in status['q_absolute_deg']]
                self.joint_pub.publish(joint)
                self.last_published = status['sequence']
            if status['state'] != self.last_status or now-self.last_status_at >= 1.0:
                self.status_pub.publish(String(data=json.dumps(status, allow_nan=False)))
                log = f"{status['state']}; q={status.get('q_absolute_deg')}; age={status['feedback_age_sec']}"
                self.get_logger().info(log)
                self.last_status, self.last_status_at = status['state'], now

    return EstimatorNode()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit', help='离线拟合独立m→q模型并保存验证指标')
    fit.add_argument('--collection-dir', type=Path, default=PACKAGE/'scripts/motor_test_logs/data_collection/20261019_170304_left_collection')
    fit.add_argument('--output', type=Path, default=MODEL_PATH)
    fit.add_argument('--refresh-cache', action='store_true')
    predict = commands.add_parser('predict', help='离线预览六路反馈位置对应的估计q')
    run = commands.add_parser('run', help='只订阅电机反馈并发布估计q，不启动硬件')
    for sub in [predict, run]:
        sub.add_argument('--model-file', type=Path, default=MODEL_PATH)
        sub.add_argument('--model', choices=['linear', 'quadratic'])
        sub.add_argument('--q-zero', type=float, nargs=4, default=Q_ZERO,
                         help='仅定义相对输出；默认-3 42 40 -36，不改变绝对q模型')
        sub.add_argument('--motor-offset', type=float, nargs=6, default=[0.0]*6,
                         help='模型输入=实际反馈-offset，按M4/6/7/8/14/27；默认不改变坐标')
    predict.add_argument('--positions', type=float, nargs=6, required=True)
    run.add_argument('--feedback-topic', '--ifeedback-topic', default='/yhand_left_feedback')
    run.add_argument('--q-topic', default='/left_thumb_estimated_q')
    run.add_argument('--relative-q-topic', default='/left_thumb_estimated_q_relative')
    run.add_argument('--joint-state-topic', default='/left_thumb_estimated_joint_states')
    run.add_argument('--valid-topic', default='/left_thumb_estimate_valid')
    run.add_argument('--status-topic', default='/left_thumb_estimate_status')
    run.add_argument('--publish-hz', type=float, default=50.0)
    run.add_argument('--feedback-timeout', type=float, default=0.5)
    run.add_argument('--allow-extrapolation', action='store_true',
                     help='超出训练范围仍输出诊断角度，valid保持false；默认暂停发布角度')
    args = parser.parse_args(argv)
    try:
        if args.command == 'fit':
            fit_model(args)
            return 0
        estimator = MotorToQEstimator(args.model_file, args.model)
        finite(args.q_zero, 4, 'q零位')
        finite(args.motor_offset, 6, '电机坐标偏移')
        if args.command == 'predict':
            m = finite([v-o for v, o in zip(args.positions, args.motor_offset)], 6, '对齐后的电机位置')
            q = estimator.predict(m)
            print(json.dumps({'q_absolute_deg': q, 'q_relative_deg': finite([v-z for v, z in zip(q, args.q_zero)], 4, '估计相对角'),
                              'outside_motor_ids': estimator.outside_motors(m),
                              'outside_q_indices': estimator.outside_q(q)}, allow_nan=False))
            return 0
        if not math.isfinite(args.publish_hz) or args.publish_hz <= 0:
            raise ValueError('publish_hz必须为正有限数')
        FeedbackState(estimator, timeout=args.feedback_timeout)
        topics = [args.feedback_topic, args.q_topic, args.relative_q_topic, args.joint_state_topic,
                  args.valid_topic, args.status_topic]
        if any(not t.strip() for t in topics) or len(set(topics)) != len(topics):
            raise ValueError('订阅和各输出话题必须非空且互不相同')
        import rclpy
        from rclpy.executors import ExternalShutdownException
        rclpy.init(args=[])
        node = None
        try:
            node = create_estimator_node(args, estimator)
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f'状态估计器错误：{exc}\n')


if __name__ == '__main__':
    sys.exit(main())
