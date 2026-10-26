"""Time-aligned diagnostic logs for the independent motor-to-q estimators."""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from left_thumb_motor_to_q_estimator import PACKAGE, finite


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


class EstimatorLog:
    """One sample row per received feedback, latest-before-receipt side inputs.

    Raw events preserve the asynchronous streams. Age checks are reception-time
    checks; source messages have no common hardware timestamp.
    """

    def __init__(self, args, models):
        self.models = models
        self.fingers = tuple(models)
        self.names = [f'{f}_q{i}' for f in models for i in range(4)]
        default = PACKAGE/'scripts/motor_test_logs/motor_to_q_estimator'/datetime.now(
            ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        self.directory = Path(getattr(args, 'log_dir', None) or default)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any((self.directory/name).exists() for name in ['metadata.json', 'samples.csv', 'events.jsonl']):
            raise FileExistsError(f'日志目录已有记录，拒绝覆盖：{self.directory}')
        self.started = time.monotonic()
        self.max_age = args.log_max_age
        self.latest = {}
        self.sequences = {}
        self.left_at = None
        self.rows = 0
        self.last_flush = self.started
        self.closed = False
        # Exclusive creation protects an earlier run from accidental overwrite.
        self.samples = (self.directory/'samples.csv').open('x', newline='', encoding='utf-8')
        self.events = None
        try:
            self.events = (self.directory/'events.jsonl').open('x', encoding='utf-8')
            (self.directory/'models').mkdir(exist_ok=True)
            snapshots = {}
            for finger, model in models.items():
                # Freeze the configuration loaded in memory, even if the on-disk
                # model was refitted while this estimator was already running.
                data = (json.dumps(model.config, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
                name = f'left_{finger}_motor_to_q_estimator.json'
                (self.directory/'models'/name).write_bytes(data)
                snapshots[finger] = {'file': f'models/{name}', 'sha256': hashlib.sha256(data).hexdigest(),
                                     'selected_model': model.model, 'motor_ids': list(model.motor_ids),
                                     'source_model_path': str(model.path)}
            self.metadata = {
                'schema_version': 1, 'log_type': 'motor_to_q_estimator',
                'started_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                'finger_order': list(models), 'q_order': self.names, 'q_units': 'degrees',
                'motor_units': 'degrees', 'model_snapshots': snapshots,
                'feedback_topic': args.feedback_topic, 'q_cmd_topic': args.q_cmd_topic,
                'glove_topic': args.glove_topic, 'glove_raw_topic': args.glove_raw_topic,
                'robot_glove_topic': args.robot_glove_topic,
                'glove_role': 'operator_input', 'robot_glove_available': bool(args.robot_glove_topic),
                'log_max_age_sec': self.max_age, 'feedback_timeout_sec': args.feedback_timeout,
                'q_zero_deg': args.q_zero or [v for m in models.values() for v in m.q_zero],
                'motor_offset_deg': args.motor_offset or [0.0]*sum(len(m.motor_ids) for m in models.values()),
                'alignment': 'One row per feedback receipt; latest earlier received q_cmd/glove with age and sequence. Raw independent events retained. Not hardware-synchronized.',
                'q_cmd_definition': 'Controller request before motor clipping, absolute and relative; actual published motor target logged separately.',
                'comparison_note': 'Operator glove and q_cmd are not measured robot q. Without an independent robot glove, estimate-minus-command is consistency/tracking deviation, not joint-estimation accuracy.',
                'rows': 0, 'closed_cleanly': False}
            self.write_metadata()
            fields = ['elapsed_sec', 'unix_time_sec', 'feedback_sequence', 'command_active', 'command_state',
                      'q_cmd_age_sec', 'q_cmd_sequence', 'q_cmd_aligned', 'glove_age_sec', 'glove_sequence',
                      'glove_left_raw_age_sec', 'glove_aligned', 'robot_glove_age_sec', 'robot_glove_sequence',
                      'robot_glove_aligned']
            fields += [f'{prefix}_{name}' for prefix in ['q_cmd_abs', 'q_cmd_rel', 'q_cmd_reference',
                       'operator_glove', 'robot_glove', 'q_est_abs', 'q_est_rel'] for name in self.names]
            fields += [f'{finger}_{key}' for finger in models for key in
                       ['state', 'valid', 'publishable', 'outside_motor_ids', 'outside_q_indices']]
            fields += [f'm{mid}_{key}' for mid in range(1, 29) for key in
                       ['position', 'velocity', 'current', 'error', 'command', 'model_position']]
            self.writer = csv.DictWriter(self.samples, fieldnames=fields)
            self.writer.writeheader()
        except Exception:
            self.samples.close()
            if self.events is not None:
                self.events.close()
            raise

    def write_metadata(self):
        self.metadata['rows'] = self.rows
        temporary = self.directory/'metadata.json.tmp'
        temporary.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2)+'\n')
        temporary.replace(self.directory/'metadata.json')

    def event(self, source, payload, now):
        seq = self.sequences.get(source, 0)+1
        self.sequences[source] = seq
        self.events.write(json.dumps(json_safe({'source': source, 'sequence': seq,
            'elapsed_sec': now-self.started, 'unix_time_sec': time.time(), 'data': payload}),
            ensure_ascii=False, allow_nan=False)+'\n')
        return seq

    def q_command(self, packet, now):
        try:
            if not isinstance(packet, dict):
                raise ValueError('q_cmd遥测需要JSON对象')
            if packet.get('schema_version') != 1 or packet.get('hand') != 'left':
                raise ValueError('q_cmd遥测版本或手侧错误')
            if not isinstance(packet.get('active'), bool) or not isinstance(packet.get('control_state'), str):
                raise ValueError('q_cmd active或control_state错误')
            fingers = packet['fingers']
            if not fingers or len(set(fingers)) != len(fingers) or any(f not in
                    ('thumb', 'index', 'middle', 'ring', 'little') for f in fingers):
                raise ValueError('q_cmd手指顺序错误')
            count = 4*len(fingers)
            finite(packet['q_reference_deg'], count, 'q_cmd参考')
            if packet['active']:
                finite(packet['q_cmd_absolute_deg'], count, 'q_cmd绝对角')
                finite(packet['q_cmd_relative_deg'], count, 'q_cmd相对角')
            if len(packet['motor_position_command_deg']) != 28:
                raise ValueError('q_cmd电机位置需要28路')
            for value in packet['motor_position_command_deg']:
                if value is not None and not math.isfinite(float(value)):
                    raise ValueError('q_cmd电机位置包含非有限数')
            seq = self.event('q_cmd', packet, now)
            self.latest['q_cmd'] = (now, seq, packet)
        except (ValueError, KeyError, TypeError) as exc:
            self.latest.pop('q_cmd', None)
            self.event('invalid_q_cmd', {'reason': str(exc)}, now)

    def glove(self, values, now, *, robot=False):
        source = 'robot_glove' if robot else 'glove'
        try:
            values = finite(values, 48, source)
            seq = self.event(source, values, now)
            self.latest[source] = (now, seq, values)
        except (ValueError, TypeError) as exc:
            self.latest.pop(source, None)
            self.event('invalid_'+source, {'reason': str(exc)}, now)

    def glove_raw(self, values, now):
        if values and len(values) % 176 == 0 and any(
                abs(values[i]) <= 1e-6 and all(math.isfinite(v) for v in values[i:i+176])
                for i in range(0, len(values), 176)):
            self.left_at = now

    def feedback(self, message, status, now):
        rows = [{'position': float(m.pos), 'velocity': float(m.vel), 'current': float(m.tor),
                 'error': int(m.error)} for m in message.motors]
        seq = self.event('feedback', {'motors': rows, 'estimation': status}, now)
        row = {'elapsed_sec': now-self.started, 'unix_time_sec': time.time(), 'feedback_sequence': seq}
        for source in ['q_cmd', 'glove', 'robot_glove']:
            sample = self.latest.get(source)
            age = None if sample is None else max(0.0, now-sample[0])
            row[source+'_age_sec'] = age
            row[source+'_sequence'] = None if sample is None else sample[1]
            aligned = age is not None and age <= self.max_age
            if source == 'glove' and self.metadata['glove_raw_topic']:
                raw_age = None if self.left_at is None else max(0.0, now-self.left_at)
                row['glove_left_raw_age_sec'] = raw_age
                aligned = aligned and raw_age is not None and raw_age <= self.max_age
            row[source+'_aligned'] = int(aligned)
            if sample is None:
                continue
            data = sample[2]
            if source == 'q_cmd':
                row['command_active'] = int(data['active'])
                row['command_state'] = data['control_state']
                for i, finger in enumerate(data['fingers']):
                    if finger not in self.models:
                        continue
                    for j in range(4):
                        name = f'{finger}_q{j}'
                        row['q_cmd_reference_'+name] = data['q_reference_deg'][4*i+j]
                        if data['active']:
                            row['q_cmd_abs_'+name] = data['q_cmd_absolute_deg'][4*i+j]
                            row['q_cmd_rel_'+name] = data['q_cmd_relative_deg'][4*i+j]
                for mid, value in enumerate(data['motor_position_command_deg'], 1):
                    row[f'm{mid}_command'] = value
            else:
                prefix = 'robot_glove' if source == 'robot_glove' else 'operator_glove'
                for finger, model in self.models.items():
                    start, _ = model.config['glove_q_slice']
                    for j in range(4):
                        row[f'{prefix}_{finger}_q{j}'] = data[start+j]
        finger_status = status['fingers'] if 'fingers' in status else {self.fingers[0]: status}
        for finger, s in finger_status.items():
            for key in ['state', 'valid', 'publishable']:
                row[f'{finger}_{key}'] = int(s[key]) if isinstance(s[key], bool) else s[key]
            for key in ['outside_motor_ids', 'outside_q_indices']:
                row[f'{finger}_{key}'] = json.dumps(s.get(key, []))
            for prefix, key in [('q_est_abs', 'q_absolute_deg'), ('q_est_rel', 'q_relative_deg')]:
                for j, value in enumerate(s.get(key, [])):
                    row[f'{prefix}_{finger}_q{j}'] = value
            for mid, value in zip(self.models[finger].motor_ids, s.get('model_motor_positions_deg', [])):
                row[f'm{mid}_model_position'] = value
        for mid, values in enumerate(rows, 1):
            if mid <= 28:
                for key, value in values.items():
                    row[f'm{mid}_{key}'] = value if math.isfinite(value) else None
        self.writer.writerow(row)
        self.rows += 1
        if now-self.last_flush >= 1:
            self.samples.flush()
            self.events.flush()
            self.write_metadata()
            self.last_flush = now

    def close(self):
        if not self.closed:
            self.samples.close()
            self.events.close()
            self.metadata['closed_cleanly'] = True
            self.metadata['elapsed_sec'] = time.monotonic()-self.started
            self.write_metadata()
            self.closed = True


def attach_log(node, args, models):
    """Logging is enabled by the new CLI; old callers keep their previous behavior."""
    if not getattr(args, 'log_enabled', False):
        return None
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Float64MultiArray, String

    log = EstimatorLog(args, models)

    def command(message):
        try:
            packet = json.loads(message.data)
        except ValueError:
            packet = {}
        log.q_command(packet, time.monotonic())

    node.create_subscription(String, args.q_cmd_topic, command, 10)
    node.create_subscription(Float64MultiArray, args.glove_topic,
                             lambda m: log.glove(m.data, time.monotonic()), qos_profile_sensor_data)
    if args.glove_raw_topic:
        node.create_subscription(Float64MultiArray, args.glove_raw_topic,
                                 lambda m: log.glove_raw(m.data, time.monotonic()), qos_profile_sensor_data)
    if args.robot_glove_topic:
        node.create_subscription(Float64MultiArray, args.robot_glove_topic,
            lambda m: log.glove(m.data, time.monotonic(), robot=True), qos_profile_sensor_data)
    node.get_logger().info(f'状态估计日志：{log.directory}；手套默认标记为操作员输入')
    return log
