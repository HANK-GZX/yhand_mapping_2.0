#!/usr/bin/env python3
"""Compare estimator logs, or replay a historical teleoperator log offline."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('MPLCONFIGDIR', '/tmp/yhand_motor_to_q_analysis_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from left_hand_motor_to_q_estimator import PACKAGE, load_models
from left_thumb_motor_to_q_estimator import FeedbackState


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return math.nan


def replay_teleop(source, output, model='linear', model_dir=PACKAGE/'config'):
    """Observed motor feedback is replayed; no synthetic tracking measurements."""
    metadata = json.loads((source/'metadata.json').read_text())
    fingers = tuple(metadata.get('fingers', [metadata['finger']]))
    models = load_models(model_dir, fingers, None if model == 'auto' else model)
    states = {f: FeedbackState(m, allow_extrapolation=True) for f, m in models.items()}
    (output/'models').mkdir(parents=True, exist_ok=True)
    snapshots = {}
    for f, m in models.items():
        target = output/'models'/m.path.name
        target.write_text(json.dumps(m.config, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        snapshots[f] = {'file': f'models/{target.name}', 'selected_model': m.model,
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'motor_ids': list(m.motor_ids)}
    converted = []
    last_sequence = None
    with (source/'samples.csv').open() as stream:
        for row in csv.DictReader(stream):
            sequence = row['feedback_sequence']
            if sequence == last_sequence:
                continue
            last_sequence = sequence
            t = number(row['elapsed_sec'])
            feedback_age = number(row.get('feedback_age_sec'))
            active = row['state'] == 'position'
            r = {'elapsed_sec': t, 'feedback_sequence': sequence, 'command_active': int(active),
                 'command_state': row['state'], 'q_cmd_age_sec': 0,
                 'q_cmd_aligned': int(math.isfinite(feedback_age) and feedback_age <= .1),
                 'glove_age_sec': row.get('manus_age_sec', ''),
                 'glove_left_raw_age_sec': row.get('manus_left_raw_age_sec', ''),
                 'robot_glove_age_sec': row.get('glove_age_sec', '')}
            glove_age = number(row.get('manus_age_sec'))
            raw_age = number(row.get('manus_left_raw_age_sec'))
            r['glove_aligned'] = int(metadata.get('input_mode') == 'manus' and glove_age <= .1 and raw_age <= .1)
            r['robot_glove_aligned'] = int(metadata.get('robot_glove_measurement_enabled', False)
                                          and number(row.get('glove_age_sec')) <= .1)
            msg = SimpleNamespace(motors=[SimpleNamespace(pos=0.0, error=0) for _ in range(28)])
            for mid in metadata['motor_ids']:
                msg.motors[mid-1].pos = number(row.get(f'm{mid}_position'))
                error = number(row.get(f'm{mid}_error_code'))
                msg.motors[mid-1].error = int(error) if math.isfinite(error) else 1
                for dest, src in [('position', 'position'), ('velocity', 'velocity'), ('current', 'current'),
                                  ('error', 'error_code'), ('command', 'command')]:
                    r[f'm{mid}_{dest}'] = row.get(f'm{mid}_{src}', '')
                r[f'm{mid}_model_position'] = msg.motors[mid-1].pos
            for i, (f, state) in enumerate(states.items()):
                state.update(msg, t)
                s = state.snapshot(t)
                if not math.isfinite(feedback_age) or feedback_age > metadata.get('feedback_timeout_sec', .5):
                    s.update(valid=False, publishable=False, state='stale_feedback')
                for key in ['state', 'valid', 'publishable']:
                    r[f'{f}_{key}'] = int(s[key]) if isinstance(s[key], bool) else s[key]
                for key in ['outside_motor_ids', 'outside_q_indices']:
                    r[f'{f}_{key}'] = json.dumps(s.get(key, []))
                for j in range(4):
                    name = f'{f}_q{j}'
                    k = 4*i+j
                    for dest, src in [('q_cmd_abs', 'target'), ('q_cmd_rel', 'target_relative'),
                                      ('operator_glove', 'manus')]:
                        r[f'{dest}_{name}'] = row.get(f'{src}_q{k}', '')
                    if metadata.get('robot_glove_measurement_enabled', False):
                        r[f'robot_glove_{name}'] = row.get(f'measured_q{k}', '')
                    r[f'q_cmd_reference_{name}'] = metadata['q_reference'][k]
                    for dest, src in [('q_est_abs', 'q_absolute_deg'), ('q_est_rel', 'q_relative_deg')]:
                        r[f'{dest}_{name}'] = s.get(src, [math.nan]*4)[j]
            converted.append(r)
    if not converted:
        raise ValueError('遥操日志没有可回放帧')
    fields = list(dict.fromkeys(k for r in converted for k in r))
    with (output/'replayed_samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(converted)
    result = {'schema_version': 1, 'log_type': 'offline_teleop_replay', 'finger_order': list(fingers),
              'source_directory': str(source), 'source_samples_sha256': hashlib.sha256((source/'samples.csv').read_bytes()).hexdigest(),
              'model_snapshots': snapshots, 'robot_glove_available': metadata.get('robot_glove_measurement_enabled', False),
              'glove_role': 'operator_input', 'alignment': 'Historical controller rows, first row per feedback_sequence. q_cmd is the same control iteration; side-source ages from controller CSV. Estimates recomputed offline, not recorded live.',
              'q_cmd_definition': 'Controller target q before motor clipping; not a measured robot pose.',
              'comparison_note': 'Command/estimate deviation is not estimation accuracy. Operator glove is not robot ground truth.'}
    (output/'replay_metadata.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return converted, result


def statistics(error, mask):
    selected = error[mask & np.isfinite(error)]
    if not len(selected):
        return {'n': 0, 'bias_deg': None, 'mae_deg': None, 'rmse_deg': None, 'p95_abs_deg': None}
    return {'n': len(selected), 'bias_deg': float(selected.mean()), 'mae_deg': float(abs(selected).mean()),
            'rmse_deg': float(np.sqrt((selected**2).mean())), 'p95_abs_deg': float(np.percentile(abs(selected), 95))}


def analyze(rows, metadata, output):
    output.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('没有日志样本')
    col = lambda name: np.array([number(r.get(name)) for r in rows])
    t = col('elapsed_sec')
    active = col('command_active') == 1
    aligned = col('q_cmd_aligned') == 1
    robot_aligned = col('robot_glove_aligned') == 1
    summary = {'source_mode': metadata['log_type'], 'rows': len(rows),
               'duration_sec': float(np.nanmax(t)-np.nanmin(t)), 'active_rows': int(active.sum()),
               'command_aligned_active_rows': int((active & aligned).sum()),
               'robot_glove_available': metadata.get('robot_glove_available', False),
               'comparison_note': metadata['comparison_note'], 'alignment': metadata['alignment'], 'fingers': {}}
    metrics_rows = []
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
    for finger in metadata['finger_order']:
        valid = col(f'{finger}_valid') == 1
        state_config = metadata['model_snapshots'][finger]
        config = json.loads((output/state_config['file']).read_text()) if metadata['log_type'] == 'offline_teleop_replay' else state_config.get('_config')
        # New logs keep snapshots in the source directory; the caller supplies that root.
        if config is None:
            config = json.loads((Path(metadata['_source_root'])/state_config['file']).read_text())
        finger_summary = {'model': state_config['selected_model'], 'valid_fraction_active': float(valid[active].mean()) if active.any() else None,
                          'q_command_deviation': [], 'robot_glove_estimation_error': [], 'motors': {}}
        fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
        for j, ax in enumerate(axes.flat):
            name = f'{finger}_q{j}'
            estimate, command = col(f'q_est_abs_{name}'), col(f'q_cmd_abs_{name}')
            metric = statistics(estimate-command, active & aligned)
            metric['valid_only'] = statistics(estimate-command, active & aligned & valid)
            metric['estimate_min_deg'] = float(np.nanmin(estimate)) if np.isfinite(estimate).any() else None
            metric['estimate_max_deg'] = float(np.nanmax(estimate)) if np.isfinite(estimate).any() else None
            lo, hi = config['q_training_bounds_deg'][f'q{j}']
            metric['outside_q_fraction_active'] = float(((estimate < lo) | (estimate > hi))[active].mean()) if active.any() else None
            finger_summary['q_command_deviation'].append(metric)
            metrics_rows.append({'finger': finger, 'joint': j, 'comparison': 'estimate_minus_command',
                                 **{k: v for k, v in metric.items() if k != 'valid_only'}})
            ax.plot(t, command, label='q_cmd absolute', linewidth=.9)
            ax.plot(t, estimate, label='q_est absolute (incl. invalid diagnostics)', linewidth=.9)
            operator = col(f'operator_glove_{name}')
            if np.isfinite(operator).any():
                ax.plot(t, operator, '--', label='operator glove INPUT', alpha=.5, linewidth=.7)
            robot = col(f'robot_glove_{name}')
            if summary['robot_glove_available']:
                robot_metric = statistics(estimate-robot, robot_aligned)
                robot_metric['valid_only'] = statistics(estimate-robot, robot_aligned & valid)
                finger_summary['robot_glove_estimation_error'].append(robot_metric)
                ax.plot(t, robot, label='robot glove measurement', linewidth=.8)
            invalid = ~valid & np.isfinite(estimate)
            ax.scatter(t[invalid], estimate[invalid], s=2, color='red', alpha=.25, label='invalid estimate')
            ax.axhspan(lo, hi, color='green', alpha=.05)
            ax.set(title=f'{finger} q{j}', ylabel='angle / deg', xlabel='time / s')
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f'{finger}: {state_config["selected_model"]}; operator glove is input, not robot ground truth')
        fig.tight_layout()
        fig.savefig(output/f'{finger}_angles.png', dpi=150)
        plt.close(fig)
        ids = config['motor_ids']
        fig, axes = plt.subplots(math.ceil(len(ids)/2), 2, figsize=(13, 3*math.ceil(len(ids)/2)), squeeze=False)
        for mid, ax in zip(ids, axes.flat):
            position, command = col(f'm{mid}_position'), col(f'm{mid}_command')
            model_position = col(f'm{mid}_model_position')
            lo, hi = config['motor_training_bounds_deg'][str(mid)]
            m = statistics(position-command, active & aligned)
            m['model_input_outside_fraction_active'] = float(((model_position < lo) | (model_position > hi))[active].mean()) if active.any() else None
            m['feedback_min_deg'] = float(np.nanmin(position)) if np.isfinite(position).any() else None
            m['feedback_max_deg'] = float(np.nanmax(position)) if np.isfinite(position).any() else None
            m['training_bounds_deg'] = [lo, hi]
            finger_summary['motors'][str(mid)] = m
            ax.plot(t, command, label='published motor position target', linewidth=.8)
            ax.plot(t, position, label='actual feedback position', linewidth=.8)
            ax.plot(t, model_position, '--', label='model input position', linewidth=.6, alpha=.6)
            ax.axhspan(lo, hi, color='green', alpha=.08, label='training input range')
            finite_values = np.r_[position[np.isfinite(position)], command[np.isfinite(command)]]
            if len(finite_values) and np.ptp(finite_values) > 1000:
                ax.set_yscale('symlog', linthresh=10)
            ax.set(title=f'M{mid}', xlabel='time / s', ylabel='motor angle / deg')
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(ids):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f'{finger}: motor commands, feedback, and estimator input')
        fig.tight_layout()
        fig.savefig(output/f'{finger}_motors.png', dpi=150)
        plt.close(fig)
        summary['fingers'][finger] = finger_summary
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (output/'angle_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics_rows[0]))
        writer.writeheader()
        writer.writerows(metrics_rows)
    lines = ['# 遥操与关节状态估计日志分析', '',
             f"来源：`{metadata.get('source_directory', metadata.get('_source_root'))}`；模式：`{metadata['log_type']}`。", '',
             f"样本数 {summary['rows']}，时长 {summary['duration_sec']:.2f} 秒，位置控制帧 {summary['active_rows']}，其中q_cmd对齐帧 {summary['command_aligned_active_rows']}。", '',
             '操作员手套角是控制输入；q_cmd是期望姿态。估计q与它们的差值不能直接解释为机器手真实关节角误差。', '',
             '本日志'+('包含独立机器手手套测量。' if summary['robot_glove_available'] else '**未包含独立机器手实测角**，因此不报告实机关节估计精度。'), '',
             f"时间对齐：{metadata['alignment']}", '',
             '| 手指 | 有效估计/位置控制帧 | 估计−q_cmd MAE，q0/q1/q2/q3（度） |',
             '| --- | ---: | --- |']
    for f, s in summary['fingers'].items():
        fraction = s['valid_fraction_active']
        mae = [m['mae_deg'] for m in s['q_command_deviation']]
        lines.append(f"| {f} | {fraction:.1%} | "+', '.join('无配对' if v is None else f'{v:.2f}' for v in mae)+' |' if fraction is not None else f'| {f} | 无控制帧 | 无配对 |')
    lines += ['', '上表包含超界诊断值，仅用于观察失真；有效子集指标在summary.json的valid_only中。电机位置跟踪偏差与模型输入超界比例也保存在summary.json。', '']
    lines += ['## 输入覆盖与电机跟踪', '',
              '| 手指 | 超出训练范围最多的电机 | 位控帧中超界比例 | 该电机反馈范围/度 | 训练范围/度 | 电机反馈−命令MAE/度 |',
              '| --- | --- | ---: | --- | --- | ---: |']
    for f, s in summary['fingers'].items():
        candidates = [(mid, m) for mid, m in s['motors'].items() if m['model_input_outside_fraction_active'] is not None]
        if candidates:
            mid, m = max(candidates, key=lambda pair: pair[1]['model_input_outside_fraction_active'])
            ranges = lambda v: '无有效值' if any(x is None for x in v) else f'[{v[0]:.2f}, {v[1]:.2f}]'
            error = '无配对' if m['mae_deg'] is None else f"{m['mae_deg']:.2f}"
            lines.append(f"| {f} | M{mid} | {m['model_input_outside_fraction_active']:.1%} | "+
                         ranges([m['feedback_min_deg'], m['feedback_max_deg']])+' | '+ranges(m['training_bounds_deg'])+f' | {error} |')
    lines += ['', '训练范围之外的反馈属于模型外推。电机反馈与命令的偏差描述驱动位置跟踪，不等同于关节角偏差；逐轴在范围内也不能证明联合姿态与预紧条件匹配。', '']
    for f in summary['fingers']:
        lines += [f'## {f}', '', f'![关节角]({f}_angles.png)', '', f'![电机位置]({f}_motors.png)', '']
    (output/'README.md').write_text('\n'.join(lines))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log_dir', type=Path, nargs='?')
    parser.add_argument('--teleop-log', type=Path, help='历史遥操日志：用实际反馈离线重算估计')
    parser.add_argument('--model', choices=['linear', 'quadratic', 'auto'], default='linear')
    parser.add_argument('--model-dir', type=Path, default=PACKAGE/'config')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if bool(args.log_dir) == bool(args.teleop_log):
        parser.error('选择一个估计器日志目录，或--teleop-log历史遥操目录')
    source = args.teleop_log or args.log_dir
    output = args.output_dir or source/'analysis'/('motor_to_q_'+args.model if args.teleop_log else 'comparison')
    output.mkdir(parents=True, exist_ok=True)
    if args.teleop_log:
        rows, metadata = replay_teleop(source, output, args.model, args.model_dir)
    else:
        metadata = json.loads((source/'metadata.json').read_text())
        metadata['_source_root'] = str(source)
        with (source/'samples.csv').open() as stream:
            rows = list(csv.DictReader(stream))
    result = analyze(rows, metadata, output)
    print(json.dumps({'report': str(output/'README.md'), 'rows': result['rows'],
                      'robot_glove_available': result['robot_glove_available']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
