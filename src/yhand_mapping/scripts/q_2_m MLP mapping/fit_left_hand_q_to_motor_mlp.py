#!/usr/bin/env python3
"""Fit five independent 4-q MLPs from paired collection results, entirely offline."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/yhand_q_to_motor_mlp_mpl')
import numpy as np

from mlp_q_to_motor import (ALIGNMENT, INPUT_COORDINATES, FINGER_SPECS, MODEL_DIR, PACKAGE,
                            SIDE_ALIGNMENT, SIDE_WINDING_LIMIT_DEG, MLPFingerMapper, forward, side_bending_policy)
from role_weighted_training import (SECONDARY_WEIGHT, PRIORITY_PENALTY, role_input_weights,
                                    build_network, input_priority_penalty, export_network)

SCRIPTS = PACKAGE/'scripts'
sys.path.insert(0, str(SCRIPTS/'q_estimator'))
from left_hand_motor_to_q_estimator import load_finger_data
from fit_left_thumb_q_to_motor import load as load_thumb

FINGERS = ('thumb', 'index', 'middle', 'ring', 'little')
SEED = 20261026


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(root, finger, refresh=False):
    """Load original absolute glove angles and unchanged measured motor labels."""
    collection = root/f'{finger}_data'
    if finger == 'thumb':
        q, m, meta, provenance = load_thumb(collection, refresh)
    else:
        q, m, meta, provenance = load_finger_data(collection, finger, refresh)
    # Same ring-side population as the reviewed current model; all four targets
    # below are actual positions from THIS run, never spliced across sessions.
    if finger == 'ring':
        keep = np.array([int(r['point_id']) < 2214 for r in meta])
        q, m, meta = q[keep], m[keep], [r for r, k in zip(meta, keep) if k]
    if not len(q) or not np.isfinite(q).all() or not np.isfinite(m).all():
        raise ValueError('训练数据为空或包含非有限值')
    return q, m, meta, dict(provenance, collection=str(collection.relative_to(PACKAGE))
                          if collection.is_relative_to(PACKAGE) else str(collection))


def delta_angles(q_absolute, reference):
    """Use the same per-finger robot zero as teleoperation, before standardization."""
    q = np.asarray(q_absolute, dtype=float)
    zero = np.asarray(reference, dtype=float)
    if q.ndim != 2 or q.shape[1] != 4 or zero.shape != (4,):
        raise ValueError('训练角度需要N×4，零位需要4个角度')
    delta = q-zero
    if not np.isfinite(delta).all():
        raise ValueError('相对训练角度包含非有限值')
    return delta


def pose_groups(q, step=2.0):
    """Keep repeated/nearby poses in the same outer and inner fold."""
    return np.unique(np.floor(np.asarray(q)/step).astype(np.int64), axis=0, return_inverse=True)[1]


def grouped_folds(groups, count=5, seed=SEED):
    unique = np.unique(groups)
    if len(unique) < count:
        raise ValueError('独立姿态组数量不足')
    rng = np.random.default_rng(seed)
    mapping = dict(zip(rng.permutation(unique).tolist(), (np.arange(len(unique)) % count).tolist()))
    return np.array([mapping[g] for g in groups])


def metrics(y, pred):
    error = pred-y
    denom = np.sum((y-y.mean(0))**2, axis=0)
    return {'mae_motor_deg': np.abs(error).mean(0).tolist(),
            'rmse_motor_deg': np.sqrt((error**2).mean(0)).tolist(),
            'p95_abs_motor_deg': np.percentile(abs(error), 95, axis=0).tolist(),
            'bias_motor_deg': error.mean(0).tolist(),
            'r2': (1-np.sum(error**2, axis=0)/np.maximum(denom, 1e-12)).tolist()}


def polynomial(q, y, test_q, degree):
    center, scale = q.mean(0), np.maximum(q.std(0), 1e-8)
    def phi(a):
        x = (a-center)/scale
        columns = [np.ones(len(x)), *x.T]
        if degree == 2:
            columns += [x[:, i]*x[:, j] for i in range(4) for j in range(i, 4)]
        return np.column_stack(columns)
    return phi(test_q) @ np.linalg.lstsq(phi(q), y, rcond=None)[0]


def train(q, y, *, epochs, seed, width=32, validation=None, patience=80,
          input_weights=None, priority_penalty=PRIORITY_PENALTY):
    """Fit delta-angle inputs to unchanged feedback labels; train-only scaling."""
    import torch
    torch.manual_seed(seed)
    x_center, x_scale = q.mean(0), np.maximum(q.std(0), 1e-8)
    y_center, y_scale = y.mean(0), np.maximum(y.std(0), 1e-8)
    tensor = lambda a: torch.tensor(a, dtype=torch.float32)
    x, target = tensor((q-x_center)/x_scale), tensor((y-y_center)/y_scale)
    net = build_network(y.shape[1], width, input_weights)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.01, weight_decay=1e-4)
    if validation is not None:
        vx, vy = validation
        vx, vy = tensor((vx-x_center)/x_scale), tensor((vy-y_center)/y_scale)
    best, best_epoch, best_state = float('inf'), 0, None
    for epoch in range(1, epochs+1):
        optimizer.zero_grad(set_to_none=True)
        loss = (net(x)-target).square().mean()+priority_penalty*input_priority_penalty(net)
        if not torch.isfinite(loss):
            raise ValueError('MLP训练损失非有限')
        loss.backward()
        optimizer.step()
        if validation is not None:
            with torch.no_grad():
                score = float((net(vx)-vy).square().mean())
            if score < best-1e-5:
                best, best_epoch = score, epoch
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            if epoch-best_epoch >= patience:
                break
    if validation is not None:
        net.load_state_dict(best_state)
    else:
        best_epoch = epochs
    network = export_network(net, x_center, x_scale, y_center, y_scale)
    # Verify the portable evaluator matches the actual trained Torch model.
    with torch.no_grad():
        expected = net(x).numpy()*y_scale+y_center
    if not np.allclose(expected, forward(network, q), atol=2e-4, rtol=1e-5):
        raise ValueError('Torch与导出推理不一致')
    return network, best_epoch


def validate(q, y, combo, *, epochs, width, seed, input_weights=None, priority_penalty=PRIORITY_PENALTY):
    groups = pose_groups(q)
    grouped = grouped_folds(groups, seed=seed)
    result, predictions, selected_epochs = {}, {}, []
    for scheme, labels in [('pose_group_5fold', grouped), ('leave_one_combo_out', combo)]:
        predicted = {name: np.zeros_like(y) for name in ['linear', 'mlp']}
        fold_details = []
        for i, label in enumerate(np.unique(labels)):
            test = labels == label
            outer_train = np.flatnonzero(~test)
            # Select epochs on an INNER pose split; outer test is never used for
            # optimization, scaling or checkpoint selection. Refit all outer rows.
            inner = grouped_folds(groups[outer_train], seed=seed+i+1)
            fit_rows, val_rows = outer_train[inner != 0], outer_train[inner == 0]
            _, chosen = train(q[fit_rows], y[fit_rows], epochs=epochs, width=width, seed=seed+i,
                              validation=(q[val_rows], y[val_rows]), input_weights=input_weights,
                              priority_penalty=priority_penalty)
            network, _ = train(q[~test], y[~test], epochs=chosen, width=width, seed=seed+i,
                               input_weights=input_weights, priority_penalty=priority_penalty)
            predicted['mlp'][test] = forward(network, q[test])
            for name, degree in [('linear', 1)]:
                predicted[name][test] = polynomial(q[~test], y[~test], q[test], degree)
            fold_details.append({'label': str(label), 'train': int((~test).sum()),
                                 'test': int(test.sum()), 'selected_epochs': chosen})
            if scheme == 'pose_group_5fold':
                selected_epochs.append(chosen)
            print(f'  {scheme} {i+1}/{len(np.unique(labels))}: epochs={chosen}', flush=True)
        result[scheme] = {name: metrics(y, pred) for name, pred in predicted.items()}
        result[scheme]['folds'] = fold_details
        predictions[scheme] = predicted
    return result, predictions, int(np.median(selected_epochs)), groups, grouped


def fit_finger(finger, args, report_dir):
    print(f'{finger}: loading paired data', flush=True)
    q_absolute, y, meta, provenance = load_data(args.data_root.resolve(), finger, args.refresh_cache)
    base_path = PACKAGE/f'config/left_{finger}_q_to_motor_models.json'
    base = json.loads(base_path.read_text())
    reference = base['q_reference']
    q = delta_angles(q_absolute, reference)
    ids = list(FINGER_SPECS[finger]['motor_inputs'])
    gains = role_input_weights(finger, ids, getattr(args, 'secondary_input_weight', SECONDARY_WEIGHT))
    penalty = getattr(args, 'input_priority_penalty', PRIORITY_PENALTY)
    combo = np.array([r['combo_id'] for r in meta])
    print(f'{finger}: {len(q)} points, {len(np.unique(combo))} combos', flush=True)
    evaluation, predictions, epochs, groups, folds = validate(q, y, combo, epochs=args.epochs,
        width=args.width, seed=args.seed, input_weights=gains, priority_penalty=penalty)
    network, _ = train(q, y, epochs=epochs, width=args.width, seed=args.seed,
                       input_weights=gains, priority_penalty=penalty)
    reference_outside = [i for i in range(4) if not q[:, i].min() <= 0 <= q[:, i].max()]
    config = {'schema_version': 3 if 'branches' in network else 2, 'hand': 'left', 'finger': finger, 'model_type': 'mlp', 'mapping': 'q_to_motor',
        'q_units': 'degrees', 'motor_position_units': 'degrees', 'motor_order': ids,
        'input_coordinates': INPUT_COORDINATES, 'q_reference': reference,
        'training_input_definition': 'delta_q = recorded_glove_angles - q_reference; before standardization',
        'training_target_definition': 'Unchanged recorded motor feedback positions in degrees',
        'input_priority': {'per_motor_weights': dict(zip(map(str, ids), gains.tolist())),
                           'applied_after_standardization': True, 'penalty_strength': penalty,
                           'policy': 'Side: q0 primary; flexion/extension: q1/q2/q3 primary. Soft preference, not a guaranteed derivative ratio.'},
        'q_reference_source': base['q_reference_source'],
        'reference_is_exact_zero_calibration': base['reference_is_exact_zero_calibration'],
        'calibration_measured_by_script': False,
        'output_alignment': ALIGNMENT,
        'side_motor_mapping': 'All motors use all four q through the independently trained finger MLP',
        'training_points': len(q), 'training_timeouts': sum(r['timeout'] == '1' for r in meta),
        'training_scope': 'Formal full (thumb) or xy (other fingers), including movement timeouts; ring point_id<2214',
        'source': provenance, 'reference_model_sha256': sha(base_path),
        'q_absolute_training_bounds_deg': np.column_stack([q_absolute.min(0), q_absolute.max(0)]).tolist(),
        'q_delta_training_bounds_deg': np.column_stack([q.min(0), q.max(0)]).tolist(),
        'motor_feedback_training_bounds_deg': np.column_stack([y.min(0), y.max(0)]).tolist(),
        'reference_outside_training_q_indices': reference_outside,
        'raw_prediction_at_reference_motor_deg': forward(network, np.zeros(4)).tolist(),
        'fit_limitation': 'Validation is raw recorded motor feedback, NOT rebased motor-zero control accuracy or hardware tracking. '
                          'Reference can lie outside sampled poses; neutral subtraction does not calibrate physical zero. '
                          'No robot test or closed-loop q control performed. Ring uses four same-session targets from ring_data, '
                          'whereas the current polynomial uses older extensor/flexor data.',
        'training': {'seed': args.seed, 'architecture': [4, args.width, args.width, len(ids)],
                     'architecture_type': network.get('architecture', 'dense_shared'),
                     'activation': 'tanh', 'optimizer': 'AdamW', 'lr': .01, 'weight_decay': 1e-4,
                     'epochs': epochs, 'maximum_selection_epochs': args.epochs, 'early_stop_patience': 80,
                     'selection': 'Median inner-selected epochs from five outer pose folds; final fit on all rows',
                     'validation_pose_bin_deg': 2.0, 'torch_version': args.torch_version},
        'validation': evaluation, 'network': network}
    policy = side_bending_policy(finger, getattr(args, 'side_bending_winding_limit', SIDE_WINDING_LIMIT_DEG))
    if policy is not None:
        config.update(schema_version=4, output_alignment=SIDE_ALIGNMENT, side_bending_policy=policy,
            side_motor_mapping='q0-only anchor plus bending correction; negative correction smoothly bounded, positive payout preserved',
            control_policy_validation_note='Stored regression MAE evaluates raw network feedback predictions, before neutral alignment and the side bending policy; not hardware tracking accuracy.')
    path = args.output_dir/f'left_{finger}_q_to_motor_mlp.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    model = MLPFingerMapper(temporary, finger=finger)
    if any(model.predict([0]*4).values()):
        raise ValueError('零输入必须对应严格零输出')
    temporary.replace(path)
    folder = report_dir/finger
    folder.mkdir(parents=True, exist_ok=True)
    (folder/path.name).write_bytes(path.read_bytes())
    (folder/'point_log_snapshot.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    np.savez_compressed(folder/'validation_predictions.npz', q=q, q_delta=q, q_absolute=q_absolute,
        q_reference=reference, input_coordinates=INPUT_COORDINATES, motor_feedback=y,
        motor_ids=ids, pose_group=groups, pose_fold=folds, point_ids=[int(r['point_id']) for r in meta], combo=combo,
        **{f'{scheme}_{name}': pred for scheme, models in predictions.items() for name, pred in models.items()})
    return {k: v for k, v in config.items() if k != 'network'}


def plot_report(report, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [(f, mid, i) for f, c in report.items() for i, mid in enumerate(c['motor_order'])]
    fig, axes = plt.subplots(2, 1, figsize=(17, 9))
    for ax, scheme in zip(axes, ['pose_group_5fold', 'leave_one_combo_out']):
        for k, name in enumerate(['linear', 'mlp']):
            values = [report[f]['validation'][scheme][name]['mae_motor_deg'][i] for f, _, i in rows]
            ax.bar(np.arange(len(rows))+(k-.5)*.35, values, width=.35, label=name)
        ax.set_xticks(np.arange(len(rows)), [f'{f}\nM{mid}' for f, mid, _ in rows], rotation=55, ha='right')
        ax.set_ylabel('Held-out motor MAE (deg)')
        ax.set_title(scheme+' | raw feedback coordinates; not hardware tracking')
        ax.legend()
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(folder/'validation_mae.png', dpi=160)
    plt.close(fig)
    text = ['# q→m MLP 离线验证', '',
        '误差单位为电机角度 °；比较模型在相同数据、相同外层留出样本上的原始反馈坐标预测。',
        'linear 是本批重新拟合的四q基线，不是直接评测当前已减零位的控制输出。', '',
        '| 手指 | 点数 | 姿态分组 MAE：linear / MLP | 留一组合 MAE：linear / MLP |',
        '|---|---:|---|---|']
    for finger, c in report.items():
        values = [' / '.join(f"{np.mean(c['validation'][s][n]['mae_motor_deg']):.2f}" for n in ['linear', 'mlp'])
                  for s in ['pose_group_5fold', 'leave_one_combo_out']]
        text.append(f"| {finger} | {c['training_points']} | {values[0]} | {values[1]} |")
    text += ['', '各指网络结构与超参数保存在模型 training 字段；tanh 隐藏层，训练和标准化不读取外层测试集。',
        '姿态按四q每维2°分箱，同箱样本保持在同一折；邻箱之间仍可能接近。留一组合评估未见过的肌肉组合。',
        '早停只读内层验证；选定轮数后在整份外层训练集重训。最终模型在全量样本上按内层轮数中位数重训。',
        '每指原始预测、折分、point_id、combo 和模型快照保存在对应子目录。', '',
        '训练输入为Δq=采集角度−本指零位角度，电机标签保持原始反馈；标准化只使用训练折。',
        '四指侧摆与伸屈使用独立32×32分支；标准化后主轴权重1、次轴默认0.25，并惩罚弱输入路径的第一层权重。拇指保持原结构。',
        '控制使用 f(Δq)-f(0)，保持零输入零输出；原始反馈 MAE 不等于实机遥操误差。',
        '四指控制输出另对侧摆应用q0锚点约束：弯曲额外收线默认最多2°，正向放线补偿保留。上表原始反馈MAE未将此控制约束混作预测成绩。',
        '无名指统一使用 ring_data 同批四电机的2170点，保留排除末尾500个密集点的规则。',
        '该指当前多项式伸屈数据来自旧批，因此数据来源也发生改变，不能把差异全归因于网络。',
        '部分参考角在采样范围外，减去网络预测零位不能代替实机标定；尚未进行实机跟踪测试。', '',
        '![各电机留出误差](validation_mae.png)', '']
    (folder/'REPORT.md').write_text('\n'.join(text))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finger', choices=['four', 'five', *FINGERS], default='five')
    parser.add_argument('--data-root', type=Path, default=SCRIPTS/'motor_test_logs/data_collection')
    parser.add_argument('--output-dir', type=Path, default=MODEL_DIR)
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--secondary-input-weight', type=float, default=SECONDARY_WEIGHT)
    parser.add_argument('--input-priority-penalty', type=float, default=PRIORITY_PENALTY)
    parser.add_argument('--side-bending-winding-limit', type=float, default=SIDE_WINDING_LIMIT_DEG,
                        help='Side winding caused by bending is smoothly bounded to this many motor degrees')
    parser.add_argument('--refresh-cache', action='store_true')
    args = parser.parse_args(argv)
    if args.epochs < 1 or not 1 <= args.width <= 256 or args.threads < 1:
        parser.error('epochs/threads必须为正，width必须在1..256')
    if not np.isfinite(args.secondary_input_weight) or not 0 < args.secondary_input_weight <= 1:
        parser.error('secondary-input-weight必须在(0,1]内')
    if not np.isfinite(args.input_priority_penalty) or args.input_priority_penalty < 0:
        parser.error('input-priority-penalty必须为非负有限数')
    if not np.isfinite(args.side_bending_winding_limit) or not 0 < args.side_bending_winding_limit <= 20:
        parser.error('side-bending-winding-limit必须在(0,20]电机度内')
    import torch
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.torch_version = str(torch.__version__)
    folder = args.report_dir or SCRIPTS/'motor_test_logs/q_to_motor_mlp'/datetime.now().strftime('%Y%m%d_%H%M%S')
    folder.mkdir(parents=True, exist_ok=False)
    report = {}
    for finger in (FINGERS if args.finger == 'five' else FINGERS[1:] if args.finger == 'four' else (args.finger,)):
        report[finger] = fit_finger(finger, args, folder)
        (folder/'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    plot_report(report, folder)
    print(f'Models: {args.output_dir}\nReport: {folder}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
