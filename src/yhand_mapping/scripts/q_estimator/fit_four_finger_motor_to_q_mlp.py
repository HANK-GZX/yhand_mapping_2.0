#!/usr/bin/env python3
"""Train independent q0/q123 feedback estimators from four-finger paired data."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault('MPLCONFIGDIR', '/tmp/yhand_inverse_mlp_mpl')
import numpy as np

from left_hand_motor_to_q_estimator import load_finger_data
from left_thumb_motor_to_q_estimator import PACKAGE, feature_row
from left_little_q_to_motor import FINGER_SPECS
from mlp_motor_to_q import FOUR_FINGERS, SIDE_MOTORS, GroupedMLP, input_gains

SEED = 20261030


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose_groups(q, step=2.0):
    return np.unique(np.floor(q/step).astype(np.int64), axis=0, return_inverse=True)[1]


def grouped_folds(groups, seed=SEED, count=5):
    unique = np.unique(groups)
    if len(unique) < count:
        raise ValueError('独立姿态组数量不足')
    order = np.random.default_rng(seed).permutation(unique)
    mapping = dict(zip(order.tolist(), (np.arange(len(order)) % count).tolist()))
    return np.array([mapping[g] for g in groups])


def metrics(target, estimate):
    e = estimate-target
    denominator = np.maximum(((target-target.mean(0))**2).sum(0), 1e-12)
    return {'n': len(target), 'mae_deg': abs(e).mean(0).tolist(),
            'rmse_deg': np.sqrt((e*e).mean(0)).tolist(),
            'p95_abs_error_deg': np.percentile(abs(e), 95, axis=0).tolist(),
            'bias_deg': e.mean(0).tolist(), 'r2': (1-(e*e).sum(0)/denominator).tolist()}


def polynomial(m, q, test, degree):
    center, scale = m.mean(0), np.maximum(m.std(0), 1e-8)
    design = lambda x: np.array([feature_row(row, degree) for row in (x-center)/scale])
    return design(test) @ np.linalg.lstsq(design(m), q, rcond=None)[0]


def train(m, q, finger, args, *, seed, validation=None, epochs=None):
    """Train-only scaling, independent checkpoints and epoch counts per branch."""
    import torch
    from torch import nn
    center, scale = m.mean(0), np.maximum(m.std(0), 1e-8)
    yc, ys = q.mean(0), np.maximum(q.std(0), 1e-8)
    tensor = lambda a: torch.tensor(a, dtype=torch.float32)
    x, y = tensor((m-center)/scale), tensor((q-yc)/ys)
    if validation is not None:
        vm, vq = validation
        vx, vy = tensor((vm-center)/scale), tensor((vq-yc)/ys)
    ids = list(FINGER_SPECS[finger]['motor_inputs'])
    gains = input_gains(finger, ids, args.secondary_input_weight)
    fit = {'architecture': 'independent_q0_q123', 'activation': 'tanh',
           'input_center': center.tolist(), 'input_scale': scale.tolist(),
           'output_center': yc.tolist(), 'output_scale': ys.tolist(), 'branches': []}
    chosen = []
    for k, indices in enumerate(([0], [1, 2, 3])):
        torch.manual_seed(seed+k)
        net = nn.Sequential(nn.Linear(len(ids), args.width), nn.Tanh(),
                            nn.Linear(args.width, args.width), nn.Tanh(), nn.Linear(args.width, len(indices)))
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        gain = tensor(gains[k])
        tx = x*gain
        target = y[:, indices]
        if validation is not None:
            val_x, val_y = vx*gain, vy[:, indices]
        maximum = args.epochs if epochs is None else int(epochs[k])
        best, best_epoch, state = float('inf'), 0, None
        for epoch in range(1, maximum+1):
            optimizer.zero_grad(set_to_none=True)
            priority = (net[0].weight.square()*(gain.reciprocal().square()-1)).mean()
            loss = (net(tx)-target).square().mean()+args.input_priority_penalty*priority
            if not torch.isfinite(loss):
                raise ValueError('训练损失包含非有限值')
            loss.backward()
            optimizer.step()
            if validation is not None:
                with torch.no_grad():
                    score = float((net(val_x)-val_y).square().mean())
                if score < best-1e-5:
                    best, best_epoch = score, epoch
                    state = {name: v.detach().clone() for name, v in net.state_dict().items()}
                if epoch-best_epoch >= args.patience:
                    break
        if validation is not None:
            net.load_state_dict(state)
        else:
            best_epoch = maximum
        layers = [{'weight': layer.weight.detach().tolist(), 'bias': layer.bias.detach().tolist(),
                   'activation': 'tanh'} for layer in net if isinstance(layer, nn.Linear)]
        layers[-1]['activation'] = 'identity'
        fit['branches'].append({'output_indices': indices, 'input_weights': gains[k], 'layers': layers})
        chosen.append(best_epoch)
        # Check every training point against the actual Torch branch before export.
        with torch.no_grad():
            expected = net(tx).numpy()*ys[indices]+yc[indices]
        actual = forward_branch(fit['branches'][-1], (m-center)/scale)*ys[indices]+yc[indices]
        np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=1e-5)
    return fit, chosen


def forward_branch(branch, standardized):
    x = standardized*np.array(branch['input_weights'])
    for layer in branch['layers']:
        x = x @ np.array(layer['weight']).T+layer['bias']
        if layer['activation'] == 'tanh':
            x = np.tanh(x)
    return x


def predict(fit, m):
    x = (m-fit['input_center'])/fit['input_scale']
    y = np.empty((len(m), 4))
    for branch in fit['branches']:
        y[:, branch['output_indices']] = forward_branch(branch, x)
    return y*fit['output_scale']+fit['output_center']


def sensitivities(fit, m):
    """Mean absolute d(standardized q)/d(standardized m), not causal effects."""
    x = (m-fit['input_center'])/fit['input_scale']
    result = np.empty((4, m.shape[1]))
    for k in range(m.shape[1]):
        plus, minus = x.copy(), x.copy()
        plus[:, k] += .001
        minus[:, k] -= .001
        for branch in fit['branches']:
            d = (forward_branch(branch, plus)-forward_branch(branch, minus))/.002
            result[branch['output_indices'], k] = abs(d).mean(0)
    return result.tolist()


def fit_finger(finger, args, folder):
    mroot = args.data_root.resolve()/f'{finger}_data'
    q, m, meta, provenance = load_finger_data(mroot, finger, args.refresh_cache)
    ids = list(FINGER_SPECS[finger]['motor_inputs'])
    combos = np.array([r['combo_id'] for r in meta])
    if len(np.unique(combos)) < 2:
        raise ValueError('留组合验证至少需要两个动作组合')
    groups = pose_groups(q)
    pose_fold = grouped_folds(groups, seed=args.seed)
    evaluation, arrays, epoch_choices = {}, {}, []
    print(f'{finger}: {len(m)} paired xy points; {len(np.unique(combos))} combos; {len(np.unique(groups))} pose groups', flush=True)
    for scheme, labels in [('pose_group_5fold', pose_fold), ('leave_one_combo_out', combos)]:
        estimates = {name: np.full_like(q, np.nan) for name in ('linear', 'quadratic', 'mlp')}
        details = []
        for i, label in enumerate(np.unique(labels)):
            test = labels == label
            outer = np.flatnonzero(~test)
            inner = grouped_folds(groups[outer], seed=args.seed+100+i)
            training, validation = outer[inner != 0], outer[inner == 0]
            _, selected = train(m[training], q[training], finger, args, seed=args.seed+i,
                                validation=(m[validation], q[validation]))
            fit, _ = train(m[outer], q[outer], finger, args, seed=args.seed+i, epochs=selected)
            estimates['mlp'][test] = predict(fit, m[test])
            for name, degree in [('linear', 1), ('quadratic', 2)]:
                estimates[name][test] = polynomial(m[outer], q[outer], m[test], degree)
            details.append({'label': str(label), 'training_rows': len(outer), 'test_rows': int(test.sum()),
                            'inner_training_rows': len(training), 'inner_validation_rows': len(validation),
                            'q0_q123_epochs': selected})
            if scheme == 'pose_group_5fold':
                epoch_choices.append(selected)
            # Preserve the held-out model and exact row membership for audit/replay.
            fold_dir = folder/finger/'folds'/scheme
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir/f'{i}.json').write_text(json.dumps({'label': str(label), 'fit': fit,
                'outer_training_indices': outer.tolist(), 'test_indices': np.flatnonzero(test).tolist(),
                'inner_training_indices': training.tolist(), 'inner_validation_indices': validation.tolist()}, allow_nan=False)+'\n')
            print(f'  {finger} {scheme} {i+1}/{len(np.unique(labels))}: epochs={selected}; MLP MAE={np.round(abs(estimates["mlp"][test]-q[test]).mean(0), 3).tolist()}', flush=True)
        evaluation[scheme] = {name: metrics(q, y) for name, y in estimates.items()}
        evaluation[scheme]['folds'] = details
        arrays.update({f'{scheme}_{name}': y for name, y in estimates.items()})
    selected = np.median(epoch_choices, axis=0).astype(int).tolist()
    fit, _ = train(m, q, finger, args, seed=args.seed, epochs=selected)
    base_path = PACKAGE/f'config/left_{finger}_q_to_motor_models.json'
    base = json.loads(base_path.read_text())
    source = dict(provenance, collection=str(mroot.relative_to(PACKAGE)) if mroot.is_relative_to(PACKAGE) else str(mroot))
    config = {'schema_version': 2, 'mapping': 'motor_feedback_to_q', 'model_type': 'grouped_mlp',
        'hand': 'left', 'finger': finger, 'motor_ids': ids, 'q_names': [f'left_{finger}_q{j}' for j in range(4)],
        'glove_q_slice': list(FINGER_SPECS[finger]['glove_slice']), 'motor_units': 'degrees', 'q_units': 'degrees',
        'input_coordinates': 'recorded_motor_feedback_positions', 'output_coordinates': 'absolute_glove_angles',
        'default_model': 'mlp', 'relative_output_reference_deg': base['q_reference'],
        'relative_reference_source': base['q_reference_source']+'; subtraction only, no fit or encoder-zero constraint.',
        'reference_model_sha256': sha(base_path), 'training_points': len(q),
        'training_timeouts': sum(r['timeout'] == '1' for r in meta),
        'training_scope': 'All paired formal active_mode=xy points, including timeouts and all 2670 ring points; actual motor feedback inputs and measured absolute glove labels.',
        'source_sha256': source,
        'motor_training_bounds_deg': {str(mid): [float(m[:, k].min()), float(m[:, k].max())] for k, mid in enumerate(ids)},
        'q_training_bounds_deg': {f'q{j}': [float(q[:, j].min()), float(q[:, j].max())] for j in range(4)},
        'input_priority': {'side_motor_ids': list(SIDE_MOTORS[finger]), 'secondary_weight': args.secondary_input_weight,
            'applied_after_standardization': True, 'first_layer_penalty_strength': args.input_priority_penalty,
            'penalty': 'mean(W_first**2 * (1/gain**2 - 1))',
            'policy': 'q0: side=1, bending=secondary; q123: bending=1, side=secondary. Soft preference, no guaranteed derivative ratio.'},
        'training': {'seed': args.seed, 'hidden_width': args.width, 'branch_shapes': [[len(ids), args.width, args.width, n] for n in (1, 3)],
            'activation': 'tanh', 'optimizer': 'AdamW', 'lr': args.lr, 'weight_decay': 1e-4,
            'maximum_epochs': args.epochs, 'early_stop_patience': args.patience, 'final_q0_q123_epochs': selected,
            'selection': 'Independent inner-validation early stopping per branch; outer-training refit. Final epochs are medians of the five pose-fold choices; all-data final refit.',
            'pose_group_definition': 'floor(absolute_q / 2 degrees), same bin never crosses pose folds; adjacent bins may be close',
            'torch_version': args.torch_version},
        'validation': evaluation, 'training_metrics': metrics(q, predict(fit, m)),
        'mean_absolute_standardized_jacobian': sensitivities(fit, m),
        'limitations': 'Same-session window-mean validation only. No dynamic/contact/cross-session accuracy claim. No command targets or synthetic labels. No automatic encoder alignment or new physical zero calibration. Marginal input bounds do not establish joint coverage.',
        'fits': {'mlp': fit}}
    portable = GroupedMLP(fit, len(ids))
    sample = m[np.linspace(0, len(m)-1, min(len(m), 100), dtype=int)]
    expected = predict(fit, sample)
    actual = np.array([portable.predict_standardized((row-fit['input_center'])/fit['input_scale']) for row in sample])
    np.testing.assert_allclose(expected, actual, atol=1e-10)
    target = args.output_dir/f'left_{finger}_motor_to_q_estimator.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(target)
    finger_dir = folder/finger
    (finger_dir/target.name).write_bytes(target.read_bytes())
    (finger_dir/'point_log_snapshot.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    np.savez_compressed(finger_dir/'validation_predictions.npz', motor_feedback=m, q_absolute=q,
        motor_ids=ids, point_ids=[int(r['point_id']) for r in meta], combo=combos,
        pose_groups=groups, pose_fold=pose_fold, **arrays)
    return {k: v for k, v in config.items() if k != 'fits'}


def report(summary, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    names = [(f, j) for f in summary for j in range(4)]
    schemes = ('pose_group_5fold', 'leave_one_combo_out')
    lines = ['# 四指 m→q 分支MLP训练与验证', '',
             '每指q0与q123两个独立网络；不含拇指。输入为采集电机反馈位置，标签为配对实测绝对手套角。',
             '正式xy点全部保留，包含超时点；无名指2670点，未排除末尾500点。',
             '标准化后主肌群输入权重1，次肌群默认0.25，额外第一层弱输入路径L2惩罚。软优先级不保证导数比例。',
             '标准化及早停仅使用内层训练/验证，外层测试不参与；两分支独立选择轮数后重训。', '',
             '所有误差单位均为关节角度。最终全量模型训练误差不作为验证成绩。', '']
    for ax, scheme in zip(axes, schemes):
        lines += [f'## {scheme}', '', '| 手指 | 点数 | 模型 | q0 MAE | q1 MAE | q2 MAE | q3 MAE |',
                  '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
        for k, name in enumerate(('linear', 'quadratic', 'mlp')):
            values = [summary[f]['validation'][scheme][name]['mae_deg'][j] for f, j in names]
            ax.bar(np.arange(len(names))+(k-1)*.25, values, width=.25, label=name)
        for f, c in summary.items():
            for name in ('linear', 'quadratic', 'mlp'):
                values = c['validation'][scheme][name]['mae_deg']
                lines.append(f'| {f} | {c["training_points"]} | {name} | '+' | '.join(f'{v:.3f}' for v in values)+' |')
        ax.set_xticks(np.arange(len(names)), [f'{f}\nq{j}' for f, j in names], rotation=45, ha='right')
        ax.set_title(scheme)
        ax.set_ylabel('held-out q MAE / deg')
        ax.legend()
        ax.grid(axis='y', alpha=.2)
        lines.append('')
    fig.tight_layout()
    fig.savefig(folder/'validation_mae.png', dpi=160)
    plt.close(fig)
    lines += ['![验证对比](validation_mae.png)', '', '## 输入优先级与适用范围', '',
              '各模型的mean_absolute_standardized_jacobian记录训练分布上的标准化局部敏感度，行是q0/q1/q2/q3，列顺序是motor_ids；它描述网络依赖，不等于因果或物理辨识。',
              '留动作组合比同姿态分箱五折更能反映未见动作的泛化；同一采集批次验证仍不能代表动态外载、其他批次或实际遥操估计精度。',
              '模型输出绝对手套角，q_relative只减参考；没有自动减去F(0)，没有重设编码器零位。运行时仍需对齐实际编码器与采集坐标。',
              '每指目录保存模型快照、point_log、验证预测、point_id和精确折分；folds保存每个外层MLP、内外层样本成员。', '']
    (folder/'REPORT.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finger', choices=['four', *FOUR_FINGERS], default='four')
    parser.add_argument('--data-root', type=Path, default=PACKAGE/'scripts/motor_test_logs/data_collection')
    parser.add_argument('--output-dir', type=Path, default=PACKAGE/'config/motor_to_q_mlp')
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.006)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--secondary-input-weight', type=float, default=.25)
    parser.add_argument('--input-priority-penalty', type=float, default=.001)
    parser.add_argument('--refresh-cache', action='store_true')
    args = parser.parse_args(argv)
    if min(args.epochs, args.patience, args.threads) < 1 or not 1 <= args.width <= 256:
        parser.error('epochs/patience/threads需要正数；width在1..256')
    if not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.input_priority_penalty) or args.input_priority_penalty < 0:
        parser.error('lr需要正有限数，penalty需要非负有限数')
    input_gains('index', [25, 3, 15, 1, 28], args.secondary_input_weight)
    import torch
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.torch_version = str(torch.__version__)
    folder = args.report_dir or PACKAGE/'scripts/motor_test_logs/motor_to_q_mlp'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'training_arguments.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2)+'\n')
    summary = {}
    for finger in FOUR_FINGERS if args.finger == 'four' else (args.finger,):
        summary[finger] = fit_finger(finger, args, folder)
        (folder/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        report(summary, folder)
    print(f'Models: {args.output_dir}\nReport: {folder}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
