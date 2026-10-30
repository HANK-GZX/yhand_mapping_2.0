#!/usr/bin/env python3
"""Refit only four-finger extensor heads using the existing measured samples.

Default output is a staged model directory inside the report, not live config.
Other outputs are copied exactly from the source model, including side policy.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np

from fit_left_hand_q_to_motor_mlp import (SCRIPTS, SEED, load_data, delta_angles,
    pose_groups, grouped_folds, metrics, sha)
from mlp_q_to_motor import MODEL_DIR, MLPFingerMapper, forward, motor_forward
from monotonic_extensor_training import train_extensor

FINGERS = ('index', 'middle', 'ring', 'little')


def replace_extensor(previous, head):
    """Preserve every non-extensor parameter and its normalization exactly."""
    return replace_motor_head(previous, head, 0)


def replace_motor_head(previous, head, output_index):
    """Replace one independent output without refitting any other output."""
    if any((b.get('shared_antagonistic_pair')or b.get('shared_thumb_flex_ext')or b.get('shared_thumb_side')) and output_index in b['output_indices']
           for b in previous['network'].get('branches',[])):
        raise ValueError('共享肌肉网络不能单独替换一个输出，请使用联合训练入口')
    allowed = (0,2) if previous['finger']=='middle' else (0,1)
    if type(output_index) is not int or output_index not in allowed:
        raise ValueError('仅支持独立替换伸肌或屈肌头')
    direction = head['branches'][0].get('monotonic_direction', 1)
    if direction != (1 if output_index == 0 else -1):
        raise ValueError('伸肌需要正向单调，屈肌需要负向单调')
    result = deepcopy(previous)
    network = result['network']
    for key in ('input_center', 'input_scale'):
        np.testing.assert_allclose(network[key], head[key], atol=1e-12, rtol=1e-12)
    branches = []
    for branch in network['branches']:
        remaining = [j for j, index in enumerate(branch['output_indices']) if index != output_index]
        if not remaining:
            continue
        branch = deepcopy(branch)
        branch['output_indices'] = [branch['output_indices'][j] for j in remaining]
        for key in ('weight', 'bias'):
            branch['layers'][-1][key] = [branch['layers'][-1][key][j] for j in remaining]
        branches.append(branch)
    replacement = deepcopy(head['branches'][0])
    replacement['output_indices'] = [output_index]
    network['branches'] = sorted([replacement, *branches], key=lambda b:b['output_indices'][0])
    for key in ('output_center', 'output_scale'):
        network[key][output_index] = head[key][0]
    result['input_priority']['per_motor_weights'][str(result['motor_order'][output_index])] = list(replacement['input_weights'])
    has_sides=any(b.get('monotonic_inputs')==[0]for b in network['branches'])
    has_flexor=any(b.get('monotonic_direction')==-1 and b.get('monotonic_inputs')==[1,2,3]for b in network['branches'])
    if previous['finger']=='middle':
        has_shallow=any(b['output_indices']==[1] and b.get('monotonic_inputs')==[1,2]
                        for b in network['branches'])
        result['schema_version']=10 if has_sides and has_shallow else 9 if has_sides else 8 if has_flexor else 5
    else:
        result['schema_version']=7 if has_sides else 6 if has_flexor else 5
    return result


def evaluate(q, y, combo, *, epochs, width, seed, direction=1, trainer=None):
    """Outer holdout is never used for scaling, optimization or epoch choice."""
    groups = pose_groups(q)
    fit_head = train_extensor if trainer is None else trainer
    folds = grouped_folds(groups, seed=seed)
    evaluation, predictions, chosen_epochs = {}, {}, []
    for scheme, labels in [('pose_group_5fold', folds), ('leave_one_combo_out', combo)]:
        predicted = np.zeros((len(y), np.asarray(y).reshape(len(y),-1).shape[1]))
        details = []
        for i, label in enumerate(np.unique(labels)):
            test = labels == label
            rows = np.flatnonzero(~test)
            inner = grouped_folds(groups[rows], seed=seed+i+1)
            fit_rows, val_rows = rows[inner != 0], rows[inner == 0]
            _, chosen = fit_head(q[fit_rows], y[fit_rows], epochs=epochs, width=width,
                seed=seed+i, validation=(q[val_rows], y[val_rows]), direction=direction)
            head, _ = fit_head(q[~test], y[~test], epochs=chosen, width=width, seed=seed+i, direction=direction)
            predicted[test] = forward(head, q[test])
            details.append(dict(label=str(label), train=int((~test).sum()), test=int(test.sum()),
                                inner_selected_epochs=chosen))
            if scheme == 'pose_group_5fold':
                chosen_epochs.append(chosen)
            print(f'  {scheme} {label}: epochs={chosen}, MAE={np.abs(predicted[test]-y[test]).mean():.3f}', flush=True)
        evaluation[scheme] = dict(metrics(y, predicted), folds=details)
        predictions[scheme] = predicted
    return evaluation, predictions, int(np.median(chosen_epochs)), groups, folds


def fit_finger(finger, args, report):
    source = args.source_model_dir/f'left_{finger}_q_to_motor_mlp.json'
    previous = MLPFingerMapper(source, finger=finger).config
    if previous['schema_version']==11:
        raise ValueError('无名指M10/M17已共享网络，请使用fit_ring_joint_mlp.py联合重训')
    if previous['schema_version'] not in (4, 5, 6, 7, 8, 9, 10):
        raise ValueError('需要带侧摆约束的四指分支模型')
    absolute, labels, meta, provenance = load_data(args.data_root.resolve(), finger)
    q = delta_angles(absolute, previous['q_reference'])
    y = labels[:, :1]
    if previous['training_points'] != len(q):
        raise ValueError('数据数量与原模型不同；本入口不允许静默改变训练样本')
    combo = np.array([r['combo_id'] for r in meta])
    print(f'{finger}: {len(q)} original samples; only M{previous["motor_order"][0]} is retrained', flush=True)
    evaluation, predictions, epochs, groups, folds = evaluate(q, y, combo,
        epochs=args.epochs, width=args.width, seed=args.seed)
    head, _ = train_extensor(q, y, epochs=epochs, width=args.width, seed=args.seed)
    config = replace_extensor(previous, head)
    config.update(raw_prediction_at_reference_motor_deg=forward(config['network'], np.zeros(4)).tolist(),
        extensor_structure=dict(type='partially_monotonic_mixed_softplus_mlp',
            motor_id=previous['motor_order'][0], nondecreasing_delta_q_indices=[1, 2, 3],
            q0_constraint='unconstrained; secondary input gain 0.25 and weak-path penalty',
            positive_weight_parameterization='softplus(raw_weight)',
            hidden_activation='first half softplus(x); second half -softplus(-x)',
            output_activation='identity', architecture=[4, args.width, args.width, 1],
            guarantee='At fixed q0, raw/neutral-aligned extensor output is nondecreasing in q1/q2/q3. '
                      'No positive lower slope or correct out-of-domain amplitude is guaranteed.'),
        preserved_outputs=dict(source_model_sha256=sha(source),
            motor_ids=previous['motor_order'][1:], source_training=previous['training'],
            note='Weights, normalization, zero alignment and side policy are copied unchanged.'),
        extensor_training=dict(seed=args.seed, epochs=epochs, maximum_selection_epochs=args.epochs,
            optimizer='AdamW', lr=.01, weight_decay=0, early_stop_patience=120,
            input_priority_penalty=.001, torch_version=args.torch_version,
            selection='Median inner-selected epochs from five outer pose folds; final fit on all original rows',
            input_definition=previous['training_input_definition'],
            target_definition=previous['training_target_definition'], source=provenance,
            no_added_samples=True, no_pseudo_labels=True, no_linear_baseline=True),
        extensor_validation=evaluation)
    config['training'] = dict(config['extensor_training'],
        architecture_type='role_weighted_branches_with_monotonic_extensor',
        architecture=[4, args.width, args.width, 1], activation='mixed_softplus (extensor only)',
        validation_pose_bin_deg=2., note='Other output branches retain source_training in preserved_outputs')
    # Existing per-motor estimates remain valid for the unchanged branches;
    # replace ONLY the extensor estimates with its newly held-out predictions.
    for scheme, result in evaluation.items():
        for metric, value in result.items():
            if metric != 'folds':
                config['validation'][scheme]['mlp'][metric][0] = value[0]
    config['control_policy_validation_note'] = ('Regression validation is raw measured motor feedback, '
        'not control-zero or hardware accuracy. Only extensor validation is newly computed; '
        'unchanged output estimates are inherited from source-model validation.')
    folder = report/finger
    folder.mkdir()
    path = args.output_dir/source.name
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    mapper = MLPFingerMapper(path, finger=finger)
    np.testing.assert_array_equal(motor_forward(config, np.zeros(4)), np.zeros(len(labels[0])))
    rng = np.random.default_rng(204)
    probe = rng.uniform([-35, -50, -30, -50], [35, 150, 160, 100], size=(1000, 4))
    preserved_difference = float(np.abs(motor_forward(previous, probe)[:, 1:]-motor_forward(config, probe)[:, 1:]).max())
    np.testing.assert_allclose(motor_forward(previous, probe)[:, 1:], motor_forward(config, probe)[:, 1:], atol=1e-12, rtol=0)
    minimum_changes = []
    for axis in (1, 2, 3):
        more = probe.copy(); more[:, axis] += .5
        change = motor_forward(config, more)[:, 0]-motor_forward(config, probe)[:, 0]
        if change.min() < -1e-10:
            raise ValueError('单调性检查失败')
        minimum_changes.append(float(change.min()))
    shutil.copyfile(path, folder/path.name)
    (folder/'point_log_snapshot.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    np.savez_compressed(folder/'validation_predictions.npz', q_absolute=absolute, q_delta=q,
        q_reference=mapper.q_reference, motor_feedback=labels, motor_ids=mapper.motor_ids,
        point_ids=[int(r['point_id']) for r in meta], combo=combo, pose_group=groups, pose_fold=folds,
        **predictions)
    result = dict(training_points=len(q), extensor_motor=mapper.motor_ids[0], epochs=epochs,
        original_extensor_validation={s: previous['validation'][s]['mlp']['mae_motor_deg'][0] for s in evaluation},
        new_extensor_validation={s: evaluation[s]['mae_motor_deg'][0] for s in evaluation},
        non_extensor_max_difference=preserved_difference, minimum_change_for_half_degree_q123=minimum_changes,
        source_sha256=sha(source), model_sha256=sha(path))
    print(f'{finger}: {result}', flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finger', choices=['four', *FINGERS], default='four')
    parser.add_argument('--source-model-dir', type=Path, default=MODEL_DIR)
    parser.add_argument('--data-root', type=Path, default=SCRIPTS/'motor_test_logs/data_collection')
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--epochs', type=int, default=2400)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    if args.epochs < 1 or not 2 <= args.width <= 256 or args.threads < 1:
        parser.error('epochs/threads必须为正，width必须在2..256')
    import torch
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.torch_version = str(torch.__version__)
    report = args.report_dir or SCRIPTS/'motor_test_logs/q_to_motor_monotonic_extensor'/datetime.now().strftime('%Y%m%d_%H%M%S')
    report.mkdir(parents=True, exist_ok=False)
    args.output_dir = args.output_dir or report/'models'
    if args.output_dir.resolve() == args.source_model_dir.resolve():
        parser.error('请先输出候选目录并离线验证，不能直接覆盖来源模型')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backup = report/'previous_models'; backup.mkdir()
    for source in args.source_model_dir.glob('left_*_q_to_motor_mlp.json'):
        shutil.copyfile(source, backup/source.name)
        target = args.output_dir/source.name
        if target.exists():
            raise ValueError(f'拒绝覆盖已有候选模型：{target}')
        shutil.copyfile(source, target)
    summary = {}
    for finger in FINGERS if args.finger == 'four' else (args.finger,):
        summary[finger] = fit_finger(finger, args, report)
        (report/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(f'Staged models: {args.output_dir}\nReport: {report}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
