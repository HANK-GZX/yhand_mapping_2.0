#!/usr/bin/env python3
"""Refit selected flexors with a decreasing pure-MLP head on original data."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np

from fit_left_hand_q_to_motor_mlp import SCRIPTS, SEED, load_data, delta_angles, sha
from fit_monotonic_extensor import evaluate, replace_motor_head
from monotonic_extensor_training import train_extensor
from mlp_q_to_motor import MODEL_DIR, MLPFingerMapper, forward, motor_forward


def fit(args, report):
    finger = getattr(args, 'finger', 'little')
    source = args.source_model_dir/f'left_{finger}_q_to_motor_mlp.json'
    previous = MLPFingerMapper(source, finger=finger).config
    if previous['schema_version']==11:
        raise ValueError('无名指M10/M17已共享网络，请使用fit_ring_joint_mlp.py联合重训')
    if previous['schema_version'] not in (5, 6, 7, 8, 9, 10):
        raise ValueError('需要保留当前单调伸肌的手指模型')
    index=2 if finger=='middle' else 1
    ids = previous['motor_order']; mid = ids[index]
    absolute, labels, meta, provenance = load_data(args.data_root.resolve(), finger)
    q = delta_angles(absolute, previous['q_reference'])
    y = labels[:, index:index+1]  # Keep recorded flexor labels unchanged, including their sign.
    if previous['training_points'] != len(q) or previous['source'] != provenance:
        raise ValueError('训练数据来源或数量与当前模型不同')
    print(f'{finger}: {len(q)} original samples; only M{mid} is retrained, direction=-1', flush=True)
    combo = np.array([r['combo_id'] for r in meta])
    evaluation, predictions, epochs, groups, folds = evaluate(q, y, combo,
        epochs=args.epochs, width=args.width, seed=args.seed, direction=-1)
    head, _ = train_extensor(q, y, epochs=epochs, width=args.width, seed=args.seed, direction=-1)
    config = replace_motor_head(previous, head, index)
    config['raw_prediction_at_reference_motor_deg'] = forward(config['network'], np.zeros(4)).tolist()
    config['flexor_structure'] = dict(type='partially_monotonic_mixed_softplus_mlp', motor_id=mid,
        nonincreasing_delta_q_indices=[1, 2, 3], input_weights=[.25, 1, 1, 1],
        architecture=[4, args.width, args.width, 1], monotonic_direction=-1,
        hidden_activation='first half softplus(x); second half -softplus(-x)',
        output_activation='identity then fixed sign -1, before output de-standardization',
        guarantee=f'At fixed other inputs, increasing q1/q2/q3 never increases M{mid} motor position '
                  '(never reduces winding). No minimum winding slope or OOD accuracy guarantee.')
    training = dict(seed=args.seed, epochs=epochs, maximum_selection_epochs=args.epochs,
        optimizer='AdamW', lr=.01, weight_decay=0, early_stop_patience=120,
        input_priority_penalty=.001, torch_version=args.torch_version,
        selection='Median inner-selected epochs from five outer pose folds; final fit on original rows',
        input_definition=previous['training_input_definition'],
        target_definition=previous['training_target_definition'], source=provenance,
        no_added_samples=True, no_pseudo_labels=True, no_linear_baseline=True)
    config['flexor_training'] = training
    config['flexor_validation'] = evaluation
    preserved=[i for i in range(len(ids))if i!=index]
    config['preserved_outputs'] = dict(source_model_sha256=sha(source), motor_ids=[ids[i] for i in preserved],
        source_training=previous['training'],
        note='Other motor weights, normalization, neutral alignment and side policy are copied unchanged.')
    config['training'] = dict(training,
        architecture_type='role_weighted_branches_with_monotonic_extensor_and_flexor',
        architecture=[4, args.width, args.width, 1], activation=f'mixed_softplus (updated M{mid} only)',
        updated_motor_id=mid, validation_pose_bin_deg=2.,
        note='Other heads retain their earlier training/validation; middle updates only the deep flexor, preserving M16.')
    for scheme, result in evaluation.items():
        for metric, value in result.items():
            if metric != 'folds':
                config['validation'][scheme]['mlp'][metric][index] = value[0]
    config['control_policy_validation_note'] = ('Regression validation is raw feedback, not hardware or '
        'neutral-aligned accuracy. Only flexor validation is recomputed in this refit; extensor and side metrics '
        'are inherited from their earlier held-out evaluations.')
    path = args.output_dir/source.name
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    mapper = MLPFingerMapper(path, finger=finger)
    probe = np.random.default_rng(209).uniform([-40, -60, -40, -60], [40, 160, 180, 120], (5000, 4))
    before, after = motor_forward(previous, probe), motor_forward(config, probe)
    difference = float(np.abs(before[:, preserved]-after[:, preserved]).max())
    np.testing.assert_allclose(before[:, preserved], after[:, preserved], atol=1e-12, rtol=0)
    changes = []
    for axis in (1, 2, 3):
        more = probe.copy(); more[:, axis] += .5
        change = motor_forward(config, more)[:, index]-after[:, index]
        assert change.max() <= 1e-10
        changes.append(float(change.max()))
    np.testing.assert_array_equal(motor_forward(config, np.zeros(4)), np.zeros(len(ids)))
    assert config['side_bending_policy'] == previous['side_bending_policy']
    np.savez_compressed(report/'validation_predictions.npz', q_absolute=absolute, q_delta=q,
        q_reference=mapper.q_reference, motor_feedback=labels, motor_ids=mapper.motor_ids,
        point_ids=[int(r['point_id']) for r in meta], combo=combo, pose_group=groups, pose_fold=folds,
        **predictions)
    (report/'point_log_snapshot.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    result = dict(training_points=len(q), updated_motor_id=mid, schema_version=config['schema_version'],
        epochs=epochs, old_validation={s:previous['validation'][s]['mlp']['mae_motor_deg'][index] for s in evaluation},
        new_validation={s:evaluation[s]['mae_motor_deg'][0] for s in evaluation},
        preserved_motor_ids=[ids[i] for i in preserved], preserved_max_difference=difference,
        maximum_half_degree_q123_changes=changes, source_sha256=sha(source), model_sha256=sha(path))
    (report/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finger', choices=['little','ring','middle'], default='little')
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
    report = args.report_dir or SCRIPTS/'motor_test_logs/q_to_motor_monotonic_flexor'/datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = args.output_dir or report/'models'
    if args.output_dir.resolve() == args.source_model_dir.resolve():
        parser.error('必须先输出候选目录，不能直接覆盖来源模型')
    report.mkdir(parents=True, exist_ok=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backup = report/'previous_models'; backup.mkdir()
    for source in args.source_model_dir.glob('left_*_q_to_motor_mlp.json'):
        target = args.output_dir/source.name
        if target.exists():raise ValueError(f'拒绝覆盖已有候选模型：{target}')
        shutil.copyfile(source, backup/source.name)
        shutil.copyfile(source, target)
    fit(args, report)
    print(f'Staged models: {args.output_dir}\nReport: {report}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
