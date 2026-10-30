#!/usr/bin/env python3
"""Stage a shared thumb MLP refinement of M6/M7; retain the M4 subgraph."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import numpy as np

from fit_left_hand_q_to_motor_mlp import SCRIPTS, sha, load_data, delta_angles, grouped_folds, metrics
from fit_thumb_spatial_mlp import workspace_keys, macro_error
from thumb_fold_models import load_thumb_fold, group_validation_reports
from thumb_extensor_training import (train_extensor_refinement, replace_extensors,
    PRIOR_DIRECTIONS, PRIOR_FLOORS, PRIOR_SCALES, M7_SLOPE_CEILINGS)
from mlp_q_to_motor import MODEL_DIR, MLPFingerMapper, forward, motor_forward
from thumb_short_extensor_training import train_short_extensor, replace_short_extensor


def fit_job(job):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    scheme, label, source, q, y, runtime, options = job
    options = dict(options)
    short = options.pop('mode', 'extensors') == 'short'
    trainer = train_short_extensor if short else train_extensor_refinement
    head, detail = trainer(source, q, y[:, :3], runtime_q=runtime, **options)
    return scheme, label, head, detail


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-model-dir', type=Path, default=MODEL_DIR)
    p.add_argument('--mode', choices=('extensors', 'short'), default='extensors')
    p.add_argument('--reference-report', type=Path,
        help='对应来源模型的验证档案；默认按mode选择')
    p.add_argument('--runtime-inputs', type=Path, default=SCRIPTS/
        'motor_test_logs/thumb_q_to_motor_test/analysis/flex_ext_opposition_20261029_1414/comparison_arrays.npz')
    p.add_argument('--report-dir', type=Path)
    p.add_argument('--runtime-extra-inputs', type=Path, help='可选NPZ中的q，只取拇指四角，不使用运行电机标签')
    p.add_argument('--epochs', type=int)
    p.add_argument('--extra-width', type=int, default=16)
    p.add_argument('--slope-penalty', type=float)
    p.add_argument('--lr', type=float)
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    short = args.mode == 'short'
    args.reference_report = args.reference_report or SCRIPTS/('motor_test_logs/thumb_extensor_mlp/20261029_m6_m7_shared_envelope'
        if short else 'motor_test_logs/thumb_coupled_mlp/20261029_linear_coupling')
    args.epochs = args.epochs if args.epochs is not None else (1400 if short else 800)
    args.lr = args.lr if args.lr is not None else (.015 if short else .008)
    args.slope_penalty = args.slope_penalty if args.slope_penalty is not None else (2. if short else .15)
    replace = replace_short_extensor if short else replace_extensors
    preserved_indices = [0, 2, 3, 4, 5] if short else [0, 3, 4, 5]
    if (args.epochs < 1 or args.extra_width < 4 or args.extra_width % 4
            or not 1 <= args.workers <= 4 or not np.isfinite(args.slope_penalty)
            or args.slope_penalty < 0 or not np.isfinite(args.lr) or args.lr <= 0):
        p.error('训练参数范围不正确')
    source = args.source_model_dir/'left_thumb_q_to_motor_mlp.json'
    old = MLPFingerMapper(source, finger='thumb').config
    reference = json.loads((args.reference_report/'summary.json').read_text())
    if sha(source) != reference['new_sha256']:
        raise ValueError('来源模型与参考验证档案不匹配')
    a = np.load(args.reference_report/'validation_predictions.npz')
    absolute, y, meta, provenance = load_data(SCRIPTS/'motor_test_logs/data_collection', 'thumb')
    q = delta_angles(absolute, old['q_reference'])
    for k, v in [('q_delta', q), ('motor_feedback', y),
                 ('point_ids', np.array([int(r['point_id']) for r in meta]))]:
        np.testing.assert_array_equal(a[k], v)
    assert old['source'] == provenance
    keep, combo = a['eligible'], a['combo']
    runtime_archive = np.load(args.runtime_inputs)
    runtime = np.concatenate([runtime_archive[k+'_q'][runtime_archive[k+'_normal']]
                              for k in ('latest', 'auxiliary', 'linear')])
    if args.runtime_extra_inputs:
        extra = np.load(args.runtime_extra_inputs)['q']
        if extra.ndim != 2 or extra.shape[1] < 4 or not np.isfinite(extra).all():
            raise ValueError('额外运行输入需要有限N×4或N×20角度')
        runtime = np.concatenate((runtime, extra[:, :4]))
    out = args.report_dir or SCRIPTS/'motor_test_logs/thumb_extensor_mlp'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('models', 'previous_models'):
        (out/folder).mkdir()
        for path in args.source_model_dir.glob('left_*_q_to_motor_mlp.json'):
            shutil.copyfile(path, out/folder/path.name)
    np.savez_compressed(out/'runtime_q_only.npz', q=runtime)
    options = dict(epochs=args.epochs, lr=args.lr, mode=args.mode,
                   slope_penalty=args.slope_penalty, seed=20261029)
    if not short:
        options['extra_width'] = args.extra_width
    schemes = [('pose_group_5fold', a['pose_fold']), ('workspace_block_3fold', a['workspace_fold']),
               ('leave_one_combo_out', combo),
               ('zero_interval_3fold', grouped_folds(a['time_group'], count=3, seed=20261028))]
    arrays = {k: a[k] for k in ('q_absolute', 'q_delta', 'q_reference', 'motor_ids',
        'motor_feedback', 'point_ids', 'combo', 'eligible', 'pose_group', 'pose_fold',
        'workspace_fold', 'time_group')}
    pending, folds, predictions, details = [], {}, {}, {}
    for scheme, labels in schemes:
        predictions[scheme] = a[scheme].copy()
        details[scheme] = []
        for i, label in enumerate(np.unique(labels)):
            test = labels == label
            train = (~test) & keep
            frozen = load_thumb_fold(old, args.reference_report, scheme, label)
            np.testing.assert_allclose(forward(frozen['network'], q[test]), a[scheme][test], atol=1e-9, rtol=0)
            key = (scheme, str(label))
            folds[key] = (test, train, frozen)
            pending.append((scheme, str(label), frozen, q[train], y[train], runtime,
                            dict(options, seed=20261029+i)))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        futures = [pool.submit(fit_job, job) for job in pending]
        for future in as_completed(futures):
            scheme, label, head, detail = future.result()
            test, train, frozen = folds[(scheme, label)]
            replacement = replace(frozen, head)
            pred = forward(replacement['network'], q[test])
            before = a[scheme][test]
            np.testing.assert_allclose(pred[:, preserved_indices], before[:, preserved_indices], atol=1e-10, rtol=0)
            np.testing.assert_array_equal(pred[:, 3:], forward(frozen['network'], q[test])[:, 3:])
            # Retain unchanged archived outputs; older archives differ from the
            # present evaluator by ~1e-13 from floating-point accumulation.
            pred[:, preserved_indices] = before[:, preserved_indices]
            predictions[scheme][test] = pred
            folder = out/'fold_models'/scheme
            folder.mkdir(parents=True, exist_ok=True)
            (folder/f'{label}.json').write_text(json.dumps(head, indent=2)+'\n')
            mask = test & keep
            detail.update(label=label, train=int(train.sum()), test=int(mask.sum()))
            details[scheme].append(detail)
            if mask.any():
                err = abs(pred[keep[test], 1:3]-y[mask, 1:3]).mean(0)
                print(scheme, label, 'M6/M7 MAE', err.round(3).tolist(), flush=True)
    boundary = json.loads((SCRIPTS/'motor_test_logs/workspace_boundaries/left/boundary_models.json').read_text())['combos']
    keys = workspace_keys(y, combo, boundary)
    validation = {}
    for scheme, _ in schemes:
        pred = predictions[scheme]
        validation[scheme] = dict(mlp=metrics(y[keep], pred[keep]),
            previous_mlp=metrics(y[keep], a[scheme][keep]), n_test=int(keep.sum()),
            folds=sorted(details[scheme], key=lambda d: d['label']),
            macro_voxel_mae_motor_deg=macro_error(y[keep], pred[keep], combo[keep], keys[keep]).tolist(),
            previous_macro_voxel_mae_motor_deg=macro_error(y[keep], a[scheme][keep], combo[keep], keys[keep]).tolist(),
            all_recorded_diagnostic_mlp=metrics(y, pred),
            all_recorded_diagnostic_previous_mlp=metrics(y, a[scheme]),
            scope='Each fold embeds only its own frozen training-fold features. Runtime angle-only slope probes are domain-informed priors; not independent hardware validation.')
        arrays[scheme] = pred
        arrays['previous_'+scheme] = a[scheme]
    _, _, head, detail = fit_job(('final', '', old, q[keep], y[keep], runtime, options))
    c = replace(old, head)
    c['previous_extensor_validation'] = deepcopy(old['validation'])
    c['validation'] = validation
    c['group_validation_reports'] = group_validation_reports(old, args.reference_report)
    c['group_validation_reports']['flex_ext'] = str(out.resolve())
    training = dict(**options, optimizer='AdamW', weight_decay=0,
        source=provenance, source_model_sha256=sha(source), reference_report=str(args.reference_report),
        runtime_probe_archive=str(args.runtime_inputs), runtime_probe_archive_sha256=sha(args.runtime_inputs),
        runtime_angle_frames=len(runtime), runtime_motor_labels_used=False,
        slope_prior_directions=PRIOR_DIRECTIONS.tolist(), slope_prior_floors=PRIOR_FLOORS.tolist(),
        slope_prior_scales=PRIOR_SCALES.tolist(), slope_units='motor degree per joint degree',
        m7_slope_prior_ceilings=M7_SLOPE_CEILINGS.tolist(),
        slope_prior_scope='Soft lower slopes on moderate deep-bend angle probes; motivated by dominant linear trends, not measured motor targets or amplitude guarantees',
        trainable_motor_ids=[6, 7], frozen_motor_ids=[4, 8, 14, 27],
        labeled_training_points=int(keep.sum()), no_added_labeled_samples=True,
        no_position_pseudo_labels=True, no_linear_baseline=True, no_runtime_compensation=True,
        detail=detail, selection='Fixed candidate; group metrics reviewed, not an independent final test')
    if args.runtime_extra_inputs:
        training.update(runtime_extra_inputs=str(args.runtime_extra_inputs), runtime_extra_inputs_sha256=sha(args.runtime_extra_inputs))
    if short:
        training.update(trainable_motor_ids=[6], frozen_motor_ids=[4, 7, 8, 14, 27],
            source_hidden_layers_frozen=True, added_neurons_per_layer=8,
            source_readout_frozen=detail['source_readout_frozen'],
            slope_prior_scope='Medium and deep bend probes, negative q1 opposition included; training-only soft slope bounds',
            slope_prior_floors=detail['slope_floors'], medium_opposition_floors=detail['medium_opposition_floors'],
            slope_prior_scales=[5., 6., 3., 2.], slope_prior_directions=[1., -1., 1., 1.],
            slope_prior_ceilings=detail['slope_ceilings'])
        training.pop('m7_slope_prior_ceilings')
        c['thumb_short_extensor_training'] = training
    else:
        c['thumb_extensor_training'] = training
    c['previous_flex_ext_training'] = deepcopy(old['thumb_flex_ext_training'])
    # Preserve reference_report: fold loader uses it for original normalization.
    c['thumb_flex_ext_training'].update(width=len(head['branches'][0]['layers'][0]['bias']),
        epochs=args.epochs, trainable_motor_ids=[6] if short else [6, 7], frozen_motor_ids=[4, 7] if short else [4],
        refinement_report=str(out.resolve()), refinement_details='thumb_short_extensor_training' if short else 'thumb_extensor_training')
    groups = deepcopy(old['training']['groups'])
    width = c['thumb_flex_ext_training']['width']
    groups[0].update(architecture=[4, width, width, 3], frozen_motor_ids=[4, 7] if short else [4], trainable_motor_ids=[6] if short else [6, 7])
    c['training'] = dict(architecture_type='thumb_three_groups_refined_extensors', groups=groups,
        details=('One shared M4/M6/M7 MLP; source hidden subgraph and M4/M7 frozen, only new signed q0/q1 neurons and M6 readout trained'
                 if short else 'One expanded shared M4/M6/M7 MLP; M4 subgraph frozen, side/opposition copied unchanged'))
    c['preserved_outputs'] = dict(motor_ids=[old['motor_order'][i] for i in preserved_indices], source_model_sha256=sha(source),
        note=('Frozen source hidden subgraph and M4/M7 readouts; side/opposition, normalization, reference and zero policy unchanged'
              if short else 'Frozen M4 subgraph and unchanged side/opposition branches; normalization, reference and zero policy unchanged'))
    c['fit_limitation'] = ('q2/q3 directions remain structural; q0/q1 coupling and minimum slopes are soft priors. '
        'No new deep-bend motor supervision; amplitude, tension, robot posture and abnormal-input handling require hardware validation.')
    path = out/'models'/source.name
    path.write_text(json.dumps(c, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    MLPFingerMapper(path, finger='thumb')
    probe = np.random.default_rng(29).uniform([-50, -70, -120, -70], [70, 90, 150, 150], (5000, 4))
    before, after = motor_forward(old, probe), motor_forward(c, probe)
    preserved = float(abs(after[:, preserved_indices]-before[:, preserved_indices]).max())
    assert preserved <= 1e-10
    np.testing.assert_array_equal(motor_forward(c, np.zeros(4)), np.zeros(6))
    for axis in (2, 3):
        more = probe.copy()
        more[:, axis] += 1
        diff = motor_forward(c, more)-after
        assert diff[:, 0].max() <= 1e-9 and diff[:, 1:3].min() >= -1e-9
    summary = dict(previous_sha256=sha(source), new_sha256=sha(path), training=training,
        training_points=int(keep.sum()), source=provenance, validation=validation,
        preserved_output_max_difference=preserved, structural_check_points=5000,
        hardware_started=False, active_model_changed=False)
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    np.savez_compressed(out/'validation_predictions.npz', **arrays)
    shutil.copyfile(args.reference_report/'point_log_snapshot.json', out/'point_log_snapshot.json')
    print('Staged', path, flush=True)


if __name__ == '__main__':
    main()
