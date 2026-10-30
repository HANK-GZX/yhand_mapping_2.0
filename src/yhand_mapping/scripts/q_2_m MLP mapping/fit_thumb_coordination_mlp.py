#!/usr/bin/env python3
"""Fit shared posture features and three M8/M14/M27 heads, preserving flex/ext."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
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
from thumb_coordination_training import train_coordination, replace_coordination, quadratic_reference_jacobian, GRADIENT_SCALES, GRADIENT_UPPER
from mlp_q_to_motor import MODEL_DIR, MLPFingerMapper, forward, motor_forward


def fit_job(job):
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    scheme, label, q, y, probes, options = job
    jac = quadratic_reference_jacobian(q, y[:, 3:], probes)
    head, detail = train_coordination(q, y[:, 3:], probe_q=probes, reference_jacobian=jac, **options)
    return scheme, label, head, detail


def make_probes(archive):
    q = np.concatenate([archive[k+'_q'][archive[k+'_normal']] for k in ('mlp', 'linear', 'linear_thumb')])
    groups = [q[(q[:, 2] >= 50) & (q[:, 3] >= 30)],
        q[(q[:, 0] >= 10) & (q[:, 1] <= -5) & (q[:, 2] >= 15) & (q[:, 2] <= 50) & (q[:, 3] >= 15)],
        q[(abs(q[:, 2]) < 15) & (abs(q[:, 3]) < 20)], q[np.max(abs(q), axis=1) > 10]]
    rng = np.random.default_rng(29)
    sampled = [g[rng.choice(len(g), min(n, len(g)), replace=False)] for g, n in zip(groups, (96, 96, 48, 48)) if len(g)]
    anchors = np.array([[a, b, c, d] for a in (0., 10., 20.) for b in (-20., -10., 0., 10.)
        for c in (20., 40., 60., 80.) for d in (30., 50., 80.)])
    return np.concatenate(sampled+[anchors])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-model-dir', type=Path, default=MODEL_DIR)
    p.add_argument('--reference-report', type=Path, default=SCRIPTS/'motor_test_logs/thumb_extensor_mlp/20261029_M6_q1_medium')
    p.add_argument('--runtime-inputs', type=Path, default=SCRIPTS/'motor_test_logs/five_q_to_motor_test/analysis/20261029_1620_vs_1623_thumb/comparison_arrays.npz')
    p.add_argument('--report-dir', type=Path)
    p.add_argument('--epochs', type=int, default=1600)
    p.add_argument('--width', type=int, default=36)
    p.add_argument('--head-widths', type=int, nargs=3, default=[12, 24, 12])
    p.add_argument('--lr', type=float, default=.008)
    p.add_argument('--gradient-penalty', type=float, default=.4)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--resume', action='store_true', help='复用相同来源及参数下已完成的验证折')
    args = p.parse_args()
    if not 1 <= args.workers <= 4:p.error('workers需要1至4')
    source = args.source_model_dir/'left_thumb_q_to_motor_mlp.json'
    old = MLPFingerMapper(source, finger='thumb').config
    reference = json.loads((args.reference_report/'summary.json').read_text())
    if sha(source) != reference['new_sha256']:raise ValueError('来源权重与参考验证档案不匹配')
    a = np.load(args.reference_report/'validation_predictions.npz')
    absolute, y, meta, provenance = load_data(SCRIPTS/'motor_test_logs/data_collection', 'thumb')
    q = delta_angles(absolute, old['q_reference'])
    for k, v in [('q_delta', q), ('motor_feedback', y), ('point_ids', np.array([int(m['point_id']) for m in meta]))]:
        np.testing.assert_array_equal(a[k], v)
    assert old['source'] == provenance
    keep, combo = a['eligible'], a['combo']
    probes = make_probes(np.load(args.runtime_inputs))
    out = args.report_dir or SCRIPTS/'motor_test_logs/thumb_coordination_mlp'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=args.resume)
    for folder in ('models', 'previous_models'):
        (out/folder).mkdir(exist_ok=args.resume)
        for path in args.source_model_dir.glob('left_*_q_to_motor_mlp.json'):shutil.copyfile(path, out/folder/path.name)
    np.savez_compressed(out/'angle_probes.npz', q=probes)
    options = dict(epochs=args.epochs, width=args.width, head_widths=tuple(args.head_widths),
        lr=args.lr, gradient_penalty=args.gradient_penalty, seed=20261029)
    manifest = dict(source_sha256=sha(source), runtime_sha256=sha(args.runtime_inputs),
        options=dict(options, head_widths=list(args.head_widths)))
    manifest_path = out/'run_options.json'
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('复用折需要相同的来源、探针及训练参数记录')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    schemes = [('pose_group_5fold', a['pose_fold']), ('workspace_block_3fold', a['workspace_fold']),
        ('leave_one_combo_out', combo), ('zero_interval_3fold', grouped_folds(a['time_group'], count=3, seed=20261028))]
    arrays = {k: a[k] for k in ('q_absolute', 'q_delta', 'q_reference', 'motor_ids', 'motor_feedback',
        'point_ids', 'combo', 'eligible', 'pose_group', 'pose_fold', 'workspace_fold', 'time_group')}
    jobs, fold_info, predictions, details = [], {}, {}, {}
    for scheme, labels in schemes:
        predictions[scheme] = a[scheme].copy();details[scheme] = []
        for i, label in enumerate(np.unique(labels)):
            test, train = labels == label, (labels != label) & keep
            frozen = load_thumb_fold(old, args.reference_report, scheme, label)
            np.testing.assert_allclose(forward(frozen['network'], q[test]), a[scheme][test], atol=1e-9, rtol=0)
            fold_info[(scheme, str(label))] = (test, train, frozen)
            jobs.append((scheme, str(label), q[train], y[train], probes, dict(options, seed=20261029+i)))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        futures = []
        for job in jobs:
            scheme, label = job[:2]
            cached = out/'fold_models'/scheme/f'{label}.json'
            if args.resume and cached.exists():
                head = json.loads(cached.read_text())
                detail_path = cached.with_suffix('.details.json')
                detail = json.loads(detail_path.read_text()) if detail_path.exists() else dict(epochs=args.epochs, resumed=True)
                future = Future();future.set_result((scheme, label, head, detail))
            else:future = pool.submit(fit_job, job)
            futures.append(future)
        for future in as_completed(futures):
            scheme, label, head, detail = future.result()
            test, train, frozen = fold_info[(scheme, label)]
            new = replace_coordination(frozen, head)
            values = forward(new['network'], q[test])
            np.testing.assert_allclose(values[:, :3], a[scheme][test, :3], atol=1e-10, rtol=0)
            predictions[scheme][test, 3:] = values[:, 3:]
            folder = out/'fold_models'/scheme;folder.mkdir(parents=True, exist_ok=True)
            (folder/f'{label}.json').write_text(json.dumps(head, indent=2)+'\n')
            valid = test & keep
            detail.update(label=label, train=int(train.sum()), test=int(valid.sum()))
            (folder/f'{label}.details.json').write_text(json.dumps(detail, indent=2)+'\n')
            details[scheme].append(detail)
            if valid.any():print(scheme, label, 'M8/M14/M27 MAE', abs(predictions[scheme][valid, 3:]-y[valid, 3:]).mean(0).round(3).tolist(), flush=True)
    boundaries = json.loads((SCRIPTS/'motor_test_logs/workspace_boundaries/left/boundary_models.json').read_text())['combos']
    keys = workspace_keys(y, combo, boundaries)
    validation = {}
    for scheme, _ in schemes:
        pred = predictions[scheme]
        validation[scheme] = dict(mlp=metrics(y[keep], pred[keep]), previous_mlp=metrics(y[keep], a[scheme][keep]),
            n_test=int(keep.sum()), folds=sorted(details[scheme], key=lambda v: v['label']),
            macro_voxel_mae_motor_deg=macro_error(y[keep], pred[keep], combo[keep], keys[keep]).tolist(),
            previous_macro_voxel_mae_motor_deg=macro_error(y[keep], a[scheme][keep], combo[keep], keys[keep]).tolist(),
            all_recorded_diagnostic_mlp=metrics(y, pred), all_recorded_diagnostic_previous_mlp=metrics(y, a[scheme]),
            scope='Encoder/heads/normalization and reference-gradient fit use training-fold labels only. Runtime q-only probes are domain-informed priors, not an independent hardware test.')
        arrays[scheme] = pred;arrays['previous_'+scheme] = a[scheme]
    _, _, head, detail = fit_job(('final', '', q[keep], y[keep], probes, options))
    c = replace_coordination(old, head)
    c['validation'] = validation;c['previous_coordination_validation'] = old['validation']
    reports = group_validation_reports(old, args.reference_report)
    c['previous_group_validation_reports'] = deepcopy(reports)
    c['group_validation_reports'] = dict(flex_ext=reports['flex_ext'], coordination=str(out.resolve()))
    c['thumb_coordination_training'] = dict(options, head_widths=list(args.head_widths),
        optimizer='AdamW', weight_decay=0, labeled_points=int(keep.sum()), source=provenance,
        source_model_sha256=sha(source), reference_report=str(args.reference_report),
        runtime_input_archive=str(args.runtime_inputs), runtime_input_sha256=sha(args.runtime_inputs),
        runtime_motor_labels_used=False, position_labels='original paired motor feedback only',
        reference_gradient='complete quadratic refit using each training fold only; never deployed',
        gradient_scales=GRADIENT_SCALES.tolist(), gradient_upper=GRADIENT_UPPER.tolist(),
        trainable_motor_ids=[8, 14, 27], preserved_motor_ids=[4, 6, 7], detail=detail,
        no_position_pseudo_labels=True, no_runtime_baseline=True, no_runtime_compensation=True,
        selection='Fixed epoch candidate assessed on known runtime inputs and grouped folds; not independent final testing')
    c['training'] = dict(architecture_type='thumb_flex_ext_plus_shared_three_heads',
        groups=[deepcopy(old['training']['groups'][0]), dict(name='side_opposition', motor_ids=[8, 14, 27],
            shared_architecture=[4, args.width, args.width], head_widths=list(args.head_widths), activation='mixed_softplus')])
    c['preserved_outputs'] = dict(motor_ids=[4, 6, 7], source_model_sha256=sha(source), note='Flex/ext network, normalization, q reference and zero protocol unchanged')
    c['fit_limitation'] = 'Structural q0 M8+/M27- and q1 M14+; other coupling slopes are soft priors. Paired positions do not measure tension or guarantee linear physical tracking.'
    path = out/'models'/source.name
    path.write_text(json.dumps(c, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    MLPFingerMapper(path, finger='thumb')
    test = np.random.default_rng(29).uniform([-50, -70, -120, -70], [70, 90, 150, 150], (5000, 4))
    before, after = motor_forward(old, test), motor_forward(c, test)
    np.testing.assert_array_equal(after[:, :3], before[:, :3])
    np.testing.assert_array_equal(motor_forward(c, np.zeros(4)), np.zeros(6))
    plus = test.copy();plus[:, 0] += 1;delta = motor_forward(c, plus)-after
    assert delta[:, 3].min() >= -1e-9 and delta[:, 5].max() <= 1e-9
    plus = test.copy();plus[:, 1] += 1;assert (motor_forward(c, plus)-after)[:, 4].min() >= -1e-9
    summary = dict(previous_sha256=sha(source), new_sha256=sha(path), source=provenance,
        training=c['thumb_coordination_training'], validation=validation, training_points=int(keep.sum()),
        preserved_output_max_difference=0., structural_check_points=len(test),
        hardware_started=False, active_model_changed=False)
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    np.savez_compressed(out/'validation_predictions.npz', **arrays)
    shutil.copyfile(args.reference_report/'point_log_snapshot.json', out/'point_log_snapshot.json')
    print('Staged', path, flush=True)


if __name__ == '__main__':main()
