#!/usr/bin/env python3
"""Stage refitted middle M23/M16 MLPs using only unchanged recorded feedback."""
import argparse
from copy import deepcopy
from datetime import datetime
from functools import partial
import json
from pathlib import Path
import shutil

import numpy as np

from fit_left_hand_q_to_motor_mlp import SCRIPTS, load_data, delta_angles, sha
from fit_monotonic_extensor import evaluate
from middle_bend_training import train_bend
from mlp_q_to_motor import MODEL_DIR, MLPFingerMapper, forward, motor_forward

SETTINGS = {
    23: dict(direction=1, activation='mixed_softplus_tanh', q3_penalty=.03, deep_weight=4.),
    16: dict(direction=-1, activation='mixed_softplus_tanh', q3_penalty=.03, deep_weight=2.),
}


def assemble(previous, heads):
    """Keep M2 and both side networks/normalization byte-for-byte equivalent."""
    if previous['finger'] != 'middle' or set(heads) != {23, 16}:
        raise ValueError('仅支持同时替换中指M23/M16')
    result = deepcopy(previous)
    network = result['network']
    for index, mid in enumerate((23, 16)):
        head = heads[mid]
        for key in ('input_center', 'input_scale'):
            np.testing.assert_allclose(network[key], head[key], atol=1e-12, rtol=1e-12)
        branch = deepcopy(head['branches'][0])
        branch['output_indices'] = [index]
        network['branches'] = [b for b in network['branches'] if b['output_indices'] != [index]]+[branch]
        for key in ('output_center', 'output_scale'):
            network[key][index] = head[key][0]
        result['input_priority']['per_motor_weights'][str(mid)] = list(branch['input_weights'])
    network['branches'].sort(key=lambda b:b['output_indices'][0])
    result['schema_version'] = 10
    result['raw_prediction_at_reference_motor_deg'] = forward(network, np.zeros(4)).tolist()
    result['input_priority']['policy'] = ('M23/M16: q1/q2 primary, q3 secondary gain 0.5, q0 gain 0.25; '
        'other motors retain their previous per-motor weights. See bend_head_training for penalties.')
    result['middle_structure_note'] = ('M23/M16 use monotonic Softplus+tanh hidden units with bounded negative tails. '
        'M23 q1/q2/q3 payout is monotonic; M16 q1/q2 winding is monotonic with weak, unconstrained q3 coupling. '
        'M2/M9/M24 weights and control policy unchanged.')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model-dir', type=Path, default=MODEL_DIR)
    parser.add_argument('--data-root', type=Path, default=SCRIPTS/'motor_test_logs/data_collection')
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--reuse-m23-report', type=Path,
                        help='Reuse an already validated identical-source M23 fit; M16 is still retrained')
    parser.add_argument('--reuse-m16-report', type=Path,
                        help='Reuse an already validated identical-source M16 fit; M23 is still retrained')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    if args.epochs < 1 or not 4 <= args.width <= 256 or args.threads < 1:
        parser.error('epochs/threads必须为正，width必须为4..256')
    import torch
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    source = args.source_model_dir/'left_middle_q_to_motor_mlp.json'
    previous = MLPFingerMapper(source, finger='middle').config
    if previous['schema_version'] not in (9, 10):
        raise ValueError('需要含中指独立M2/侧摆头的版本9或10模型')
    absolute, labels, meta, provenance = load_data(args.data_root.resolve(), 'middle')
    if len(absolute) != previous['training_points'] or provenance != previous['source']:
        raise ValueError('原始数据来源或数量改变，拒绝静默改变数据')
    q = delta_angles(absolute, previous['q_reference'])
    combo = np.array([row['combo_id'] for row in meta])
    report = args.report_dir or SCRIPTS/'motor_test_logs/middle_bend_mlp'/datetime.now().strftime('%Y%m%d_%H%M%S')
    report.mkdir(parents=True, exist_ok=False)
    backup, models = report/'previous_models', report/'models'
    backup.mkdir()
    models.mkdir()
    for path in args.source_model_dir.glob('left_*_q_to_motor_mlp.json'):
        shutil.copyfile(path, backup/path.name)
        shutil.copyfile(path, models/path.name)
    heads, results = {}, {}
    print(f'Report: {report}; unchanged measured samples: {len(q)}', flush=True)
    for index, mid in enumerate((23, 16)):
        folder = report/f'M{mid}'
        reuse_report = args.reuse_m23_report if mid == 23 else args.reuse_m16_report
        if reuse_report is not None:
            reuse = reuse_report.resolve()
            old_summary = json.loads((reuse/'summary.json').read_text())
            if old_summary['previous_sha256'] != sha(source):
                raise ValueError(f'复用M{mid}的来源模型与当前来源不同')
            saved = np.load(reuse/f'M{mid}/validation_predictions.npz')
            for key,value in [('q_absolute',absolute),('q_delta',q),('motor_feedback',labels),
                              ('point_ids',[int(row['point_id']) for row in meta])]:
                np.testing.assert_array_equal(saved[key], value)
            if any(old_summary['heads'][str(mid)]['training'][key] != value
                   for key,value in SETTINGS[mid].items()):
                raise ValueError(f'复用M{mid}的结构或训练设置不一致')
            if (old_summary['heads'][str(mid)]['training']['width'] != args.width
                    or old_summary['heads'][str(mid)]['training']['seed'] != args.seed
                    or old_summary['heads'][str(mid)]['training']['maximum_selection_epochs'] != args.epochs):
                raise ValueError(f'复用M{mid}的宽度、种子或轮数上限不一致')
            shutil.copytree(reuse/f'M{mid}', folder)
            heads[mid] = json.loads((folder/'head.json').read_text())
            results[str(mid)] = old_summary['heads'][str(mid)]
            print(f'M{mid} reused identical-source validated head from {reuse}', flush=True)
            continue
        folder.mkdir()
        settings = SETTINGS[mid]
        trainer = partial(train_bend, **{key:value for key,value in settings.items() if key != 'direction'})
        print(f'M{mid}: {settings}', flush=True)
        evaluation, predictions, epochs, groups, folds = evaluate(q, labels[:, index:index+1], combo,
            epochs=args.epochs, width=args.width, seed=args.seed,
            direction=settings['direction'], trainer=trainer)
        head, _ = trainer(q, labels[:, index], epochs=epochs, width=args.width,
                          seed=args.seed, direction=settings['direction'])
        heads[mid] = head
        training = dict(**settings, seed=args.seed, epochs=epochs, maximum_selection_epochs=args.epochs,
            width=args.width, optimizer='AdamW', lr=.01, early_stop_patience=150,
            input_weights=[.25, 1., 1., .5], q0_first_layer_penalty=.015,
            original_boundary_definition='q1 and q2 both >= their training-fold 80th percentiles',
            boundary_weight_normalized=True, torch_version=str(torch.__version__),
            no_added_samples=True, no_pseudo_labels=True, no_linear_baseline=True,
            source=provenance, input_definition=previous['training_input_definition'],
            target_definition=previous['training_target_definition'],
            selection='Median inner-selected epochs from five pose folds; final fit on all original rows')
        results[str(mid)] = dict(training=training, validation=evaluation,
            old_mae={scheme:previous['validation'][scheme]['mlp']['mae_motor_deg'][index] for scheme in evaluation},
            new_mae={scheme:values['mae_motor_deg'][0] for scheme,values in evaluation.items()})
        (folder/'head.json').write_text(json.dumps(head, indent=2)+'\n')
        (folder/'summary.json').write_text(json.dumps(results[str(mid)], ensure_ascii=False, indent=2)+'\n')
        np.savez_compressed(folder/'validation_predictions.npz', q_absolute=absolute, q_delta=q,
            q_reference=previous['q_reference'], motor_feedback=labels,
            point_ids=[int(row['point_id']) for row in meta], combo=combo,
            pose_group=groups, pose_fold=folds, **predictions)
        print(f'M{mid} result: epochs={epochs}; old MAE {results[str(mid)]["old_mae"]}; '
              f'new MAE {results[str(mid)]["new_mae"]}', flush=True)
    config = assemble(previous, heads)
    config['mapping_revision'] = 'middle_m23_m16_bend_20261027'
    config['bend_head_training'] = {mid:item['training'] for mid,item in results.items()}
    config['bend_head_validation'] = {mid:item['validation'] for mid,item in results.items()}
    config['extensor_structure'] = dict(type='monotonic_softplus_tanh_mlp', motor_id=23,
        architecture=[4, args.width, args.width, 1], nondecreasing_delta_q_indices=[1, 2, 3],
        activation='half softplus; half tanh; bounded negative hidden tails',
        input_weights=[.25,1.,1.,.5], output_activation='identity',
        guarantee='Monotonic payout in q1/q2/q3. No correct OOD amplitude or hardware accuracy guarantee.')
    config['extensor_training'] = results['23']['training']
    config['extensor_validation'] = results['23']['validation']
    config['shallow_flexor_structure'] = dict(type='partially_monotonic_softplus_tanh_mlp', motor_id=16,
        architecture=[4, args.width, args.width, 1], nonincreasing_delta_q_indices=[1, 2],
        unconstrained_delta_q_indices=[0, 3], input_weights=[.25,1.,1.,.5],
        hidden_activation='half softplus; half tanh, both monotonic; bounded negative hidden tails',
        q3_constraint='Soft weight penalty only; not forced to the deep-flexor q3 direction',
        guarantee='Monotonic winding in q1/q2 only. No OOD amplitude or q3 derivative guarantee.')
    config['preserved_outputs'] = dict(motor_ids=[2,9,24], source_model_sha256=sha(source),
        source_training=previous['training'], note='Their neural weights, scales, side policy and reference are unchanged.')
    config['training'] = dict(architecture_type='role_weighted_middle_bend_monotonic_mlp',
        updated_motor_ids=[23,16], architecture=[4,args.width,args.width,1],
        details='bend_head_training; M2/M9/M24 training fields and validation retained')
    for index, mid in enumerate((23,16)):
        for scheme,values in results[str(mid)]['validation'].items():
            for key,value in values.items():
                if key != 'folds':
                    config['validation'][scheme]['mlp'][key][index] = value[0]
    config['control_policy_validation_note'] = ('M23/M16 raw-feedback validation recomputed; other estimates inherited. '
        'Outer folds do not enter scaling, gradients or epoch selection. Architecture exploration was done before this '
        'evaluation; this is not a fully nested architecture-search accuracy estimate or hardware validation.')
    path = models/source.name
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    mapper = MLPFingerMapper(path, finger='middle')
    probe = np.random.default_rng(20261027).uniform([-60,-70,-120,-120],[75,150,150,120],(5000,4))
    old, new = motor_forward(previous, probe), motor_forward(config, probe)
    preserved = float(abs(old[:,2:]-new[:,2:]).max())
    np.testing.assert_allclose(new[:,2:], old[:,2:], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(motor_forward(config, np.zeros(4)), np.zeros(5))
    checks = dict(preserved_outputs_max_difference=preserved, monotonic_half_degree={})
    for axis in (1,2,3):
        more=probe.copy()
        more[:,axis]+=.5
        change=motor_forward(config,more)-new
        assert change[:,0].min()>=-1e-10
        if axis in (1,2):
            assert change[:,1].max()<=1e-10
        checks['monotonic_half_degree'][str(axis)]=[float(change[:,0].min()),float(change[:,1].max())]
    checks['same_zero_reference'] = previous['q_reference'] == config['q_reference']
    checks['same_side_policy'] = previous['side_bending_policy'] == config['side_bending_policy']
    summary = dict(training_points=len(q), source=provenance, previous_sha256=sha(backup/source.name),
        new_sha256=sha(path), heads=results, checks=checks, models=str(models), hardware_started=False)
    (report/'point_log_snapshot.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    (report/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(f'Staged and verified: {path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
