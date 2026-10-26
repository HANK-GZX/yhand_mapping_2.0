#!/usr/bin/env python3
"""Fit the left-thumb mixed polynomial model offline; never imports ROS/SDK.

By default uses the reviewed full-result cache, verifying its point-log snapshot.
--refresh-cache reads the original result/data XLSX pairs instead. Changing the
reference changes the expansion of f(delta_q)-f(0), so export again after changing it.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
PACKAGE = SCRIPTS.parent
DEFAULT_COLLECTION = PACKAGE / 'scripts/motor_test_logs/data_collection/20261019_170304_left_collection'
IDS = [4, 6, 7, 8, 14, 27]
ROLES = ['flex_long', 'ext_short', 'ext_long', 'adduction', 'opposition', 'abduction']
PAIRS = [(i, j) for i in range(4) for j in range(i, 4)]
SEED = 20261023
ZERO_REFERENCE = [8.491,38.262,26.028,-26.219]


def features(q, degree):
    x = np.c_[np.ones(len(q)), q]
    return np.c_[x, np.column_stack([q[:, i]*q[:, j] for i, j in PAIRS])] if degree == 2 else x


def fit(q, m, degree, *, zero_constrained=False):
    center, scale = (np.zeros(4) if zero_constrained else q.mean(0)), q.std(0)
    scale[scale < 1e-12] = 1
    phi = features((q-center)/scale, degree)
    beta = (np.vstack([np.zeros((1,m.shape[1])),np.linalg.lstsq(phi[:,1:],m,rcond=None)[0]])
            if zero_constrained else np.linalg.lstsq(phi,m,rcond=None)[0])
    return center, scale, beta


def predict(q, fitted, degree):
    center, scale, beta = fitted
    return features((q-center)/scale, degree) @ beta


def rebase_coefficients(fitted, degree):
    """Expand f(delta_q) - f(0) into raw-degree monomials exactly."""
    center, scale, beta = fitted
    linear = beta[1:5] / scale[:, None]
    quadratic = []
    if degree == 2:
        for k, (i, j) in enumerate(PAIRS):
            c = beta[5+k] / (scale[i]*scale[j])
            quadratic.append(c)
            linear[i] -= c*center[j]
            linear[j] -= c*center[i]  # Diagonal terms contribute twice.
    return np.vstack([np.zeros((1, beta.shape[1])), linear, *[c[None, :] for c in quadratic]])


def errors(y, predicted):
    e = y-predicted
    return {'n':len(y), 'r2':float(1-np.sum(e*e)/np.sum((y-y.mean())**2)),
            'mae':float(abs(e).mean()), 'rmse':float(np.sqrt(np.mean(e*e))),
            'p95_abs_error':float(np.percentile(abs(e), 95))}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(collection, refresh):
    finger = collection / 'left_hand/thumb'
    analysis = collection / 'analysis/thumb_linearity'
    cache, snapshot = analysis / 'data_cache.npz', analysis / 'point_log_snapshot.csv'
    with (finger / 'point_log.csv').open(encoding='utf-8-sig') as f:
        current_meta = list(csv.DictReader(f))
    if not refresh and cache.exists() and snapshot.exists():
        with snapshot.open(encoding='utf-8-sig') as f:
            meta = list(csv.DictReader(f))
        if meta != current_meta:
            raise ValueError('采样日志与缓存不同，请使用--refresh-cache重读原始结果')
        with np.load(cache) as data:
            if not np.array_equal(data['point_ids'], [int(r['point_id']) for r in meta]):
                raise ValueError('缓存point_id与日志不匹配')
            q, feedback = data['glove'][:, 8:12].copy(), data['feedback'].copy()
        provenance = {'cache_sha256':digest(cache), 'point_log_sha256':digest(finger/'point_log.csv')}
    else:
        from plot_collection_combo_distribution import load_points
        points, report = load_points(finger/'result', finger/'point_log.csv', 'left', 'thumb', [], False)
        if (report['invalid_points'] or report['missing_glove_point_ids'] or report['missing_motor_point_ids']
                or report['unlabelled_point_ids']):
            raise ValueError('原始结果文件存在缺失或无效配对')
        meta = [p['meta'] for p in points]
        q, feedback = np.stack([p['q'] for p in points]), np.stack([p['motor'] for p in points])
        provenance = {'point_log_sha256':digest(finger/'point_log.csv'), 'load_report':report}
    m = feedback[:, np.array(IDS)-1, 0]
    if not np.isfinite(q).all() or not np.isfinite(m).all():
        raise ValueError('q或电机反馈包含非有限数值')
    formal = np.array([r['active_mode']=='full' for r in meta])
    if formal.sum() < 30:
        raise ValueError('完整组合点不足以拟合四q二次模型')
    return q[formal], m[formal], [r for r, keep in zip(meta, formal) if keep], provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-dir', type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument('--output', type=Path, default=PACKAGE/'config/left_thumb_q_to_motor_models.json')
    parser.add_argument('--q-reference', type=float, nargs=4, default=ZERO_REFERENCE,
                        help='多项式展开参考；默认原采集启动帧，不是实测电机零位')
    parser.add_argument('--refresh-cache', action='store_true')
    args = parser.parse_args(argv)
    collection = args.collection_dir.resolve()
    q_abs, m, meta, provenance = load(collection, args.refresh_cache)
    reference = np.array(args.q_reference)
    if not np.isfinite(reference).all():
        raise ValueError('角度参考必须有限')
    q = q_abs-reference
    combo = np.array([r['combo_id'] for r in meta])
    rng = np.random.default_rng(SEED)
    fold_ids = np.empty(len(q), int)
    for c in np.unique(combo):
        ii = np.flatnonzero(combo == c)
        fold_ids[rng.permutation(ii)] = np.arange(len(ii)) % 5
    fitted, rebased, validation = {}, {}, {}
    for name, degree in [('linear', 1), ('quadratic', 2)]:
        fitted[name] = fit(q, m, degree)
        rebased[name] = rebase_coefficients(fitted[name], degree)
        # Check re-expansion at every training point before exporting control coefficients.
        expected = predict(q, fitted[name], degree)-predict(np.zeros((1, 4)), fitted[name], degree)
        if not np.allclose(features(q, degree)@rebased[name], expected, atol=1e-8, rtol=1e-10):
            raise ValueError('多项式零位展开校验失败')
        validation[name] = {}
        for scheme, labels in [('stratified_5fold', fold_ids), ('leave_one_combo_out', combo)]:
            predicted = np.empty_like(m)
            for label in np.unique(labels):
                test = labels == label
                predicted[test] = predict(q[test], fit(q[~test], m[~test], degree), degree)
            validation[name][scheme] = [errors(m[:, j], predicted[:, j]) for j in range(6)]
    config = {
        'schema_version':1, 'hand':'left', 'finger':'thumb', 'q_units':'degrees',
        'motor_position_units':'degrees', 'motor_order':IDS, 'default_model':'mixed',
        'q_reference':reference.tolist(),
        'q_reference_source':'Run startup glove frame; provisional expansion reference, NOT exact motor-zero calibration',
        'reference_is_exact_zero_calibration':False, 'calibration_measured_by_script':False,
        'zero_position_constraint':True, 'zero_motor_positions':{str(mid):0.0 for mid in IDS},
        'output_alignment':'delta_m = f(q_reference + delta_q) - f(q_reference); zero motor target at delta_q=0. Coefficients are expanded into unscaled delta_q monomials.',
        'model_policy':'M4/M7 always linear; M6/M8/M14/M27 always complete quadratic including all squares and pairwise cross terms, regardless of global model option.',
        'side_motor_mapping':'M8/M27: complete four-q quadratic compensation',
        'source_collection':str(collection.relative_to(PACKAGE)) if collection.is_relative_to(PACKAGE) else str(collection),
        'training_scope':'All formal active_mode=full result/data pairs, including movement timeouts; excludes single-motor preuniform points',
        'training_points':len(q), 'training_timeouts':sum(r['timeout']=='1' for r in meta),
        'fitting_method':'Affine OLS in training-standardized coordinates, exactly expanded as f(delta_q)-f(0); no origin-constrained refit',
        'fit_limitation':'Validation metrics describe original affine feedback coordinates, NOT rebased control accuracy. Default startup reference lies outside some sampled q ranges; no exact motor-zero alignment or hardware tracking validation has been performed.',
        'validation_seed':SEED, 'source_sha256':provenance,
        'q_relative_training_bounds':{f'q{i}':[float(q[:, i].min()), float(q[:, i].max())] for i in range(4)},
        'motor_position_training_bounds':{str(mid):[float(m[:, j].min()), float(m[:, j].max())] for j, mid in enumerate(IDS)},
        'motor_position_limits_deg':{},
        'motor_position_limit_source':'No additional thumb position limits; same target handling as current index/middle controller',
        'motors':{}}
    order = ['1', 'x0', 'x1', 'x2', 'x3']
    for j, (mid, role) in enumerate(zip(IDS, ROLES)):
        selected = 'linear' if mid in (4, 7) else 'quadratic'
        target = {'role':role, 'input_q_indices':[0, 1, 2, 3], 'control_model':selected, 'fits':{}}
        for name, degree in [('linear', 1), ('quadratic', 2)]:
            center, scale, beta = fitted[name]
            target['fits'][name] = {
                'degree':degree, 'effective_degree':degree,
                'input_center':[0.0]*4, 'input_scale':[1.0]*4,
                'feature_order':order+([f'x{i}*x{k}' for i, k in PAIRS] if degree == 2 else []),
                'coefficients':rebased[name][:, j].tolist(),
                'source_fit':{'input_center':center.tolist(), 'input_scale':scale.tolist(),
                              'coefficients':beta[:, j].tolist(),
                              'prediction_at_reference_motor_deg':float(predict(np.zeros((1, 4)), fitted[name], degree)[0, j]),
                              'descriptive_metrics':errors(m[:, j], predict(q, fitted[name], degree)[:, j])},
                'source_coordinate_validation':{scheme:rows[j] for scheme, rows in validation[name].items()}}
        config['motors'][str(mid)] = target
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'Wrote {args.output}: {len(q)} points; M4/M7 linear; M6/M8/M14/M27 quadratic')
    return 0


if __name__ == '__main__':
    sys.exit(main())
