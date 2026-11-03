#!/usr/bin/env python3
"""Build manual delta-q references from recorded tuples, with whole-log holdout."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from joint_coupling import DEFAULT_REFERENCE
from hand_motor_map import HAND_MOTOR_MAP


COLLECTION_CACHES = {
    'thumb': 'analysis/thumb_linearity/data_cache.npz',
    'index': 'analysis/index_linearity/data_cache.npz',
    'middle': 'analysis/middle_linearity/data_cache.npz',
    'ring': 'analysis/ring_distribution/result_arrays.npz',
    'little': 'analysis/motor_to_q_estimator/little/data_cache.npz',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_reference(data, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    text = re.sub(r'\[\n\s+(-?[\d.]+),\n\s+(-?[\d.]+),\n\s+(-?[\d.]+),\n\s+(-?[\d.]+)\n\s+\]',
                  r'[\1, \2, \3, \4]', text)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(text)
    temporary.replace(output)


def suspect_angle_branches(values):
    """Known alternate representations plus bracketed deep q2 sign flips."""
    known = (values[:, 1] > 100) & (values[:, 2] < -60) & (values[:, 3] < -60)
    bracketed = np.zeros(len(values), dtype=bool)
    negative = np.flatnonzero(values[:, 2] < -60)
    if len(negative):
        groups = np.split(negative, np.flatnonzero(np.diff(negative) > 1) + 1)
        for group in groups:
            start, end = group[0], group[-1]
            if start == 0 or end+1 == len(values):
                continue
            before, after = values[start-1], values[end+1]
            # Both sides return to positive flexion, with a >90-degree q2 jump
            # at each boundary. No generic negative-angle or speed clipping.
            if (before[2] > 40 and after[2] > 40
                    and before[2]-values[start, 2] > 90 and after[2]-values[end, 2] > 90):
                bracketed[group] = True
    return known | bracketed, int(known.sum()), int((bracketed & ~known).sum())


def read_delta_log(source):
    """Read operator deltas only; arbitrary manual requests never augment support."""
    source = Path(source)
    meta = json.loads((source / 'metadata.json').read_text())
    if meta.get('hand') != 'left' or meta.get('input_mode') != 'manus' or not meta.get('manus_zero_q'):
        raise ValueError('需要已完成零位采样的左手Manus日志')
    fingers = meta.get('fingers') or [meta['finger']]
    slices = meta.get('finger_q_slices') or {finger: [4*i, 4*i+4] for i, finger in enumerate(fingers)}
    size = 4 * len(fingers)
    if sorted(slices.values()) != [[4*i, 4*i+4] for i in range(len(fingers))]:
        raise ValueError('手指角度分组必须为连续的四角组合')
    reference = np.asarray(meta['q_reference'], dtype=float)
    if reference.shape != (size,) or not np.isfinite(reference).all():
        raise ValueError('模型参考角缺失或非有限')
    rows, times, seen, counts = [], [], set(), Counter()
    with (source / 'samples.csv').open() as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            counts['csv_rows'] += 1
            if row.get('state') != 'position':
                counts['inactive'] += 1
                continue
            try:
                q = np.asarray([float(row[f'target_relative_q{i}']) for i in range(size)])
                absolute = np.asarray([float(row[f'target_q{i}']) for i in range(size)])
                elapsed = float(row['elapsed_sec'])
            except (ValueError, KeyError):
                counts['incomplete_target'] += 1
                continue
            if not np.isfinite(q).all() or not np.isfinite(absolute).all() or not np.isfinite(elapsed):
                counts['nonfinite'] += 1
                continue
            keys = ['manus_age_sec', 'feedback_age_sec']
            # Some older logs predate the raw-skeleton age column.
            keys += ['manus_left_raw_age_sec'] if 'manus_left_raw_age_sec' in row else []
            if any(not row.get(key) or not np.isfinite(float(row[key])) or float(row[key]) > .5 for key in keys):
                counts['stale_or_missing_age'] += 1
                continue
            if np.max(abs(absolute - reference - q)) > 1e-5:
                counts['coordinate_mismatch'] += 1
                continue
            identity = row.get('target_id') or str(elapsed)
            if identity in seen:
                counts['duplicate_target_id'] += 1
                continue
            seen.add(identity)
            rows.append(q)
            times.append(elapsed)
    if not rows:
        raise ValueError('没有完整、新鲜且坐标一致的活动目标')
    return meta, slices, np.asarray(rows), dict(counts)


def collection_finger(root, finger, q_reference, *, seed=20261103, verify_raw=True):
    """Read paired measured result means; split all members of each pose cell together."""
    folder = Path(root)/(finger+'_data')
    point_folder = next((p for p in [folder/'left_hand'/finger, folder/finger]
                         if (p/'point_log.csv').is_file()), None)
    if point_folder is None:
        raise ValueError(f'{finger}采集目录没有point_log.csv')
    log_path = point_folder/'point_log.csv'
    with log_path.open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row['finger'] != finger for row in rows):
        raise ValueError(f'{finger}采集日志为空或手指标记不一致')
    ids = np.asarray([int(row['point_id']) for row in rows])
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f'{finger}采集point_id重复')
    cache = folder/COLLECTION_CACHES[finger]
    snapshot = cache.parent/'point_log_snapshot.csv'
    with snapshot.open(encoding='utf-8-sig') as stream:
        if list(csv.DictReader(stream)) != rows:
            raise ValueError(f'{finger}采集缓存的日志快照已过期，不能直接补充')
    with np.load(cache, allow_pickle=False) as data:
        cache_ids = data['point_ids'] if 'point_ids' in data else data['ids']
        if not np.array_equal(ids, cache_ids):
            raise ValueError(f'{finger}缓存point_id与原始日志不一致')
        glove = data['glove'].copy() if 'glove' in data else data['glove48'].copy()
        feedback = data['feedback'].copy()
        sample_key = next((key for key in ['sample_count','sample_counts'] if key in data), None)
        sample_counts = data[sample_key].copy() if sample_key else None
    if glove.shape != (len(ids),48) or feedback.shape != (len(ids),28,5):
        raise ValueError(f'{finger}采集缓存形状错误')
    offset = 8 + 4*list(COLLECTION_CACHES).index(finger)
    absolute = glove[:,offset:offset+4]
    reference = np.asarray(q_reference,dtype=float)
    if reference.shape != (4,) or not np.isfinite(reference).all():
        raise ValueError('采集角度转换需要当前模型的四个有限参考角')
    delta = absolute-reference
    motor_indices = np.array(list(HAND_MOTOR_MAP['left'][finger].values()))-1
    finite = np.isfinite(delta).all(axis=1) & np.isfinite(feedback[:,motor_indices,:]).all(axis=(1,2))
    valid_count = (np.isfinite(sample_counts) & (sample_counts>0) & (sample_counts==np.floor(sample_counts))
                   if sample_counts is not None else np.ones(len(ids),dtype=bool))
    if valid_count.shape != (len(ids),):
        raise ValueError(f'{finger}采集sample_count形状错误')
    # The points are independent sampled poses, not a time-continuous teleop trace.
    # Do not apply the temporal sign-flip heuristic across independent points.
    branch = (delta[:,1]>100) & (delta[:,2]<-60) & (delta[:,3]<-60)
    valid = finite & valid_count & ~branch
    if valid.sum() < 5:
        raise ValueError(f'{finger}有效采集角度不足5点')
    poses, group = np.unique(np.round(delta[valid]/2).astype(int),axis=0,return_inverse=True)
    if len(poses)<5:
        raise ValueError(f'{finger}有效独立姿态格不足5个')
    rng = np.random.default_rng(seed+offset)
    held_cells = rng.choice(len(poses),max(1,len(poses)//5),replace=False)
    held = np.zeros(len(ids),dtype=bool)
    held[valid] = np.isin(group,held_cells)
    train = valid & ~held
    startup_path = point_folder.parent/'startup_first_frame_glove48.csv'
    with startup_path.open(encoding='utf-8-sig') as stream:
        startup = next(csv.DictReader(stream))
    startup_q = [float(startup[f'q_{i}']) for i in range(offset,offset+4)]
    if not np.isfinite(startup_q).all():
        raise ValueError(f'{finger}采集启动角无效')
    spot_checks = []
    if verify_raw:
        from openpyxl import load_workbook
        for i in rng.choice(np.flatnonzero(valid),min(8,int(valid.sum())),replace=False):
            raw_path = point_folder/'result/glove_data'/f'{ids[i]:05d}.xlsx'
            workbook = load_workbook(raw_path,read_only=True,data_only=True)
            try:
                raw = np.asarray(next(workbook['data'].values),dtype=float)[offset:offset+4]
            finally:
                workbook.close()
            np.testing.assert_allclose(raw,absolute[i],rtol=0,atol=1e-8,
                                       err_msg=f'{finger}原始glove与缓存不一致')
            motor_path = point_folder/'result/motor_feedback'/f'{ids[i]:05d}.xlsx'
            workbook = load_workbook(motor_path,read_only=True,data_only=True)
            try:
                raw_rows = list(workbook['data'].values)
            finally:
                workbook.close()
            raw_count = float(raw_rows[0][1])
            if not np.isfinite(raw_count) or raw_count<=0 or not raw_count.is_integer():
                raise ValueError(f'{finger}原始结果窗口sample_count无效')
            if sample_counts is not None and sample_counts[i]!=raw_count:
                raise ValueError(f'{finger}原始sample_count与缓存不一致')
            np.testing.assert_allclose(np.asarray(raw_rows[1:],dtype=float)[motor_indices],feedback[i,motor_indices],
                                       rtol=0,atol=1e-7,err_msg=f'{finger}原始feedback与缓存不一致')
            spot_checks.append(dict(point_id=int(ids[i]),glove_sha256=digest(raw_path),feedback_sha256=digest(motor_path)))
    def counts(mask, key):
        return dict(Counter(row[key] for row,keep in zip(rows,mask) if keep))
    info = dict(finger=finger, point_log=str(log_path.relative_to(root)), point_log_sha256=digest(log_path),
                cache=str(cache.relative_to(root)), cache_sha256=digest(cache),
                snapshot_sha256=digest(snapshot), startup_sha256=digest(startup_path),
                q_columns_0_based=[offset,offset+4], q_reference_deg=reference.tolist(),
                reference_rule='delta_q = measured result mean - current mapping q_reference; no extra zero alignment',
                startup_q_deg=startup_q,startup_minus_mapping_reference_deg=(np.array(startup_q)-reference).tolist(),
                total_points=len(ids), valid_points=int(valid.sum()), training_points=int(train.sum()),
                held_out_points=int(held.sum()), unique_pose_cells=len(poses), held_out_pose_cells=len(held_cells),
                seed=seed, split_rule='20% of 2-degree 4D pose cells; all repeated members assigned together',
                training_point_ids=ids[train].tolist(),held_out_point_ids=ids[held].tolist(),
                rejected_point_ids=ids[~valid].tolist(),
                invalid_finite_points=int((~finite).sum()),invalid_sample_count_points=int((~valid_count).sum()),
                excluded_angle_branch_points=int(branch.sum()),
                training_reached_flags=counts(train,'reached'),training_timeout_flags=counts(train,'timeout'),
                training_zero_states=counts(train,'zero_calib_state'),training_modes=counts(train,'active_mode'),
                sample_count_validation='full cache' if sample_counts is not None else 'legacy cache lacks count; raw spot checks only',
                raw_spot_checks=spot_checks,
                measurement_note='Recorded result/glove_data mean paired with measured motor feedback; not requested joint angles. '
                                 'Timeout and zero-calibration flags are retained, not treated as invalid measured q.')
    return dict(ids=ids,absolute=absolute,delta=delta,valid=valid,train=train,held=held,info=info)


def augment_collection(root, baseline_reference, output, *, seed=20261103):
    root = Path(root).resolve()
    data = json.loads(Path(baseline_reference).read_text())
    if data['source'].get('collection_supplement'):
        raise ValueError('baseline-reference应为补充采集角度前的参考，避免重复入库')
    sources = {}
    for finger in data['fingers']:
        profile = data['fingers'][finger]
        batch = collection_finger(root,finger,profile['q_reference_deg'],seed=seed)
        old = np.asarray(profile['observed_delta_q_deg'])
        train = batch['delta'][batch['train']]
        _, index = np.unique(np.round(train/2).astype(int),axis=0,return_index=True)
        existing = {tuple(v) for v in np.round(old/2).astype(int)}
        additions = np.asarray([train[i] for i in sorted(index)
                                if tuple(np.round(train[i]/2).astype(int)) not in existing]).reshape(-1,4)
        points = np.vstack([old,np.round(additions,6)])
        profile['pre_collection_representative_count'] = len(old)
        profile['collection_training_samples'] = int(batch['train'].sum())
        profile['collection_held_out_samples'] = int(batch['held'].sum())
        profile['collection_added_representative_count'] = len(additions)
        profile['representative_count'] = len(points)
        profile['observed_delta_q_deg'] = points.tolist()
        sources[finger] = batch['info']
        print(f'{finger}: collection train={batch["train"].sum()}, held={batch["held"].sum()}, added={len(additions)}',flush=True)
    data['source']['kind'] = 'historical_and_collection_union'
    data['source']['collection_supplement'] = dict(parent_reference_sha256=digest(baseline_reference),
        seed=seed, fingers=sources, point_cloud_units='delta_q_deg',
        historical_holdout_note='Existing 32 held-out teleop logs remain excluded from construction.',
        measurement_note='Collection result means add measured-angle coverage; combined reference still includes operator input records.')
    data['note'] = ('Union of observed operator deltas and collection joint-angle result means; '
                    'not a certified mechanical workspace. Same paired-neighbor rule and unchanged tolerances.')
    write_reference(data,output)
    print(json.dumps(dict(output=str(output),representatives={f:p['representative_count'] for f,p in data['fingers'].items()}),ensure_ascii=False))
    return data


def augment_history(root, validation_manifest, baseline_reference, output):
    """Only unselected Manus logs may add points; frozen holdout logs stay unseen."""
    root = Path(root).resolve()
    manifest = json.loads(Path(validation_manifest).read_text())
    data = json.loads(Path(baseline_reference).read_text())
    if data.get('source', {}).get('kind') in ('historical_log_union','historical_and_collection_union'):
        raise ValueError('baseline-reference应为补充前的单段参考，避免嵌套补充或验证泄漏')
    held = {entry['log'] for entry in manifest['selected_logs']}
    anchor = data['source']['log_name']
    pools = {finger: [] for finger in data['fingers']}
    counts = {finger: Counter() for finger in pools}
    provenance, excluded = [], []
    for meta_path in sorted(root.glob('*q_to_motor_test/20*/metadata.json')):
        source = meta_path.parent
        relative = str(source.relative_to(root))
        if relative in held:
            continue
        if source.name == anchor:
            assert digest(source / 'samples.csv') == data['source']['samples_sha256']
            assert digest(meta_path) == data['source']['metadata_sha256']
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get('input_mode') != 'manus':
            continue
        try:
            meta, slices, q, quality = read_delta_log(source)
        except ValueError as exc:
            excluded.append(dict(log=relative, reason=str(exc)))
            continue
        entry = dict(log=relative, model=meta['model'], samples_sha256=digest(source / 'samples.csv'),
                     metadata_sha256=digest(meta_path), fresh_unique_target_updates=len(q),
                     q_reference_deg=meta['q_reference'], operator_zero_q_deg=meta['manus_zero_q'],
                     finger_q_slices=slices, quality_counts=quality, fingers={})
        for finger, (start, end) in slices.items():
            if finger not in pools:
                raise ValueError(f'历史日志包含未知手指{finger}')
            values = q[:, start:end]
            suspect, known_count, bracket_count = suspect_angle_branches(values)
            good = values[~suspect]
            pools[finger].append(good)
            counts[finger].update(source_samples=len(values), excluded_angle_branch_samples=int(suspect.sum()),
                                  bracketed_sign_flip_samples=bracket_count, additional_log_count=1)
            entry['fingers'][finger] = dict(source_samples=len(values), excluded_angle_branch_samples=int(suspect.sum()),
                                           known_angle_branch_samples=known_count, bracketed_sign_flip_samples=bracket_count)
        provenance.append(entry)
    if not provenance:
        raise ValueError('没有未参与验证的历史Manus日志可补充')
    assert held.isdisjoint(entry['log'] for entry in provenance)
    baseline_source = data['source']
    data['source'] = dict(kind='historical_log_union', log_name='historical_delta_q_union_20261103',
                          baseline_sha256=digest(baseline_reference), anchor=baseline_source,
                          validation_manifest_sha256=digest(validation_manifest),
                          held_out_logs=sorted(held), additional_logs=provenance,
                          excluded_logs=excluded,
                          fresh_unique_target_updates=baseline_source['fresh_unique_target_updates']
                              + sum(log['fresh_unique_target_updates'] for log in provenance),
                          robot_joint_measurement_enabled=False,
                          build_note='Preserve anchor points, add real tuples from non-holdout Manus logs; '
                                     'no manual targets, synthetic tuples, interpolation or tolerance widening.')
    for finger, profile in data['fingers'].items():
        baseline = np.asarray(profile['observed_delta_q_deg'])
        new = np.concatenate(pools[finger]) if pools[finger] else np.empty((0, 4))
        _, indices = np.unique(np.round(new / 2).astype(int), axis=0, return_index=True)
        # Preserve every old point exactly: same-cell additions cannot replace old support.
        occupied = {tuple(v) for v in np.round(baseline / 2).astype(int)}
        additions = [new[i] for i in sorted(indices) if tuple(np.round(new[i] / 2).astype(int)) not in occupied]
        added = np.asarray(additions).reshape(-1, 4)
        points = np.vstack([baseline, np.round(added, 6)])
        profile['anchor_source_samples'] = profile['source_samples']
        profile['anchor_excluded_angle_branch_samples'] = profile['excluded_angle_branch_samples']
        profile['anchor_representative_count'] = len(baseline)
        profile['anchor_operator_zero_q_deg'] = profile['operator_zero_q_deg']
        profile['operator_zero_q_deg'] = None
        profile['coordinate_note'] = 'Each historical tuple uses its own session delta; never subtract anchor zero again.'
        profile['source_samples'] += counts[finger]['source_samples']
        profile['excluded_angle_branch_samples'] += counts[finger]['excluded_angle_branch_samples']
        profile['excluded_bracketed_sign_flip_samples'] = counts[finger]['bracketed_sign_flip_samples']
        profile['exclusion_rule'] = ('q1>100 AND q2<-60 AND q3<-60; or q2<-60 segment bracketed '
                                     'by q2>40 on both sides with >90-degree boundary jumps')
        profile['additional_log_count'] = counts[finger]['additional_log_count']
        profile['additional_representative_count'] = len(added)
        profile['representative_count'] = len(points)
        profile['observed_delta_q_deg'] = points.tolist()
    write_reference(data, output)
    print(json.dumps(dict(output=str(output), added_logs=len(provenance), held_out_logs=len(held),
                          representatives={finger: profile['representative_count'] for finger, profile in data['fingers'].items()}),
                     ensure_ascii=False))
    return data


def build(source, output):
    meta_path, csv_path = source / 'metadata.json', source / 'samples.csv'
    meta = json.loads(meta_path.read_text())
    if meta['finger'] != 'five' or meta['input_mode'] != 'manus' or not meta.get('manus_zero_q'):
        raise ValueError('需要已完成操作员零位采样的五指Manus日志')
    rows, seen = [], set()
    with csv_path.open() as stream:
        for row in csv.DictReader(stream):
            if row['state'] != 'position' or row['target_id'] in seen:
                continue
            if not all(row[f'target_relative_q{i}'] for i in range(20)):
                continue
            ages = [row.get(key) for key in ['manus_age_sec', 'manus_left_raw_age_sec', 'feedback_age_sec']]
            if any(not v or not np.isfinite(float(v)) or float(v) > .5 for v in ages):
                continue
            if not np.isfinite([float(row[f'target_relative_q{i}']) for i in range(20)]).all():
                continue
            seen.add(row['target_id'])
            rows.append(row)
    if not rows:
        raise ValueError('没有完整、新鲜的五指角度目标')
    q = np.array([[float(row[f'target_relative_q{i}']) for i in range(20)] for row in rows])
    data = dict(schema_version=1, coordinates='delta_q_deg', tolerance_deg=[5., 10., 10., 10.],
                representative_voxel_deg=2.,
                source=dict(log_name=source.name, model=meta['model'], thumb_version=meta.get('thumb_version'),
                            samples_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                            metadata_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                            fresh_unique_target_updates=len(rows),
                            robot_joint_measurement_enabled=meta['robot_glove_measurement_enabled']),
                note='Observed operator delta-q tuples, not verified robot reachability. '
                     'Reject unsupported manual requests; keep q0/q1 and never project angles automatically.',
                fingers={})
    for finger, (start, end) in meta['finger_q_slices'].items():
        x = q[:, start:end]
        suspect = (x[:, 1] > 100) & (x[:, 2] < -60) & (x[:, 3] < -60)
        good = x[~suspect]
        # Retain a real observed tuple from each 2-degree 4D cell, never its centroid.
        _, indices = np.unique(np.round(good / 2).astype(int), axis=0, return_index=True)
        points = good[np.sort(indices)]
        data['fingers'][finger] = dict(source_samples=len(x), excluded_angle_branch_samples=int(suspect.sum()),
                                      exclusion_rule='q1>100 AND q2<-60 AND q3<-60; observed alternate-angle branch',
                                      representative_count=len(points),
                                      q_reference_deg=meta['q_reference'][start:end],
                                      operator_zero_q_deg=meta['manus_zero_q'][start:end],
                                      observed_delta_q_deg=np.round(points, 6).tolist())
    write_reference(data, output)
    print(json.dumps(dict(output=str(output), representatives={
        f: p['representative_count'] for f, p in data['fingers'].items()}), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, nargs='?')
    parser.add_argument('--history-root', type=Path)
    parser.add_argument('--collection-root', type=Path)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--validation-manifest', type=Path)
    parser.add_argument('--baseline-reference', type=Path)
    parser.add_argument('--output', type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    if args.collection_root:
        if args.source or args.history_root or not args.baseline_reference:
            parser.error('采集补充需要--baseline-reference，不能同时指定单段source或--history-root')
        augment_collection(args.collection_root,args.baseline_reference,args.output,seed=args.seed)
    elif args.history_root:
        if args.source or not args.validation_manifest or not args.baseline_reference:
            parser.error('历史补充需要--validation-manifest和--baseline-reference，不能同时指定单段source')
        augment_history(args.history_root, args.validation_manifest, args.baseline_reference, args.output)
    elif args.source:
        build(args.source, args.output)
    else:
        parser.error('请指定单段source，或使用--history-root补充历史日志')
