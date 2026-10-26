#!/usr/bin/env python3
"""Plot thumb-tip XYZ from the skeletons paired with the current model's q data.

Node 0 is wrist; node 4 is thumb tip, according to this collection's 25-node
layout. Express tip positions in the wrist frame using normalized [x,y,z,w]
quaternions. No robot forward kinematics or execution of fitted commands is
implied. Default length units remain the recorded SDK units, since sender
unitScale is absent from this repository.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/yhand_thumb_xyz_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent/'q_2_m linear mapping'))
from left_little_q_to_motor import FingerMapper
from replace_right_non_thumb_q_from_skeleton import read_xlsx_first_row

PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_RUN = PACKAGE/'scripts/motor_test_logs/data_collection/20261019_170304_left_collection'
DOCS = {
    'coordinate_and_unit_configuration':'https://docs.manus-meta.com/2.4.0/Plugins/SDK/SDK%20Minimal%20Client/',
    'raw_skeleton_semantics':'https://docs.manus-meta.com/latest/Plugins/SDK/C%2B%2B/SDK%20Client/',
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wrist_local_positions(skeleton):
    """Transform global bone poses into the wrist's own frame (N,25,3)."""
    a = np.asarray(skeleton, dtype=float)
    if a.ndim != 2 or a.shape[1] != 176 or not np.isfinite(a).all():
        raise ValueError('骨架必须是N×176有限数值矩阵')
    if not np.allclose(a[:, 0], 0, atol=1e-6):
        raise ValueError('存在非左手骨架')
    nodes = a[:, 1:].reshape(-1, 25, 7)
    norm = np.linalg.norm(nodes[:, 0, 3:7], axis=1)
    if (norm < 1e-8).any():
        raise ValueError('腕部四元数无效')
    rotation = Rotation.from_quat(nodes[:, 0, 3:7]/norm[:, None]).as_matrix()
    relative = nodes[:, :, :3]-nodes[:, 0:1, :3]
    return np.einsum('nji,nkj->nki', rotation, relative)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_skeletons(run, out, ids, refresh):
    folder = run/'left_hand/thumb/result/Glove Skeleton Data'
    files = {int(p.stem):p for p in folder.glob('*.xlsx') if p.stem.isdigit()}
    missing = sorted(set(map(int, ids))-files.keys())
    if missing:
        raise ValueError(f'缺少骨架结果文件：{missing[:20]}')
    manifest = [(int(pid), files[int(pid)].stat().st_size, files[int(pid)].stat().st_mtime_ns) for pid in ids]
    manifest_hash = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
    cache = out/'skeleton_cache.npz'
    if cache.exists() and not refresh:
        with np.load(cache) as z:
            if np.array_equal(z['point_ids'], ids) and z['manifest_sha256'].item()==manifest_hash:
                return z['skeletons'].copy(), manifest_hash
    skeletons = []
    for i, pid in enumerate(ids, 1):
        skeletons.append(read_xlsx_first_row(files[int(pid)]))
        if i%1000==0 or i==len(ids):
            print(f'Read skeleton {i}/{len(ids)}', flush=True)
    skeletons = np.array(skeletons, dtype=float)
    np.savez_compressed(cache, point_ids=ids, skeletons=skeletons, manifest_sha256=manifest_hash)
    return skeletons, manifest_hash


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--model-file', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--refresh-cache', action='store_true')
    parser.add_argument('--length-scale', type=float, default=1,
                        help='仅改变显示尺度；默认1保留SDK原始长度。确认原始单位为m后可用1000转为mm')
    parser.add_argument('--unit-label', default='SDK length unit', help='显示单位名称，不自动推定物理单位')
    args = parser.parse_args(argv)
    if not np.isfinite(args.length_scale) or args.length_scale<=0:
        raise ValueError('length-scale必须为正有限值')
    run = args.collection_dir.resolve()
    out = args.output_dir or run/'analysis/thumb_xyz_workspace'
    out.mkdir(parents=True, exist_ok=True)
    analysis = run/'analysis/thumb_linearity'
    with (analysis/'point_log_snapshot.csv').open(encoding='utf-8-sig') as f:
        meta = list(csv.DictReader(f))
    with (run/'left_hand/thumb/point_log.csv').open(encoding='utf-8-sig') as f:
        current_meta = list(csv.DictReader(f))
    if meta!=current_meta:
        raise ValueError('角度缓存日志与当前采样日志不匹配')
    mapper = FingerMapper(args.model_file, finger='thumb')
    q_cache = analysis/'data_cache.npz'
    expected_hash = mapper.config.get('source_sha256', {}).get('cache_sha256')
    if expected_hash and expected_hash!=sha256(q_cache):
        raise ValueError('角度缓存与当前模型训练来源不匹配')
    with np.load(q_cache) as z:
        ids, q_abs = z['point_ids'].copy(), z['glove'][:, 8:12].copy()
    if not np.array_equal(ids, [int(r['point_id']) for r in meta]):
        raise ValueError('角度缓存point_id不匹配')
    skeleton, manifest_hash = load_skeletons(run, out, ids, args.refresh_cache)
    local = wrist_local_positions(skeleton)
    nodes = skeleton[:, 1:].reshape(-1, 25, 7)
    # Validate global-position interpretation from fixed child offsets in parent
    # orientation frames. Constant bone lengths alone would not validate rotations.
    vectors = np.diff(nodes[:, :5, :3], axis=1)
    parent_rotations = Rotation.from_quat(nodes[:, :4, 3:7].reshape(-1, 4))
    parent_offsets = parent_rotations.inv().apply(vectors.reshape(-1, 3)).reshape(-1, 4, 3)
    offset_p95 = np.percentile(np.linalg.norm(parent_offsets-np.median(parent_offsets, axis=0), axis=2), 95, axis=0)
    lengths = np.linalg.norm(vectors, axis=2)
    # Preserve valid preuniform points in CSV but show formal training support in figures.
    formal = np.array([r['active_mode']=='full' for r in meta])
    if not formal.any():
        raise ValueError('没有正式full采样点')
    predicted = np.array([[result[f'M{mid}'] for mid in mapper.motor_ids]
                          for row in q_abs for result in [mapper.predict(row, relative=False)]])
    rows = []
    for i, record in enumerate(meta):
        row = {'point_id':int(ids[i]), 'combo_id':record['combo_id'], 'active_mode':record['active_mode'],
               'reached':record['reached'], 'timeout':record['timeout']}
        row.update({f'q{j}_absolute_deg':float(q_abs[i, j]) for j in range(4)})
        for j, axis in enumerate('xyz'):
            row[f'tip_wrist_{axis}_raw_unit'] = float(local[i, 4, j])
            row[f'tip_stream_{axis}_raw_unit'] = float(nodes[i, 4, j])
        row.update({f'M{mid}_model_command_deg':float(predicted[i, j]) for j, mid in enumerate(mapper.motor_ids)})
        rows.append(row)
    write_csv(out/'thumb_tip_xyz_points.csv', rows)
    xyz = local[formal, 4]*args.length_scale
    xyz_bounds = {axis:{'minimum':float(xyz[:, i].min()), 'maximum':float(xyz[:, i].max()),
                       'span':float(np.ptp(xyz[:, i]))} for i, axis in enumerate('XYZ')}
    summary = {'collection':str(run), 'model_file':str(mapper.model_path), 'model_sha256':sha256(mapper.model_path),
               'q_cache_sha256':sha256(q_cache), 'skeleton_manifest_sha256':manifest_hash,
               'all_points':len(ids), 'formal_points':int(formal.sum()), 'preuniform_points':int((~formal).sum()),
               'invalid_points':0, 'tip_node_id':4, 'wrist_node_id':0,
               'frame':'p_tip_wrist = R_wrist^T @ (p_tip_stream - p_wrist_stream), normalized mean wrist quaternion',
               'length_unit':args.unit_label, 'display_scale':args.length_scale,
               'sender_unitScale_verified':False, 'XYZ_bounds_in_display_units':xyz_bounds,
               'median_thumb_chain_lengths_raw_units':np.median(lengths[formal], axis=0).tolist(),
               'parent_frame_bone_offset_p95_deviation_raw_units':offset_p95.tolist(),
               'scope':'MANUS skeleton thumb-tip coverage paired with current model training q vectors; not robot FK or tracking of model commands',
               'average_transform_note':'Uses mean result skeletons. Transform of mean positions/orientations approximates, but is not generally equal to, the mean per-frame wrist-relative tip.',
               'no_hull_volume':'No physical workspace volume or convex-envelope fill is inferred from sparse/incomplete coverage.',
               'sources':DOCS}
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    plot(out, xyz, q_abs[formal, 3], args.unit_label)
    plot_chains(out, local[formal, :5]*args.length_scale, q_abs[formal, 3], args.unit_label)
    report(out, summary)
    print(json.dumps({'output_dir':str(out), 'formal_points':int(formal.sum()), 'bounds':xyz_bounds,
                      'parent_offset_p95_deviation':offset_p95.tolist()}, ensure_ascii=False, indent=2))
    return 0


def save(out, fig, name):
    fig.savefig(out/f'{name}.png', dpi=170, bbox_inches='tight')
    fig.savefig(out/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def equal3(ax, points):
    center = (points.min(0)+points.max(0))/2
    radius = max(np.ptp(points, axis=0).max()/2, 1e-10)*1.08
    ax.set(xlim=(center[0]-radius, center[0]+radius),
           ylim=(center[1]-radius, center[1]+radius), zlim=(center[2]-radius, center[2]+radius))
    ax.set_box_aspect((1, 1, 1))


def plot(out, xyz, color, unit):
    plt.rcParams.update({'font.family':'Noto Sans CJK JP', 'axes.unicode_minus':False,
                         'font.size':10, 'axes.grid':True, 'grid.alpha':.2})
    fig = plt.figure(figsize=(15, 11), layout='constrained')
    for k, angles in enumerate([(24, -55), (23, 130)]):
        ax = fig.add_subplot(2, 2, k+1, projection='3d')
        sc = ax.scatter(*xyz.T, c=color, cmap='viridis', s=4, alpha=.5, rasterized=True)
        equal3(ax, xyz)
        ax.view_init(*angles)
        ax.set(xlabel=f'X / {unit}', ylabel=f'Y / {unit}', zlabel=f'Z / {unit}',
               title='腕部局部坐标：三维视角 '+str(k+1))
        fig.colorbar(sc, ax=ax, shrink=.65, pad=.08, label='q3 / °')
    for k, (i, j, title) in enumerate([(0, 1, 'XY 投影'), (0, 2, 'XZ 投影')]):
        ax = fig.add_subplot(2, 2, k+3)
        ax.scatter(xyz[:, i], xyz[:, j], c=color, cmap='viridis', s=3, alpha=.4, rasterized=True)
        ax.set(xlabel=f'{"XYZ"[i]} / {unit}', ylabel=f'{"XYZ"[j]} / {unit}', title=title)
        ax.set_aspect('equal', adjustable='box')
    fig.suptitle(f'左手拇指指尖 XYZ 样本覆盖：MANUS骨架（n={len(xyz):,}）\n'
                 '腕部局部坐标；与当前拟合模型的正式训练样本对应；长度保留SDK原始单位', fontsize=13)
    save(out, fig, 'thumb_tip_xyz_workspace')
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
    for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        ax.scatter(xyz[:, i], xyz[:, j], c=color, cmap='viridis', s=4, alpha=.4, rasterized=True)
        ax.set(xlabel=f'{"XYZ"[i]} / {unit}', ylabel=f'{"XYZ"[j]} / {unit}', title=f'{"XYZ"[i]}{"XYZ"[j]} 投影')
        ax.set_aspect('equal', adjustable='box')
    fig.suptitle('拇指指尖三平面投影：点云空白保留，不用凸包填满', fontsize=13)
    save(out, fig, 'thumb_tip_xyz_projections')


def plot_chains(out, nodes, color, unit):
    fig = plt.figure(figsize=(10, 8), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*nodes[:, 4].T, c=color, cmap='viridis', s=4, alpha=.2, rasterized=True)
    # Select actual poses at q3 percentiles; no interpolation between skeletons.
    representatives = []
    for quantile in [.02, .25, .5, .75, .98]:
        i = int(np.argmin(abs(color-np.quantile(color, quantile))))
        representatives.append(nodes[i])
        ax.plot(*nodes[i].T, marker='o', ms=3, lw=1.5, label=f'q3={color[i]:.1f}°')
    ax.scatter([0], [0], [0], c='black', marker='s', s=55, label='Wrist node 0')
    equal3(ax, np.vstack(representatives))
    ax.view_init(23, -60)
    ax.set(xlabel=f'X / {unit}', ylabel=f'Y / {unit}', zlabel=f'Z / {unit}',
           title='拇指实际样本骨架：腕 → 掌骨 → 近节 → 远节 → 指尖')
    ax.legend(loc='upper left')
    save(out, fig, 'thumb_tip_xyz_with_skeleton')


def report(out, s):
    bounds = '\n'.join(f"| {a} | {r['minimum']:.6f} | {r['maximum']:.6f} | {r['span']:.6f} |"
                       for a, r in s['XYZ_bounds_in_display_units'].items())
    text = f"""# 拇指指尖 XYZ 样本工作空间

与当前混合模型正式训练点对应的{s['formal_points']}个指尖位置，来自指定批次`result/Glove Skeleton Data`的`data`表。全部{s['all_points']}个骨架文件有效，另有{s['preuniform_points']}个预采点保存在CSV，主图只画正式full点。

![三维点云与投影](thumb_tip_xyz_workspace.png)

![三平面投影](thumb_tip_xyz_projections.png)

![骨架与指尖](thumb_tip_xyz_with_skeleton.png)

## 坐标和节点

本批25节点布局按仓库`minimal_example_dual_stream_commented.py`解释：node 0是Wrist，node 4是Thumb Tip，176维向量按`1+7*node`起始，先XYZ再qx/qy/qz/qw。指尖原始位置切片为`[29:32]`，腕部位置为`[1:4]`，腕部四元数为`[4:8]`。

每点按`p_tip_wrist=R_wrist.T@(p_tip_stream-p_wrist_stream)`移除腕部平移和旋转，绘制腕部自身的XYZ方向；不额外把某个轴命名为解剖学内收/前后方向。四元数先归一化。依据全体数据中子节点位移在父节点姿态坐标内近似常量，检查本批位置是共同坐标中的累积节点位置，而非父子局部偏移；各骨段检验值见`summary.json`。

结果文件是记录窗口的均值。使用均值位置和归一化均值四元数作坐标变换，是腕部局部均值位置的近似，不声称与逐帧先变换后平均完全相同。

## 长度与范围

图中单位：`{s['length_unit']}`，显示倍率{s['display_scale']}。发送端SDK的`unitScale`没有记录在当前仓库，默认保留原始长度数值，未直接标成mm。MANUS官方说明单位及world/local输出取决于SDK初始化配置，详见[SDK坐标与单位设置]({DOCS['coordinate_and_unit_configuration']})。若确认发送端unitScale=1且原始单位为m，可用`--length-scale 1000 --unit-label mm`重新生成毫米图。

| 轴 | 最小值 | 最大值 | 跨度 |
| --- | --- | --- | --- |
{bounds}

轴范围不能任意组合为可达的长方体。图中保留点云的分支和空缺，没有以凸包补满或计算物理可达体积。

## 与拟合模型的关系

当前模型是`m=f(q)`，长屈/长伸一次，短伸/内收/外展/对掌二次。本图逐point_id配对该模型训练所用的q与对应骨架指尖位置，CSV还保存这些q算出的六路模型命令；没有将模型命令施加到设备再测量XYZ。

这份骨架由MANUS估计。官方将raw skeleton定义为手套数据施加于标准MANUS手模型后的输出，详见[骨架流说明]({DOCS['raw_skeleton_semantics']})。当前仓库没有该机器手拇指的连杆尺寸/正运动学模型，也没有当前新映射执行后的指尖跟踪数据，因此本图不能直接视为机器人实物采用新映射后的可达包络。若需该包络，应加入机器人正运动学或执行后的指尖位置测量。

## 复现

在工作空间根目录运行：`python3 src/yhand_mapping/scripts/plot_left_thumb_xyz_workspace.py`。`--refresh-cache`重读骨架Excel；`--output-dir`可另存图。图同时输出PNG与PDF；`thumb_tip_xyz_points.csv`含全部原始单位XYZ、q和模型命令；`summary.json`记录模型和数据哈希、坐标、单位及范围。
"""
    (out/'README.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
