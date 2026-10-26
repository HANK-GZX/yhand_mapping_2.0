#!/usr/bin/env python3
"""Plot the current thumb model over recorded q support, without ROS/hardware.

These are joint-angle/motor-coordinate projections, not a fingertip XYZ workspace
or proof of physical reachability. No Cartesian product of independent q bounds
is sampled: only recorded full-combo q vectors are evaluated by FingerMapper.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/yhand_thumb_workspace_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent/'q_2_m linear mapping'))
from left_little_q_to_motor import FingerMapper

PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_RUN = PACKAGE/'scripts/motor_test_logs/data_collection/20261019_170304_left_collection'
MOTOR_LABELS = ['M4 长屈', 'M6 短伸', 'M7 长伸', 'M8 内收', 'M14 对掌', 'M27 外展']
PAIRS = [(i, j) for i in range(4) for j in range(i+1, 4)]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--model-file', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    run = args.collection_dir.resolve()
    out = args.output_dir or run/'analysis/thumb_model_workspace'
    out.mkdir(parents=True, exist_ok=True)
    analysis = run/'analysis/thumb_linearity'
    cache = analysis/'data_cache.npz'
    with (analysis/'point_log_snapshot.csv').open(encoding='utf-8-sig') as f:
        meta = list(csv.DictReader(f))
    with (run/'left_hand/thumb/point_log.csv').open(encoding='utf-8-sig') as f:
        current_meta = list(csv.DictReader(f))
    if meta != current_meta:
        raise ValueError('缓存日志与当前采样日志不同，需先刷新thumb_linearity分析缓存')
    mapper = FingerMapper(args.model_file, finger='thumb')
    source_hashes = mapper.config.get('source_sha256', {})
    if source_hashes.get('cache_sha256') and sha256(cache) != source_hashes['cache_sha256']:
        raise ValueError('缓存与当前模型训练缓存哈希不一致')
    data = np.load(cache)
    if not np.array_equal(data['point_ids'], [int(r['point_id']) for r in meta]):
        raise ValueError('缓存point_id不匹配')
    formal = np.array([r['active_mode']=='full' for r in meta])
    q_abs = data['glove'][formal, 8:12]
    q = q_abs-np.asarray(mapper.q_reference)
    m = data['feedback'][formal][:, np.array(mapper.motor_ids)-1, 0]
    predicted = np.array([[result[f'M{mid}'] for mid in mapper.motor_ids]
                          for row in q for result in [mapper.predict(row)]])
    if not np.isfinite(q).all() or not np.isfinite(predicted).all():
        raise ValueError('输入或模型输出不是有限数值')
    selected = mapper.effective_models()
    offsets = np.array([mapper.config['motors'][str(mid)]['fits'][selected[mid]]
                        ['source_fit']['prediction_at_reference_motor_deg'] for mid in mapper.motor_ids])
    # Put feedback and model predictions on the same translated coordinate basis.
    # This constant shift is visualization alignment, not a measured motor-zero pose.
    aligned_feedback = m-offsets
    rows = []
    for i, meta_i in enumerate(np.flatnonzero(formal)):
        row = {'point_id':int(data['point_ids'][meta_i]), 'combo':meta[meta_i]['combo_id']}
        row.update({f'q{j}_absolute_deg':float(q_abs[i, j]) for j in range(4)})
        row.update({f'delta_q{j}_deg':float(q[i, j]) for j in range(4)})
        for j, mid in enumerate(mapper.motor_ids):
            row.update({f'M{mid}_current_model_output_deg':float(predicted[i, j]),
                        f'M{mid}_recorded_feedback_deg':float(m[i, j]),
                        f'M{mid}_feedback_shifted_for_display_deg':float(aligned_feedback[i, j])})
        rows.append(row)
    write_csv(out/'workspace_points.csv', rows)
    ranges = []
    for j in range(4):
        ranges.append({'coordinate':f'delta_q{j}', 'unit':'joint_deg',
                       'minimum':float(q[:, j].min()), 'maximum':float(q[:, j].max()),
                       'zero_within_marginal_range':bool(q[:, j].min()<=0<=q[:, j].max())})
    for j, mid in enumerate(mapper.motor_ids):
        ranges.append({'coordinate':f'M{mid}_model_output', 'unit':'motor_deg',
                       'minimum':float(predicted[:, j].min()), 'maximum':float(predicted[:, j].max()),
                       'zero_within_marginal_range':bool(predicted[:, j].min()<=0<=predicted[:, j].max())})
    write_csv(out/'workspace_ranges.csv', ranges)
    summary = {'source_collection':str(run), 'model_file':str(mapper.model_path),
               'model_sha256':sha256(mapper.model_path), 'cache_sha256':sha256(cache),
               'formal_points':len(q), 'q_reference':list(mapper.q_reference),
               'motor_ids':list(mapper.motor_ids), 'effective_models':selected,
               'feedback_display_offsets':offsets.tolist(), 'ranges':ranges,
               'controller_enforces_q_bounds':False,
               'controller_enforces_thumb_motor_bounds':bool(mapper.config.get('motor_position_limits_deg')),
               'teleop_cmc_gain':[1.0,1.0],
               'scope':'Current mixed-model outputs at recorded 4D q vectors; finite coverage of data, not physical XYZ reachability or enforced limits',
               'zero_note':'Black stars mark delta_q=0 / motor_target=0. Reference is from model metadata; some recorded marginal q bounds may not include it.',
               'feedback_note':'Gray feedback points are translated by source f(reference) to share model output coordinates; this is not motor-zero calibration.'}
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    plot(out, q, predicted, aligned_feedback)
    report(out, summary)
    print(json.dumps({'output_dir':str(out), 'formal_points':len(q), 'ranges':ranges}, ensure_ascii=False, indent=2))
    return 0


def plot(out, q, predicted, feedback):
    plt.rcParams.update({'font.family':'Noto Sans CJK JP', 'axes.unicode_minus':False,
                         'font.size':10, 'axes.grid':True, 'grid.alpha':.2})
    def save(fig, name):
        fig.savefig(out/f'{name}.png', dpi=170, bbox_inches='tight')
        fig.savefig(out/f'{name}.pdf', bbox_inches='tight')
        plt.close(fig)
    def zero3(ax):
        ax.scatter([0], [0], [0], marker='*', s=150, c='black', depthshade=False)
    fig = plt.figure(figsize=(16, 13), layout='constrained')
    panels = [(q, [0, 2, 3], q[:, 1], 'Δq1 / °', '输入覆盖：Δq0 / Δq2 / Δq3',
               [f'Δq{i} / °' for i in [0, 2, 3]]),
              (q, [1, 2, 3], q[:, 0], 'Δq0 / °', '输入覆盖：Δq1 / Δq2 / Δq3',
               [f'Δq{i} / °' for i in [1, 2, 3]]),
              (predicted, [0, 1, 2], q[:, 3], 'Δq3 / °', '模型输出：长屈 / 短伸 / 长伸',
               [MOTOR_LABELS[i]+' / °' for i in [0, 1, 2]]),
              (predicted, [3, 5, 4], q[:, 0], 'Δq0 / °', '模型输出：内收 / 外展 / 对掌',
               [MOTOR_LABELS[i]+' / °' for i in [3, 5, 4]])]
    for i, (values, indices, color, color_label, title, labels) in enumerate(panels):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        sc = ax.scatter(*values[:, indices].T, c=color, cmap='viridis', s=3, alpha=.5, rasterized=True)
        zero3(ax)
        ax.set(xlabel=labels[0], ylabel=labels[1], zlabel=labels[2], title=title)
        ax.view_init(elev=23, azim=-55)
        ax.tick_params(labelsize=8)
        fig.colorbar(sc, ax=ax, shrink=.65, pad=.08, label=color_label)
    fig.suptitle(f'当前拇指混合模型的样本覆盖（n={len(q):,}）\n'
                 'M4/M7一次；M6/M8/M14/M27完整二次。★ = 控制零输入/零输出', fontsize=14)
    save(fig, 'thumb_model_workspace_3d')
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout='constrained')
    for ax, (i, j) in zip(axes.flat, PAIRS):
        ax.scatter(q[:, i], q[:, j], s=3, alpha=.25, c='#2878b5', rasterized=True)
        ax.scatter([0], [0], marker='*', s=130, c='black', label='控制展开点')
        ax.set(xlabel=f'Δq{i} / °', ylabel=f'Δq{j} / °', title=f'Δq{i} – Δq{j}')
    axes[0, 0].legend()
    fig.suptitle('四q的六个二维投影：点云为实际采集角度组合，单轴最小/最大值不能任意组合', fontsize=13)
    save(fig, 'thumb_q_workspace_projections')
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout='constrained')
    for ax, (i, j) in zip(axes.flat, [(0, 1), (0, 2), (1, 2), (3, 5), (3, 4), (5, 4)]):
        ax.scatter(feedback[:, i], feedback[:, j], s=3, alpha=.18, c='#777777',
                   label='反馈经同常量平移', rasterized=True)
        ax.scatter(predicted[:, i], predicted[:, j], s=3, alpha=.3, c='#d95f02',
                   label='当前模型输出', rasterized=True)
        ax.scatter([0], [0], marker='*', s=130, c='black', label='控制零目标')
        ax.set(xlabel=MOTOR_LABELS[i]+' / °', ylabel=MOTOR_LABELS[j]+' / °')
    axes[0, 0].legend(markerscale=2)
    fig.suptitle('六路m输出投影：模型与反馈使用相同平移坐标；★不代表已实测零位', fontsize=13)
    save(fig, 'thumb_motor_workspace_projections')


def report(out, s):
    rows = ['| 坐标 | 样本上的最小值/° | 样本上的最大值/° |', '| --- | --- | --- |']
    rows += [f"| {r['coordinate']} | {r['minimum']:.2f} | {r['maximum']:.2f} |" for r in s['ranges']]
    text = f"""# 当前拇指拟合模型工作空间图

模型：`{s['model_file']}`。使用{s['formal_points']}个正式full点的真实四维q组合，调用当前控制器同一`FingerMapper.predict()`计算六路输出。M4/M7一次，M6/M8/M14/M27完整二次。单位均为度；q是关节角，m是电机位置角。

![三维投影](thumb_model_workspace_3d.png)

![四q二维覆盖](thumb_q_workspace_projections.png)

![六电机二维覆盖](thumb_motor_workspace_projections.png)

## 如何解释

- 这是模型在已有数据支持区域的输入/输出覆盖。四维q与六维m分别用多组投影显示；三维图没有把4个q或6台电机降成3个自由度。
- 点云的空白与分支保留。不把每个q的独立最小/最大值做笛卡尔积，不生成未经采集支持的联合姿态，也不以凸包填满空缺。
- 图中展示物理q对应的模型输出，未施加操作员CMC增益或发送窗口。当前控制器拇指输出窗口启用状态为`{s['controller_enforces_thumb_motor_bounds']}`，遥操CMC增益为`{s['teleop_cmc_gain']}`；这些点云边界不是代码强制边界。
- 黑色星号是相对q全0及六路m目标全0。模型参考为原始q={s['q_reference']}°，具体标定来源见模型配置。零点与采集点云之间仍有未被联合采样覆盖的区域。
- 橙色是当前零位对齐后的模型输出。灰色反馈也减去对应原始模型在展开参考的预测值，使二者处于同一平移坐标；这只是可视化对齐，不是另一次物理标定。CSV同时保留原始反馈，避免误认坐标。
- 指尖XYZ范围需要运动学或指尖位置数据与模型的进一步对应，此处没有将关节/电机投影称作指尖物理可达域。

## 覆盖范围

{chr(10).join(rows)}

范围仅对本次真实q样本成立，不代表所有区间组合都可达。训练覆盖也不等于模型输出在真实设备上经过验证。

## 文件

每张图同时导出PNG与PDF。`workspace_points.csv`保留每点q、当前模型输出、原始反馈、平移后反馈；`workspace_ranges.csv`与`summary.json`记录范围及模型/缓存哈希。

复现：在工作空间根目录运行`python3 src/yhand_mapping/scripts/plot_left_thumb_model_workspace.py`。
"""
    (out/'README.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
