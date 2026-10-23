#!/usr/bin/env python3
"""Compare thumb feedback, fitted predictions and saved boundary in motor space.

Boundary recordings contain only motor feedback, so no fingertip XYZ boundary
is inferred. Show current rebased commands and original-coordinate fits in
separate figures, retaining the original saved coordinate values in each.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/yhand_thumb_boundary_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from left_little_q_to_motor import FingerMapper
from left_little_q_to_motor_test import prepare_target

PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_RUN = PACKAGE/'scripts/motor_test_logs/data_collection/20261019_170304_left_collection'
DEFAULT_BOUNDARY = PACKAGE/'scripts/motor_test_logs/workspace_boundaries/left'
IDS = [4, 6, 7, 8, 14, 27]
NAMES = ['M4 长屈', 'M6 短伸', 'M7 长伸', 'M8 内收', 'M14 对掌', 'M27 外展']
GROUPS = [(role, direction) for direction in ['in', 'out'] for role in ['flex_long', 'ext_short', 'ext_long']]
ROLE_LABELS = {'flex_long':'长屈 M4', 'ext_short':'短伸 M6', 'ext_long':'长伸 M7'}
KINDS = [('collection_feedback', '采集反馈'), ('source_fit', '零位对齐前拟合'),
         ('current_model_command', '当前模型命令'), ('boundary_raw_feedback', 'boundary原始反馈')]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def inside_envelope(motors, fit):
    axes = fit['axes']
    xi, yi, zi = [IDS.index(axes[k]) for k in ['x_motor', 'y_motor', 'z_motor']]
    xg, zg = np.array(fit['x_grid']), np.array(fit['z_grid'])
    if not (np.diff(xg)>0).all() or not (np.diff(zg)>0).all():
        raise ValueError('boundary网格必须严格递增')
    low, high = np.array(fit['y_min']), np.array(fit['y_max'])
    if low.shape!=(len(xg),len(zg)) or high.shape!=low.shape:
        raise ValueError('boundary曲面形状与x/z网格不匹配')
    locations = motors[:, [xi, zi]].copy()
    # Boundary CSV rounds each value to 9 decimals. Repeated extrema can land
    # 1e-10 outside the full-precision JSON grid; correct only this rounding.
    for j, grid in enumerate([xg, zg]):
        near = (locations[:, j]>=grid[0]-1e-8) & (locations[:, j]<=grid[-1]+1e-8)
        locations[near, j] = np.clip(locations[near, j], grid[0], grid[-1])
    lo = RegularGridInterpolator((xg,zg), low, bounds_error=False, fill_value=np.nan)(locations)
    hi = RegularGridInterpolator((xg,zg), high, bounds_error=False, fill_value=np.nan)(locations)
    # Raw CSV feedback is rounded to 9 decimals; allow only numerical rounding.
    valid = np.isfinite(lo) & np.isfinite(hi)
    in_y = valid & (motors[:,yi]>=lo-1e-8) & (motors[:,yi]<=hi+1e-8)
    return in_y, valid, lo, hi


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--boundary-dir', type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument('--model-file', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    run, boundary_dir = args.collection_dir.resolve(), args.boundary_dir.resolve()
    out = args.output_dir or run/'analysis/thumb_boundary_comparison'
    out.mkdir(parents=True, exist_ok=True)
    model = FingerMapper(args.model_file, finger='thumb')
    analysis = run/'analysis/thumb_linearity'
    meta = read_csv(analysis/'point_log_snapshot.csv')
    if meta!=read_csv(run/'left_hand/thumb/point_log.csv'):
        raise ValueError('数据缓存与当前采样日志不一致')
    cache = analysis/'data_cache.npz'
    if digest(cache)!=model.config['source_sha256']['cache_sha256']:
        raise ValueError('模型和采样缓存来源不一致')
    with np.load(cache) as z:
        if not np.array_equal(z['point_ids'], [int(r['point_id']) for r in meta]):
            raise ValueError('缓存point_id不匹配')
        mask = np.array([r['active_mode']=='full' for r in meta])
        q = z['glove'][mask, 8:12]
        feedback = z['feedback'][mask][:, np.array(IDS)-1]
        ids = z['point_ids'][mask]
    labels = np.array([r['combo_id'] for r, keep in zip(meta, mask) if keep])
    raw_prediction = np.array([[p[f'M{mid}'] for mid in IDS] for row in q for p in [model.predict(row,relative=False)]])
    current = np.array([[p['positions'][mid] for mid in IDS] for row in q
                        for p in [prepare_target(model,row,relative=False)]])
    offsets = np.array([model.config['motors'][str(mid)]['fits'][name]['source_fit']['prediction_at_reference_motor_deg']
                        for mid,name in model.effective_models().items()])
    source = raw_prediction+offsets
    boundary_file = boundary_dir/'boundary_models.json'
    boundary = json.loads(boundary_file.read_text())
    if boundary['hand']!='left':
        raise ValueError('boundary必须为左手')
    raw_files = sorted((boundary_dir/'raw/thumb').glob('*.csv'))
    if not raw_files:
        raise ValueError('找不到拇指boundary原始反馈')
    raw_rows = [r for path in raw_files for r in read_csv(path)]
    raw = np.array([[float(r[f'pos_{mid}']) for mid in IDS] for r in raw_rows])
    raw_current = np.array([[float(r[f'tor_{mid}']) for mid in IDS] for r in raw_rows])
    for values in [raw, current, source, feedback[:,:,0]]:
        if not np.isfinite(values).all():
            raise ValueError('比较数据包含非有限数值')
    sets = {'collection_feedback':feedback[:,:,0], 'source_fit':source, 'current_model_command':current,
            'boundary_raw_feedback':raw}
    fits, group_masks, inclusion, point_flags = {}, {}, [], []
    for role, direction in GROUPS:
        prefix = f'thumb_{role}_{direction}_opp_'
        fit = boundary['combos'][prefix+'off']
        other = boundary['combos'][prefix+'on']
        if fit.get('status')!='ok' or fit.get('dims')!=3:
            raise ValueError(f'{prefix}没有可用三维boundary')
        for key in ['axes','x_grid','z_grid','y_min','y_max']:
            if fit[key]!=other[key]:
                raise ValueError('on/off包络不相同，不能合并绘制')
        if fit['point_count']!=len(raw):
            raise ValueError('原始boundary采样点数与当前拟合包络不一致，不能混合历史段')
        key = f'{role}_{direction}'
        fits[key] = fit
        group_masks[key] = np.array([c.startswith(prefix) for c in labels])
        for kind, _ in KINDS:
            points = sets[kind] if kind=='boundary_raw_feedback' else sets[kind][group_masks[key]]
            inside, in_grid, low, high = inside_envelope(points, fit)
            inclusion.append({'group':key, 'dataset':kind, 'n':len(points),
                              'inside_saved_envelope':int(inside.sum()),
                              'inside_percent':float(inside.mean()*100),
                              'outside_xz_grid':int((~in_grid).sum()),
                              'outside_y_within_xz_grid':int((in_grid & ~inside).sum())})
            if kind!='boundary_raw_feedback':
                for i, pid in enumerate(ids[group_masks[key]]):
                    point_flags.append({'point_id':int(pid), 'group':key, 'dataset':kind,
                                        'inside_saved_envelope':int(inside[i]), 'within_xz_grid':int(in_grid[i]),
                                        'interpolated_y_min':float(low[i]) if in_grid[i] else '',
                                        'interpolated_y_max':float(high[i]) if in_grid[i] else ''})
    write_csv(out/'saved_envelope_inclusion.csv', inclusion)
    write_csv(out/'point_inclusion.csv', point_flags)
    ranges = []
    for kind, title in KINDS:
        for j, mid in enumerate(IDS):
            ranges.append({'dataset':kind, 'label':title, 'motor':mid, 'n':len(sets[kind]),
                           'minimum_deg':float(sets[kind][:,j].min()), 'maximum_deg':float(sets[kind][:,j].max())})
    write_csv(out/'motor_ranges.csv', ranges)
    points = []
    for i, pid in enumerate(ids):
        row = {'point_id':int(pid), 'combo_id':labels[i]}
        for j, mid in enumerate(IDS):
            for kind in ['collection_feedback','source_fit','current_model_command']:
                row[f'M{mid}_{kind}_deg'] = float(sets[kind][i,j])
        points.append(row)
    write_csv(out/'collection_and_model_points.csv', points)
    summary = {'collection_dir':str(run), 'boundary_file':str(boundary_file),
               'boundary_sha256':digest(boundary_file), 'boundary_raw_files':[str(p) for p in raw_files],
               'boundary_raw_sha256':{p.name:digest(p) for p in raw_files},
               'model_file':str(model.model_path), 'model_sha256':digest(model.model_path),
               'collection_formal_points':len(q), 'boundary_points':len(raw),
               'coordinate':'Saved SDK motor position coordinates, degrees; no cross-session fitted registration',
               'source_fit_minus_current_command_offsets_deg':dict(zip(map(str,IDS),map(float,offsets))),
               'offset_note':'Reference predictions only. Current commands additionally include CMC gains, exterior policy and final command clipping; differences need not be constant.',
               'cmc_gain':[1.0,1.0],
               'collection_median_measured_currents_A':dict(zip(map(str,IDS),map(float,np.median(feedback[:,:,2],axis=0)))),
               'boundary_median_measured_currents_A':dict(zip(map(str,IDS),map(float,np.median(raw_current,axis=0)))),
               'inclusion':inclusion, 'ranges':ranges,
               'xyz_overlay_available':False, 'reason':'Boundary raw CSV contains motor pos/vel/tor only, no glove q or skeleton XYZ',
               'limitation':'Envelope inclusion is geometric membership of a historical fitted motor envelope; not a physical reachability/safety metric. Cross-session zero alignment is unverified and tension conditions differ.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    plot(out, sets, fits, group_masks, inclusion)
    report(out, summary)
    print(json.dumps({'output_dir':str(out), 'collection_points':len(q), 'boundary_points':len(raw),
                      'inclusion':inclusion},ensure_ascii=False,indent=2))
    return 0


def save(out, fig, name):
    fig.savefig(out/f'{name}.png',dpi=155,bbox_inches='tight')
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def plot(out, sets, fits, masks, inclusion):
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,
                         'font.size':9,'axes.grid':True,'grid.alpha':.18})
    def scatter(ax, data, indices, color, label, alpha, size=4):
        ax.scatter(*data[:,indices].T,c=color,s=size,alpha=alpha,label=label,rasterized=True)
    raw = sets['boundary_raw_feedback']
    raw_show = raw[np.linspace(0,len(raw)-1,min(1800,len(raw)),dtype=int)]
    for kind, name, title in [('current_model_command','thumb_current_model_collection_boundary',
                               '当前控制模型命令（Δq全0→m全0）'),
                              ('source_fit','thumb_source_fit_collection_boundary','原物理q对应的拟合（未施加CMC增益或发送窗口）')]:
        fig = plt.figure(figsize=(21,13),layout='constrained')
        for i, (role, direction) in enumerate(GROUPS):
            key = f'{role}_{direction}'
            fit, mask = fits[key], masks[key]
            axes = fit['axes']
            indices = [IDS.index(axes[k]) for k in ['y_motor','x_motor','z_motor']]
            ax = fig.add_subplot(2,3,i+1,projection='3d')
            scatter(ax,raw_show,indices,'#bbbbbb','boundary采集',.12,2)
            scatter(ax,sets['collection_feedback'][mask],indices,'#e17c05','本批实际反馈',.45,5)
            scatter(ax,sets[kind][mask],indices,'#2878b5','当前模型命令' if kind=='current_model_command' else '原坐标拟合',.55,5)
            yy,zz = np.meshgrid(fit['x_grid'],fit['z_grid'],indexing='ij')
            low,high = np.array(fit['y_min']),np.array(fit['y_max'])
            for surface in [low,high]:
                ax.plot_wireframe(surface,yy,zz,rstride=3,cstride=3,color='#555555',alpha=.5,linewidth=.6)
            # Matching limits between original and rebased plots expose the translation.
            allpoints = np.vstack([raw,*[sets[k][mask] for k in ['collection_feedback','source_fit','current_model_command']]])
            for setter, col, extra in [(ax.set_xlim,indices[0],np.r_[low.ravel(),high.ravel()]),
                                       (ax.set_ylim,indices[1],np.asarray(fit['x_grid'])),
                                       (ax.set_zlim,indices[2],np.asarray(fit['z_grid']))]:
                v = np.r_[allpoints[:,col],extra];span=max(np.ptp(v),1)
                setter(v.min()-.04*span,v.max()+.04*span)
            stat = next(r for r in inclusion if r['group']==key and r['dataset']==kind)
            ax.set(title=f'{ROLE_LABELS[role]} × {"内收" if direction=="in" else "外展"}（合并opp on/off）\n'
                         f'n={mask.sum()}；该模型点落入旧包络 {stat["inside_percent"]:.1f}%',
                   xlabel=f'侧摆 M{axes["y_motor"]} / °',ylabel=f'伸屈 M{axes["x_motor"]} / °',
                   zlabel=f'对掌 M{axes["z_motor"]} / °')
            ax.view_init(18,-58)
        handles,labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.035),ncol=3,markerscale=2)
        fig.suptitle(f'拇指：采集数据＋拟合结果＋boundary同图比较\n{title}；灰网格为旧boundary上下包络，坐标均为电机位置度',fontsize=14)
        save(out,fig,name)
    fig,axs = plt.subplots(2,3,figsize=(17,9),layout='constrained')
    colors = ['#e17c05','#009e73','#2878b5','#777777']
    for j,ax in enumerate(axs.flat):
        for i,((kind,label),color) in enumerate(zip(KINDS,colors)):
            lo,hi = sets[kind][:,j].min(),sets[kind][:,j].max()
            ax.plot([lo,hi],[i,i],color=color,lw=5,solid_capstyle='round')
            ax.text(lo,i+.12,f'{lo:.1f}',fontsize=8,ha='center')
            ax.text(hi,i+.12,f'{hi:.1f}',fontsize=8,ha='center')
        ax.set(yticks=range(4),yticklabels=[label for _,label in KINDS],xlabel='电机位置 / °',
               title=NAMES[j],ylim=(-.5,3.6))
    fig.suptitle('六电机范围对比：采集、原坐标拟合、当前命令、boundary（各轴范围不代表可任意组合）',fontsize=13)
    save(out,fig,'thumb_motor_range_comparison')


def report(out,s):
    table = ['| 伸屈×侧摆 | 本批反馈落入旧包络 | 原坐标拟合落入旧包络 | 当前命令落入旧包络 |',
             '| --- | --- | --- | --- |']
    for role,direction in GROUPS:
        key=f'{role}_{direction}'
        values=[next(r['inside_percent'] for r in s['inclusion'] if r['group']==key and r['dataset']==k)
                for k in ['collection_feedback','source_fit','current_model_command']]
        table.append(f'| {ROLE_LABELS[role]}×{direction} | '+' | '.join(f'{v:.2f}%' for v in values)+' |')
    txt=f"""# 拇指采集数据、当前模型和boundary叠加比较

本批{s['collection_formal_points']}个正式full结果均值点；boundary为`{s['boundary_file']}`和其9581个原始连续反馈帧。对掌on/off在旧boundary中是相同曲面，因此合并成6个三维视图，避免重复12次。数据与模型点按各自采集combo的伸屈肌和侧摆方向分组；每幅图的旧boundary点云均来自同一组整指人工采集。

## 当前控制模型直接输出与采集反馈、boundary

![当前输出叠加](thumb_current_model_collection_boundary.png)

橙色：本批实际电机反馈。蓝色：将本批相对q当作控制输入，经当前`prepare_target()`计算的命令，包含配置中的CMC增益`{s['cmc_gain']}`、外推策略和输出窗口。灰色点：旧boundary原始反馈（图中仅均匀抽样显示1800点，统计使用全部9581点）。灰色网格：boundary保存的上下包络。没有移动任意一批坐标以使点云重合。

## 排除零位平移后，检查拟合本身

![原坐标拟合叠加](thumb_source_fit_collection_boundary.png)

本图的蓝色使用同一混合模型在原物理q上的反馈预测，未施加操作员CMC增益或发送窗口；旧自由截距模型会补回参考点预测，新零位约束模型的参考点预测为0。这是拟合诊断，不是当前发送的命令。两张图对应子图使用同样的坐标显示范围。

六路模型在参考点的预测为：`{s['source_fit_minus_current_command_offsets_deg']}`°。当前命令还受CMC增益、外推处理和发送窗口影响，因此它与原物理q拟合结果的差异不必是常量；具体模型版本以本目录JSON哈希为准。

![各电机范围](thumb_motor_range_comparison.png)

## 落入旧包络的统计

{chr(10).join(table)}

按每组`x_grid,z_grid`对侧摆上下界双线性插值。超出x/z网格的点明确计为包络外，不把边缘值外推。CSV分开保存“x/z网格外”和“网格内但侧摆超出上下界”的点数；`point_inclusion.csv`保存逐点结果。原始boundary反馈也按相同算法计算，见`summary.json`，用于检查网格解释。

这些比例仅表示落在历史拟合包络中的几何比例，**不是机械可达率、控制成功率或安全率**。本批数据本身也有相当部分不在旧包络中，不能把模型点与旧包络的差异全部归因于拟合。

两批张紧条件不同：本批六路实测电流中位数为`{s['collection_median_measured_currents_A']}` A；旧boundary为`{s['boundary_median_measured_currents_A']}` A。跨批次的物理零位一致性尚未验证；张紧条件、归零、历史与拟合包络近似均可能影响比较，本图不区分其因果贡献。旧包络也不是当前遥操的硬限位。

## 与指尖XYZ图的关系

boundary的CSV只有`pos_1..28/vel_1..28/tor_1..28`与时间，没有q、指尖XYZ或骨架。旧boundary JSON的三维轴是“伸屈电机、侧摆电机、对掌电机”，不是物理XYZ。因此可以直接叠加的是上述电机空间。

本批实际采集的骨架指尖XYZ仍见[XYZ采样图](../thumb_xyz_workspace/README.md)。当前缺少boundary同步骨架或机器手正运动学，不能把旧电机boundary直接画成指尖XYZ外壳，也不通过未经验证的逆回归补造这个外壳。

## 文件与复现

每张图均有PNG和PDF。`collection_and_model_points.csv`保存实际反馈、原坐标拟合、当前命令；`motor_ranges.csv`保存六路范围；`saved_envelope_inclusion.csv`、`point_inclusion.csv`保存包络统计；`summary.json`记录数据来源和哈希。

运行`python3 src/yhand_mapping/scripts/compare_left_thumb_collection_boundary.py`复现。可使用`--boundary-dir`选择另一组boundary。脚本仅读文件，不改变控制模型或边界文件。
"""
    (out/'README.md').write_text(txt,encoding='utf-8')


if __name__=='__main__':
    raise SystemExit(main())
