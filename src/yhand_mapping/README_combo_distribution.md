# 按 combo 分析采集结果

脚本：`scripts/plot_collection_combo_distribution.py`。依赖 `numpy`、`matplotlib`、`openpyxl`，不依赖 ROS，也不需要 pandas。

在仓库根目录运行中指、无名指新批次（直接传入手目录）：

```bash
python3 src/yhand_mapping/scripts/plot_collection_combo_distribution.py \
  src/yhand_mapping/scripts/motor_test_logs/data_collection/20261013_150832_left_collection/left_hand \
  --fingers middle ring
```

脚本也支持仅传入某个手指的 `result` 目录，例如旧批次：

```bash
python3 src/yhand_mapping/scripts/plot_collection_combo_distribution.py \
  src/yhand_mapping/scripts/motor_test_logs/data_collection/20261012_135157_left_collection/left_hand/index/result
```

默认输出到输入目录下的 `combo_distribution/`：

输入为手目录时，自动发现各手指的 `result/glove_data` 与 `result/motor_feedback`，分别输出到 `combo_distribution/<finger>/`。每个手指使用自己的 `point_log.csv`，点号不会跨手指合并。默认处理发现的全部手指，可用 `--fingers middle ring` 筛选。手目录输出另包含 `collection_finger_summary.csv`（每指点数/超时比例）、`collection_combo_summary.csv`（各 combo 统计）及 `collection_report.json`（结果索引）。以下文件位于各手指输出目录中：

- `combo_overview.png`：每个 combo 的两个主轴电机反馈位置，蓝色圆点表示 `reached=1, timeout=0`，红色叉号表示其他点。
- `combo_overview_zoom.png`：各 combo 的 P01–P99 主体区间放大图，坐标范围按各组分别确定，标题标注视窗外点数；完整点仍保留在总览和 CSV 中。
- `<combo>/motor_position_distribution.png`：电机位置按采集顺序着色、二维密度、按四个手套关节角着色。
- `<combo>/motor_feedback_histograms.png`：该手指全部电机的 `pos / vel / tor` 直方图，按采集状态分开显示。
- `<combo>/glove_distribution.png`：该手指四个关节角的直方图，以及随点号变化的散点图。
- `combo_summary.csv`：combo 点数、到达/超时计数、主轴电机编号、active_mode 计数。
- `field_statistics.csv`：该手指每个电机全部五项反馈和四个手套关节角的有效/缺失数量、最小值、P05、中位数、P95、最大值、均值和总体标准差。
- `points.csv`：按点号排列的该手指关节角、电机反馈与采集标记，方便后续分析。
- `load_report.json`：输入路径、手套列索引、均值窗口、文件配对情况、跳过的损坏文件、combo 统计，以及本指四个角度/完整 48 维手套向量的重复数量与最长连续重复段。

每次运行会覆盖所选 combo 对应的输出文件；对比不同筛选结果时请指定不同的 `--output-dir`。

## 读取与统计口径

只读取 `result/glove_data/*.xlsx` 与 `result/motor_feedback/*.xlsx` 的 `data` 工作表。根据文件名中的点号，与上一级 `point_log.csv` 关联 combo 和采集状态；不读取 `motor_cmd`、骨架或 `process` 数据，不从电机位置推断 combo。

手套文件必须为一行 48 列；电机反馈第一行为 `[window, sample_count]`，后续 28 行按 1-based 电机编号排列，五列依次为 `pos, vel, tor, kp, kd`。空数值计作缺失，不补零。无法读取或格式错误的配对会记录在 `load_report.json`；没有合法数据时直接报错。

左手手套角位于 48 维数组的 `[8:28]`，右手位于 `[28:48]`；按 `thumb/index/middle/ring/little` 顺序，每指四列。左手食指使用 Excel 第 13–16 列，中指使用第 17–20 列，无名指使用第 21–24 列，图中均记为该指的 `q0–q3`。保持原始数据单位。电机编号和 combo 主轴使用仓库 `hand_motor_map.py` 的映射，适用于这批采集所用的新腕带接线；历史不同接线的数据需要先确认映射。

**每个 result 文件只有一个采样窗口均值。图和标准差描述的是 combo 内不同采集点之间的分布，不能表示单个点采样窗口内的噪声或稳定性。** 默认保留全部点，包括超时点。直方图的纵轴是点数，各 combo 共用分箱和对应电机的位置范围；关节角颜色也共用范围，便于比较。

批量处理时，各手指分别确定比较范围。电机位置平面图的共享范围只使用该电机作为 combo 的 x/y 主轴时的数值；直方图仍展示该电机在本指全部 combo 中的分布。放大图使用每组自己的 P01–P99 范围并加少量边距，适合看点云主体，不能直接用其坐标尺度比较组间跨度。手套重复统计使用有限数值的精确相等比较，不自动删除重复数据，也不把重复直接判定为设备故障。

## 筛选与其他手指

只分析一个 combo，并输出到独立目录：

```bash
python3 src/yhand_mapping/scripts/plot_collection_combo_distribution.py \
  src/yhand_mapping/scripts/motor_test_logs/data_collection/20261012_135157_left_collection/left_hand/index/result \
  --combo index_deep_in --output-dir /tmp/index_deep_in_distribution
```

只看日志中到达且未超时的点，或者增加 `kp/kd` 的直方图：

```bash
# 在上面的命令后追加对应参数
--reached-only --output-dir /tmp/index_reached_distribution
--motor-fields pos vel tor kp kd
```

可用 `--bins 60` 修改分箱数量，`--dpi 200` 修改图片清晰度。其他手指直接替换 `result` 路径；手别和手指通常从路径自动识别，自定义目录可指定 `--hand left --finger index --point-log <csv路径>`。拇指的电机平面图使用 combo 的 x/y 主轴，对掌电机反馈包含在全部本指电机的直方图和统计表中。
