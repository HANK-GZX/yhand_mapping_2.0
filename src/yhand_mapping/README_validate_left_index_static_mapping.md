# 左手食指静态映射重复扫描

当前电机编号见 [新腕带电机对应表](README_motor_mapping.md)。模型轴顺序按肌肉保留，不能按新编号排序。

脚本：`scripts/validate_left_index_static_mapping.py`

实验固定使用左手食指深屈电机 `motor 15`。默认执行 10 轮扫描，每轮只回零一次，
随后依次运动到：

```text
-10, -20, ..., -200
```

单点控制复用 `data_collection_main.py` 的 `_execute_point()`，因此拮抗电机放线、
回卷、到位判断、小电流张紧和结果窗口采样均与正常数采保持一致。

## 运行

运行前需要：

- 编译并 source ROS2 工作空间；
- 左手 `Y_Hand_Node` 已启动；
- `/glove_joint_data` 已发布 48D 数据；
- 没有其他程序向 `/yhand_left_cmd` 发布命令。

执行：

```bash
source install/setup.bash
python3 -u src/yhand_mapping/scripts/validate_left_index_static_mapping.py
```

脚本显示运动范围后，需要输入大写 `RUN` 才会开始。

小规模联调：

```bash
python3 -u src/yhand_mapping/scripts/validate_left_index_static_mapping.py \
  --positions 3 \
  --first-target -10 \
  --last-target -30 \
  --sweeps 2
```

## 输出

默认输出目录：

```text
scripts/motor_test_logs/static_mapping_validation/<timestamp>_left_index_deep_sweeps/
```

- `result_window_samples.csv`：每个结果窗口的逐帧食指 `q` 和相关电机 `pos/vel/tor`。
- `trials.csv`：每轮每个目标位置的均值、标准差、极差和到位状态。
- `repeatability_summary.csv`：同一目标位置在 10 轮扫描之间的 `q` 标准差。
- `stability_summary.csv`：单次结果窗口内 `m/q` 的最大值减最小值。
- `summary.json`：全局最差重复标准差和窗口极差。
- `metadata.json`：电机映射、目标点和本次运行参数。

默认只有回零成功、到位成功且结果窗口至少包含 20 个有限配对样本的记录才进入
重复性汇总。timeout 数据仍保存在原始 CSV 中。
