# Workspace Boundary 采集说明

当前仅保留**全电机同时张紧、按整根手指采集各组合**的方式。电机编号来自 [新腕带电机对应表](README_motor_mapping.md)。

## 启动

```bash
# 左手
ros2 launch yhand_mapping workspace_boundary_launch.py
# 右手
ros2 launch yhand_mapping workspace_boundary_launch.py target_hand:=right
```

Launch 读取 `config/left_hand_device_config.json` 或 `config/right_hand_device_config.json`，启动对应手的 `Y_Hand_Node`，随后打开独立终端运行 `scripts/workspace_boundary_collect.py`。无需指定采集模式。

左手话题：`/yhand_left_cmd`、`/yhand_left_feedback`；右手：`/yhand_right_cmd`、`/yhand_right_feedback`。

## 电流控制

四指所有电机持续纯电流张紧，菜单等待和录制期间均使用相同配置：

| 电机范围 | 电流 |
| --- | --- |
| 食指、中指深屈与浅屈，无名指、小指屈肌 | **−0.03 A** |
| 四指伸肌 | **−0.05 A** |
| 四指侧摆（内收、外展） | **−0.05 A** |
| 拇指全部六路 | 当前默认 **−0.04 A** |
| 腕部四路 | 当前默认 **−0.03 A** |

全部 `pos=vel=kp=kd=0`、`pretightenforce.flag=False`。选择手指只决定记录数据的归属，全手电流不会随手指选择改变。

四指电流常量在 `scripts/workspace_boundary_config.py`：`SIMULTANEOUS_FLEXOR_CURRENT`、`SIMULTANEOUS_EXTENSOR_CURRENT`、`SIMULTANEOUS_SIDE_CURRENT`。拇指和腕部读取 `DEFAULT_TENSION_CURRENT_BY_HAND`；由 `get_simultaneous_tension_currents()` 生成最终 28 路电流。

## 操作与组合

一级菜单：`1..5` 选择拇指、食指、中指、无名指、小指；`f` 拟合保存；`p` 刷新图；`x` 退出。

选择手指后：

1. 输入 `r` 或直接回车开始记录，人工掰动该手指覆盖可达区域。
2. 再次回车停止本段并保存原始 CSV。
3. 可重复 `r` 追加采样段。
4. `b` 返回手指选择；`f` 拟合保存全部本次样本；`x` 退出。
5. `o` 清空当前整根手指的内存样本和原始 `session_*.csv`，其他手指不受影响。

| 手指 | 每段样本投影到的组合 | 数量 |
| --- | --- | --- |
| 食指 | 深屈/浅屈/伸肌 × 内收/外展 | 6 |
| 中指 | 深屈/浅屈/伸肌 × 内收/外展 | 6 |
| 无名指 | 屈肌/伸肌 × 内收/外展 | 4 |
| 小指 | 屈肌/伸肌 × 内收/外展 | 4 |
| 拇指 | 短伸/长伸/长屈 × 内收/外展 × opp_off/opp_on | 12 |

每根手指的所有组合共用该手指采集的样本。拇指的 opp_off/opp_on 保留原来的控制模式标签，两者使用相同采样点，不分别录制。

**启动时不自动读取已有 CSV 或模型。** 本次进程内按手指累积样本，`f` 覆盖默认目录中的 `boundary_models.json`；未采集的组合标记为 `insufficient_points`。退出不会自动拟合，完成后需要按 `f`。

## 默认保存路径及数采使用

保存路径恢复为：

```text
src/yhand_mapping/scripts/motor_test_logs/workspace_boundaries/<hand>/
├── raw/<finger>/session_YYYYmmdd_HHMMSS_xxxxxx.csv
├── run_index.json
├── boundary_models.json
└── plots/panel_<finger>_YYYYmmdd_HHMMSS.png
```

已有同时张紧采集的数据从 `workspace_boundaries_simultaneous` 迁入此默认目录，旧默认目录的数据保存在 `backups/workspace-boundary-unified-20261010-*/previous_workspace_boundaries/`。

之前单独组合采集的图另行保留在 `<hand>/plots/per_combo_comparison/`，保持原文件名，便于与 `plots/` 下的同时张紧采集图对比。重新采集、清空本指数据及拟合不会覆盖该对比目录，原备份也仍保留。

`data_collection_main.py` 默认读取同一路径的 `boundary_models.json`，因此数采无需额外指定 `--boundary-path`。它在启动时加载模型，重新拟合后需要重启数采。

CSV 保存系统时间和全部 28 路 `pos_i/vel_i/tor_i` 反馈。`run_index.json` 保存每根手指的实际样本数、投影到每个二维组合的点数和 `combo_axis_motors`。

JSON 保留 `collection_mode="per_finger_simultaneous"`、`tension_current_scope="all_collection"` 和 28 路张紧电流。各四指组合的 `collection.mode="per_finger"`，记录所用电流和样本归属；组合 ID、坐标轴、网格格式与数采兼容。

## 拟合和显示

四指 20 个二维组合拟合 `y_min(x)`、`y_max(x)`，至少需要 20 个点；拇指 12 个三维组合标签拟合 `y_min(x,z)`、`y_max(x,z)`，至少需要 45 个点，共导出 32 个组合。

Launch 仍使用 `max_outward_expand=14`。各手指面板显示该手指所有组合，拟合时导出所有手指面板。人工采集点的包络用于数采生成目标，不保证包络内部所有点在自动控制下都可到达。

## 参数和排查

脚本支持 `--hand`、`--sample-hz`、`--max-outward-expand`、`--cmd-topic`、`--feedback-topic`、`--output-dir`、`--disable-plot`；launch 支持 `target_hand`。

默认目标采样率 200 Hz，实际速度受反馈频率限制。终端打印每段实际采样率。无反馈时检查连接和话题；`insufficient_points` 时补采对应手指后按 `f`；导入 `devices_pkg.msg` 失败时先 source ROS2 和工作空间环境。
