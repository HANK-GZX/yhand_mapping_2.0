# right_24_motor_position_replay_from_xlsx 使用说明（含回放记录模式）

当前电机编号见 [新腕带电机对应表](README_motor_mapping.md)。模型轴顺序按肌肉保留，不能按新编号排序。

对应脚本：

- `scripts/right_24_motor_position_replay_from_xlsx.py`

---

## 1. 功能概述

该脚本有两种模式：

1. 普通回放模式（默认）  
   读取 `xlsx` 位姿文件并持续发布目标电机位置，用于 24 电机位姿保持测试。

2. 回放记录模式（`--verify-mode`）  
   逐个读取 `motor_feedback` 文件（支持 `xlsx/csv`）作为 cmd 输入并发布到手，订阅实时手套 raw skeleton，按同名保存为 `xlsx`。  
   本模式不做误差统计与对比，仅做“发布 + 采样记录”。

---

## 2. 前置条件

1. 已 source ROS2 工作空间。
2. 手驱动节点在线（`/yhand_right_feedback`、`/yhand_right_cmd` 或左手对应话题）。
3. 手套 raw skeleton 话题在线（默认 `/glove_joint_data_raw_skeleton`）。

---

## 3. 普通回放模式（原模式）

```bash
python3 src/yhand_mapping/scripts/right_24_motor_position_replay_from_xlsx.py \
  --hand right \
  --source-motor-map legacy \
  --folder src/yhand_mapping/scripts/2026.5.18 \
  --cmd-topic /yhand_right_cmd \
  --feedback-topic /yhand_right_feedback
```

说明：

1. 当前实现是“单文件保持”模式：即使目录有多个文件，也只使用第一个目标并持续发布。
2. `--loop`、`--frame-interval-sec` 在该模式下保留但不生效（脚本会提示）。

---

## 4. 回放记录模式（新增）

### 4.1 你当前数据路径的推荐命令

```bash
python3 src/yhand_mapping/scripts/right_24_motor_position_replay_from_xlsx.py \
  --verify-mode \
  --hand right \
  --source-motor-map legacy \
  --verify-motor-feedback-dir /home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/scripts/motor_test_logs/data_collection/20260529_095459_right_collection/right_hand/csv_full_hand/result/motor_feedback \
  --verify-glove-skeleton-dir "/home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/scripts/motor_test_logs/data_collection/20260529_095459_right_collection/right_hand/csv_full_hand/result/Glove Skeleton Data" \
  --cmd-topic /yhand_right_cmd \
  --feedback-topic /yhand_right_feedback \
  --glove-raw-skeleton-topic /glove_joint_data_raw_skeleton
```

### 4.2 输入规则

1. `motor_feedback` 作为 cmd 源：  
   - `xlsx`：读取电机位姿行（A2~A29）  
   - `csv`：读取第一列前 28 个有效数字作为 28 电机位置
2. 若提供 `Glove Skeleton Data` 目录，仅用于文件名 stem 对齐筛选（例如 `00001.xlsx` 对 `00001.xlsx/csv`），不做数值比较。

### 4.3 输出结果

默认输出目录：

- `<motor_feedback目录上级>/replay_record_YYYYmmdd_HHMMSS/`

目录内包含：

1. `record_manifest.csv`  
   每个文件一行，含 `seq`、`file`、`motor_file`、`sample_count`、`recorded_glove_xlsx`。
2. `record_summary.json`  
   全局记录摘要（record-only）。
3. `recorded_glove_skeleton/<stem>.xlsx`  
   每个点采样平均后的实测 176 维 glove skeleton（单行 xlsx，与 stem 对应）。

---

## 5. 关键参数（校验模式）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--verify-mode` | `false` | 开启回放记录模式（record-only） |
| `--verify-motor-feedback-dir` | `""` | motor_feedback 目录；空则优先 `<folder>/motor_feedback` |
| `--verify-glove-skeleton-dir` | `""` | 可选，仅用于同名筛选；空则用 motor 目录同级 `Glove Skeleton Data` |
| `--glove-raw-skeleton-topic` | `/glove_joint_data_raw_skeleton` | 实时手套骨架话题 |
| `--ready-timeout-sec` | `20.0` | 启动时等待 glove_raw 与 cmd 订阅者就绪超时 |
| `--verify-glove-sample-count` | `5` | 每个点采样多少帧 glove_raw 再平均 |
| `--verify-glove-sample-timeout-sec` | `1.5` | 单点采样等待超时 |
| `--verify-post-reach-wait-sec` | `0.2` | 每个点发布后额外等待，再采样（实际会至少保持 30 个发布周期） |

---

## 6. 说明

1. 当前 `--verify-mode` 已改为记录模式，不输出误差统计。
2. 若 `start_driver_node:=false` 且没有其他节点订阅 `cmd_topic`，脚本会在 ready 阶段超时退出。

以上 2026 年 5 月的数据示例使用 `legacy`。新腕带下记录的文件使用默认 `current`。
