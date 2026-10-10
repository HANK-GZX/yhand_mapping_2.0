# data_collection_from_motor_feedback_csv 使用说明

当前电机编号见 [新腕带电机对应表](README_motor_mapping.md)。模型轴顺序按肌肉保留，不能按新编号排序。

本文档对应：

- `launch/data_collection_from_motor_feedback_csv_launch.py`
- `scripts/data_collection_from_motor_feedback_csv.py`
- `scripts/data_collection_main.py`

这条链路用于“按 five_finger 导出的 motor feedback CSV 回放目标点，并重新采集 glove / raw skeleton / motor feedback 数据”。

补充说明：

- 这里使用的原始 `motor feedback CSV`，最初是通过 `scripts/five_finger_delta_mapping_main.py` 摇操记录得到的。
- 如果需要追踪 CSV 的生成/落盘逻辑，可直接参考 `five_finger_delta_mapping_main.py` 里的 five_finger 记录流程，尤其是 `record_motor_feedback_csv`、`motor_feedback_csv_path` 以及 `motor_feedback_frames/` 相关逻辑。

---

## 1. 整体用途

输入是 five_finger mapping 记录出来的 `motor_feedback_frames/*.csv`。

这些 CSV 默认对应 `scripts/five_finger_delta_mapping_main.py` 摇操过程中记录下来的 motor feedback 帧；如果想确认表头、输出目录和记录时机，建议直接对照 five_finger 的 CSV 记录实现。

脚本会把 CSV 中每一行电机位置当作一个目标点：

1. 读取 CSV 行，得到本点每个受控电机的目标位置。
2. 发布一条 28 电机命令，让手运动到该目标。
3. 持续采样 glove、raw skeleton、motor feedback。
4. 判断所有受控电机是否连续到位。
5. 到位或超时后，切到收紧线电流模式。
6. 在稳定窗口内采样并求平均。
7. 为这个点保存 process/result xlsx，并追加 point log。

---

## 2. 输入 CSV

输入方式二选一：

- 旧版单 CSV：使用 `cmd_csv_path`。
- 按手指输入：使用任意 1～5 个 `<finger>_cmd_csv_path`。

每个路径都必须指向包含电机位置列的 CSV。

支持列名：

- `motor_<id>_pos`
- `m<id>_pos`

可选列：

- `frame_idx`

关键规则：

- 只识别 `1..28` 号电机列。
- 表头里出现的电机列会组成本次采集的受控电机集合。
- 每个被选中的数据行必须包含所有受控电机的有效值，否则该行跳过。
- `start_row`、`stride`、`max_points` 会先筛选 CSV 行，再开始执行。
- 通过 launch 启动时，默认 `random_cmd_publish:=true`，筛选后的点还会被随机重排。

随机重排规则：

- 默认 `min_cmd_frame_gap:=30`。
- 任意两个连续发布的 CSV 点，其 `frame_idx` 差值至少为 30。
- `cmd_random_seed:=0` 表示用当前时间自动生成随机种子。
- 如果无法满足最小 frame gap，会直接报错，需要降低 `min_cmd_frame_gap` 或减少筛选约束。

---

## 3. 推荐启动

右手最小示例：

```bash
cd /home/yhand-demo/Yhand_Mapping_2.0
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch yhand_mapping data_collection_from_motor_feedback_csv_launch.py \
  target_hand:=right \
  cmd_csv_path:=src/yhand_mapping/scripts/motor_test_logs/motor_feedback_frames/right_motor_feedback_xxx.csv
```

当前终端直接看采集日志：

```bash
ros2 launch yhand_mapping data_collection_from_motor_feedback_csv_launch.py \
  target_hand:=right \
  cmd_csv_path:=src/yhand_mapping/scripts/motor_test_logs/motor_feedback_frames/right_motor_feedback_xxx.csv \
  collect_in_terminal:=false
```

任意 1～5 根手指 CSV 连续采集，例如只采集食指和无名指：

```bash
ros2 launch yhand_mapping data_collection_from_motor_feedback_csv_launch.py \
  target_hand:=right \
  index_cmd_csv_path:=/abs/path/index.csv \
  ring_cmd_csv_path:=/abs/path/ring.csv
```

执行顺序不受命令行参数书写顺序影响，固定从下面的顺序中选择已提供的 CSV：

```text
thumb -> index -> middle -> ring -> little
```

相邻两个实际执行的 CSV 之间默认回 0。上面的示例实际执行为
`index -> 回0 -> ring`。提供手指路径时不要再传 `cmd_csv_path`。
如需关闭 CSV 之间的回零，可显式传 `zero_between_csv:=false`。

常用参数示例：

```bash
ros2 launch yhand_mapping data_collection_from_motor_feedback_csv_launch.py \
  target_hand:=right \
  cmd_csv_path:=/abs/path/right_motor_feedback.csv \
  start_row:=100 \
  stride:=2 \
  max_points:=500 \
  collection_name:=csv_replay_r1 \
  point_arrival_tol:=7.0 \
  arrival_confirm_sec:=1.0 \
  result_window_sec:=1.0 \
  zero_calibrate_every_points:=50 \
  zero_calibrate_at_start:=false \
  use_antagonist_release:=false
```

不在本 launch 内启动 driver：

```bash
ros2 launch yhand_mapping data_collection_from_motor_feedback_csv_launch.py \
  target_hand:=right \
  cmd_csv_path:=/abs/path/right_motor_feedback.csv \
  start_driver_node:=false
```

---

## 4. 当前关键行为

通过 `data_collection_from_motor_feedback_csv_launch.py` 启动时，launch 外层 runtime wrapper 会改两个行为：

- 默认随机发布 CSV 点：`random_cmd_publish:=true`
- 到位后的收紧线阶段强制使用 `-0.08A`
- 到位后开始 result window 前的等待强制为 `1.0s`

注意：

- launch 参数里仍有 `post_reach_pretighten_sec`，默认值显示为 `2.0`。
- 当前通过 launch 启动时，wrapper 会强制传给脚本 `--post-reach-pretighten-sec 1.0`。
- 所以实际运行日志会显示 `pretighten_wait=1.000s`。
- 如果直接运行 `data_collection_from_motor_feedback_csv.py`，才会完全按脚本参数值执行。

---

## 5. 启动到进入循环

启动后流程：

1. launch 可选启动 `Y_Hand_Node`。
2. launch 可选启动手套节点 `minimal_example_dual_stream_commented.py`。
3. 延迟 `control_script_start_delay_sec` 后启动采集脚本。
4. 脚本读取 CSV，筛选有效行，得到 `frames` 和 `motor_union`。
5. 创建输出目录。
6. 创建 `DataCollectionNode`，订阅 feedback、glove48、raw skeleton。
7. 等待启动 ready。
8. 保存 startup first frame。
9. 如果 `zero_calibrate_at_start:=true`，先做一次回零校准。
10. 开始逐点执行 CSV rows。

启动 ready 默认要求：

- feedback 已收到
- glove48 已收到
- raw skeleton 已收到

相关放宽参数：

- `disable_glove_required:=true`：glove48 和 raw skeleton 未就绪也继续。
- `disable_glove_raw_required:=true`：仍要求 glove48，但不要求 raw skeleton。

---

## 6. 单点执行到记录的完整流程

下面是每个 CSV 点的时间线。

### 6.1 周期性回零检查

每个点开始前，会先检查是否需要回零：

- `zero_calibrate_every_points:=50` 表示每执行完 50 个点后，在第 51 个点开始前回零一次。
- `zero_calibrate_every_points<=0` 表示关闭周期回零。
- 回零事件只写入 `zero_calibration_events.csv`。
- 回零本身不会保存成一个 point xlsx。

### 6.2 构造本点目标

脚本从当前 CSV frame 得到：

- `target_by_motor`：`motor_id -> target_pos`
- `active_motors`：本次受控电机集合，也就是 CSV 表头里的电机列集合
- `combo_id`：`csv_frame_<frame_idx>`

然后调用 `data_collection_main._execute_point(...)`。

### 6.3 生成 28 电机命令矩阵

本点会构造一个 28 行命令矩阵，每行是：

```text
[pos, vel, tor, kp, kd]
```

规则：

- CSV 中有目标的电机：使用 CSV 目标位置 `target_pos`，并使用 defaults 里的 `vel/tor/kp/kd`。
- CSV 中没有目标的电机：不做位置控制，进入 current-only 收紧状态。
- 默认 `use_antagonist_release:=false`，所以不会执行拮抗肌 release/rewind 时间线。
- 如果 `use_antagonist_release:=true`，移动阶段会对部分拮抗电机临时给 release/rewind 电流。

### 6.4 发布目标并采样 process

脚本发布目标命令后进入移动循环。

循环中会持续做这些事：

- 采样 glove48。
- 采样 raw skeleton 176D。
- 采样 motor feedback 28x5。
- 计算受控电机当前反馈位置和目标位置的最大误差。
- 判断是否到位。
- 检查是否超时。

process 采样频率来自 `PROCESS_SAMPLE_HZ`，当前配置是 `30Hz`。

feedback 如果超过 `FEEDBACK_STALE_SEC`，当前样本不会计入 process，当前配置是 `1.0s`。

### 6.5 到位判断

到位判断只看本次受控电机：

```text
max(abs(feedback_pos[motor] - target_pos[motor])) <= point_arrival_tol
```

并且需要连续满足 `arrival_confirm_sec` 秒。

默认 launch 参数：

- `point_arrival_tol:=7.0`
- `arrival_confirm_sec:=1.0`

如果误差中途变大，连续计时会清零重新开始。

### 6.6 超时判断

如果在 `MOVE_TIMEOUT_SEC` 内没有满足到位条件，本点标记为 timeout。

当前配置：

- `MOVE_TIMEOUT_SEC=1.5s`

即使 timeout，本点也会继续进入收紧线和 result window 采样，并保存文件，只是在 `point_log.csv` 中标记 timeout。

### 6.7 到位或超时后的收紧线

到位或超时后，脚本不再维持位置控制目标，而是切到 idle tor matrix。

通过当前 launch 启动时，这个 idle tor matrix 被 wrapper 改成：

```text
[0.0, 0.0, -0.08, 0.0, 0.0]
```

也就是所有电机进入 current-only 的 `-0.08A` 收紧线状态。

### 6.8 稳定采样和 result window

切到收紧线后，会继续采样一个稳定阶段。

当前通过 launch 启动时，默认 result 时间线是：

```text
0.0s       切到 -0.08A 收紧线
1.0s       result window 开始
2.0s       result window 结束
```

说明：

- `1.0s` 之前是收紧线等待，不参与 result 平均。
- `[1.0s, 2.0s]` 之间的样本会作为 result window。
- result 会对 glove48、raw skeleton、motor feedback 分别求平均。
- CSV 模式下 `process_include_post_stable=True`，所以 process xlsx 会包含移动阶段和到 result window 结束为止的稳定阶段。

### 6.9 保存本点 xlsx

每个点会保存 8 个 xlsx 文件：

- `process/glove_data/<point_id>.xlsx`
- `process/Glove Skeleton Data/<point_id>.xlsx`
- `process/motor_cmd/<point_id>.xlsx`
- `process/motor_feedback/<point_id>.xlsx`
- `result/glove_data/<point_id>.xlsx`
- `result/Glove Skeleton Data/<point_id>.xlsx`
- `result/motor_cmd/<point_id>.xlsx`
- `result/motor_feedback/<point_id>.xlsx`

文件名格式：

```text
00001.xlsx
00002.xlsx
...
```

### 6.10 追加 point 日志

保存 xlsx 后，会追加两份 CSV：

- `point_log.csv`
- `point_source_map.csv`

`point_log.csv` 记录：

- point_id
- global_point_id
- reached
- timeout
- move_elapsed_sec
- max_error
- active_motors
- zero_calib_state

`point_source_map.csv` 记录：

- point_id
- global_point_id
- csv_row_index
- csv_frame_idx
- active_motor_count
- active_motors

---

## 7. 每类 xlsx 记录什么

`process/glove_data`：

- 多行，每行是一次采样的 48D glove 数据。
- 覆盖本点移动阶段和后续稳定阶段。

`process/Glove Skeleton Data`：

- 每个采样展开成 25 个节点行。
- 每个节点行格式是 `[side_code, node_id, x, y, z, qx, qy, qz, qw]`。
- 采样之间用空行分隔。

`process/motor_cmd`：

- 固定 28 行 x 5 列。
- 内容是本点发布的目标命令矩阵。
- 注意它不是每一时刻的动态命令日志。

`process/motor_feedback`：

- 多个 28 行 x 5 列的反馈矩阵。
- 每个采样是一个 28x5 group。
- group 之间用空行分隔。

`result/glove_data`：

- 单行 48D。
- 是 result window 内 glove48 的平均值。

`result/Glove Skeleton Data`：

- 单行 176D。
- 是 result window 内 raw skeleton 的平均值。

`result/motor_cmd`：

- 固定 28 行 x 5 列。
- 内容和本点目标命令矩阵一致。

`result/motor_feedback`：

- 第一行写 result window label 和 sample_count。
- 后面是 result window 内 motor feedback 的平均 28x5 矩阵。

---

## 8. 输出结构

默认输出目录位于：

```text
scripts/motor_test_logs/data_collection/<timestamp>_<hand>_collection/
```

典型结构：

```text
<run_root>/
  <hand>_hand/
    startup_first_frame.json
    zero_calibration_events.csv
    run_summary.json
    <collection_name>/
      point_log.csv
      point_source_map.csv
      process/
        glove_data/
        Glove Skeleton Data/
        motor_cmd/
        motor_feedback/
      result/
        glove_data/
        Glove Skeleton Data/
        motor_cmd/
        motor_feedback/
```

---

## 9. Launch 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `target_hand` | `right` | 目标手，`left` 或 `right` |
| `cmd_csv_path` | `""` | 旧版单 CSV 模式输入路径；使用手指路径时留空 |
| `thumb_cmd_csv_path` | `""` | 可选，拇指 CSV |
| `index_cmd_csv_path` | `""` | 可选，食指 CSV |
| `middle_cmd_csv_path` | `""` | 可选，中指 CSV |
| `ring_cmd_csv_path` | `""` | 可选，无名指 CSV |
| `little_cmd_csv_path` | `""` | 可选，小指 CSV |
| `start_driver_node` | `true` | 是否在本 launch 内启动 `Y_Hand_Node` |
| `start_glove_node` | `true` | 是否启动手套发布节点 |
| `disable_glove_required` | `false` | 启动时 glove 未就绪是否继续 |
| `disable_glove_raw_required` | `false` | 仍要求 glove48，但不要求 raw skeleton |
| `random_cmd_publish` | `true` | 是否随机重排 CSV 点执行顺序 |
| `min_cmd_frame_gap` | `30` | 随机重排后相邻点最小 frame_idx 间隔 |
| `cmd_random_seed` | `0` | 随机种子，`<=0` 表示自动 |
| `start_row` | `0` | CSV 起始数据行，0-based，不含表头 |
| `stride` | `1` | 每 N 行取 1 行 |
| `max_points` | `0` | 最大执行点数，`0` 表示不限制 |
| `collection_name` | `csv_full_hand` | 本次采集子目录名 |
| `point_arrival_tol` | `7.0` | 到位误差阈值 |
| `arrival_confirm_sec` | `1.0` | 连续满足误差阈值的确认时长 |
| `post_reach_pretighten_sec` | `2.0` | launch 声明默认值；当前 wrapper 实际强制为 `1.0` |
| `result_window_sec` | `1.0` | result 平均窗口长度 |
| `zero_calibrate_every_points` | `50` | 每执行多少点后回零，`<=0` 关闭 |
| `zero_calibrate_at_start` | `false` | 第一个点前是否先回零 |
| `zero_between_csv` | `true` | 多 CSV 模式下，相邻已提供 CSV 之间是否回零 |
| `use_antagonist_release` | `false` | 移动阶段是否启用拮抗肌 release/rewind |
| `cmd_topic` | `""` | 覆盖命令话题，空则用手别默认 |
| `feedback_topic` | `""` | 覆盖反馈话题，空则用手别默认 |
| `glove_topic` | `/glove_joint_data` | glove48 话题 |
| `glove_raw_skeleton_topic` | `/glove_joint_data_raw_skeleton` | raw skeleton 话题 |
| `defaults_path` | `""` | defaults 文件路径，空则用脚本默认 |
| `output_dir` | `""` | 输出根目录，空则用脚本默认 |
| `collect_in_terminal` | `auto` | 是否在 `gnome-terminal` 中运行采集脚本 |
| `shutdown_on_collect_exit` | `true` | 采集脚本退出时是否关闭整个 launch |
| `control_script_start_delay_sec` | `1.5` | 启动 driver 后延迟多久启动采集 |

手指 CSV 模式只创建一个运行时间戳目录，数据分别保存到已提供的手指目录：

```text
<run_root>/<hand>_hand/thumb/
<run_root>/<hand>_hand/index/
<run_root>/<hand>_hand/middle/
<run_root>/<hand>_hand/ring/
<run_root>/<hand>_hand/little/
```

CSV 间回零与周期回零都记录在同一份
`<run_root>/<hand>_hand/zero_calibration_events.csv` 中；总汇总写入
`<run_root>/<hand>_hand/run_summary.json`。

---

## 10. 常见问题

### 10.1 每 50 次回零会不会生成 xlsx？

不会。周期回零只写 `zero_calibration_events.csv`，不会作为 point 保存，也不会生成 process/result xlsx。

### 10.2 为什么日志里 pretighten 是 1 秒，不是 launch 默认 2 秒？

当前 launch wrapper 强制把脚本参数改为 `--post-reach-pretighten-sec 1.0`，所以实际是 1 秒。

### 10.3 为什么点的执行顺序和 CSV 顺序不同？

因为 launch 默认 `random_cmd_publish:=true`。如果要按 CSV 顺序执行，传：

```bash
random_cmd_publish:=false
```

### 10.4 启动后因 raw skeleton 未就绪退出怎么办？

如果只需要 glove48，可以传：

```bash
disable_glove_raw_required:=true
```

如果 glove48 也暂时不要求，可以传：

```bash
disable_glove_required:=true
```

旧接线下的 CSV 需要增加 `source_motor_map:=legacy`（launch）或 `--source-motor-map legacy`（脚本）。新记录默认 `current`，读取时不改写源文件。
