# five_finger_delta_mapping_main 使用说明

当前电机编号见 [新腕带电机对应表](README_motor_mapping.md)。模型轴顺序按肌肉保留，不能按新编号排序。

对应文件：

- `scripts/five_finger_delta_mapping_main.py`
- `launch/five_finger_mapping_launch.py`

这份文档只保留五指 mapping 的常用入口、记录方式、关键逻辑和排查点。每根手指的模型路径、motor id、q index 以代码为准：

- launch 默认：`launch/five_finger_mapping_launch.py` 的 `FINGER_DEFAULTS_BY_HAND`
- 脚本 fallback 默认：`scripts/five_finger_delta_mapping_main.py` 的 `FINGER_DEFAULTS_BY_HAND`

---

## 1. 功能概览

`five_finger_delta_mapping_main.py` 是“五指 delta 多模型 + 单节点整手发布”方案。

核心点：

- 一个节点同时推理 `thumb/index/middle/ring/little`
- 每个周期只发布一条 28 电机整手命令，避免多个单指节点互相覆盖
- 支持 `target_hand:=left/right/both`
- 支持 motor feedback CSV 记录
- 右手非拇指默认可从 raw skeleton 解算 q；拇指仍走 glove48 q

---

## 2. 常用启动

右手：

```bash
ros2 launch yhand_mapping five_finger_mapping_launch.py target_hand:=right
```

左手：

```bash
ros2 launch yhand_mapping five_finger_mapping_launch.py target_hand:=left
```

左右手同时：

```bash
ros2 launch yhand_mapping five_finger_mapping_launch.py target_hand:=both
```

只跑右手拇指：

```bash
ros2 launch yhand_mapping five_finger_mapping_launch.py \
  target_hand:=right \
  enable_thumb:=true \
  enable_index:=false \
  enable_middle:=false \
  enable_ring:=false \
  enable_little:=false
```

只跑右手非拇指四指：

```bash
ros2 launch yhand_mapping five_finger_mapping_launch.py \
  target_hand:=right \
  enable_thumb:=false \
  enable_index:=true \
  enable_middle:=true \
  enable_ring:=true \
  enable_little:=true
```

按键记录 motor feedback：

```bash
ros2 launch yhand_mapping five_finger_mapping_launch.py \
  target_hand:=right \
  record_motor_feedback_csv:=false \
  mapping_in_terminal:=auto \
  motor_feedback_csv_path:=auto
```

启动后在 `five_finger_mapping_right` 终端里按键：

- `r`：开始记录
- `p`：停止记录

---

## 3. 每帧逻辑

每个控制周期：

1. 从 `/glove_joint_data` 的 48D 数据取当前手 20D q。

- 左手：`data[8:28]`
- 右手：`data[28:48]`

2. 如果是右手非拇指且 raw q 开启，则用 `/glove_joint_data_raw_skeleton` 覆盖 `index/middle/ring/little` 的 q。

3. 每根启用手指构造模型输入：

```text
[q_now, q_prev, m_prev, dq]
```

其中：

- `q_now`：当前手指 q
- `q_prev`：上一帧手指 q，首帧用 `q_now`
- `m_prev`：该手指目标电机当前反馈位置
- `dq = q_now - q_prev`

4. 模型输出 `pred_delta`，得到：

```text
pred_m = m_prev + pred_delta
```

5. 对 `pred_m` 做后处理后发布整手 28 电机命令：

- 加 `motor_cmd_offset_map_json`
- 套左右手电机限位
- 写回 28 电机命令矩阵
- 发布到 `/yhand_left_cmd` 或 `/yhand_right_cmd`

手套数据会只消费最新帧，避免旧帧堆积导致延迟。

---

## 4. motor feedback 记录

当前记录功能只保留 `motor_feedback` CSV。旧版本里的 `trace` CSV 是模型推理过程调试日志，不是实际电机反馈轨迹，不能作为 `data_collection_from_motor_feedback_csv.py` 的输入；相关 trace 启动参数已经删除。

用途：

- 记录每一帧 `/yhand_*_feedback` 的 24 个手指电机反馈位置
- 后续可作为 `data_collection_from_motor_feedback_csv.py` 的 cmd CSV 输入

推荐方式是运行中按键控制：

- `r` 开始记录
- `p` 停止记录

路径：

- `motor_feedback_csv_path:=auto` 写到 `scripts/motor_test_logs/motor_feedback_frames/`
- 支持 `{hand}`、`{pid}`、`{timestamp}` 占位符

行为：

- `record_motor_feedback_csv:=false`：启动不记录，等按 `r`
- `record_motor_feedback_csv:=true`：启动立即记录，之后仍可按 `p` 停、按 `r` 继续
- 同一次运行内多次 `r/p` 写到同一个 CSV
- 停止时会 flush/close 文件

`mapping_in_terminal:=auto` 是默认值。有 `gnome-terminal` 和显示环境时，mapping 会进入单独终端，方便接收 `r/p`。

---

## 5. 右手 raw skeleton q

默认规则：

- 右手 `index/middle/ring/little`：默认从 raw skeleton 解算 q
- 右手 `thumb`：仍使用 `/glove_joint_data` 的 glove48 q
- 左手：默认使用 glove48 q

launch 开关：

```bash
right_index_q_from_raw_skeleton:=true
```

这个参数名沿用旧名字，但现在含义是“右手非拇指四指使用 raw skeleton q”。

关闭 raw，改回 glove48：

```bash
ros2 launch yhand_mapping five_finger_mapping_launch.py \
  target_hand:=right \
  right_index_q_from_raw_skeleton:=false
```

raw 话题：

```bash
glove_raw_skeleton_topic:=/glove_joint_data_raw_skeleton
```

raw 数据格式：

- 单手：`176D = 1 hand flag + 25 node * 7`
- 双手：`352D = 2 * 176D`
- 右手 hand flag 约 `1.0`
- 左手 hand flag 约 `0.0`

运行时确认：

- 日志出现 `right_non_thumb_q_raw_ready=true`
- 开启 raw 时会周期性打印 `right_non_thumb_q16=...`

---

## 6. 默认值与覆盖

使用 `ros2 launch` 时，优先级从高到低：

1. 命令行显式传入的 launch 参数
2. `five_finger_mapping_launch.py` 的 hand-specific 默认
3. `five_finger_delta_mapping_main.py` 的 hand-specific fallback 默认

常用参数：

| 参数 | 作用 |
| --- | --- |
| `target_hand` | `left`、`right` 或 `both` |
| `enable_thumb/index/middle/ring/little` | 控制每根手指是否启用 |
| `*_model_ckpt` | 覆盖某根手指模型路径 |
| `*_model_motor_ids` | 覆盖某根手指控制的电机 id |
| `*_q_indices` | 覆盖某根手指使用的 q index |
| `*_mirror_right_from_left` | 控制右手是否镜像左手模型输出 |
| `right_index_q_from_raw_skeleton` | 右手非拇指四指是否使用 raw skeleton q |
| `glove_topic` | glove48 话题 |
| `glove_raw_skeleton_topic` | raw skeleton 话题 |
| `glove_accept_hz` | 优先使用的手套接收频率限制 |
| `glove_max_hz` | `glove_accept_hz` 为空时的 fallback |
| `glove_sub_queue_size` | 手套 ROS subscription queue |
| `glove_pending_maxlen` | 内部待处理手套帧缓存长度 |
| `record_motor_feedback_csv` | 是否启动后立即记录 motor feedback CSV |
| `motor_feedback_csv_path` | motor feedback CSV 路径 |
| `mapping_in_terminal` | `auto/true/false`，用于让 `r/p` 按键进入 mapping 进程 |
| `q_offset_map_json` | 加在 glove q 输入上的 offset |
| `motor_cmd_offset_map_json` | 加在模型输出电机命令上的 offset |
| `motor_feedback_limits_left_json` | 覆盖左手电机限位 |
| `motor_feedback_limits_right_json` | 覆盖右手电机限位 |
| `device` | `auto/cpu/cuda` |
| `require_ready` | 启动时是否强制检查 topic ready |

---

## 7. 直接运行脚本

一般推荐使用 launch。需要单独调试脚本时：

```bash
python3 scripts/five_finger_delta_mapping_main.py --hand right
```

关闭右手非拇指 raw q：

```bash
python3 scripts/five_finger_delta_mapping_main.py \
  --hand right \
  --no-right-non-thumb-q-from-raw-skeleton
```

直跑脚本时，`r/p` 按键记录只有在 stdin 是 TTY 时才可用。

---

## 8. 排查

模型没生效：

- 检查 `target_hand`
- 检查对应手指 `enable_*`
- 检查是否显式传了 `*_model_ckpt`
- 检查 `*_mirror_right_from_left`

按 `r/p` 没反应：

- 确认按键是在 `five_finger_mapping_*` 终端里输入
- 图形环境下可强制 `mapping_in_terminal:=true`
- SSH 或无图形环境可用 `record_motor_feedback_csv:=true` 启动即记录

感觉卡顿：

- 看状态日志里的 `infer_last_ms`、`infer_ema_ms`、`infer_load`
- 看是否频繁出现 `pred_clip`
- 必要时降低 `glove_accept_hz` 或 `glove_max_hz`

raw q 没工作：

- 确认 `right_index_q_from_raw_skeleton:=true`
- 确认 `/glove_joint_data_raw_skeleton` 有数据
- 看日志是否出现 `right_non_thumb_q_raw_ready=true`
