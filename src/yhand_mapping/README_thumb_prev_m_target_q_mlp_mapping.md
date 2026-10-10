# prev_m_target_q_mlp_mapping 使用说明

当前电机编号见 [新腕带电机对应表](README_motor_mapping.md)。模型轴顺序按肌肉保留，不能按新编号排序。

对应文件：

- `scripts/thumb_prev_m_target_q_mlp_mapping_main.py`
- `launch/thumb_prev_m_target_q_mlp_mapping_launch.py`

这个运行链路支持“五指 prev-m + target-q MLP 单节点发布”。文件名里仍保留 `thumb`，是为了兼容最开始的右手拇指版本；当前脚本已经不是只能跑拇指。

默认右手模型路径：

- `thumb`: `scripts/mp_right/thumb_delta_multihead_masked/best_thumb_prev_m_target_q_mlp.pt`
- `index`: `scripts/mp_right/index_delta_multihead_masked/best_index_prev_m_target_q_mlp.pt`
- `middle`: `scripts/mp_right/middle_delta_multihead_masked/best_middle_prev_m_target_q_mlp.pt`
- `ring`: `scripts/mp_right/ring_delta_multihead_masked/best_ring_prev_m_target_q_mlp.pt`
- `little`: `scripts/mp_right/little_delta_multihead_masked/best_little_prev_m_target_q_mlp.pt`

因此默认启动时只启用右手拇指。要跑五指，只需要打开对应 `enable_*` 开关；前提是这些默认路径下已经放好了同格式 checkpoint。

---

## 1. 功能概览

每根启用的手指各加载一个 MLP：

```text
input  = [history_m_0.., target_q_0..]
output = target_m_0..
```

核心点：

- 单个 ROS 节点同时推理多根手指
- 每个控制周期只发布一条 28 电机整手命令，避免多个单指节点互相覆盖
- 每根手指可独立配置 ckpt、motor ids、q indices、enable 开关
- 模型输出直接作为目标电机位置，不做 `delta_m` 累加
- 支持 `target_hand:=left/right/both`
- 支持电机限位、q offset、motor command offset
- 手套数据只消费最新帧，避免旧帧堆积造成延迟

---

## 2. 常用启动

默认右手启动，当前只启用右手拇指：

```bash
ros2 launch yhand_mapping thumb_prev_m_target_q_mlp_mapping_launch.py
```

如果 driver 和 glove publisher 已经在运行：

```bash
ros2 launch yhand_mapping thumb_prev_m_target_q_mlp_mapping_launch.py \
  start_driver_node:=false \
  start_glove_node:=false
```

启用右手五指：

```bash
ros2 launch yhand_mapping thumb_prev_m_target_q_mlp_mapping_launch.py \
  target_hand:=right \
  enable_thumb:=true \
  enable_index:=true \
  enable_middle:=true \
  enable_ring:=true \
  enable_little:=true
```

如果某根手指需要临时换模型，仍可只覆盖那根手指的 ckpt：

```bash
ros2 launch yhand_mapping thumb_prev_m_target_q_mlp_mapping_launch.py \
  target_hand:=right \
  enable_index:=true \
  index_model_ckpt:=/path/to/another_index_prev_m_target_q_mlp.pt
```

左右手同时运行：

```bash
ros2 launch yhand_mapping thumb_prev_m_target_q_mlp_mapping_launch.py \
  target_hand:=both
```

注意：左手默认没有启用模型，需要显式传入左手 ckpt 或合适的 mirror 配置。

---

## 3. 输入输出

checkpoint 中常见输入列：

```text
history_m_0, history_m_1, ...,
target_q_0, target_q_1, ...
```

运行时：

- `history_m` 来自最新 `/yhand_*_feedback` 的电机位置
- `target_q` 来自 `/glove_joint_data` 的 hand-local glove20
- 如果某个 feedback 位置是 `NaN`，脚本使用上一帧 command position 兜底

输出列：

```text
target_m_0, target_m_1, ...
```

输出会直接写入对应电机目标位置：

```text
target_m_i -> model_motor_ids[i]
```

这里没有旧 delta 模型里的：

```text
pred_m = m_prev + delta_m
```

新脚本等价于：

```text
pred_m = target_m
```

---

## 4. 每帧逻辑

每个控制周期：

1. 从 `/glove_joint_data` 取 48D 手套数据。

- 左手：`data[8:28]`
- 右手：`data[28:48]`

2. 对每根启用手指按 `*_q_indices` 取目标 q。

3. 对每根启用手指按 `*_model_motor_ids` 取当前电机 feedback 位置。

4. 按 checkpoint 的 `input_cols` 组装输入。

5. 使用 checkpoint 的 `x_scaler` 标准化。

6. MLP 推理。

7. 使用 checkpoint 的 `y_scaler` 反标准化，得到目标电机位置。

8. 对目标位置做后处理：

- 加 `motor_cmd_offset_map_json`
- 套 `motor_feedback_limits_*_json` 电机限位
- 写回 28 电机命令矩阵

9. 发布到 `/yhand_left_cmd` 或 `/yhand_right_cmd`。

如果没有新手套帧，会保持上一帧 command；如果还没有 feedback，则发布 idle torque。

---

## 5. 常用参数

| 参数 | 作用 |
| --- | --- |
| `target_hand` | `left`、`right` 或 `both` |
| `start_driver_node` | 是否启动 `Y_Hand_Node` |
| `start_glove_node` | 是否启动 glove publisher |
| `glove_topic` | 48D 手套话题，默认 `/glove_joint_data` |
| `enable_thumb/index/middle/ring/little` | 是否启用对应手指 |
| `*_model_ckpt` | 对应手指 checkpoint |
| `*_model_motor_ids` | 对应手指输出 motor 顺序 |
| `*_q_indices` | 对应手指 hand-local glove20 q 顺序 |
| `*_mirror_right_from_left` | 是否镜像 motor ids |
| `control_period_sec` | 命令发布周期，默认 `0.01` |
| `status_period_sec` | 状态日志周期，默认 `2.0` |
| `glove_max_hz` | 手套最大接收频率，`<=0` 表示不限制 |
| `device` | Torch device：`auto/cpu/cuda` |
| `require_ready` | 启动时是否等待 glove + feedback 就绪 |
| `ready_timeout_sec` | ready 等待超时 |

旧拇指版本的参数仍可用，作为 thumb alias：

- `model_ckpt` -> `thumb_model_ckpt`
- `model_motor_ids` -> `thumb_model_motor_ids`
- `q_indices` -> `thumb_q_indices`
- `mirror_right_from_left` -> `thumb_mirror_right_from_left`

---

## 6. 默认 motor 与 q 顺序

右手：

```text
thumb:  motors 20,18,13,21,22,28    q 0,1,2,3
index:  motors 17,11,14,15,1        q 4,5,6,7
middle: motors 2,9,16,10,23         q 8,9,10,11
ring:   motors 3,7,24,8              q 12,13,14,15
little: motors 27,25,6,4           q 16,17,18,19
```

左手：

```text
thumb:  motors 27,7,8,14,6,4       q 0,1,2,3
index:  motors 1,15,3,25,28         q 4,5,6,7
middle: motors 2,24,9,16,23          q 8,9,10,11
ring:   motors 17,21,10,22          q 12,13,14,15
little: motors 11,20,18,13          q 16,17,18,19
```

---

## 7. 运行时日志

启动时会打印：

- 每根启用手指的模型路径
- checkpoint architecture
- MLP 维度，例如 `10->[128, 128, 64]->6`
- `input_cols`
- `target_m_cols`
- `q_indices`
- `motor_ids`
- motor limit mode

周期状态日志会打印：

- glove/feedback 是否 ready
- glove 和 feedback age
- 推理耗时
- clip 次数
- 每根启用手指的 command、target_q、pred_m

如果怀疑输入不对，优先看日志里的：

```text
target_q=...
pred_m=...
```

---

## 8. 和 five_finger_delta 的区别

`five_finger_delta_mapping_main.py`：

```text
input = [q_now, q_prev, m_prev, dq]
output = delta_m
pred_m = m_prev + delta_m
```

当前脚本：

```text
input = [history_m, target_q]
output = target_m
pred_m = target_m
```

因此同一个 checkpoint 不能在这两个脚本之间直接混用。
