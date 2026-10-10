# curl_m0_m2_mlp_mapping 使用说明

当前电机编号见 [新腕带电机对应表](README_motor_mapping.md)。模型轴顺序按肌肉保留，不能按新编号排序。

对应文件：

- `scripts/curl_m0_m2_mlp_mapping_main.py`
- `launch/curl_m0_m2_mlp_mapping_launch.py`

说明：

- 当前这套脚本已经同时支持 `index`、`middle`、`ring` 和 `little`
- 当前模型形式是 `q0 + curl multihead MLP`

---

## 1. 功能概览

这条链路用于控制单根非拇指手指：

- `finger:=index` 控制食指
- `finger:=middle` 控制中指
- `finger:=ring` 控制无名指
- `finger:=little` 控制小指

运行逻辑：

```text
raw q0/q1/q2/q3
 -> 用 q1/q2/q3 计算 supervised curl
 -> 模型推理
 -> 输出 target_m_i
 -> 发布到该手指对应的主动电机
```

当前支持：

- 右手直接使用右手模型
- 左手通过右手模型镜像控制
- 同一只手可同时启用 `index`、`middle`、`ring`、`little`
- 可选从 `raw_skeleton` 解算该手指的 `q0..q3`
- 也可退回直接读取 glove20 对应 `q_indices`

---

## 2. 默认模型

默认右手模型路径：

- `index`: `scripts/mp_curve_q0_right/index/best_index_q0_curl_multihead_mlp.pt`
- `middle`: `scripts/mp_curve_q0_right/middle/best_middle_q0_curl_multihead_mlp.pt`
- `ring`: `scripts/mp_curve_q0_right/ring/best_ring_q0_curl_multihead_mlp.pt`
- `little`: `scripts/mp_curve_q0_right/little/best_little_q0_curl_multihead_mlp.pt`

默认 curl meta：

- `index`: `scripts/curve_function/index/index_target_q0_supervised_curl_1570.meta.json`
- `middle`: `scripts/curve_function/middle/middle_target_q0_supervised_curl_1357.meta.json`
- `ring`: `scripts/curve_function/ring/ring_target_q0_supervised_curl_1704.meta.json`
- `little`: `scripts/curve_function/little/little_target_q0_supervised_curl_1320.meta.json`

默认 glove20 q 顺序：

- `index`: `4,5,6,7`
- `middle`: `8,9,10,11`
- `ring`: `12,13,14,15`
- `little`: `16,17,18,19`

默认右手电机顺序：

- `index`: `17,11,14,15,1`
- `middle`: `2,9,16,10,23`
- `ring`: `3,7,24,8`
- `little`: `27,25,6,4`

默认左手镜像后的电机顺序：

- `index`: 由运行时根据右手电机顺序逐项镜像得到
- `middle`: 由运行时根据右手电机顺序逐项镜像得到
- `ring`: `17,21,10,22`
- `little`: `13,11,20,18`

---

## 3. 常用启动

右手食指：

```bash
ros2 launch yhand_mapping curl_m0_m2_mlp_mapping_launch.py \
  target_hand:=right \
  finger:=index
```

右手中指：

```bash
ros2 launch yhand_mapping curl_m0_m2_mlp_mapping_launch.py \
  target_hand:=right \
  finger:=middle
```

默认右手无名指：

```bash
ros2 launch yhand_mapping curl_m0_m2_mlp_mapping_launch.py
```

右手小指：

```bash
ros2 launch yhand_mapping curl_m0_m2_mlp_mapping_launch.py \
  target_hand:=right \
  finger:=little
```

左手无名指镜像：

```bash
ros2 launch yhand_mapping curl_m0_m2_mlp_mapping_launch.py \
  target_hand:=left \
  finger:=ring
```

左手小指镜像：

```bash
ros2 launch yhand_mapping curl_m0_m2_mlp_mapping_launch.py \
  target_hand:=left \
  finger:=little
```

同一只手同时启用食指、中指、无名指和小指：

```bash
ros2 launch yhand_mapping curl_m0_m2_mlp_mapping_launch.py \
  target_hand:=right \
  enable_index:=true \
  enable_middle:=true \
  enable_ring:=true \
  enable_little:=true
```

左手同时启用食指、中指、无名指和小指镜像：

```bash
ros2 launch yhand_mapping curl_m0_m2_mlp_mapping_launch.py \
  target_hand:=left \
  enable_index:=true \
  enable_middle:=true \
  enable_ring:=true \
  enable_little:=true
```

如果 driver 和 glove 节点已经在运行：

```bash
ros2 launch yhand_mapping curl_m0_m2_mlp_mapping_launch.py \
  target_hand:=left \
  finger:=little \
  start_driver_node:=false \
  start_glove_node:=false
```

左右手同时启动时，建议分两个 terminal：

```bash
# terminal 1: 先起右手，带 driver 和 glove
ros2 launch yhand_mapping curl_m0_m2_mlp_mapping_launch.py \
  target_hand:=right \
  finger:=index
```

```bash
# terminal 2: 再起左手，只起推理，避免重复启动 driver/glove
ros2 launch yhand_mapping curl_m0_m2_mlp_mapping_launch.py \
  target_hand:=left \
  finger:=index \
  start_driver_node:=false \
  start_glove_node:=false
```

说明：

- 左手和右手分别使用不同的话题：
  - 左手：`/yhand_left_cmd`、`/yhand_left_feedback`
  - 右手：`/yhand_right_cmd`、`/yhand_right_feedback`
- 所以左右手各起一份推理通常不会互相覆盖
- 真正容易冲突的是：
  - 同一只手同时起两份推理
  - 左右两边都重复启动 driver 或 glove publisher

---

## 4. 左手镜像说明

左手时默认：

```text
mirror_left_from_right = true
```

这表示：

- 仍然使用右手训练得到的模型
- 输出电机编号会自动镜像到左手
- 如果模型输入里包含 `q0`，左手会自动对 `q0` 取反，再送进右手模型

这样做是为了让左手的内收外展方向和右手模型的坐标约定一致。

如果你想关闭镜像：

```bash
ros2 launch yhand_mapping curl_m0_m2_mlp_mapping_launch.py \
  target_hand:=left \
  finger:=ring \
  mirror_left_from_right:=false
```

通常不建议关闭，除非你已经准备好了左手专用模型或左手专用电机顺序。

---

## 5. 输入来源

默认优先使用 raw skeleton：

```text
right_non_thumb_q_from_raw_skeleton = true
```

这时脚本会：

1. 从 `/glove_joint_data_raw_skeleton` 读取 176D 或 352D 数据
2. 只取当前 `target_hand` 对应的那一只手
3. 调用 `replace_right_non_thumb_q_from_skeleton.py`
4. 读取：

- `index_q_0..index_q_3`
- `middle_q_0..middle_q_3`
- `ring_q_0..ring_q_3`
- `little_q_0..little_q_3`

如果关闭这个开关，就改为直接从 glove20 读取：

```bash
ros2 launch yhand_mapping curl_m0_m2_mlp_mapping_launch.py \
  target_hand:=right \
  finger:=little \
  right_non_thumb_q_from_raw_skeleton:=false
```

---

## 6. 输出与日志

输出：

- 发布到 `/yhand_left_cmd` 或 `/yhand_right_cmd`
- 只改当前启用手指对应的主动电机

当前状态日志只保留三类信息：

- `raw_angles`
- `active_cmd`
- `active_torque_mag`

其中：

- `raw_angles` 是五指 glove20 角度分组显示
- `active_cmd` 是当前主动电机的目标位置
- `active_torque_mag` 是当前主动电机 feedback torque 的绝对值

---

## 7. 常用参数

| 参数 | 作用 |
| --- | --- |
| `target_hand` | `left` 或 `right` |
| `finger` | `index`、`middle`、`ring` 或 `little` |
| `enable_index` | 是否启用食指模型；为空时走 `finger` 的兼容默认行为 |
| `enable_middle` | 是否启用中指模型；为空时走 `finger` 的兼容默认行为 |
| `enable_ring` | 是否启用无名指模型；为空时走 `finger` 的兼容默认行为 |
| `enable_little` | 是否启用小指模型；为空时走 `finger` 的兼容默认行为 |
| `start_driver_node` | 是否启动 `Y_Hand_Node` |
| `start_glove_node` | 是否启动 glove publisher |
| `glove_topic` | 手套角度话题，默认 `/glove_joint_data` |
| `glove_raw_skeleton_topic` | raw skeleton 话题，默认 `/glove_joint_data_raw_skeleton` |
| `right_non_thumb_q_from_raw_skeleton` | 是否从 raw skeleton 解算 q |
| `model_ckpt` | 模型路径；留空则走当前手指默认值 |
| `curl_meta` | curl meta 路径；留空则走当前手指默认值 |
| `model_motor_ids` | 电机顺序；留空则走当前手指默认值 |
| `q_indices` | glove20 索引；留空则走当前手指默认值 |
| `mirror_left_from_right` | 左手是否镜像右手模型，默认 `true` |
| `index_model_ckpt` / `middle_model_ckpt` / `ring_model_ckpt` / `little_model_ckpt` | 分别覆盖食指/中指/无名指/小指模型路径 |
| `index_curl_meta` / `middle_curl_meta` / `ring_curl_meta` / `little_curl_meta` | 分别覆盖食指/中指/无名指/小指 curl meta |
| `index_model_motor_ids` / `middle_model_motor_ids` / `ring_model_motor_ids` / `little_model_motor_ids` | 分别覆盖食指/中指/无名指/小指电机顺序 |
| `index_q_indices` / `middle_q_indices` / `ring_q_indices` / `little_q_indices` | 分别覆盖食指/中指/无名指/小指 q 索引 |
| `control_period_sec` | 发布周期，默认 `0.01` |
| `status_period_sec` | 状态日志周期，默认 `2.0` |
| `device` | `auto/cpu/cuda` |
| `require_ready` | 启动时是否等待 ready |
| `ready_timeout_sec` | ready 等待超时 |

---

## 8. 自定义模型

如果你要替换成自己的模型，可以只覆盖对应参数：

```bash
ros2 launch yhand_mapping curl_m0_m2_mlp_mapping_launch.py \
  target_hand:=right \
  finger:=little \
  model_ckpt:=/your/best_little_q0_curl_multihead_mlp.pt \
  curl_meta:=/your/little_target_q0_supervised_curl.meta.json
```

要求：

- checkpoint 格式与当前 `q0+curl multihead` 一致
- checkpoint 中应包含 `curl_head_target_m_cols`、`q0curl_head_target_m_cols`
- output scaler 命名要和输出列一致，例如：
  - `index`: `y_m014`、`y_m23`
  - `middle`: `y_m012`、`y_m34`
  - `ring`: `y_m02`、`y_m13`
  - `little`: `y_m01`、`y_m23`

---

## 9. 排查建议

如果启动后手指不动，优先检查：

1. `finger` 是否选对
2. `target_hand` 是否选对
3. `model_ckpt` 和 `curl_meta` 路径是否存在
4. `/yhand_*_feedback` 是否正常
5. `/glove_joint_data_raw_skeleton` 是否真的有当前这只手的数据

如果左手方向不对，优先检查：

1. 是否仍然是 `mirror_left_from_right:=true`
2. 当前模型是否确实是右手模型
3. raw skeleton 解算出来的 `q0` 方向是否和预期一致

如果你只是想快速验证左手镜像，最常用的是：

```bash
ros2 launch yhand_mapping curl_m0_m2_mlp_mapping_launch.py target_hand:=left finger:=index
```

或：

```bash
ros2 launch yhand_mapping curl_m0_m2_mlp_mapping_launch.py target_hand:=left finger:=middle
```

或：

```bash
ros2 launch yhand_mapping curl_m0_m2_mlp_mapping_launch.py target_hand:=left finger:=ring
```

```bash
ros2 launch yhand_mapping curl_m0_m2_mlp_mapping_launch.py \
  target_hand:=left \
  finger:=little
```
