# Mapping 功能说明（当前实现，2026-05）

当前电机编号见 [新腕带电机对应表](README_motor_mapping.md)。模型轴顺序按肌肉保留，不能按新编号排序。

本文档对应当前代码：

- `launch/thumb_mapping_launch.py`
- `launch/thumb_delta_multihead_mapping_launch.py`
- `launch/thumb_v2_mapping_launch.py`
- `launch/little_mapping_launch.py`
- `launch/lefthand_full_mapping_launch.py`
- `launch/lefthand_unified_mapping_launch.py`
- `scripts/thumb_nn_mapping_main.py`
- `scripts/thumb_delta_multihead_mapping_main.py`
- `scripts/thumb_v2_nn_mapping_main.py`
- `scripts/little_nn_mapping_main.py`
- `scripts/lefthand_full_mapping_main.py`
- `scripts/yhand_unified_mapping_main.py`
- `scripts/muscle_adjustment_defaults.py`

---

## 1. 总览

当前 mapping 运行时统一流程：

1. 启动目标手 `Y_Hand_Node`（左右手读取 `config/left_hand_device_config.json` / `config/right_hand_device_config.json` 的连接参数）。
2. 可选启动手套节点 `minimal_example_dual_stream_commented.py`。
3. 启动对应 mapping 脚本，将 glove48 数据切片为 hand-local glove20，再推理输出电机目标。

默认话题：

- 左手：`/yhand_left_feedback`、`/yhand_left_cmd`
- 右手：`/yhand_right_feedback`、`/yhand_right_cmd`
- 手套：`/glove_joint_data`

---

## 2. 各 launch 用途

1. `thumb_mapping_launch.py`
   用于拇指/无名指/食指映射（支持多模型组合），并支持 `thumb_backend` 选择：

- `nn`：`thumb_nn_mapping_main.py`（inverse MLP）
- `delta`：`thumb_delta_multihead_mapping_main.py`（delta multihead）

2. `thumb_delta_multihead_mapping_launch.py`
   仅拇指 delta-multihead 专用 launch。

3. `thumb_v2_mapping_launch.py`
   拇指 v2 模型专用 launch（兼容 .pt/.pth 多种格式）。

4. `little_mapping_launch.py`
   小指模型 launch，支持 inverse-delta / delta-multihead / simple-mlp 三类 ckpt 格式自动识别。

5. `lefthand_full_mapping_launch.py`
   左手五指独立模型组合映射（thumb/index/middle/ring/little 五个 ckpt）。

6. `lefthand_unified_mapping_launch.py`
   统一大模型映射（20q(+onehot) -> 24 电机）。
7. `five_finger_mapping_launch.py`
   五指 delta 多模型单节点映射（一次发布整手命令，避免多节点互相覆盖）。

### 2.1 右手五指同时映射（delta，多模型）

```bash
ros2 launch yhand_mapping five_finger_mapping_launch.py \
  target_hand:=right \
  thumb_model_ckpt:=/home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/scripts/mp/runs/thumb_1/best_thumb_delta_multihead.pt \
  thumb_model_motor_ids:=27,7,8,14,6,4 \
  thumb_q_indices:=0,1,2,3 \
  thumb_mirror_right_from_left:=true \
  index_model_ckpt:=/home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/scripts/mp_right/index_delta_multihead/best_index_delta_multihead.pt \
  index_model_motor_ids:=17,11,14,15,1 \
  index_q_indices:=4,5,6,7 \
  index_mirror_right_from_left:=false \
  middle_model_ckpt:=/home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/scripts/mp_right/middle_delta_multihead/best_middle_delta_multihead.pt \
  middle_model_motor_ids:=2,9,16,10,23 \
  middle_q_indices:=8,9,10,11 \
  middle_mirror_right_from_left:=false \
  ring_model_ckpt:=/home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/scripts/mp/runs/ring_delta_multihead/best_ring_delta_multihead.pt \
  ring_model_motor_ids:=17,21,10,22 \
  ring_q_indices:=12,13,14,15 \
  ring_mirror_right_from_left:=true \
  little_model_ckpt:=/home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/scripts/mp_right/little_delta_multihead/best_little_delta_multihead.pt \
  little_model_motor_ids:=27,25,6,4 \
  little_q_indices:=16,17,18,19 \
  little_mirror_right_from_left:=false
```

### 2.2 five_finger 默认配置来源

`five_finger` 链路有两层默认值：

1. 脚本默认：`scripts/five_finger_delta_mapping_main.py`（`FINGER_DEFAULTS`、`Q_OFFSET_MAP_DEFAULTS`）。
2. launch 默认：`launch/five_finger_mapping_launch.py`（`DeclareLaunchArgument(...)` 和 launch 内配置表）。

使用 `ros2 launch yhand_mapping five_finger_mapping_launch.py ...` 时，实际生效链路是：

- 先取 launch 默认（或你命令行传入的 launch 参数）
- 再由 launch 转发给 `five_finger_delta_mapping_main.py`

当前 `five_finger_mapping_launch.py` 默认重点：

- 五指模型参数（enable/ckpt/motor_ids/q_indices/mirror）：由 launch 内 `FINGER_DEFAULTS` 配置表统一管理。
- q 偏置（会传递给脚本）：统一通过 `q_offset_map_json`（按 hand + q_index 配置）。
- 右手非拇指 raw skeleton q：`right_index_q_from_raw_skeleton` 默认 `true`（即默认用 raw 解算 index/middle/ring/little 的 `q_0..q_3`）；传 `right_index_q_from_raw_skeleton:=false` 可改回 glove48 切片。

### 2.3 参数覆盖优先级（five_finger）

按最终生效顺序（高 -> 低）：

1. 你在命令行显式传入的 launch 参数  
   例如：`ring_model_motor_ids:=17,21,10,22`
2. `five_finger_mapping_launch.py` 中的 `DeclareLaunchArgument` 默认值
3. `five_finger_delta_mapping_main.py` 的 argparse 默认值  
   仅在“直接 `python3 scripts/five_finger_delta_mapping_main.py ...` 运行脚本”时直接生效

### 2.4 five_finger motor_feedback 记录开关（默认值）

`five_finger_mapping_launch.py` 当前只保留 motor_feedback 记录：

1. `record_motor_feedback_csv:=false` + `motor_feedback_csv_path:=auto`
   默认关闭启动即录制；运行时可在 mapping 终端按 `r` 开始记录、按 `p` 停止记录。`auto` 输出到 `scripts/motor_test_logs/motor_feedback_frames/`。
2. `mapping_in_terminal:=auto`
   默认有图形终端时把 mapping 进程放到 `gnome-terminal`，用于接收 `r/p` 录制按键。

## 3. 拇指模型切换（nn / delta）

当前 `thumb_mapping_launch.py` 支持通过 `thumb_backend` 在两种后端切换：

- `nn`（inverse MLP）：`scripts/mp/runs/inverse_raw_thumb/best_inverse.pt`
- `delta`（delta multihead）：`scripts/mp/runs/thumb_1/best_thumb_delta_multihead.pt`

并且支持自动选模型路径：

- `thumb_model_ckpt:=auto` 时，`nn` 自动使用 `thumb_nn_model_ckpt`
- `thumb_model_ckpt:=auto` 时，`delta` 自动使用 `thumb_delta_model_ckpt`

### 3.1 左手 nn（inverse_raw_thumb）

```bash
ros2 launch yhand_mapping thumb_mapping_launch.py \
  target_hand:=left \
  target_finger:=thumb \
  thumb_backend:=nn \
  thumb_model_ckpt:=auto \
  enable_thumb:=true \
  enable_ring:=false \
  enable_index:=false \
  thumb_model_motor_ids:=27,7,8,14,6,4 \
  thumb_q_indices:=0,1,2,3
```

### 3.2 左手 delta（thumb_1）

```bash
ros2 launch yhand_mapping thumb_mapping_launch.py \
  target_hand:=left \
  target_finger:=thumb \
  thumb_backend:=delta \
  thumb_model_ckpt:=auto \
  glove_max_hz:=100 \
  thumb_model_motor_ids:=27,7,8,14,6,4 \
  thumb_q_indices:=0,1,2,3
```

说明：

- `thumb_backend:=nn` 会走 `thumb_nn_mapping_main.py`，可多模型组合（thumb/ring/index）。
- `thumb_backend:=delta` 会走 `thumb_delta_multihead_mapping_main.py`，并强制只启用拇指模型。
- 右手使用时只需改 `target_hand:=right`，通常保留 `mirror_right_from_left:=true`。
- `thumb_model_motor_ids` 按模型输出 `m_0..m_5` 的顺序填写。

---

## 4. 左右手每根手指启动指令

下面给出按“单根手指”启动的常用指令。默认都使用 `/glove_joint_data`。

### 4.1 拇指（thumb，delta）

左手：

```bash
ros2 launch yhand_mapping thumb_mapping_launch.py \
  target_hand:=left \
  target_finger:=thumb \
  thumb_backend:=delta \
  thumb_model_ckpt:=/home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/scripts/mp/runs/thumb_1/best_thumb_delta_multihead.pt \
  thumb_model_motor_ids:=27,7,8,14,6,4 \
  thumb_q_indices:=0,1,2,3
```

右手：

```bash
ros2 launch yhand_mapping thumb_mapping_launch.py \
  target_hand:=right \
  target_finger:=thumb \
  thumb_backend:=delta \
  thumb_model_ckpt:=/home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/scripts/mp/runs/thumb_1/best_thumb_delta_multihead.pt \
  thumb_model_motor_ids:=27,7,8,14,6,4 \
  thumb_q_indices:=0,1,2,3 \
  mirror_right_from_left:=true
```

### 4.2 食指（index，nn）

左手：

```bash
ros2 launch yhand_mapping thumb_mapping_launch.py \
  target_hand:=left \
  target_finger:=index \
  thumb_backend:=nn \
  enable_thumb:=false \
  enable_ring:=false \
  enable_index:=true \
  index_model_ckpt:=/home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/scripts/mp_test/index/best_inverse.pt \
  index_model_motor_ids:=1,15,3,25,28 \
  index_q_indices:=4,5,6,7
```

右手：

```bash
ros2 launch yhand_mapping thumb_mapping_launch.py \
  target_hand:=right \
  target_finger:=index \
  thumb_backend:=nn \
  enable_thumb:=false \
  enable_ring:=false \
  enable_index:=true \
  index_model_ckpt:=/home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/scripts/mp_test/index/best_inverse.pt \
  index_model_motor_ids:=1,15,3,25,28 \
  index_q_indices:=4,5,6,7 \
  mirror_right_from_left:=true
```

### 4.3 无名指（ring，nn）

左手：

```bash
ros2 launch yhand_mapping thumb_mapping_launch.py \
  target_hand:=left \
  target_finger:=ring \
  thumb_backend:=nn \
  enable_thumb:=false \
  enable_ring:=true \
  enable_index:=false \
  ring_model_ckpt:=/home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/scripts/mp_test/ring/best_inverse.pt \
  ring_model_motor_ids:=17,21,10,22 \
  ring_q_indices:=12,13,14,15
```

右手：

```bash
ros2 launch yhand_mapping thumb_delta_multihead_mapping_launch.py \
  target_hand:=right \
  model_ckpt:=/home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/scripts/mp_right/ring_delta_multihead/best_ring_delta_multihead.pt \
  model_motor_ids:=3,7,24,8 \
  q_indices:=12,13,14,15 \
  mirror_right_from_left:=false

```

### 4.4 小指（little，little_mapping_launch）

左手：

```bash
ros2 launch yhand_mapping little_mapping_launch.py \
  target_hand:=left \
  model_ckpt:=/home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/scripts/mp/runs/little_delta_multihead/best_little_delta_multihead.pt \
  model_motor_ids:=11,13,20,18 \
  q_indices:=16,17,18,19
```

右手：

```bash
ros2 launch yhand_mapping little_mapping_launch.py \
  target_hand:=right \
  model_ckpt:=/home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/scripts/mp/runs/little_delta_multihead/best_little_delta_multihead.pt \
  model_motor_ids:=11,13,20,18 \
  q_indices:=16,17,18,19 \
  mirror_right_from_left:=true
```

如果使用“右手数据直训”的小指模型（不走左手镜像）：

```bash
ros2 launch yhand_mapping little_mapping_launch.py \
  target_hand:=right \
  model_ckpt:=/home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/scripts/mp_right/little_delta_multihead/best_little_delta_multihead.pt \
  model_motor_ids:=27,25,6,4 \
  q_indices:=16,17,18,19 \
  mirror_right_from_left:=false
```

### 4.5 中指（middle）

当前没有 `middle_mapping_launch.py` 独立 launch，可用下面命令启动中指 delta 模型：

左手：

```bash
ros2 launch yhand_mapping thumb_mapping_launch.py \
  target_hand:=left \
  target_finger:=middle \
  thumb_backend:=delta \
  thumb_model_ckpt:=/home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/scripts/mp/runs/middle_delta_multihead/best_middle_delta_multihead.pt \
  thumb_model_motor_ids:=2,24,9,16,23 \
  thumb_q_indices:=8,9,10,11
```

右手：

```bash
ros2 launch yhand_mapping thumb_mapping_launch.py \
  target_hand:=right \
  target_finger:=middle \
  thumb_backend:=delta \
  thumb_model_ckpt:=/home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/scripts/mp/runs/middle_delta_multihead/best_middle_delta_multihead.pt \
  thumb_model_motor_ids:=2,24,9,16,23 \
  thumb_q_indices:=8,9,10,11 \
  mirror_right_from_left:=true
```

---

## 5. thumb_mapping_launch 关键参数

常用参数（核心）：

- `target_hand`：`left|right`
- `target_finger`：`ring|thumb|index|both|custom`
- `thumb_backend`：`nn|delta`
- `thumb_model_ckpt`：拇指模型路径；`auto` 表示按 `thumb_backend` 自动选择
- `thumb_nn_model_ckpt`：`nn` 后端默认模型路径
- `thumb_delta_model_ckpt`：`delta` 后端默认模型路径
- `thumb_model_motor_ids`：拇指输出映射电机 ID（逗号分隔）
- `thumb_q_indices`：拇指输入 glove20 索引（逗号分隔）
- `mirror_right_from_left`：右手是否镜像左手电机映射
- `control_period_sec`：控制周期
- `status_period_sec`：日志打印周期
- `glove_max_hz`：仅 `delta` 后端生效，手套最大接收频率（Hz），默认 `100.0`；`<=0` 表示不限频
- `device`：`auto|cpu|cuda`
- `require_ready`：启动时是否强制等待 glove+feedback 就绪

`nn` 后端下还会用到：

- `enable_thumb`、`enable_ring`、`enable_index`
- `ring_model_ckpt/ring_model_motor_ids/ring_q_indices`
- `index_model_ckpt/index_model_motor_ids/index_q_indices`

---

## 6. 参数对齐规则（避免维度错误）

1. `len(q_indices)` 必须等于模型输入维度 `q_dim`。
2. `len(model_motor_ids)` 必须等于模型输出维度 `m_dim`（或与 `m_cols` 可解析规则一致）。
3. 若开启 `mirror_right_from_left=true` 且 `target_hand=right`，电机 ID 会自动按左右手映射表转换。
4. 在 `thumb_nn_mapping_main.py` 与 `thumb_delta_multihead_mapping_main.py` 中，若 `hand=right` 且 `mirror_right_from_left=true`，`MOTOR_FEEDBACK_LIMITS` 也会自动按同一映射表镜像成右手电机键后再裁剪。
5. glove48 在运行时会先切片为 hand-local glove20：

- 左手使用 `[8:28]`
- 右手使用 `[28:48]`

---

## 7. 常见问题

1. 报 `Output dim mismatch`：

- 检查 `thumb_model_motor_ids` 数量是否与 ckpt 输出维度一致（你的 `inverse_raw_thumb` 为 6 维）。

2. 报 `Input dim mismatch`：

- 检查 `thumb_q_indices` 数量是否为 4（你的 `inverse_raw_thumb` 为 4 维）。

3. 右手动作不对：

- 先确认 `mirror_right_from_left` 是否符合你的训练假设（默认 `true`）。

4. 启动后一直 idle 不动：

- 检查是否收到 glove48（`/glove_joint_data`）与手反馈话题。

5. `Cannot import torch` 或 `Cannot import devices_pkg.msg`：

- 确认已 source 对应 ROS2 与工作空间环境，并使用与 launch 一致的 Python 环境。

6. `delta` 出现延迟/抽搐：

- 优先降低 `glove_max_hz`（例如 `100 -> 80 -> 60`）观察改善。
- 运行日志会打印 `glove_rx/glove_accept/glove_drop_rl` 与 `infer_last_ms/infer_ema_ms/infer_load`，若 `infer_load` 持续接近或超过 `1.0`，说明推理负载已接近控制周期上限。

---

## 8. wrist + five_finger 融合频率问题复盘（100Hz）

适用范围：

- `launch/wrist_five_finger_fusion_right_launch.py`
- `launch/wrist_five_finger_fusion_mapping_right_launch.py`
- `scripts/wrist_five_finger_fusion_control.py`
- `scripts/wrist_five_finger_fusion_mapping_main.py`

### 8.1 现象

典型现象是：

- 单独跑 wrist：`/yhand_right_cmd` 可到 `100Hz`
- 单独跑 five_finger：`/yhand_right_cmd` 可到 `100Hz`
- 融合后频率明显下降（例如 `10~30Hz`）

### 8.2 根因（本次定位结论）

导致融合频率低的主要原因不是拼接数学逻辑，而是控制链路调度方式：

- 链路拆分过多（跨 topic / 跨进程），增加同步点与调度抖动
- 订阅回调做了较重工作，阻塞新帧接收，输入有效频率被拖低
- 发布节拍受输入回调牵制，而不是固定控制周期驱动
- 启动阶段对 feedback 的阻塞等待会进一步拉低有效输出频率

### 8.3 已采用的稳定方案（不改数学逻辑）

当前融合控制改为与单 wrist 控制一致的“单节点固定调度”：

1. 单节点发布  
`wrist_five_finger_fusion_control.py` 的 `main()` 直接运行  
`WristFiveFingerFusionMappingNode`，不再走旧的双节点拼接框架。

2. 回调只缓存，控制循环处理  
`_raw_skeleton_cb()` 与 `_glove48_cb()` 仅缓存最新帧；  
非拇指 raw-skeleton q 解算在 `_control_timer_cb()` 中执行（`_update_non_thumb_q16`）。

3. 100Hz timer 固定发布  
由 `output_publish_rate_hz` 驱动控制循环；没有新帧时保持上一帧命令，不追历史帧。

4. feedback 未就绪不阻塞  
在 feedback 缺失/过期时按 idle tension 周期发布，避免启动卡住。

一句话：数学拼接逻辑保持不变，只把“回调重活”改为“控制循环重活”，并由固定周期统一发布。

### 8.4 关键日志字段判读（先看输入还是看输出）

融合日志重点看：

- `raw_hz`：raw skeleton 有效接收频率
- `glove_hz`：glove48 有效接收频率
- `step_hz` / `recompute_hz`：发生融合重算的频率
- `out_hz`：最终 `/yhand_right_cmd` 发布频率
- `infer_ema_ms` / `infer_load`：推理耗时与负载占比

快速判断规则：

1. `raw_hz/glove_hz` 先低：输入侧瓶颈（上游手套流或回调阻塞）
2. 输入高但 `step_hz` 低：计算侧瓶颈（融合计算或解算耗时）
3. `step_hz` 高但 `out_hz` 低：发布/驱动链路瓶颈
4. `infer_load >= 1.0` 持续出现：当前控制周期已超载

### 8.5 现场排查命令（建议顺序）

1. 看手套原始输入频率：

```bash
ros2 topic hz /glove_joint_data_raw_skeleton
```

2. 看融合输出频率：

```bash
ros2 topic hz /yhand_right_cmd
```

3. 同步观察融合日志中的  
`out_hz / step_hz / raw_hz / glove_hz / infer_ema_ms / infer_load`，先定位掉频发生在哪一层。

### 8.6 100Hz 推荐配置

- `output_publish_rate_hz = 100`
- `right_non_thumb_q_from_raw_skeleton = true`（需要非拇指 raw q 解算时）
- 若负载高可短时设 `right_non_thumb_q_from_raw_skeleton = false` 做 A/B 对比，确认瓶颈是否来自非拇指 raw q 解算

注意：

- `/glove_joint_data_raw_skeleton` 实测若只有 `~65Hz`，融合重算频率上限会受输入限制；
- 但只要控制循环固定 `100Hz` 且采用“无新帧保持上一帧”，`/yhand_right_cmd` 仍可稳定高频发布，跟手性主要由输入新帧率决定。
