# joint_angle_prev_m_target_q_mlp_mapping 使用说明

当前电机编号见 [新腕带电机对应表](README_motor_mapping.md)。模型轴顺序按肌肉保留，不能按新编号排序。

对应文件：

- `launch/joint_angle_prev_m_target_q_mlp_mapping_launch.py`
- `scripts/joint_angle_terminal_input.py`
- 底层映射复用：`launch/thumb_prev_m_target_q_mlp_mapping_launch.py`

这个运行链路是对 `thumb_prev_m_target_q_mlp_mapping` 的单指 4D 关节角封装。
它不会从 48D 手套数据里取整手 `glove20`，而是只接收一根手指的 4 个目标角度，
再把这 4 个角度映射成该手指的目标电机位置。

---

## 1. 功能概览

核心逻辑：

```text
input topic  = [q0, q1, q2, q3]   # 仅一根手指
enabled finger = target_finger
other fingers  = disabled
output         = target_m_0..
```

特点：

- 只控制一根手指
- 输入消息是 `std_msgs/msg/Float64MultiArray`
- 消息长度固定为 4
- 可选自动弹出一个终端手动输入四个角度
- 底层仍使用 `prev_m + target_q -> target_m` MLP
- 仍然支持左手/右手、电机限位、offset、mirror、driver 复用

---

## 2. 和 base launch 的关系

这份 launch 本质上做了三件事：

1. 包装并 include `thumb_prev_m_target_q_mlp_mapping_launch.py`
2. 只启用 `target_finger` 对应的一个模型，其他手指全部禁用
3. 把 `glove_topic` 改成一个 4D 关节角话题，默认 `/target_joint_angles`

因此：

- `target_finger` 决定唯一启用的手指
- `selected_model_ckpt` 会传给所有 `*_model_ckpt` 参数，但只有启用的那根手指实际使用
- 如果 `selected_model_ckpt` 为空，则回退到底层 base launch 的默认 ckpt

---

## 3. 输入格式

默认话题：

```text
/target_joint_angles
```

消息类型：

```text
std_msgs/msg/Float64MultiArray
```

消息内容：

```text
[q0, q1, q2, q3]
```

要求：

- 必须正好 4 个数
- 这 4 个数会被解释为 `target_finger` 的 4 个目标关节角
- 因为只有一根手指启用，所以脚本会把这 4 个值自动扩展到 hand-local glove20 的对应 `q_indices`
- 其他未启用手指不会参与推理

例如：

- `target_finger:=thumb` 时，4 个数写入 thumb 的 `q_indices`
- `target_finger:=index` 时，4 个数写入 index 的 `q_indices`

---

## 4. 常用启动

默认右手拇指，并自动打开输入终端：

```bash
ros2 launch yhand_mapping joint_angle_prev_m_target_q_mlp_mapping_launch.py
```

启动右手食指：

```bash
ros2 launch yhand_mapping joint_angle_prev_m_target_q_mlp_mapping_launch.py \
  target_hand:=right \
  target_finger:=index
```

启动左手中指：

```bash
ros2 launch yhand_mapping joint_angle_prev_m_target_q_mlp_mapping_launch.py \
  target_hand:=left \
  target_finger:=middle \
  selected_model_ckpt:=/path/to/middle_prev_m_target_q_mlp.pt
```

如果 driver 已经在跑：

```bash
ros2 launch yhand_mapping joint_angle_prev_m_target_q_mlp_mapping_launch.py \
  start_driver_node:=false
```

如果不想自动弹出输入终端，而是由你自己的节点发布 4D 角度：

```bash
ros2 launch yhand_mapping joint_angle_prev_m_target_q_mlp_mapping_launch.py \
  start_angle_input_terminal:=false \
  joint_angle_topic:=/my_joint_angles
```

指定当前手指模型：

```bash
ros2 launch yhand_mapping joint_angle_prev_m_target_q_mlp_mapping_launch.py \
  target_finger:=thumb \
  selected_model_ckpt:=/path/to/thumb_prev_m_target_q_mlp.pt
```

---

## 5. 交互终端

当 `start_angle_input_terminal:=true` 时，launch 会在约 2 秒后打开一个 `gnome-terminal`，
运行 `joint_angle_terminal_input.py`。

终端输入规则：

- 输入 `4` 个角度，空格或逗号分隔
- 输入 `zero` 或 `z`：发送 `[0, 0, 0, 0]`
- 输入 `q` / `quit` / `exit` / `x`：退出终端，并关闭整个 launch

示例：

```text
10 20 30 40
```

或：

```text
10,20,30,40
```

每次输入后不会一直连续发布，而是按短 burst 方式发送若干帧。

相关参数：

- `input_publish_hz`：burst 发送频率
- `input_publish_burst_count`：每次输入后发送多少帧

默认：

```text
input_publish_hz = 50.0
input_publish_burst_count = 5
```

---

## 6. 常用参数

| 参数 | 作用 |
| --- | --- |
| `target_hand` | `left` 或 `right` |
| `target_finger` | `thumb/index/middle/ring/little` |
| `joint_angle_topic` | 4D 关节角输入话题 |
| `selected_model_ckpt` | 当前手指使用的 checkpoint；空则走底层默认 |
| `start_driver_node` | 是否启动 `Y_Hand_Node` |
| `start_angle_input_terminal` | 是否打开交互输入终端 |
| `input_publish_hz` | 每次输入后的 burst 发布频率 |
| `input_publish_burst_count` | 每次输入后的 burst 帧数 |

注意：

- 这份 wrapper 当前**只显式暴露**上表这些 launch 参数
- 底层 base launch 里更多高级参数，例如 `device`、`require_ready`、电机限位 JSON、`*_model_motor_ids`、`*_q_indices`，当前没有继续向外转发
- 如果你需要这些高级配置，建议直接改这份 launch，或直接使用底层 `thumb_prev_m_target_q_mlp_mapping_launch.py`

---

## 7. 默认行为说明

默认值：

```text
target_hand = right
target_finger = thumb
joint_angle_topic = /target_joint_angles
start_driver_node = true
start_angle_input_terminal = true
```

注意：

- 这份 launch 不支持 `target_hand:=both`
- 这是单指输入链路，不是整手 glove48 链路
- `selected_model_ckpt` 对右手通常可以留空并使用底层默认模型
- `selected_model_ckpt` 对左手建议显式传入；否则会落到底层 base launch 的左手默认配置，而左手默认通常没有可直接使用的 ckpt
- 如果启用了输入终端，系统需要安装 `gnome-terminal`
- 如果 `gnome-terminal` 不存在且仍设置 `start_angle_input_terminal:=true`，launch 会直接报错

---

## 8. 运行时排查

如果启动后手指不动，优先检查：

1. 当前手指模型是否存在且可加载
2. `/target_joint_angles` 是否确实收到了 4 个数
3. `target_finger` 是否和模型语义一致
4. feedback 是否正常到达 `/yhand_*_feedback`

如果输入终端弹不出来，优先检查：

```bash
which gnome-terminal
```

如果你是自己发消息，可以用：

```bash
ros2 topic pub /target_joint_angles std_msgs/msg/Float64MultiArray "{data: [10.0, 20.0, 30.0, 40.0]}"
```

---

## 9. 和 thumb_prev_m_target_q_mlp_mapping_launch.py 的区别

`thumb_prev_m_target_q_mlp_mapping_launch.py`：

- 面向 glove 输入
- 可以单指，也可以多指
- 输入通常是 glove20 / glove48

当前这份 `joint_angle_prev_m_target_q_mlp_mapping_launch.py`：

- 面向一根手指的 4D 显式关节角输入
- 每次只启用一根手指
- 适合调模型、单指验证、手工输入测试
