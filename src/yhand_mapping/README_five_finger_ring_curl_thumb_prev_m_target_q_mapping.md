# five_finger_ring_curl_thumb_prev_m_target_q_mapping 使用说明

当前电机编号见 [新腕带电机对应表](README_motor_mapping.md)。模型轴顺序按肌肉保留，不能按新编号排序。

对应文件：

- `scripts/five_finger_ring_curl_thumb_prev_m_target_q_mapping_main.py`
- `launch/five_finger_ring_curl_thumb_prev_m_target_q_mapping_launch.py`

这套链路是一个“五指融合单节点发布”版本：

- 拇指使用 `thumb_prev_m_target_q_mlp_mapping_main.py` 的 `prev_m + target_q MLP`
- 食指 / 中指 / 无名指 / 小指使用 `curl_m0_m2_mlp_mapping_main.py` 的 `q0 + curl multihead MLP`

这样做的目的，是避免把两个独立脚本同时发到同一个 `/yhand_*_cmd` 话题时互相覆盖。

---

## 1. 功能概览

单个 ROS 节点同时完成 5 指推理，并最终只发布一条 28 电机整手命令。

运行逻辑：

```text
thumb:
  history_m + target_q -> prev_m_target_q MLP -> target_m

index/middle/ring/little:
  q0/q1/q2/q3 -> supervised curl
  q0 + curl -> q0_curl_multihead MLP -> target_m

最后：
  5 指结果拼成一条 28 电机命令发布
```

特点：

- 单节点、整手发布，不会有多节点互抢命令的问题
- 支持右手直接控制
- 支持左手使用右手模型镜像控制
- 支持从 raw skeleton 解算非拇指 `q0..q3`
- 支持保留 `thumb` 旧参数别名

---

## 2. 默认模型

默认右手模型路径：

- `thumb`: `scripts/mp_curve_q0_right/thumb/best_thumb_prev_m_target_q_mlp.pt`
- `index`: `scripts/mp_curve_q0_right/index/best_index_q0_curl_multihead_mlp.pt`
- `middle`: `scripts/mp_curve_q0_right/middle/best_middle_q0_curl_multihead_mlp.pt`
- `ring`: `scripts/mp_curve_q0_right/ring/best_ring_q0_curl_multihead_mlp.pt`
- `little`: `scripts/mp_curve_q0_right/little/best_little_q0_curl_multihead_mlp.pt`

默认非拇指 curl meta：

- `index`: `scripts/curve_function/index/index_target_q0_supervised_curl_1570.meta.json`
- `middle`: `scripts/curve_function/middle/middle_target_q0_supervised_curl_1357.meta.json`
- `ring`: `scripts/curve_function/ring/ring_target_q0_supervised_curl_1704.meta.json`
- `little`: `scripts/curve_function/little/little_target_q0_supervised_curl_1320.meta.json`

---

## 3. 默认启用规则

右手默认：

- `thumb`: 启用
- `index/middle/ring/little`: 启用

左手默认：

- `thumb`: 使用右手 thumb 默认模型，并自动镜像到左手
- `index/middle/ring/little`: 使用右手非拇指默认模型，并自动镜像到左手

也就是说，当前这套 5 指融合版：

- `target_hand:=right` 时默认就是右手 5 指
- `target_hand:=left` 时默认就是左手 5 指镜像

---

## 4. 常用启动

默认右手 5 指：

```bash
ros2 launch yhand_mapping five_finger_ring_curl_thumb_prev_m_target_q_mapping_launch.py
```

左手 5 指镜像：

```bash
ros2 launch yhand_mapping five_finger_ring_curl_thumb_prev_m_target_q_mapping_launch.py \
  target_hand:=left
```

如果 driver 和 glove 已经在运行：

```bash
ros2 launch yhand_mapping five_finger_ring_curl_thumb_prev_m_target_q_mapping_launch.py \
  target_hand:=left \
  start_driver_node:=false \
  start_glove_node:=false
```

只关掉某根非拇指：

```bash
ros2 launch yhand_mapping five_finger_ring_curl_thumb_prev_m_target_q_mapping_launch.py \
  enable_ring:=false
```

只换某根手指的模型：

```bash
ros2 launch yhand_mapping five_finger_ring_curl_thumb_prev_m_target_q_mapping_launch.py \
  index_model_ckpt:=/path/to/another_index_q0_curl_multihead_mlp.pt
```

关闭非拇指 raw skeleton q 替换：

```bash
ros2 launch yhand_mapping five_finger_ring_curl_thumb_prev_m_target_q_mapping_launch.py \
  right_non_thumb_q_from_raw_skeleton:=false
```

说明：

- 参数名虽然叫 `right_non_thumb_q_from_raw_skeleton`，但在当前融合脚本里，实际表示“对当前选中的 hand，非拇指是否从 raw skeleton 解 q”
- 只是名字沿用了旧版本

---

## 5. 输入输出

输入话题：

- glove：`/glove_joint_data`
- raw skeleton：`/glove_joint_data_raw_skeleton`
- hand feedback：
  - 左手：`/yhand_left_feedback`
  - 右手：`/yhand_right_feedback`

输出话题：

- 左手：`/yhand_left_cmd`
- 右手：`/yhand_right_cmd`

---

## 6. 左手镜像规则

### 6.1 拇指

左手拇指默认行为：

- 默认使用右手 thumb checkpoint
- 默认使用右手 thumb 的 `q_indices`
- 默认使用右手 thumb 的 motor 顺序
- 运行时自动镜像到左手 motor id

### 6.2 非拇指

左手非拇指默认行为：

- 默认使用右手非拇指 checkpoint
- 默认使用右手非拇指 curl meta
- 默认使用右手非拇指 motor 顺序
- 运行时自动镜像到左手 motor id
- `q0` 在左手镜像模式下会按原四指脚本逻辑取反

如果你要手动覆盖镜像：

```bash
ros2 launch yhand_mapping five_finger_ring_curl_thumb_prev_m_target_q_mapping_launch.py \
  target_hand:=left \
  thumb_mirror_right_from_left:=true \
  index_mirror_left_from_right:=true \
  middle_mirror_left_from_right:=true \
  ring_mirror_left_from_right:=true \
  little_mirror_left_from_right:=true
```

通常不需要显式传，因为默认就已经这样做了。

---

## 7. 关键参数

| 参数 | 作用 |
| --- | --- |
| `target_hand` | `left` 或 `right` |
| `start_driver_node` | 是否启动 `Y_Hand_Node` |
| `start_glove_node` | 是否启动 glove publisher |
| `glove_topic` | glove 数据话题，默认 `/glove_joint_data` |
| `glove_raw_skeleton_topic` | raw skeleton 话题 |
| `enable_thumb` | 是否启用拇指 |
| `enable_index/middle/ring/little` | 是否启用非拇指 |
| `thumb_model_ckpt` | 拇指 checkpoint |
| `index/middle/ring/little_model_ckpt` | 非拇指 checkpoint |
| `index/middle/ring/little_curl_meta` | 非拇指 curl meta |
| `*_model_motor_ids` | 某根手指对应的 motor 顺序 |
| `*_q_indices` | 某根手指对应的 glove20 q 顺序 |
| `thumb_mirror_right_from_left` | 拇指镜像开关 |
| `index/middle/ring/little_mirror_left_from_right` | 非拇指镜像开关 |
| `right_non_thumb_q_from_raw_skeleton` | 非拇指是否从 raw skeleton 解 q |
| `control_period_sec` | 发布周期 |
| `status_period_sec` | 状态日志周期 |
| `glove_max_hz` | glove 最大接收频率 |
| `device` | `auto/cpu/cuda` |
| `require_ready` | 启动时是否等待就绪 |

拇指旧参数别名仍可用：

- `enable_model` -> `enable_thumb`
- `model_ckpt` -> `thumb_model_ckpt`
- `model_motor_ids` -> `thumb_model_motor_ids`
- `q_indices` -> `thumb_q_indices`
- `mirror_right_from_left` -> `thumb_mirror_right_from_left`

---

## 8. 运行时日志

启动后会打印：

- 当前 hand
- thumb 是否启用
- 非拇指启用了哪些 finger
- 每根 finger 的 checkpoint、motor ids、q indices

状态日志里会打印：

- `feedback_ready`
- `glove_ready`
- `raw_q_ready`
- `angles`
- `active_diag`

---

## 9. 依赖文件与迁移清单

如果只是想在新电脑上启用
`scripts/five_finger_ring_curl_thumb_prev_m_target_q_mapping_main.py`
这套 5 指融合映射需要同时迁移 `src/devices_pkg` 和 `src/yhand_mapping` 两个包，然后重新 `colcon build`。

如果只想迁移最小必需文件，至少需要下面这些。

### 9.1 直接运行主脚本的最小集合

主脚本和直接依赖脚本：

- `scripts/five_finger_ring_curl_thumb_prev_m_target_q_mapping_main.py`
- `scripts/curl_m0_m2_mlp_mapping_main.py`
- `scripts/thumb_prev_m_target_q_mlp_mapping_main.py`
- `scripts/data_collection_config.py`
- `scripts/workspace_boundary_config.py`
- `scripts/muscle_adjustment_defaults.py`
- `scripts/replace_right_non_thumb_q_from_skeleton.py`

说明：

- 当前 5 指融合脚本默认启用了 `right_non_thumb_q_from_raw_skeleton`
- 所以 `replace_right_non_thumb_q_from_skeleton.py` 也是实际运行依赖

模型文件：

- `scripts/mp_curve_q0_right/thumb/best_thumb_prev_m_target_q_mlp.pt`
- `scripts/mp_curve_q0_right/index/best_index_q0_curl_multihead_mlp.pt`
- `scripts/mp_curve_q0_right/middle/best_middle_q0_curl_multihead_mlp.pt`
- `scripts/mp_curve_q0_right/ring/best_ring_q0_curl_multihead_mlp.pt`
- `scripts/mp_curve_q0_right/little/best_little_q0_curl_multihead_mlp.pt`

非拇指 curl meta 文件：

- `scripts/curve_function/index/index_target_q0_supervised_curl_1570.meta.json`
- `scripts/curve_function/middle/middle_target_q0_supervised_curl_1357.meta.json`
- `scripts/curve_function/ring/ring_target_q0_supervised_curl_1704.meta.json`
- `scripts/curve_function/little/little_target_q0_supervised_curl_1320.meta.json`

### 9.2 如果还要用 launch 一键启动

除了上面的文件，再加上：

- `launch/five_finger_ring_curl_thumb_prev_m_target_q_mapping_launch.py`
- `config/left_hand_device_config.json` / `config/right_hand_device_config.json`
- `scripts/minimal_example_dual_stream_commented.py`

说明：

- `five_finger_ring_curl_thumb_prev_m_target_q_mapping_launch.py` 会读取左右手 JSON 配置
  里的设备连接参数
- 如果 `start_glove_node:=true`，launch 还会启动
  `minimal_example_dual_stream_commented.py`

### 9.3 ROS 包与环境依赖

新电脑上除了文件本身，还需要这些运行环境：

- ROS 2 Python 环境，可导入 `rclpy` 和 `std_msgs`
- Python 依赖：`numpy`、`torch`
- `devices_pkg.msg` 已经可导入
- 如果要通过 launch 同时启动手部驱动，还需要 `devices_pkg` 里的 `Y_Hand_Node`

说明：

- `devices_pkg.msg` 可导入，意味着 `Motor.msg`、`YHandMsg.msg` 所在 ROS 包需要正常编译
- 如果只是跑映射主脚本，但不启动真实手部驱动，也至少需要消息类型已经 build 成功

### 9.4 推荐迁移方式

推荐优先使用下面的方式迁移：

- 同时迁移 `src/devices_pkg` 和 `src/yhand_mapping`
- 在新电脑重新执行 `colcon build`
- 然后再运行 `ros2 launch yhand_mapping five_finger_ring_curl_thumb_prev_m_target_q_mapping_launch.py`

这样最不容易漏掉 `msg`、launch 配置、默认参数文件和模型相对路径依赖。

其中 `active_diag` 会显示每个被控电机的：

- `target_pos`
- `published_pos`
- `feedback_pos`
- `torque`
- `pure_tension`

---

## 9. 常见问题

### 9.1 左手启动后拇指不动

优先检查：

- 是否使用了最新的融合脚本
- 左手启动时是否仍然被旧参数覆盖掉了 thumb 模型或镜像参数
- 启动日志里 `thumb_enabled=True` 是否出现
- 启动日志里 `thumb_motor_ids` 是否已经变成左手顺序 `6,4,27,7,8,14`

### 9.2 两个脚本同时发同一只手，动作互相覆盖

不要把：

- `thumb_prev_m_target_q_mlp_mapping_launch.py`
- `curl_m0_m2_mlp_mapping_launch.py`

同时发到同一只手的 `/yhand_*_cmd`。

如果要 5 指一起跑，使用本 README 对应的融合版 launch。

### 9.3 `AttributeError` 找不到上游脚本常量

这通常表示：

- `thumb_prev_m_target_q_mlp_mapping_main.py`
- `curl_m0_m2_mlp_mapping_main.py`

已经更新了，但融合脚本还没同步。

如果你近期改了上游脚本，建议同步检查：

- 默认常量名
- 镜像参数名
- 非拇指后处理逻辑
- 发布前 target/cmd 矩阵逻辑

### 9.4 raw skeleton q 一直 not ready

检查：

- `/glove_joint_data_raw_skeleton` 是否真的在发
- 数据维度是否是 `176` 或 `352`
- packet 里是否包含当前 hand 的 chunk

---

## 10. 推荐命令

右手 5 指：

```bash
ros2 launch yhand_mapping five_finger_ring_curl_thumb_prev_m_target_q_mapping_launch.py
```

左手 5 指镜像：

```bash
ros2 launch yhand_mapping five_finger_ring_curl_thumb_prev_m_target_q_mapping_launch.py \
  target_hand:=left
```

左手 5 指镜像，不重复起 driver 和 glove：

```bash
ros2 launch yhand_mapping five_finger_ring_curl_thumb_prev_m_target_q_mapping_launch.py \
  target_hand:=left \
  start_driver_node:=false \
  start_glove_node:=false
```
