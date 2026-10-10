# wrist_five_finger_hybrid_fusion_mapping 使用说明

当前电机编号见 [新腕带电机对应表](README_motor_mapping.md)。模型轴顺序按肌肉保留，不能按新编号排序。

对应文件：

- `scripts/wrist_five_finger_hybrid_fusion_mapping_main.py`
- `launch/wrist_five_finger_hybrid_fusion_right_launch.py`

这套链路是“手腕 + 五指”的单节点融合控制版本：

- 手腕部分沿用 `wrist_angle_motor_control.py` 的角度解算与 28 电机 wrist delta 逻辑
- 拇指使用 `thumb_prev_m_target_q_mlp_mapping_main.py` 的 `prev_m + target_q MLP`
- 食指 / 中指 / 无名指 / 小指使用 `curl_m0_m2_mlp_mapping_main.py` / `five_finger_ring_curl_thumb_prev_m_target_q_mapping_main.py` 这一套 `q0 + curl` 非拇指逻辑
- 最终在 28 电机命令层做融合输出

当前推荐配置已经切成：

- 手腕：`mocap_tf`
- 手指：继续使用 Manus 的 `/glove_joint_data` + `/glove_joint_data_raw_skeleton`

---

## 1. 当前逻辑

运行逻辑：

```text
wrist:
  manus raw quaternion 或 mocap TF quaternion -> alpha/beta -> wrist 28-motor delta

thumb:
  prev_m + target_q -> thumb MLP -> thumb target_m

index/middle/ring/little:
  raw q0/q1/q2/q3 或 glove20 q -> q0 + curl -> non-thumb MLP -> target_m

最后：
  手指输出 cmd + wrist delta -> 发布整手 28 电机命令
```

补充说明：

- 手腕专用电机直接由 wrist delta 控制
- 手指模型输出的是各自目标位，再叠加 wrist delta
- 当只有手腕更新、五指未更新时，脚本会在上一帧五指结果基础上只平移手腕部分，不重复跑整套手指推理
- 当前脚本支持 wrist-only 输入源切换：
  - `manus_raw_skeleton`
  - `mocap_tf`
- 推荐使用 `mocap_tf` 作为手腕输入，同时保留 Manus 手套 raw 数据做手指控制

---

## 2. 左右手支持情况

### 2.1 脚本

`scripts/wrist_five_finger_hybrid_fusion_mapping_main.py` 当前已经支持：

- `--hand right`
- `--hand left`

并且会随 `hand` 自动切换：

- `cmd_topic / feedback_topic`
- glove side code
- `glove48` 左右手切片
- 手腕电机 ID 集合
- 左右手各自的电机限位 / offset / 默认参数

### 2.2 launch

`launch/wrist_five_finger_hybrid_fusion_right_launch.py` 现在也已经支持：

- `target_hand:=right`
- `target_hand:=left`

它会根据 `target_hand` 自动：

- 选择 `config/right_hand_device_config.json` 或 `config/left_hand_device_config.json`
- 校验 config 中声明的 hand 是否匹配
- 切换 driver node name、`cmd_topic`、`feedback_topic`
- 切换 glove side code
- 把 controller 参数从旧的右手专用参数改为当前通用参数

说明：

- 文件名仍然保留 `..._right_launch.py`
- 但实际入口已经是左右手通用

---

## 3. 输入输出

输入话题：

- wrist mocap TF：`/tf`
- raw skeleton：`/glove_joint_data_raw_skeleton`
- glove48：`/glove_joint_data`
- hand feedback：
  - 右手：`/yhand_right_feedback`
  - 左手：`/yhand_left_feedback`

输出话题：

- 右手：`/yhand_right_cmd`
- 左手：`/yhand_left_cmd`

默认情况下，脚本会根据 `--hand` 自动选择对应的话题，不需要手动再传一次。

补充：

- 当 `wrist_input_source=mocap_tf` 时，手腕四元数来自 `mocap_main_stickman_ros2.py` 发布的 `/tf`
- Manus 手套节点仍然必须保留，因为五指还要继续使用：
  - `/glove_joint_data`
  - `/glove_joint_data_raw_skeleton`

---

## 4. 常用启动

### 4.1 直接启动脚本

先编译并 source：

```bash
cd /home/yhand-demo/Yhand_Mapping_2.0
colcon build --packages-up-to yhand_mapping
source install/setup.bash
```

右手：

```bash
python3 src/yhand_mapping/scripts/wrist_five_finger_hybrid_fusion_mapping_main.py --hand right
```

左手：

```bash
python3 src/yhand_mapping/scripts/wrist_five_finger_hybrid_fusion_mapping_main.py --hand left
```

如果 driver 和 glove 已经由其他 launch 启动，上面这条命令就够了。

### 4.2 使用 launch

右手：

```bash
ros2 launch yhand_mapping wrist_five_finger_hybrid_fusion_right_launch.py
```

左手：

```bash
ros2 launch yhand_mapping wrist_five_finger_hybrid_fusion_right_launch.py target_hand:=left
```

当前 launch 会默认同时启动：

- YHand driver
- Manus 手套发布节点
- mocap TF 发布节点 `mocap_main_stickman_ros2.py`
- hybrid fusion 控制节点

如果只想临时关闭 mocap，可以这样：

```bash
ros2 launch yhand_mapping wrist_five_finger_hybrid_fusion_right_launch.py \
  target_hand:=left \
  start_mocap_node:=false
```

### 4.3 启动手腕 / 不启动手腕

启动“手腕 + 手指”一起运行：

```bash
ros2 launch yhand_mapping wrist_five_finger_hybrid_fusion_right_launch.py target_hand:=left
```

只运行手指链路，不启动 mocap 手腕节点：

```bash
ros2 launch yhand_mapping wrist_five_finger_hybrid_fusion_right_launch.py \
  target_hand:=left \
  start_mocap_node:=false
```

说明：

- `start_mocap_node:=true`
  - 会启动 `mocap_main_stickman_ros2.py`
  - 手腕四元数来自 `/tf`
- `start_mocap_node:=false`
  - 不启动 mocap TF 发布器
  - 手指仍然继续使用 Manus 的 `/glove_joint_data` 和 `/glove_joint_data_raw_skeleton`

注意：

- 当前这个开关控制的是“是否启动 mocap 节点”
- 它不是“彻底关闭 hybrid 脚本里的 wrist 分支”
- 因此关闭 mocap 后，常见现象是：
  - `wrist_cb_hz=0.0`
  - `wrist_valid_hz=0.0`
  - 手指 `raw_cb_hz / glove48_hz` 仍然正常
- 如果后面需要“完全纯手指模式”，需要再额外增加单独的 `disable_wrist` 或等效逻辑开关

---

## 5. 左手指令

当前左手推荐指令有两种。

直接启动脚本：

```bash
cd /home/yhand-demo/Yhand_Mapping_2.0
source install/setup.bash
python3 src/yhand_mapping/scripts/wrist_five_finger_hybrid_fusion_mapping_main.py --hand left
```

或者直接用 launch：

```bash
cd /home/yhand-demo/Yhand_Mapping_2.0
source install/setup.bash
ros2 launch yhand_mapping wrist_five_finger_hybrid_fusion_right_launch.py target_hand:=left
```

如果你还没起 glove 发布和左手驱动，需要先分别保证：

- 有人在发 `/glove_joint_data_raw_skeleton`
- 有人在发 `/glove_joint_data`
- 有人在发 `/tf`
- 左手 driver 已连到：
  - `/yhand_left_feedback`
  - `/yhand_left_cmd`

---

## 6. 关键参数

| 参数 | 作用 |
| --- | --- |
| `--hand` | `left` 或 `right`，默认 `right` |
| `--wrist-input-source` | 手腕输入源：`manus_raw_skeleton` / `mocap_tf` |
| `--mocap-tf-topic` | mocap TF 话题，默认 `/tf` |
| `--mocap-hand-frame` | mocap 手腕 frame，左手默认 `LeftHand`，右手默认 `RightHand` |
| `--cmd-topic` | 手动指定输出命令话题；默认按 `hand` 自动取 |
| `--feedback-topic` | 手动指定反馈话题；默认按 `hand` 自动取 |
| `--glove-side-code` | 手套 side code；默认左手 `0.0`，右手 `1.0` |
| `--non-thumb-q-from-raw-skeleton` | 非拇指是否优先从 raw skeleton 解 `q0..q3` |
| `--no-non-thumb-q-from-raw-skeleton` | 关闭上面的行为，改为直接从 glove20 取非拇指 q |
| `--alpha-limit-deg` | 手腕 `alpha` 限位 |
| `--beta-limit-deg` | 手腕 `beta` 限位 |
| `--alpha-scale / --beta-scale` | 手腕角度比例 |
| `--alpha-offset-deg / --beta-offset-deg` | 手腕角度整体偏置 |
| `--motor-feedback-limits-left-json` | 左手电机限位 |
| `--motor-feedback-limits-right-json` | 右手电机限位 |
| `--q-offset-map-json` | q offset 配置 |
| `--motor-cmd-offset-map-json` | 电机命令 offset 配置 |
| `--device` | `auto/cpu/cuda` |
| `--dry-run` | 仅跑逻辑，不实际下发命令 |
| `--zero-on-exit` | 退出时是否执行回 0 |

---

## 7.1 launch 级别开关

`launch/wrist_five_finger_hybrid_fusion_right_launch.py` 当前常用开关：

| 参数 | 作用 |
| --- | --- |
| `target_hand:=left/right` | 选择左手或右手 |
| `start_driver:=true/false` | 是否启动 YHand 驱动 |
| `start_glove_node:=true/false` | 是否启动 Manus 手套发布节点 |
| `start_mocap_node:=true/false` | 是否启动 mocap TF 发布节点 |
| `start_controller:=true/false` | 是否启动 hybrid fusion 控制节点 |
| `dry_run:=true/false` | 仅跑逻辑，不实际下发命令 |

---

## 8. 运行模式

控制循环里主要有这几种模式：

- `idle_no_feedback`
  - 没有反馈或反馈超时，只发默认张紧
- `wrist_only_no_glove48`
  - 有手腕、没有 glove48，只做手腕控制
- `fusion`
  - glove48 有新数据，完整跑一次手腕 + 五指融合
- `wrist_shift_only`
  - 只有手腕变了，在上一帧五指结果基础上只平移 wrist delta
- `hold_last`
  - 没有新输入，保持上一帧目标

---

## 9. 当前频率现状

当前链路实测现象：

- Manus 手套发布节点本地轮询频率：`100Hz`
- Manus 实际收到并发布的：
  - `/glove_joint_data`
  - `/glove_joint_data_raw_skeleton`
  通常约为 `65Hz`
- hybrid fusion 控制命令输出频率：`100Hz`
- 如果 `mocap_main_stickman_ros2.py` 正常运行，手腕 mocap TF 频率通常高于 Manus 手套频率

这意味着：

- 手指输入是 `~65Hz`
- 控制输出仍然可以保持 `100Hz`
- 在没有新 glove 帧时，控制节点会进入 `hold_last` 或仅做手腕平移

---

## 10. 频率与排查

### 10.1 手套发布频率

`minimal_example_dual_stream_commented.py` 现在会打印：

```text
[glove_pub_rate] timer_hz=..., ergo_recv_hz=..., ergo_pub_hz=..., raw_recv_hz=..., raw_pub_hz=...
```

重点看：

- `timer_hz`
- `raw_recv_hz`
- `raw_pub_hz`

如果出现：

- `timer_hz≈100`
- `raw_recv_hz≈65`
- `raw_pub_hz≈65`

说明是 Manus 上游实际只给到了 `~65Hz`，不是 hybrid 控制节点把频率降下来的。

### 10.2 hybrid 控制频率

`wrist_five_finger_hybrid_fusion_mapping_main.py` 状态日志里重点看：

- `raw_cb_hz`
- `raw_valid_hz`
- `wrist_cb_hz`
- `wrist_valid_hz`
- `glove48_hz`
- `cmd_hz`

典型含义：

- `raw_cb_hz / glove48_hz`：手指输入频率
- `wrist_cb_hz / wrist_valid_hz`：mocap 手腕输入频率
- `cmd_hz`：最终控制输出频率

如果：

- `wrist_input_source=mocap_tf`
- 但 `wrist_cb_hz=0.0`

通常表示 mocap TF 节点没有启动，或者 `/tf` 中没有对应 `LeftHand / RightHand` frame。

---

## 11. 常见说明

1. 为什么文件名还是 `..._right_launch.py`，但却能起左手？

因为这次只把 launch 入口改成了左右手通用，文件名暂时没有改。

2. 脚本切到左手后，会不会还是发右手话题？

不会。脚本内部会根据 `--hand left` 自动切到左手默认话题。

3. 左手时非拇指模型是不是重新训练了一套？

当前脚本层面已经支持左右手通用；具体模型是否为镜像复用，取决于下层五指子逻辑和模型配置。

4. 为什么手指数据看起来只有 `~65Hz`？

因为当前 Manus 上游实际发送到 Linux 的 `/glove_joint_data` 和 `/glove_joint_data_raw_skeleton` 就是约 `65Hz`。hybrid 节点本身不会把 topic 发布频率改成 `65Hz`。

5. 为什么控制输出还能保持 `100Hz`？

因为 hybrid 控制循环本身按 `100Hz` 跑；在 glove 没有新帧的周期，会保持上一帧手指结果，并继续输出当前目标命令。
