# 左拇指电机反馈 m→q 状态估计

已增加[五指统一状态估计入口](README_left_hand_motor_to_q_estimator.md)，默认运行`ros2 launch yhand_mapping left_hand_motor_to_q_estimator_launch.py`；旧单拇指入口和输出话题保持兼容。运行五指节点前先退出单拇指估计节点，避免重复发布同一话题。

独立脚本为 `scripts/q_estimator/left_thumb_motor_to_q_estimator.py`，独立模型为 `config/left_thumb_motor_to_q_estimator.json`。使用 M4、M6、M7、M8、M14、M27 的实际位置反馈，直接估计左手 glove48 `[8:12]` 对应的四个绝对角。默认完整二次模型有 28 个特征：常数、6 个一次项、6 个平方项和 15 个交叉项；也保存线性基线供对比。

当前 q→m 模型、控制器、操作员零位和增益均不修改。此节点仅订阅反馈、发布状态，不启动 SDK、不向电机发送命令、不在线自训练。没有实测 q 标签时，仅靠实时电机反馈不能继续监督拟合；训练在 `fit` 子命令中离线完成。

## 直接运行

已有遥操或 SDK 驱动发布 `/yhand_left_feedback` 时，另开终端：

```bash
source install/setup.bash
python3 src/yhand_mapping/scripts/q_estimator/left_thumb_motor_to_q_estimator.py run
```

首次安装新增文件：

```bash
colcon build --packages-select yhand_mapping --symlink-install
source install/setup.bash
ros2 launch yhand_mapping left_thumb_motor_to_q_estimator_launch.py
```

输入消息为 `devices_pkg/msg/YHandMsg`，读取 `motors[mid-1].pos`，单位沿用现有驱动和采集数据的度。默认话题与当前五指控制器一致。若自己的驱动话题名是 `/ifeedback`：

```bash
python3 src/yhand_mapping/scripts/q_estimator/left_thumb_motor_to_q_estimator.py run --feedback-topic /ifeedback
```

`--ifeedback-topic` 是 `--feedback-topic` 的同义参数。驱动需要持续发布反馈；当前 SDK 驱动是在处理命令后发布反馈，单独启动估计器不会促使驱动发送反馈。

## 输出约定

| 话题 | 类型 | 内容 |
| --- | --- | --- |
| `/left_thumb_estimated_q` | `std_msgs/Float64MultiArray` | `[q0,q1,q2,q3]` 绝对角，度，保持采集手套列定义 |
| `/left_thumb_estimated_q_relative` | `std_msgs/Float64MultiArray` | 上述绝对角减 `[-3,42,40,-36]°` |
| `/left_thumb_estimated_joint_states` | `sensor_msgs/JointState` | 同一绝对角，转为弧度；名称 `left_thumb_q0`～`left_thumb_q3` |
| `/left_thumb_estimate_valid` | `std_msgs/Bool` | 当前反馈新鲜、选中电机无错误、计算有限且各电机与估计角均位于训练边界内 |
| `/left_thumb_estimate_status` | `std_msgs/String` | JSON 状态、接收序号、反馈年龄、估计角、超界电机或错误原因 |

默认最高发布频率 50 Hz；每个反馈接收序号最多发布一次角度，跳过积压帧，不重复给旧样本打新时间戳。`YHandMsg` 不含原始采样时间戳，JointState 的时间戳是本节点收到对应反馈的 ROS 时间。状态定期输出，也在状态切换时更新。

从未收到反馈、所选电机出现非有限位置或错误码、反馈超过 0.5 秒未更新时，`valid=false`，停止发布角度。恢复新鲜有效反馈后自动恢复。其他手指电机的错误不影响本拇指估计。

默认超出训练电机位置范围或估计角超出采集角范围时也暂停角度输出，状态中分别给出 `outside_motor_ids` 和 `outside_q_indices`。想观察外推数值，可显式加 `--allow-extrapolation`，或 launch 参数 `allow_extrapolation:=true`。此时超界数值仅作诊断，`valid` 仍为 false。输入不裁剪，输出不伪装为边界姿态。逐轴范围检查不能保证联合姿态、负载或预紧条件与训练一致，`valid=true` 也不是实机误差保证。

```bash
ros2 topic echo /left_thumb_estimated_q
ros2 topic echo /left_thumb_estimate_status
```

两个话题需分别在终端运行。此输出不写入 `/glove_joint_data`，也不自动代替控制器里的真实手套测量。

## 拟合与离线检查

仓库已附拟合好的二次模型。复算：

```bash
python3 src/yhand_mapping/scripts/q_estimator/left_thumb_motor_to_q_estimator.py fit
```

默认使用 `20261019_170304_left_collection` 的 8149 个正式 full 采样点，其中 3126 个超时点也保留；每个 result 配对为一个窗口平均样本。复用校验 point_log 与 point_id 的缓存读取器，`--refresh-cache` 可强制重读原始 Excel。`--collection-dir` 选择同结构拇指采集批次，`--output` 可另存模型。

先拟合标准化输入的自由截距线性／完整二次模型，再保存系数、训练范围、数据哈希、训练指标、按动作组合分层的五折验证和留一动作组合验证。验证时归一化参数与回归系数只从训练折计算；训练目标是实际手套角，输入不是电机命令位置。

本批二次模型留组合 MAE 约 `[0.55,0.89,2.09,1.40]°`；这些是相对采集手套标签的单批次平均点误差，不是动态跟踪或跨批次精度。实时使用时模型只加载一次。

单点计算不依赖 ROS：

```bash
python3 src/yhand_mapping/scripts/q_estimator/left_thumb_motor_to_q_estimator.py predict \
  --positions -100 30 100 40 -80 10
```

输入顺序固定为 M4、M6、M7、M8、M14、M27；单点预览会返回电机和估计角的超界列表，即使超界也显示数值。上述任意组合用于展示诊断：它的电机位置逐轴在范围内，但估计 q2 超出采集范围，不能用作有效反馈。

## 零位与使用范围

绝对输出模型直接学习采集时的编码器坐标与手套角，不使用 q→m 控制模型的增量展开，也不强制电机全 0 对应某个角。`[-3,42,40,-36]°` 只用于相对输出的减法；它在部分关节超出本批角度覆盖，不能用这个参考值把模型在零位附近的误差自动消除。

若电机重新预紧、编码器归零或机械装配改变，需要确认与训练批次的电机坐标一致。只有已知坐标偏移时，才用 `--motor-offset O4 O6 O7 O8 O14 O27`；模型输入为 `实际反馈-offset`。默认偏移为 0，不自动把启动帧当作训练零位。`--q-zero Q0 Q1 Q2 Q3` 仅调整相对输出。

应先同时记录估计 q 与独立实测 q，检查跨姿态、动态、正反向及不同预紧条件下的误差，再决定怎样用于闭环。松绳、绳索弹性、摩擦和接触载荷不一定能由静态位置独立辨识。
