# 左手五指 m→q 状态估计

`scripts/q_estimator/left_hand_motor_to_q_estimator.py` 在原单拇指估计器基础上扩展为五指。读取同一帧 `/yhand_left_feedback`，逐指用实际电机位置计算四个绝对手套角，拼接为20角。节点只订阅反馈、发布估计，不启动SDK、不发送电机命令、不修改现有遥操或q→m系数，也不自动闭合关节反馈环。

## 运行

先保持现有五指遥操运行，在另一终端启动估计器。先退出旧单拇指估计进程，避免它与五指节点重复发布拇指话题。

```bash
source install/setup.bash
ros2 launch yhand_mapping left_hand_motor_to_q_estimator_launch.py
```

## 遥操对比日志与分析

估计器现在默认记录日志，原命令即可启用线性模型和记录：

```bash
ros2 launch yhand_mapping left_hand_motor_to_q_estimator_launch.py model:=linear
```

为了记录真正采用的q_cmd，遥操入口新增 `/left_five_q_command_state`（单指／其他组合对应 `/left_<finger>_q_command_state`）只读遥测话题。它与原电机指令同时发布，包含控制器当前请求的绝对q、相对q、参考角、操作员零位、是否处于位控、控制状态、最终发布的28路电机位置目标及裁剪电机。q_cmd为电机裁剪前的请求，最终电机位置另行记录。此修改不改变映射系数或原电机控制帧。

**首次更新后，需要重新启动遥操入口一次**，让它发布该遥测，再启动估计器。只重启估计器不能让旧遥操进程新增话题。沿用原运行参数，例如：

```bash
source install/setup.bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py finger:=five input_mode:=2
```

然后在另一终端启动上面的线性估计命令。退出估计器时自动关闭并刷新日志，位置打印在终端，默认目录为：

```text
scripts/motor_test_logs/motor_to_q_estimator/YYYYMMDD_HHMMSS_microseconds/
  metadata.json
  samples.csv
  events.jsonl
  models/left_<finger>_motor_to_q_estimator.json
```

`samples.csv`每收到一帧电机反馈记录一行，包含：

- 控制器q_cmd绝对角、相对角与参考角；位控是否激活。
- 操作员手套左手角，来自 `/glove_joint_data` 的对应20角。
- 全部28台电机的位置、速度、电流、错误码，以及最终电机位置命令和估计器对齐后的输入。
- 四／二十个估计绝对角和相对角，逐指valid、超界原因。即使估计无效，诊断值仍记录；非有限或无法计算的位置留空。
- 各数据源的接收序号、年龄及对齐标记。默认使用本电机反馈之前最近接收到的q_cmd和手套，年龄≤0.1秒标记为对齐，不做未来插值。

`events.jsonl`另外保存q_cmd、完整48维手套和反馈／估计状态的独立原始事件；`models/`固定本次使用的模型和哈希，避免后续重新拟合后无法复现。时间戳为接收时间，不能视为硬件严格同步。操作员手套还利用raw skeleton的左手存在信息标记过期；没有raw话题时可显式设 `glove_raw_topic:=''` 仅检查角度消息年龄。

默认手套被标记为**操作员输入**，q_cmd也只表示期望角。它们与估计角的偏差用于检查映射一致性及跟踪，不直接等于真实关节角估计误差。只有另有机器人佩戴手套或等价独立测量，并指定 `robot_glove_topic:=/your_robot_glove_48`，才计算独立实测角误差；该话题必须与操作员输入不同。不要把操作者与机器人测量配置成同一来源。

常用记录参数：`log_dir:=/path/to/new_run`指定新目录，`log_enabled:=false`关闭记录，`q_cmd_topic:=/left_five_q_command_state`选择控制器遥测源，`glove_topic:=/glove_joint_data`选择操作员角度源，`log_max_age:=0.1`设置接收时间对齐阈值。单指估计器旁路监听五指遥操时，应显式将q_cmd_topic设为五指遥测话题。已有日志文件的目录拒绝覆盖。

采集结束后分析：

```bash
python3 src/yhand_mapping/scripts/q_estimator/analyze_motor_to_q_log.py /完整路径/日志目录
```

输出在该日志的 `analysis/comparison/`，含逐指角度叠图、电机指令／反馈／训练范围叠图、指标CSV、JSON与报告。角度图保留无效诊断点；报告分别列出全部对齐帧与有效子集，并明确是否存在独立机器人实测标签。

也支持原遥操日志的实际反馈回放：

```bash
python3 src/yhand_mapping/scripts/q_estimator/analyze_motor_to_q_log.py \
  --teleop-log /完整路径/five_q_to_motor_test/某次日志 --model linear
```

此模式按反馈接收序号去重，使用实际保存的电机反馈重新计算估计；它不会把旧日志伪装成本次估计器现场记录。历史模式的模型选择由`--model`指定，原生日志使用采集时的模型快照和已记录估计。

直接使用Python：

```bash
python3 src/yhand_mapping/scripts/q_estimator/left_hand_motor_to_q_estimator.py run
```

单指也使用同一入口，例如 `finger:=index`，或Python参数 `--finger index`。`feedback_topic:=/ifeedback` 可切换实际反馈话题。无需运行第二个驱动或连接操作员手套作为估计输入。

## 输出

| 手指 | 输入电机位置顺序 | 20角数组位置 | 训练配对数 | 默认模型 |
| --- | --- | --- | --- | --- |
| 拇指 | M4/M6/M7/M8/M14/M27 | `[0:4]` | 8149 | 完整二次 |
| 食指 | M25/M3/M15/M1/M28 | `[4:8]` | 5606 | 完整二次 |
| 中指 | M23/M16/M2/M9/M24 | `[8:12]` | 5889 | 完整二次 |
| 无名指 | M10/M17/M22/M21 | `[12:16]` | 2670 | 完整二次 |
| 小指 | M11/M13/M18/M20 | `[16:20]` | 2227 | 完整二次 |

腕部M5/M12/M19/M26不参与。绝对角的定义沿用原始glove48 `[8:28]`，不是经过新标定的解剖角。

- `/left_hand_estimated_q`：20个绝对角，`Float64MultiArray`，度。
- `/left_hand_estimated_q_relative`：20个相对角，度，按各模型参考相减。
- `/left_hand_estimated_joint_states`：20个绝对角，`JointState`，弧度，名称为 `left_<finger>_q0`～`q3`。
- `/left_hand_estimate_valid`：全部五指当前均满足质量检查。
- `/left_hand_estimate_status`：JSON，含 `finger_order`、各指状态、绝对／相对诊断角、电机位置、超界原因、接收序号与反馈年龄。

每指另外发布 `/left_<finger>_estimated_q`、`/left_<finger>_estimated_q_relative`、`/left_<finger>_estimated_joint_states` 和 `/left_<finger>_estimate_valid`。原拇指四个输出话题保持兼容。

某一指发生错误码、非有限输入或超出该指训练范围时，默认暂停该指角度输出，其他有效手指继续输出。此时完整20角向量暂停发布，全手 `valid=false`；不会把旧值、0或NaN填入坏指位置拼成完整向量。反馈全帧长度不正确或断流超过0.5秒时，五指均失效。重新收到有效帧后恢复。最高输出50Hz，每个反馈接收序号最多发布一次角度；不重复给旧样本打新时间戳。JointState时间戳为本节点收到反馈的时间，因为YHandMsg没有采样时间戳。

`allow_extrapolation:=true` 或 `--allow-extrapolation` 可以显示超出范围的诊断角；`valid`仍为false。错误码、非有限值和断流时仍不发布。超界角不裁剪成100°或其他数值。

```bash
ros2 topic echo /left_hand_estimate_status
```

## 训练与模型

每指独立拟合自由截距线性和完整二次模型，输入为该指所有电机的实际位置，输出为四个实测手套角。线性与二次的输入中心／尺度分别从训练折计算；以留一动作组合的四角平均MAE选取默认模型。此批五指均选择二次。没有把正向模型斜率反解或将遥操目标角当作实测标签。

```bash
python3 src/yhand_mapping/scripts/q_estimator/left_hand_motor_to_q_estimator.py fit
```

按下列真实数据目录加载，正式full/xy点保留运动超时点，排除单轴预采。`--finger ring`可仅重拟合一指，`--collection-dir`为单指选择同结构采集目录，`--model-dir`选择模型输出目录。

- 拇指：`data_collection/20261019_170304_left_collection`，正式full点，沿用此前拇指数据。
- 食指：`data_collection/left_hand/index`，正式xy点。
- 中指：`data_collection/middle_data/left_hand/middle`，正式xy点。
- 无名指：`data_collection/ring_data/left_hand/ring`，全部2670个正式xy点，包括此前侧摆分析排除的最后500个密集点。四路反馈来自同一批次；没有混入旧伸屈模型的其他批次。
- 小指：`data_collection/little_data/left_hand/little`，全部2227个正式xy点，未使用只覆盖前147点的旧审计快照代替全量数据。

缓存读取校验point_id与日志；存在日志快照时也比较完整日志。数据源哈希、范围、归一化参数、系数、设计矩阵秩、五折分层验证和留组合验证均保存在独立的 `config/left_<finger>_motor_to_q_estimator.json`。`--refresh-cache`强制重新读取原始配对Excel，缓存只保存在采集目录的analysis中。

本批默认模型留一动作组合的MAE（按q0/q1/q2/q3，度）：

| 手指 | q0 | q1 | q2 | q3 |
| --- | ---: | ---: | ---: | ---: |
| 拇指 | 0.55 | 0.89 | 2.09 | 1.40 |
| 食指 | 1.44 | 4.23 | 7.34 | 3.29 |
| 中指 | 0.62 | 1.35 | 2.94 | 1.95 |
| 无名指 | 0.68 | 0.82 | 10.19 | 15.54 |
| 小指 | 1.38 | 1.99 | 4.62 | 4.20 |

无名指后两角误差明显，当前应作为观察估计使用。所有数字均为同采集批次的窗口平均点验证，不是当前遥操工况下的实机精度。本次扩展不解决此前拇指出现214°的失真，不自动对齐编码器零位、预紧状态或遥操模型的增量坐标。

## 参考角与坐标

相对输出默认参考：

| 手指 | 参考角/度 | 来源 |
| --- | --- | --- |
| 拇指 | `[-3,42,40,-36]` | 用户提供的电机零位，仅用于相减 |
| 食指 | `[3.627,-6.019,0.028,-2.929]` | 原采集启动帧，未确认精确电机零位 |
| 中指 | `[4.396,-8.581,-0.02,0.443]` | 原采集启动帧，未确认精确电机零位 |
| 无名指 | `[-2,-6,0,-12]` | 当前配置中的用户参考 |
| 小指 | `[-5,-9,7,-10]` | 当前配置中的用户参考 |

参考只影响相对输出，不约束回归在电机全0时的绝对角。`--q-zero`／launch `q_zero`可显式指定单指4角或五指20角；`--motor-offset`／launch `motor_offset`可指定单指各电机或五指24路的已知坐标偏移，输入为实际位置减去偏移。默认偏移全0。不要把正向模型的拟合截距自动当作编码器偏移。

范围内仅表示当前通过逐轴范围检查，不证明联合姿态或工况与训练一致。小指的正向线性模型秩不足，独立反向回归也可能依赖采集数据中的关节耦合，不能据此宣称任意外载荷下四角均可唯一辨识。接入闭环前仍需用机器手独立实测q验证；操作员期望角不作为真实关节反馈。
