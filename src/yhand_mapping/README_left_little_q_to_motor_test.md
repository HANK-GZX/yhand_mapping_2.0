# 左手小指/无名指q→m实机测试

入口：`scripts/left_little_q_to_motor_test.py`。通过现有`devices_pkg/Y_Hand_Node`调用官方SDK，同一个脚本和launch支持选择左小指或无名指。每次可选择单指，或以`finger:=both`同时控制无名指和小指。单指模式其余24路、双指模式其余20路保持默认张紧电流。

| `finger` | 手指 | 伸肌 | 屈肌 | 内收 | 外展 | Manus角度索引 | 机器人零位q/° |
|---|---|---|---|---|---|---|---|
| `little`（默认） | 小指 | M11 | M13 | M18 | M20 | `[24:28]` | `[-5,-9,7,-10]` |
| `ring` | 无名指 | M10 | M17 | M22 | M21 | `[20:24]` | `[-2,-6,0,-12]` |

同时遥操无名指和小指（Manus）：

```bash
source install/setup.bash
ros2 launch yhand_mapping left_little_q_to_motor_test_launch.py finger:=both input_mode:=2
```

先结束原来的单指遥操，再运行这条指令。双指模式只有一个控制节点和一个SDK驱动，将两指8台电机合并到同一28路指令中；不能分别启动两个单指控制进程共用指令话题，否则它们的完整指令帧会互相覆盖，发布者检查也会拒绝运行。

两指使用各自现有模型和电机限位。无名指M10/M17固定前馈0A，M21/M22保留KP/KD=800/100和±0.05A方向偏置；小指M11/M13/M18/M20保留−0.08A前馈和原增益。默认`current`留空才按两指分别取值；显式设置`current:=数值`会统一覆盖两指固定前馈，无名指侧摆仍由方向偏置覆盖。

启动后两指保持零位姿态；8个角同时连续稳定满默认1秒、每个角极差≤3°，才锁定各自Manus参考，随后按各自Δq控制。反馈或左手Manus失效会一起停止8台电机，其余20路继续默认张紧。

直接脚本也支持`--finger both`。双指手动输入需要8个角，顺序为“无名指q0/q1/q2/q3、小指q0/q1/q2/q3”；launch现有输入终端只支持4角，所以双指推荐Manus模式。若需要双指手动话题，使用`input_mode:=1 start_angle_input_terminal:=false`，向默认`/left_both_target_q`发送8维数组。

双指日志保存在`scripts/motor_test_logs/both_q_to_motor_test/<时间>/`。CSV中q0–q3对应无名指，q4–q7对应小指；metadata中的`finger_q_slices`和`model_files`保存角度分组及两份模型路径。

两指伸屈均使用全部四个Δq。小指侧摆仍只使用Δq0；无名指侧摆M22/M21改为四个Δq的一次补偿，四台电机同时位控，不再执行启动放线2秒。默认线性模型；选择二次模型时，仅伸屈按原二次模型计算，无名指侧摆仍使用同一四q一次补偿。手动和Manus流程保持相对输入、50Hz直接目标和默认KP/KD。无名指伸屈默认前馈0A，侧摆按运动方向使用±0.05A偏置；小指保持−0.08A。

无名指侧摆斜率取自新批次`20261021_171709_left_collection`的2170个正式xy点，排除43个单电机预采和最后500个密集补采点。为保持“相对q全0→电机目标全0”，使用 `m(Δq)=f(Δq)−f(0)`：

```text
M22 = -1.27305 Δq0 +0.65828 Δq1 -0.07069 Δq2 -0.08362 Δq3
M21 = +1.36390 Δq0 -0.34749 Δq1 +0.04464 Δq2 -0.00860 Δq3
```

这是对带截距模型的零位对齐，不是强制过原点重新拟合。此前带截距模型的留出MAE不直接代表零位对齐后的实机控制精度。本批精确零位尚未重新测定；实际效果需要本机验证。M10/M17保留原模型系数，小指模型不变。

无名指Manus遥操：

```bash
source install/setup.bash
ros2 launch yhand_mapping left_little_q_to_motor_test_launch.py finger:=ring input_mode:=2
```

无名指手动输入：

```bash
ros2 launch yhand_mapping left_little_q_to_motor_test_launch.py finger:=ring input_mode:=1
```

小指使用`finger:=little`或省略此参数。两种手指的直接Python入口均支持`--finger`。

无名指完整系数与分电机来源保存在`config/left_ring_q_to_motor_models.json`；M10/M17来自旧批4307个正式点，M22/M21来自上述新批2170点。四q与q0的离线对比见[四q侧摆分析](scripts/motor_test_logs/data_collection/20261021_171709_left_collection/analysis/ring_distribution/SIDE_FOUR_Q.md)。未执行新的物理标定。

下列未指定`finger`的示例均选择小指。

M18/M20仅使用q0，M11/M13使用全部四q。默认线性模型；`--model quadratic`切换二次模型。输入支持手动相对角度和左手Manus实时相对角度两种模式。

## 启动

从工作空间根目录运行，先source已编译的工作空间：

```bash
source install/setup.bash
ros2 launch yhand_mapping left_little_q_to_motor_test_launch.py input_mode:=1
```

模式1（默认，`manual`也可）会自动启动SDK控制程序，并打开四个q的输入终端；默认输入相对角度，默认线性模型。输入`q`退出窗口会关闭整个测试，启动launch的终端Ctrl+C也会先停止小指再关闭SDK。输入窗口支持`zero`发送四个零角度。

模式2（`manus`也可）使用左手Manus遥操，不打开角度输入窗口：

```bash
source install/setup.bash
ros2 launch yhand_mapping left_little_q_to_motor_test_launch.py input_mode:=2
```

先完成机器手的预紧归零，让电机0对应约定的机器人零位姿态。启动后将操作员左手套保持在希望作为零位的姿态，SDK反馈就绪后连续采样1秒均值作为`q_manus_zero`，终端提示零位已记录后再运动：

```text
Δq = q_manus_current − q_manus_zero
电机目标 = 小指映射(Δq)
```

操作员手套零位无需等于机器手的`[-5,-9,7,-10]°`；这两者分别定义各自的参考姿态。相对变化为0时，四台电机映射目标为0。零位只在启动时记录一次，之后不会跟随手套变化更新，也不会调用SDK `Set_Zero`。无名指另外要求零位采样期间四个q各自最大值−最小值≤3°；检测到运动会重新计时，在记录零位前本指四路保持KP/KD/I=0。请伸直无名指后保持静止，直到终端提示“Manus零位已记录”；超过ready-timeout仍不稳定则退出，不能用运动过程的均值作零位。小指保持原来的采样规则。

Manus模式复用`wrist_five_finger_hybrid_fusion_right_launch.py`中的`minimal_example_dual_stream_commented.py`双流发布节点，默认自动启动。Windows端Manus SDK需保持运行；如果ROS手套节点已经启动，使用：

```bash
ros2 launch yhand_mapping left_little_q_to_motor_test_launch.py \
  input_mode:=2 start_glove_node:=false
```

角度取`/glove_joint_data[24:28]`，与当前模型训练数据定义一致；`/glove_joint_data_raw_skeleton`仅用于确认左手在线，不改用骨架推算的另一套角度。只收到右手raw不能标定或刷新左手输入。角度或左手raw超过0.5秒未更新、或角度出现非有限值时，退出并停止小指输出。四个关节角不限位；四台电机的映射结果按下述位置范围分别裁剪，超出时发送边界值并继续更新。

保持50Hz发送最新映射目标，不插值、不增加限速。其余24路始终维持默认张紧电流。重新记录手套参考姿态时，退出后重新启动Manus模式。

| launch参数 | 默认值 | 含义 |
|---|---|---|
| `finger` | `little` | `little`小指；`ring`无名指；`both`两指同时控制，自动选择模型和Manus角度列 |
| `input_mode` | `1` | `1`/`manual`手动；`2`/`manus`左手套遥操 |
| `start_glove_node` | `true` | 仅Manus模式生效；启动双流发布节点 |
| `manus_topic` | `/glove_joint_data` | 48维Manus角度话题 |
| `manus_raw_topic` | `/glove_joint_data_raw_skeleton` | 左手在线检测话题 |
| `manus_zero_seconds` | `1.0` | 启动零位采样时长秒 |
| `manus_timeout` | `0.5` | 左手角度和raw最大允许年龄秒 |

自定义话题时，应确保发布节点也向对应话题发布；自动启动的现有发布节点使用上述默认话题。

launch的`current`默认留空，无名指伸屈默认0A、小指默认−0.08A；显式设置`current:=数值`可覆盖固定前馈。无名指侧摆始终采用下述方向偏置，覆盖固定前馈，不与`current`叠加。实际电流仍包含KP/KD对位置/速度误差产生的控制输出。

可选参数：`model:=quadratic`使用二次模型，手动模式下`relative:=false`输入原始glove角度，`start_driver_node:=false`复用已经启动的SDK。手动模式默认目标q话题是`/left_little_target_q`，与机器人手套实测话题分开。Manus模式始终使用相对变化。

首次新增launch后需要构建一次：

```bash
colcon build --packages-select yhand_mapping --symlink-install
source install/setup.bash
```

原来的直接Python交互方式仍可使用，支持`stop`、`status`：

```bash
python3 src/yhand_mapping/scripts/left_little_q_to_motor_test.py --relative
```

默认自动启动左手SDK驱动，连接信息读取`config/left_hand_device_config.json`。脚本先发送完整28路纯电流帧以获取反馈；小指四路为0电流，其他24路为默认张紧电流。SDK收到指令才返回反馈，因此启动不是被动等待。

收到反馈后，终端输入四个相对角度，例如：

```text
0 0 0 0
0 10 20 5
```

每次输入后持续保持该目标，可以继续输入下一个目标。支持空格、逗号、分号分隔。

| 输入 | 行为 |
|---|---|
| 四个数 | 计算四台电机位置，在下一个发送周期直接下发新目标 |
| `status` | 打印当前小指电机位置和运行状态 |
| `stop` | 小指四路KP/KD/I=0；其他24路仍为默认电流 |
| `quit` / Ctrl+C | 先发送0.5秒停止小指输出的指令，再关闭本脚本启动的SDK |

上表的`stop`和`status`是直接Python交互方式的命令；launch打开的输入窗口使用现有四角度输入工具，只提供四个数、`zero`和`q`退出。`stop`后可以输入新的目标并直接下发映射结果。该测试没有每点放线、收线3秒或切换纯电流采集的阶段；整个目标保持期间使用位控，便于验证将来遥操实际采用的控制方式。

## 输入坐标

launch、直接Python控制和映射脚本均默认输入相对角度，无需添加`--relative`。输入定义为：

```text
输入Δq = 原始glove q − [-5, -9, 7, -10]°
```

默认输入不会再次减上述参考。若要显式输入原始glove角度，直接Python使用`--absolute`，launch使用`relative:=false`。该参考是用户提供的电机零位对应角度。当前线性和二次模型均施加原点约束：相对输入`0 0 0 0`时，M11/M13/M18/M20的位置目标均为0。

当前系数使用20261020的原始电机feedback标签，并根据用户指定零位重新拟合。零位目标仍通过普通位控下发，此脚本不调用`Set_Zero`。每次预紧归零后应保持相同的物理参考；单纯读数均为0不能验证其余姿态的映射精度。

本次零位约束没有解决侧摆旧数据中的位置分歧。M20仅使用q0时，线性模型五折电机位置MAE约15.08°、二次约13.24°，不能将本次更新视为侧摆精度已改善。完整数据对比见[重拟合报告](scripts/motor_test_logs/data_collection/20261020_165513_left_collection/analysis/little_zero_constrained/ANALYSIS.md)。

## 控制参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--publish-hz` | 50 | 指令与日志频率Hz |
| `--current` | 无名指伸屈0；小指−0.08 | 固定位控前馈A；无名指侧摆由±0.05A方向偏置覆盖 |
| `--feedback-timeout` | 0.5 | SDK反馈最大允许年龄秒 |
| `--ready-timeout` | 15 | 启动等待有效反馈的超时秒 |
| `--model` | linear | `linear`或`quadratic`；小指侧摆用q0，无名指侧摆固定四q一次补偿 |

小指四路KP/KD读取`muscle_adjustment_defaults.py`，目标速度字段为0。每次输入四个q后，把按电机位置范围裁剪后的映射结果作为位置目标；50Hz仅用于重复发送当前目标，没有位置插值或额外的30°/秒限速。实际运动响应由电机控制器及KP/KD决定。默认KP/KD为M11=800/50、M13=300/30、M18=800/30、M20=800/30；实际以启动时默认参数文件为准。

拇指、其余三指及腕部共24路始终`KP/KD=0`、`tor=各自默认张紧电流`，不参与位置目标。停止帧也保持此规则。

无名指M22内收、M21外展的遥操KP=800、KD=100（`TELEOP_POSITION_KP_OVERRIDES`、`TELEOP_POSITION_KD_OVERRIDES`），无名指伸屈固定前馈默认0A，侧摆使用按方向切换的±0.05A偏置。增益覆盖只应用于本遥操脚本加载的参数副本，不修改数采的全局肌肉参数。

无名指侧摆每个发送周期根据**裁剪后的目标位置−最新有效电机反馈位置**判断方向：

| 位置误差 | M22/M21的位控电流偏置 |
|---|---:|
| > +0.5° | +0.05A，放线 |
| < −0.5° | −0.05A，收线 |
| 在[−0.5,+0.5]°内 | 0A，保留KP/KD保持位置 |

例如目标−20°、反馈−30°，应放线+0.05A；目标+10°、反馈+20°，应收线−0.05A。判断不依赖目标自身正负，也不只在收到新q时计算；同一个目标下反馈越过目标后，偏置会在后续发送周期切换。±0.5°死区用于减少到位附近反复换向；它不是关节角输入限位。启动等待、零位采样和停止时本指四路仍为KP/KD/I=0。

手动和Manus模式的q0/q1/q2/q3均不做角度限位。位控电机范围定义在`FINGER_MOTOR_POSITION_LIMITS_DEG`中。两种手指均无启动放线阶段：

| 肌肉 | 小指电机/目标范围° | 无名指电机/控制方式 |
|---|---|---|
| 伸肌 | M11 `[−5,450]` | M10 `[−5,450]` |
| 屈肌 | M13 `[−400,20]` | M17 `[−400,20]` |
| 内收 | M18 `[−15,50]` | M22 `[−15,25]` |
| 外展 | M20 `[−80,30]` | M21 `[−40,10]` |

每台位控电机独立裁剪：低于下限发送下限，高于上限发送上限，其余电机继续按最新结果更新。最终SDK位控帧也应用同一裁剪。无名指侧摆与伸屈同时下发位置目标，目标速度字段为0，侧摆KP/KD=800/100，前馈按下述规则切换；启动等待与停止时本指四路均为KP/KD/I=0。配置中的`q_relative_training_bounds`、`motor_position_training_bounds`仅保留为数据描述，不参与控制判断。

例如Δq0=−25°时线性模型预测M20≈−91.21°，实际发送−80°。输入数量、NaN/Inf、计算溢出、电机反馈失效或错误标志、Manus断流仍按原逻辑处理。这里限制的是发送的目标位置，实际反馈位置由电机运动响应决定。

反馈过期、非有限值、SDK错误标志或多驱动会结束测试并发送小指停止帧；检测到其他指令发布者时停止继续发送，避免两个控制器争用同一话题。Ctrl+C、SIGTERM及终端挂断都会先执行停止流程；未关闭本脚本以外的驱动。

## 已有SDK驱动时

若已有独立SDK驱动，使用：

```bash
source install/setup.bash
python3 src/yhand_mapping/scripts/left_little_q_to_motor_test.py \
  --relative --no-start-driver
```

已有驱动必须只有驱动本身，没有数采或映射等其他程序向`/yhand_left_cmd`发送指令。此模式退出后不关闭已有驱动。

## 单点测试与预览

单点模式：首次直接下发模型目标后持续发送5秒，然后自动结束。此计时不代表电机实际到位：

```bash
source install/setup.bash
python3 src/yhand_mapping/scripts/left_little_q_to_motor_test.py \
  --relative --q 0 10 20 5 --hold-seconds 5
```

只预览位置和完整MIT参数，不加载ROS或启动SDK：

```bash
python3 src/yhand_mapping/scripts/left_little_q_to_motor_test.py \
  --dry-run --relative --q 0 10 20 5
```

## 记录与验证

日志默认按手指分别保存：`scripts/motor_test_logs/little_q_to_motor_test/<时间>/`或`ring_q_to_motor_test/<时间>/`：

- `metadata.json`：模型、角度参考、控制参数、每次输入的目标及退出原因。
- `samples.csv`：目标原始q、机器人手套实测q、关节角误差，以及四台电机的映射目标、实际下发目标、反馈位置/速度/电流/错误码。

电机`m*_prediction`为原始模型预测，`m*_target`为裁剪后位置目标，`m*_command`为实际下发的位置目标，`m*_clipped`标记是否裁剪。`m*_control_mode`为position/release/zero_current，`m*_command_current`为下发电流。无名指侧摆正常位控时同样记录target/command；等待和停止时没有位置目标，列值留空。

metadata另保存`directional_side_motor_ids`、`side_direction_current_A`、`side_direction_deadband_deg`和方向规则。CSV的侧摆`command_current`逐帧记录实际下发的+0.05/−0.05/0A，`control_mode`仍为position，不是纯电流放线模式。

metadata另保存零位稳定阈值`manus_zero_max_range_deg`、被重启的采样窗口数和实际接受窗口的四角极差。

metadata保存`position_motor_ids`、每台电机的`motor_input_q_indices`、`side_motor_mapping`、实际启用的KP/KD、位控裁剪范围及次数。为兼容旧日志阅读工具，仍保留放线相关字段，但`release_currents_A={}`、`side_release_duration_sec=0`且没有放线起止时间。实际发送电流以CSV的command_current列为准。`--dry-run`可检查完整28路位控帧；预览假设四电机反馈位置均为0，并输出`preview_feedback_positions`与说明，不能当作实际反馈。运行时使用最新有效SDK反馈决定侧摆方向。

Manus模式另外记录操作员手套零位、输入源、成功更新和被拒绝目标次数；CSV增加`manus_q*`、`manus_relative_q*`、`target_relative_q*`和两路Manus输入年龄。实时目标保存在CSV中，不逐帧写入metadata的手动命令列表。

Manus模式下，如果`--glove-topic`与操作员输入话题相同，则关闭机器人实测角度记录，`measured_q*`和`error_q*`留空，避免把操作员输入当成机器人测量。需要独立机器人手套时，通过直接Python入口用`--glove-topic`指定另一测量话题。

`/glove_joint_data`只作测量记录，不会作为控制输入。要验证关节角误差，手套必须测量机器人实际姿态；如果该话题来自操作员手套，不能把它当作机器人实际q。缺失/过期的手套数据留空，不会伪造零误差。记录使用最近收到的反馈，并保存反馈序号及年龄，不宣称硬件同步。

计算每个姿态保持末尾的q均值，与目标q比较；同时检查电机目标误差，区分电机没有跟到目标与映射本身有偏差。超时和失败记录需要保留。

## 离线验证

```bash
source install/setup.bash
python3 -m pytest -q src/yhand_mapping/tests/test_left_little_q_to_motor_test.py
```

测试使用独立本机ROS域和假SDK驱动，覆盖28路消息顺序、其他24路默认张紧、首帧直接下发映射位置、反馈失效/错误处理，以及退出停止后再关闭驱动。另覆盖Manus启动参考、相对变化控制、左手断流/无效输入、右手隔离、关节不限位及电机边界裁剪、裁剪后继续更新目标和机器人测量隔离。两指分别验证电机编号、角度列、模型选择、默认电流和停止流程。不会启动实际SDK可执行文件。
