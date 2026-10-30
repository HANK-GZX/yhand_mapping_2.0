# 左手 q→m 遥操控制

本页对应 `launch/left_hand_q_to_motor_launch.py`，支持单指及多指遥操。模型计算与 Python 接口见[左手 q→m 模型说明](README_left_hand_q_to_motor_models.md)。

当前统一入口只提供 `linear` 与 `mlp` 两种模式。

五指MLP使用每指四个Δq作为输入。当前网络结构、训练数据、输入权重、零位处理、侧摆2°规则、验证结果及历史修改，统一见[五指MLP README](<scripts/q_2_m MLP mapping/README.md>)。

```bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py finger:=five input_mode:=2 model:=mlp
```

单指和四指等组合沿用下文的`finger`参数。`mlp_model_dir:=/完整路径/模型目录`可指定另一套权重；更新权重后需重启进程。电流、KP/KD、限位和断流停止策略由同一控制入口执行。

下文拇指既有一次／二次混合标定属于`linear`模式的内部策略。
遥操入口现在同步发布只读q_cmd遥测 `/left_<finger>_q_command_state`，供[五指状态估计日志](README_left_hand_motor_to_q_estimator.md)记录控制器真正采用的目标。遥测包含绝对／相对目标、参考角、位控状态与实际电机位置命令；不改变q→m系数或原控制帧。已有进程需要重启才能发布新增话题。

拇指已接入同一入口：`finger:=thumb`单独控制拇指六路，`finger:=five`同时控制五指24路。五指Manus遥操：

```bash
source install/setup.bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py finger:=five input_mode:=2
```

已按用户要求恢复扩展前的拇指模型与控制行为：撤回新零位约束重拟合、CMC幅度补偿、切线外推和新增拇指输出窗口。保留单拇指和五指统一遥操。模型文件更新后，需要重启控制入口才能加载恢复后的系数。

单拇指Manus遥操：

```bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py finger:=thumb input_mode:=2
```

拇指固定采用以下混合模型，使用`model:=linear`：

| 电机 | 肌肉 | 输入 | 实际模型 |
|---|---|---|---|
| M4 | 长屈 | 拇指四q | 一次 |
| M7 | 长伸 | 拇指四q | 一次 |
| M6 | 短伸 | 拇指四q | 二次 |
| M8 | 内收 | 拇指四q | 二次 |
| M27 | 外展 | 拇指四q | 二次 |
| M14 | 对掌 | 拇指四q | 二次 |

完整二次模型包含4个一次项、4个平方项、6个交叉项；展开顺序为`1,x0,x1,x2,x3,x0²,x0x1,x0x2,x0x3,x1²,x1x2,x1x3,x2²,x2x3,x3²`。`model`参数继续控制其余四指的原有选项，对拇指六路始终使用表中约定。

系数文件为`config/left_thumb_q_to_motor_models.json`，已从扩展前的完整备份恢复，来源仍为8149个正式full点，包含3126个运动超时点。参考角恢复为采集启动帧`[8.491,38.262,26.028,-26.219]°`。源模型使用自由截距最小二乘，然后展开为`f(q_reference+Δq)-f(q_reference)`，保证相对四q全0时六路目标均0；这不是按用户提供的新零位做原点约束重拟合。用户提供的`[-3,42,40,-36]°`保留在历史实验记录中，本次恢复不使用该参考角。

拇指输入四q保持原幅度，使用原完整多项式；不施加CMC增益、切线外推或新增拇指位置窗口。固定前馈0A，KP/KD读取现有`muscle_adjustment_defaults.py`；不添加无名指的侧摆方向电流偏置。其余四指策略不变。

五指模式只有一个控制进程、一个SDK驱动、一个Manus节点，50Hz把24路位置目标合并为同一28路指令，腕部M5/M12/M19/M26保持默认张紧。请结束原有单指或四指控制进程后再启动五指模式。启动时全部20个角各自极差≤3°且连续稳定1秒后锁定操作员零位；之后各指按自己的四个相对q计算。输入或反馈失效、退出时，24路一并停止，腕部继续默认电流。

五指输入顺序为“拇指4角、食指4角、中指4角、无名指4角、小指4角”，对应glove48的`[8:28]`。日志在`scripts/motor_test_logs/five_q_to_motor_test/<时间>/`；metadata保存五份模型、各指角度分组、逐电机实际模型名称和阶数。CSV的`target_relative_q*`记录原幅度目标，`manus_relative_q*`保留操作员相对变化。`finger:=four`仍是原来的四指模式。

单拇指手动输入使用`finger:=thumb input_mode:=1`，默认话题为`/left_thumb_target_q`。五指手动20角话题输入使用`finger:=five input_mode:=1 start_angle_input_terminal:=false`，向`/left_five_target_q`发布20维Float64MultiArray。

不连接硬件的拇指目标预览：

```bash
python3 "src/yhand_mapping/scripts/q_2_m linear mapping/left_little_q_to_motor_test.py" \
  --finger thumb --dry-run --q 0 0 0 0
```

离线复算使用`scripts/q_2_m linear mapping/fit_left_thumb_q_to_motor.py`，默认参考为原采集启动帧，使用自由截距拟合后做增量展开。默认读取校验过的本批结果缓存；`--refresh-cache`重读原始Excel；`--q-reference Q0 Q1 Q2 Q3`修改多项式展开参考，建议用`--output /tmp/left_thumb_models.json`先检查。拟合脚本不连接设备。首次安装或脚本清单变化后运行`colcon build --packages-select yhand_mapping --symlink-install`。

入口：`scripts/q_2_m linear mapping/left_little_q_to_motor_test.py`。通过现有`devices_pkg/Y_Hand_Node`调用官方SDK，同一个脚本和launch支持选择拇指、食指、中指、无名指或小指。每次可选择单指，或以`finger:=both`同时控制无名指和小指、`finger:=three`同时控制中指/无名指/小指、`finger:=four`同时控制四指、`finger:=five`同时控制五指。拇指单指模式其余22路、食指/中指单指模式其余23路、小指/无名指单指模式其余24路、双指模式其余20路、三指模式其余15路、四指模式其余10路、五指模式腕部4路保持默认张紧电流。

四指同时Manus遥操：

```bash
source install/setup.bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py finger:=four input_mode:=2
```

先退出已有控制。四指模式由一个控制节点、一个SDK驱动、一个Manus节点统一发送18台位控电机，名义频率50Hz；其余拇指6路和腕部4路默认张紧。各指沿用自己的模型、增益、电流与限位：食指/中指不新增位置限位、固定前馈0A；无名指保留原限位和侧摆方向偏置；小指保留原限位及−0.08A前馈。中指M2仍为已调整的300/50。

启动时16个q各自极差≤3°并连续稳定1秒才锁定操作员零位。顺序为食指4角、中指4角、无名指4角、小指4角，对应glove48的`[12:28]`。反馈/Manus失效或退出时一并停止18台电机。日志在`scripts/motor_test_logs/four_q_to_motor_test/<时间>/`，q0～q3对应食指、q4～q7中指、q8～q11无名指、q12～q15小指，metadata保存角度分组及四份模型。若使用手动16角话题，设置`finger:=four input_mode:=1 start_angle_input_terminal:=false`，向默认`/left_four_target_q`发送16维Float64MultiArray；单指弹出输入终端仍为4角。

三指同时Manus遥操：

```bash
source install/setup.bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py finger:=three input_mode:=2
```

先退出原来的单指或双指控制。三指模式只启动一个控制节点、一个SDK驱动和一个Manus发布节点；50Hz将13台位控电机合并进同一个28路消息。中指五路继续不添加电机限位、前馈0A；无名指继续现有限位及侧摆±0.05A方向偏置，小指继续现有限位及−0.08A前馈。其余15路默认张紧。`current`留空时分别采用上述默认，显式设置时统一覆盖固定前馈，无名指侧摆仍由方向偏置覆盖。

启动后中指、无名指、小指一起保持零位姿态，12个q各自极差≤3°并连续稳定满默认1秒才锁定参考；之后各自4个Δq独立推理，反馈或Manus断流时一起停止13台电机。角度顺序为“中指4角、无名指4角、小指4角”，对应glove48的`[16:28]`。

日志目录为`scripts/motor_test_logs/three_q_to_motor_test/<时间>/`；CSV的q0～q3为中指、q4～q7为无名指、q8～q11为小指。metadata记录三份模型和各指角度分组，中指五路标记为`unlimited_position_motor_ids`。需要手动12角话题时，使用`finger:=three input_mode:=1 start_angle_input_terminal:=false`，向`/left_three_target_q`发送12维Float64MultiArray；现有弹出输入窗口仅用于单指4角。

| `finger` | 手指 | 伸肌 | 屈肌 | 内收 | 外展 | Manus角度索引 | 机器人零位q/° |
|---|---|---|---|---|---|---|---|
| `little`（默认） | 小指 | M11 | M13 | M18 | M20 | `[24:28]` | `[-5,-9,7,-10]` |
| `ring` | 无名指 | M10 | M17 | M22 | M21 | `[20:24]` | `[-2,-6,0,-12]` |
| `middle` | 中指 | M23 | M16浅屈、M2深屈 | M9 | M24 | `[16:20]` | 暂无精确中指零位；见下文 |
| `index` | 食指 | M25 | M3浅屈、M15深屈 | M1 | M28 | `[12:16]` | 暂无精确食指零位；见下文 |

食指Manus遥操：

```bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py finger:=index input_mode:=2
```

食指手动相对四q输入：

```bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py finger:=index input_mode:=1
```

食指五路均使用四q一次模型，来自`data_collection/left_hand/index`的5606个正式xy点，包含3299个运动阶段超时点。使用分析得到的带截距拟合斜率计算增量`Δm=f(q_neutral+Δq)−f(q_neutral)=BΔq`，Δq全0时五路电机目标均0：

```text
M25 = -0.51221 Δq0 +1.81761 Δq1 +1.46775 Δq2 +0.92823 Δq3
M3  = +1.12725 Δq0 -2.07872 Δq1 -1.46277 Δq2 +0.12617 Δq3
M15 = +0.16910 Δq0 -2.06644 Δq1 -1.71283 Δq2 -1.10040 Δq3
M1  = +2.06561 Δq0 -0.13520 Δq1 -0.09968 Δq2 -0.21285 Δq3
M28 = -2.14885 Δq0 -0.75558 Δq1 +0.13754 Δq2 +0.30824 Δq3
```

配置为`config/left_index_q_to_motor_models.json`，保留原始截距、分析来源校验值和原坐标验证指标。它们不能直接当作增量模型的实机控制精度。训练参考`[3.627,-6.019,0.028,-2.929]°`只是启动帧，不是本机精确电机零位q；只用于显式absolute输入和日志参考。手动模式默认输入Δq，不再次减参考；Manus采用操作员自己的启动零位，只按变化控制。

食指先采用中指相同的无限位策略：不新增q或电机位置范围检查，保持非有限数、反馈错误和断流检查。五路固定前馈默认0A，默认KP/KD为M25=300/50、M3=200/30、M15=200/30、M1=800/30、M28=800/30，实际从`muscle_adjustment_defaults.py`读取；不为食指添加无名指的侧摆±0.05A偏置。启动等待/停止时食指五路KP/KD/I=0，其他23路默认张紧。默认手动话题为`/left_index_target_q`，日志目录为`scripts/motor_test_logs/index_q_to_motor_test/<时间>/`。

中指Manus遥操与手动输入使用同一入口：

```bash
source install/setup.bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py finger:=middle input_mode:=2
```

```bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py finger:=middle input_mode:=1
```

中指五台电机均使用四个相对Δq的一次模型，系数取自`20261016_161639_left_collection`的5889个正式xy点，包含4085个运动超时点。控制采用增量关系`Δm=f(q_neutral+Δq)−f(q_neutral)=BΔq`，以当前约定的电机零位0作为目标参考：

```text
M23 = -0.24029 Δq0 +1.73683 Δq1 +1.47099 Δq2 +1.24415 Δq3
M16 = +1.02029 Δq0 -1.58628 Δq1 -1.52510 Δq2 +0.06461 Δq3
M2  = -0.27993 Δq0 -1.47894 Δq1 -1.81639 Δq2 -1.19039 Δq3
M9  = +2.45499 Δq0 -0.23385 Δq1 -0.05576 Δq2 +0.03790 Δq3
M24 = -2.23981 Δq0 -0.23499 Δq1 +0.04494 Δq2 +0.01315 Δq3
```

这是使用原带截距拟合的斜率计算位置变化，不是强制过原点重新拟合。配置`config/left_middle_q_to_motor_models.json`保存原始截距、原坐标验证指标和来源；原坐标MAE不能直接当成增量控制后的实机误差。

中指手动输入默认直接接受Δq，不再减训练参考。Manus启动时四个角各自连续稳定1秒、极差≤3°后锁定操作员零位，之后仅用相对变化。Δq全0时五台电机目标均为0，不调用Set_Zero。配置中的`q_reference=[4.396,-8.581,-0.020,0.443]°`只是整批采集启动帧，用于显式`--absolute`转换和日志参考，不是本机精确电机零位标定；Manus不使用它作为操作员零位。

按用户要求，中指四个输入q和五台电机目标均不添加位置范围限制，模型结果直接发送；NaN/Inf、反馈失效、错误码及Manus断流检查保留。五路固定前馈默认0A，M2深屈遥操增益覆盖为KP=300、KD=50，单指和三指模式均生效；其余读取当前`muscle_adjustment_defaults.py`：M23=300/50、M16=200/30、M9=800/30、M24=300/30。中指不添加无名指的侧摆±0.05A偏置。停止/等待时本指五路KP/KD/I=0，其余23路默认张紧。无名指、小指及`both`现有模型、限位和电流策略保留。

中指默认手动话题为`/left_middle_target_q`，日志位于`scripts/motor_test_logs/middle_q_to_motor_test/<时间>/`，记录全部五路目标/指令/反馈。`finger:=both`仍仅代表无名指+小指；不要同时启动独立中指进程和已有控制进程向同一个28路指令话题发布。

同时遥操无名指和小指（Manus）：

```bash
source install/setup.bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py finger:=both input_mode:=2
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
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py finger:=ring input_mode:=2
```

无名指手动输入：

```bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py finger:=ring input_mode:=1
```

小指使用`finger:=little`或省略此参数。两种手指的直接Python入口均支持`--finger`。

无名指完整系数与分电机来源保存在`config/left_ring_q_to_motor_models.json`；M10/M17来自旧批4307个正式点，M22/M21来自上述新批2170点。四q与q0的离线对比见[四q侧摆分析](scripts/motor_test_logs/data_collection/ring_data/analysis/ring_distribution/SIDE_FOUR_Q.md)。未执行新的物理标定。

下列未指定`finger`的示例均选择小指。

M18/M20仅使用q0，M11/M13使用全部四q。默认`linear`；`--model mlp`切换神经网络。输入支持手动相对角度和左手Manus实时相对角度两种模式。

## 启动

从工作空间根目录运行，先source已编译的工作空间：

```bash
source install/setup.bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py input_mode:=1
```

模式1（默认，`manual`也可）会自动启动SDK控制程序，并打开四个q的输入终端；默认输入相对角度，默认线性模型。输入`q`退出窗口会关闭整个测试，启动launch的终端Ctrl+C也会先停止小指再关闭SDK。输入窗口支持`zero`发送四个零角度。

模式2（`manus`也可）使用左手Manus遥操，不打开角度输入窗口：

```bash
source install/setup.bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py input_mode:=2
```

先完成机器手的预紧归零，让电机0对应约定的机器人零位姿态。启动后将操作员左手套保持在希望作为零位的姿态，SDK反馈就绪后连续采样1秒均值作为`q_manus_zero`，终端提示零位已记录后再运动：

```text
Δq = q_manus_current − q_manus_zero
电机目标 = 小指映射(Δq)
```

操作员手套零位无需等于机器手的`[-5,-9,7,-10]°`；这两者分别定义各自的参考姿态。相对变化为0时，四台电机映射目标为0。零位只在启动时记录一次，之后不会跟随手套变化更新，也不会调用SDK `Set_Zero`。无名指另外要求零位采样期间四个q各自最大值−最小值≤3°；检测到运动会重新计时，在记录零位前本指四路保持KP/KD/I=0。请伸直无名指后保持静止，直到终端提示“Manus零位已记录”；超过ready-timeout仍不稳定则退出，不能用运动过程的均值作零位。小指保持原来的采样规则。

Manus模式复用`wrist_five_finger_hybrid_fusion_right_launch.py`中的`minimal_example_dual_stream_commented.py`双流发布节点，默认自动启动。Windows端Manus SDK需保持运行；如果ROS手套节点已经启动，使用：

```bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py \
  input_mode:=2 start_glove_node:=false
```

角度取`/glove_joint_data[24:28]`，与当前模型训练数据定义一致；`/glove_joint_data_raw_skeleton`仅用于确认左手在线，不改用骨架推算的另一套角度。只收到右手raw不能标定或刷新左手输入。角度或左手raw超过0.5秒未更新、或角度出现非有限值时，退出并停止小指输出。四个关节角不限位；四台电机的映射结果按下述位置范围分别裁剪，超出时发送边界值并继续更新。

保持50Hz发送最新映射目标，不插值、不增加限速。其余24路始终维持默认张紧电流。重新记录手套参考姿态时，退出后重新启动Manus模式。

| launch参数 | 默认值 | 含义 |
|---|---|---|
| `finger` | `little` | `thumb`拇指；`index`食指；`middle`中指；`ring`无名指；`little`小指；`both`双指；`three`三指；`four`四指；`five`五指，自动选择模型和Manus角度列 |
| `input_mode` | `1` | `1`/`manual`手动；`2`/`manus`左手套遥操 |
| `start_glove_node` | `true` | 仅Manus模式生效；启动双流发布节点 |
| `manus_topic` | `/glove_joint_data` | 48维Manus角度话题 |
| `manus_raw_topic` | `/glove_joint_data_raw_skeleton` | 左手在线检测话题 |
| `manus_zero_seconds` | `1.0` | 启动零位采样时长秒 |
| `manus_timeout` | `0.5` | 左手角度和raw最大允许年龄秒 |

自定义话题时，应确保发布节点也向对应话题发布；自动启动的现有发布节点使用上述默认话题。

launch的`current`默认留空，无名指伸屈默认0A、小指默认−0.08A；显式设置`current:=数值`可覆盖固定前馈。无名指侧摆始终采用下述方向偏置，覆盖固定前馈，不与`current`叠加。实际电流仍包含KP/KD对位置/速度误差产生的控制输出。

可选参数：`model:=mlp`使用神经网络，手动模式下`relative:=false`输入原始glove角度，`start_driver_node:=false`复用已经启动的SDK。手动模式默认目标q话题是`/left_little_target_q`，与机器人手套实测话题分开。Manus模式始终使用相对变化。

首次新增launch后需要构建一次：

```bash
colcon build --packages-select yhand_mapping --symlink-install
source install/setup.bash
```

原来的直接Python交互方式仍可使用，支持`stop`、`status`：

```bash
python3 "src/yhand_mapping/scripts/q_2_m linear mapping/left_little_q_to_motor_test.py" --relative
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

本次零位约束没有解决侧摆旧数据中的位置分歧。M20仅使用q0时，线性模型五折电机位置MAE约15.08°、二次约13.24°，不能将本次更新视为侧摆精度已改善。当前系数和验证指标见[小指配置](config/left_little_q_to_motor_models.json)。

## 控制参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--publish-hz` | 50 | 指令与日志频率Hz |
| `--current` | 无名指伸屈0；小指−0.08 | 固定位控前馈A；无名指侧摆由±0.05A方向偏置覆盖 |
| `--feedback-timeout` | 0.5 | SDK反馈最大允许年龄秒 |
| `--ready-timeout` | 15 | 启动等待有效反馈的超时秒 |
| `--model` | linear | `linear`或`mlp`；linear保留原标定，mlp使用对应手指神经网络 |

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
| 外展 | M20 `[−80,30]` | M21 `[−40,25]` |

每台位控电机独立裁剪：低于下限发送下限，高于上限发送上限，其余电机继续按最新结果更新。最终SDK位控帧也应用同一裁剪。无名指侧摆与伸屈同时下发位置目标，目标速度字段为0，侧摆KP/KD=800/100，前馈按下述规则切换；启动等待与停止时本指四路均为KP/KD/I=0。配置中的`q_relative_training_bounds`、`motor_position_training_bounds`仅保留为数据描述，不参与控制判断。

例如Δq0=−25°时线性模型预测M20≈−91.21°，实际发送−80°。输入数量、NaN/Inf、计算溢出、电机反馈失效或错误标志、Manus断流仍按原逻辑处理。这里限制的是发送的目标位置，实际反馈位置由电机运动响应决定。

反馈过期、非有限值、SDK错误标志或多驱动会结束测试并发送小指停止帧；检测到其他指令发布者时停止继续发送，避免两个控制器争用同一话题。Ctrl+C、SIGTERM及终端挂断都会先执行停止流程；未关闭本脚本以外的驱动。

## 已有SDK驱动时

若已有独立SDK驱动，使用：

```bash
source install/setup.bash
python3 "src/yhand_mapping/scripts/q_2_m linear mapping/left_little_q_to_motor_test.py" \
  --relative --no-start-driver
```

已有驱动必须只有驱动本身，没有数采或映射等其他程序向`/yhand_left_cmd`发送指令。此模式退出后不关闭已有驱动。

## 单点测试与预览

单点模式：首次直接下发模型目标后持续发送5秒，然后自动结束。此计时不代表电机实际到位：

```bash
source install/setup.bash
python3 "src/yhand_mapping/scripts/q_2_m linear mapping/left_little_q_to_motor_test.py" \
  --relative --q 0 10 20 5 --hold-seconds 5
```

只预览位置和完整MIT参数，不加载ROS或启动SDK：

```bash
python3 "src/yhand_mapping/scripts/q_2_m linear mapping/left_little_q_to_motor_test.py" \
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

拇指可选的“外展绕弧→内收对掌”combo 控制层见
[标定与使用说明](scripts/thumb_combo_control/README.md)。传入
`thumb_combo_calibration:=<实测标定JSON绝对路径>` 后，在同一节点内使用姿态关键点补偿
M4/M14 并协调伸肌释放，四指继续使用所选映射。模板含未标定数值，不能直接上机。
新日志记录 combo 权重、实际修正、KP/KD 和标定快照；不传参数保留原控制。

```bash
source install/setup.bash
python3 -m pytest -q src/yhand_mapping/tests/test_left_little_q_to_motor_test.py
```

测试使用独立本机ROS域和假SDK驱动，覆盖28路消息顺序、其他24路默认张紧、首帧直接下发映射位置、反馈失效/错误处理，以及退出停止后再关闭驱动。另覆盖Manus启动参考、相对变化控制、左手断流/无效输入、右手隔离、关节不限位及电机边界裁剪、裁剪后继续更新目标和机器人测量隔离。两指分别验证电机编号、角度列、模型选择、默认电流和停止流程。不会启动实际SDK可执行文件。
