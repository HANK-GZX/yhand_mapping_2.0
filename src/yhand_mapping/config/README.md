# 配置索引

以下路径相对于 `config/`。`q` 表示手指关节角，`m` 表示电机位置。

| 目录 | 内容和使用场景 |
| --- | --- |
| [`devices/`](devices/) | 设备连接和话题。`left_hand_device_config.json`、`right_hand_device_config.json` 分别用于左右单手，`bimanual_hand_device_config.json` 用于双手场景和软件零位 launch。更换设备时在此检查 SN、板 IP、端口和 ROS 话题。 |
| [`calibration/`](calibration/) | 软件零位参数、手动关节耦合参考和拇指组合轨迹标定模板。下方说明三者的入口。 |
| [`q_to_motor_polynomial/`](q_to_motor_polynomial/) | 左手五指的关节角到电机目标位置映射，包含 `model:=linear` 遥操入口使用的逐指模型、拇指 V3/V4 和拇指动作组合的离线拟合模型。 |
| [`q_to_motor_mlp/`](q_to_motor_mlp/) | 左手五指的 q→m 神经网络权重，对应 `model:=mlp`。可用 `mlp_model_dir` 指定自定义目录。 |
| [`motor_to_q_polynomial/`](motor_to_q_polynomial/) | 左手五指的电机反馈位置到关节角估计，含一次和二次模型。对应估计节点的 `model:=auto`、`linear` 或 `quadratic`。 |
| [`motor_to_q_mlp/`](motor_to_q_mlp/) | 左手食指、中指、无名指和小指的 m→q 神经网络权重。五指估计使用 `model:=mlp` 时，拇指使用多项式估计。 |

## 拇指 q→m 模型

- [`left_thumb_linear_coordination_v3.json`](q_to_motor_polynomial/left_thumb_linear_coordination_v3.json) 由 `model:=linear thumb_version:=v3` 选择。
- [`left_thumb_linear_v4.json`](q_to_motor_polynomial/left_thumb_linear_v4.json) 由 `model:=linear thumb_version:=v4` 选择。
- [`left_thumb_combo_q_to_motor_linear.json`](q_to_motor_polynomial/left_thumb_combo_q_to_motor_linear.json) 是按动作组合计算的普通最小二乘模型，用于离线比较。
- [`left_thumb_task_combo_q_to_motor_linear.json`](q_to_motor_polynomial/left_thumb_task_combo_q_to_motor_linear.json) 是按动作组合加入任务方向约束的拟合模型，用于离线比较。

## 标定文件

- [`software_zero_calibration_config.json`](calibration/software_zero_calibration_config.json) 定义软件零位流程的电流、控制参数和位置限位。参见 [软件零位说明](../Yhand_software_zero_calibration_README.md)。
- [`manual_joint_coupling.json`](calibration/manual_joint_coupling.json) 保存实采姿态构建的相对关节角参考，用于手动输入的关节耦合检查。可通过 `coupling_reference` 指定替代文件。
- [`left_thumb_combo_template.json`](calibration/left_thumb_combo_template.json) 是拇指组合轨迹的标定模板。完成实机标定后，通过 `thumb_combo_calibration` 指定生成的标定文件。参见 [拇指组合控制说明](../scripts/thumb_combo_control/README.md)。

使用节点的默认参数时会按上表读取配置。已有命令如果显式指定了原 `config/` 下的 JSON 文件，请把路径改到对应分组。五指 m→q 估计的 `model_dir` 仍可指向 `config/`，也支持直接传入包含模型 JSON 的自定义目录。
