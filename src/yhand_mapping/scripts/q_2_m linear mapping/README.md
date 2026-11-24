# q → m 遥操映射

- `left_little_q_to_motor_test.py`：单指、四指和五指遥操控制入口。
- `left_little_q_to_motor.py`：加载各指模型，将关节角转换为电机位置。
- `fit_left_thumb_q_to_motor.py`：离线拟合拇指模型。
- `thumb_linear_coordination.py`：v3完整四角侧摆／q1对掌协调。
- `motor_trajectory.py`：候选的首次运动启动柔化与持续轨迹约束，默认模型不启用。

拇指默认使用分电机的一次／二次混合模型，并支持 V3、V4 两个可选版本。
模型保留在 `config/`，共享输入脚本、默认参数和日志保留在 `scripts/`。

同一控制入口添加 `model:=mlp` 可切换为[五指独立 MLP](<../q_2_m MLP mapping/README.md>)。

五指遥操启动命令：

```bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py finger:=five input_mode:=2
```

五指中的拇指切换为v3，其余四指保持原linear：

```bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py finger:=five input_mode:=2 model:=linear thumb_version:=v3
```

省略`thumb_version`或设置`thumb_version:=default`保留默认版本；单拇指也支持该选项。

直接运行时需给含空格的脚本路径加引号，例如离线检查：

```bash
python3 "src/yhand_mapping/scripts/q_2_m linear mapping/left_little_q_to_motor_test.py" --finger five --dry-run --q 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

详细说明见[遥操控制](../../README_left_hand_q_to_motor.md)
与[模型计算](../../README_left_hand_q_to_motor_models.md)。

## 拇指 V4

`thumb_version:=v4` 使用一套 6×4 一次矩阵连续映射四角。拟合过程、验证指标和复现命令见
[左拇指 V4](../../README_left_thumb_linear_v4.md)。

## 显式选组的拇指一次模型

两份离线实验模型均按 12 个采集组合分别拟合四个手套角到六路电机反馈的映射。

- [普通最小二乘模型](../../config/experiments/left_thumb_combo_q_to_motor_linear.json)逐组独立拟合。
- [任务方向约束模型](../../config/experiments/left_thumb_task_combo_q_to_motor_linear.json)记录了 q2/q3 伸屈、q0 侧摆和 `opp_on` 下 q1 收放线的拟合约束。实际系数中有 3 个 `opp_on` 组合的 M14 q1 斜率小于其他电机的最大绝对 q1 斜率。

从 `combos` 中按采集工况选组。输入 `q` 为四个绝对手套角，单位为度。
预测式为 `m = motor_at_q_center_deg + motor_per_joint_deg @ (q - q_center_deg)`，其中 `@` 表示矩阵乘法。
输出顺序为 M4、M6、M7、M8、M14、M27，电机位置的正增量表示放线。
`recorded_position_motor_ids` 和 `passive_reference_motor_ids` 分别记录该组的主动位置路和被动参考路。

`thumb_version:=v4` 加载 [V4 遥操模型](../../config/experiments/left_thumb_linear_v4.json)。
