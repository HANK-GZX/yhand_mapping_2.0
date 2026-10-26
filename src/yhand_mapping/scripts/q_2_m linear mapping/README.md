# q → m 遥操映射

- `left_little_q_to_motor_test.py`：单指、四指和五指遥操控制入口。
- `left_little_q_to_motor.py`：加载各指模型，将关节角转换为电机位置。
- `fit_left_thumb_q_to_motor.py`：离线拟合拇指模型。

目录整理保留现有模型选择，拇指仍使用分电机的一次／二次混合模型。
模型保留在 `config/`，共享输入脚本、默认参数和日志保留在 `scripts/`。

五指遥操启动命令：

```bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py finger:=five input_mode:=2
```

直接运行时需给含空格的脚本路径加引号，例如离线检查：

```bash
python3 "src/yhand_mapping/scripts/q_2_m linear mapping/left_little_q_to_motor_test.py" --finger five --dry-run --q 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

详细说明见[遥操控制](../../README_left_hand_q_to_motor.md)
与[模型计算](../../README_left_hand_q_to_motor_models.md)。
