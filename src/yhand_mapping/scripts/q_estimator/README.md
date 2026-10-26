# 电机反馈 → 关节角状态估计

- `left_hand_motor_to_q_estimator.py`：五指估计入口，支持拟合、实时估计和离线预测。
- `left_thumb_motor_to_q_estimator.py`：单拇指入口及共用估计逻辑。
- `motor_to_q_logging.py`：记录 q_cmd、手套角、电机反馈及估计角。
- `analyze_motor_to_q_log.py`：日志对比分析和历史遥操日志回放。

ROS 启动命令保持不变：

```bash
ros2 launch yhand_mapping left_hand_motor_to_q_estimator_launch.py model:=linear
```

模型仍在 `config/`，日志仍在 `scripts/motor_test_logs/motor_to_q_estimator/`。
详细用法见[五指估计器说明](../../README_left_hand_motor_to_q_estimator.md)
和[拇指估计器说明](../../README_left_thumb_motor_to_q_estimator.md)。
