# Y-hand Back-Delta 软件零位功能

当前电机编号见 [新腕带电机对应表](README_motor_mapping.md)。模型轴顺序按肌肉保留，不能按新编号排序。

当前正式软件零位方案只保留 back-delta 流程，不再使用旧版多特征动作、人工模式或反复采样流程。

## 核心思路

Reference 阶段在机械零位附近记录上电初始位置，然后让手腕向后弯曲到电流极限位置并拉紧稳定，保存：

```text
delta = back_max - initial_power_on
```

Auto 阶段重新执行同样的后弯极限动作，得到当前后弯位置 `back_max_auto`，直接计算零位：

```text
zero_estimate = back_max_auto - delta
```

随后 28 个电机位置控制回到 `zero_estimate`，再 28 个电机一起负电流拉紧稳定，最终保存 `zero_offset`。

## 文件

正式入口：

```text
launch/software_zero_calibration_launch.py
scripts/software_zero_calibration.py
```

Proxy：

```text
launch/software_zero_proxy_launch.py
scripts/software_zero_proxy.py
```

公共函数：

```text
scripts/software_zero_common.py
```

配置：

```text
config/software_zero_calibration_config.json
config/bimanual_hand_device_config.json
scripts/muscle_adjustment_defaults.py
```

生成数据：

```text
scripts/software_zero/reference/<hand>_back_delta_reference.json
scripts/software_zero/state/<hand>_back_delta_state.json
```

## 运行指令

左手采集 reference：

```bash
ros2 launch yhand_mapping software_zero_calibration_launch.py \
  mode:=collect-reference \
  hand_side:=left \
  start_driver:=true \
  calibration_in_terminal:=auto
```

左手自动回零：

```bash
ros2 launch yhand_mapping software_zero_calibration_launch.py \
  mode:=calibrate-auto \
  hand_side:=left \
  start_driver:=true \
  calibration_in_terminal:=auto
```

右手采集 reference：

```bash
ros2 launch yhand_mapping software_zero_calibration_launch.py \
  mode:=collect-reference \
  hand_side:=right \
  start_driver:=true \
  calibration_in_terminal:=auto
```

右手自动回零：

```bash
ros2 launch yhand_mapping software_zero_calibration_launch.py \
  mode:=calibrate-auto \
  hand_side:=right \
  start_driver:=true \
  calibration_in_terminal:=auto
```

双手一起采集 reference：

```bash
ros2 launch yhand_mapping software_zero_calibration_launch.py \
  mode:=collect-reference \
  hand_side:=both \
  start_driver:=true \
  calibration_in_terminal:=auto
```

双手一起自动回零：

```bash
ros2 launch yhand_mapping software_zero_calibration_launch.py \
  mode:=calibrate-auto \
  hand_side:=both \
  start_driver:=true \
  calibration_in_terminal:=auto
```

`hand_side:=both` 会同时启动左右 raw driver，但校准脚本按顺序执行左手再右手，避免两个动作完全并发。

## Proxy

Proxy 默认读取：

```text
scripts/software_zero/state/<hand>_back_delta_state.json
```

启动左手 proxy：

```bash
ros2 launch yhand_mapping software_zero_proxy_launch.py \
  hand_side:=left \
  start_driver:=true \
  start_proxy:=true
```

启动双手 proxy：

```bash
ros2 launch yhand_mapping software_zero_proxy_launch.py \
  hand_side:=both \
  start_driver:=true \
  start_proxy:=true
```

注意：state 文件带有 boot id。每次重启后需要重新运行 `calibrate-auto`，再启动 proxy。

## 当前后弯动作参数

参数位置：

```text
config/software_zero_calibration_config.json
```

关键参数：

```text
前侧放线电流：+0.1A
前侧放线时间：8.0s
后侧腕部/伸肌收缩电流：-0.2A
后侧侧摆收缩电流：-0.1A
后侧到位判定：window=5.0s, threshold=3.0deg, timeout=10.0s
后弯最终全组拉紧：5.0s + stable window=3.0s, threshold=3.0deg, timeout=15.0s
最终零位 28 电机拉紧：3.0s + stable window=5.0s, threshold=3.0deg, timeout=10.0s
```
