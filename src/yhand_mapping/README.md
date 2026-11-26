# yhand_mapping

当前电机编号见 [新腕带电机对应表](README_motor_mapping.md)。模型轴顺序按肌肉保留，不能按新编号排序。

本包包含五指与腕部映射、数据采集、标定、回放、预紧业务启动文件，以及对应配置、模型和数据。

五指当前q→m神经网络的结构、训练、遥操和历史修改统一见[MLP README](<scripts/q_2_m MLP mapping/README.md>)。
设备驱动和 ROS 消息由同工作区的 `devices_pkg` 提供。

```bash
colcon build --packages-up-to yhand_mapping --symlink-install
source install/setup.bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py finger:=five input_mode:=2 model:=mlp
```

上面的命令使用 Manus 手套控制左手五指，参数见[左手 q→m 遥操说明](README_left_hand_q_to_motor.md)。

映射、采集、标定与辅助启动的左手连接配置统一读取 `config/devices/left_hand_device_config.json`，右手统一读取 `config/devices/right_hand_device_config.json`。
其他标定文件和模型权重的位置见 [配置目录说明](config/README.md)。
官方设备 launch 位于 `devices_pkg/launch/`。本包的 `yhand_pretighten_launch.py` 默认控制双手回到绝对零位，可通过 `target_hand:=left`、`target_hand:=right` 或 `target_hand:=both` 切换，并读取所选手对应的设备配置。
构建 `yhand_mapping` 时使用 `src/devices_pkg/sdk` 的官方 M2 Demo 源码生成 `Y_Hand_M2_Demo`。`scripts/yhand_pretighten.py` 启动这个可执行文件，并传入 M2 SDK 配置及所选手的连接参数。28 路的 `pos=0°` 和 `vel=0` 由 Demo 的控制循环发送，统一 `KP=500`、`KD=10`。SDK Demo 的前馈电流从零起步，随后由反馈误差积分更新。每路限流默认 `0.5 A`，单独运行脚本时可用 `--max-current-limit` 后跟 28 个正数覆盖。
脚本读取 Demo 的 SDK 反馈，逐路打印 `M01` 到 `M28` 的实际位置、相对 `0°` 的绝对误差、错误码和到位状态。默认要求连续 3 帧均满足 28 路 `|pos|≤1°` 且 `error=0`。通过后持续保持 `0°`，10 秒内未确认则打印最新反馈并关闭 Demo。可用 `zero_tolerance` 和 `zero_timeout` 调整角度容差及等待秒数。误差直接使用多圈位置计算。
Ctrl+C 或 SIGTERM 通知 Demo 释放位置控制。Demo 在退出前发送 0.5 秒的零增益和零电流指令。
通用左手 q→m 控制脚本也向驱动传入相同的 28 路电流上限。本地 `Y_Hand_Node` 对未传该参数的旧 launch 使用每路 `0.5 A` 的默认值，显式 ROS 参数优先；长度或数值无效时在硬件初始化前报错退出。
通用左手 q→m 控制以 SensorData QoS（BEST_EFFORT）订阅电机反馈，与当前 SDK 一致。
Python 工具及模型在 `scripts/`。替换 SDK 时保留本包，详见工作区根目录的 `README.md`。
