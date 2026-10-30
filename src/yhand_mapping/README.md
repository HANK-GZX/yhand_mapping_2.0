# yhand_mapping

当前电机编号见 [新腕带电机对应表](README_motor_mapping.md)。模型轴顺序按肌肉保留，不能按新编号排序。

本包包含五指与腕部映射、数据采集、标定、回放、预紧业务启动文件，以及对应配置、模型和数据。

五指当前q→m神经网络的结构、训练、遥操和历史修改统一见[MLP README](<scripts/q_2_m MLP mapping/README.md>)。
设备驱动和 ROS 消息由同工作区的 `devices_pkg` 提供。

```bash
colcon build --packages-up-to yhand_mapping --symlink-install
source install/setup.bash
ros2 launch yhand_mapping five_finger_mapping_launch.py target_hand:=right
```

映射、采集、标定与辅助启动的左手连接配置统一读取 `config/left_hand_device_config.json`，右手统一读取 `config/right_hand_device_config.json`。
官方设备 launch 位于 `devices_pkg/launch/`；本包的 `yhand_pretighten_launch.py` 分别读取 `left_hand_device_config.json` 和 `right_hand_device_config.json`。
预紧使用本包的 `scripts/yhand_pretighten.py` 发送 `tor=-0.05`，目前仅启用左手。Ctrl+C 时先连续发送 0.5 秒的全部 28 路零电流指令，再通过一次 `pretightenforce.flag=True、torque=0` 的消息调用官方 SDK 的全部 28 路 `Set_Zero`，将停止收线后的当前位置定义为零点；随后使用零电流命令获取反馈，全部 28 路连续 3 帧满足 `|pos|≤1°` 且 `error=0` 才打印检查通过，最后关闭驱动。这是重新定义坐标，不是转回旧零位。

归零反馈检查最多等待 90 秒（SDK 逐路归零在设备无应答时会阻塞），超时或失败会打印异常电机并以非零退出码关闭驱动，不宣称归零成功。驱动在此期间不会被终端的 Ctrl+C 提前结束。SIGTERM 只停止收线并关闭驱动，不触发归零。SDK 的 `Set_Zero_In_Init` 保持 `false`，其他 launch 不会因此在启动时归零。

该检查使用归零请求之后收到的 ROS 反馈；SDK 反馈不含设备时间戳或归零应答标记，连续零反馈不能替代逐电机硬件应答验证。归零改变电机位置坐标，之前使用旧零点采集的 workspace 和映射数据需要与新的零点保持一致。
Python 工具及模型在 `scripts/`。替换 SDK 时保留本包，详见工作区根目录的 `README.md`。
