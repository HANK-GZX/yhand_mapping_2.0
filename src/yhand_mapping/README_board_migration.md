# 板卡迁移 README

适用范围：

- `launch/wrist_five_finger_hybrid_fusion_right_launch.py`
- `scripts/wrist_five_finger_hybrid_fusion_mapping_main.py`
- `scripts/mocapapi_python-main/`
- `scripts/start_wrist_five_finger_hybrid_fusion_desktop.sh`
- `scripts/start_wrist_five_finger_hybrid_fusion.desktop`

本文重点回答两个问题：

1. 当前功能迁移到板卡后，`mocapapi_python-main` 是否还需要重新配置 `venv`
2. 桌面一键启动是否可以直接复制到板卡使用

---

## 1. 结论先看

### 1.1 `venv` 不是必须，但运行环境必须重新保证

迁移到板卡后，**不要求必须重新创建一个单独的 `venv`**。

但是，板卡上最终执行 `ros2 launch` 时所使用的 `python3` 环境，必须至少满足下面条件：

- 能导入 ROS2 Python 包：
  - `rclpy`
  - `tf2_ros`
  - `geometry_msgs`
- 能导入 mocap 依赖：
  - `docutils`
  - `pynput`
- 能导入当前控制链路依赖：
  - `numpy`
  - `torch`
- 能导入工作区编译产物：
  - `devices_pkg.msg`

也就是说：

- 你可以不用 `venv`
- 也可以继续使用 `venv`
- 但无论是否使用 `venv`，**都要保证 launch 真正调用到的是这个可用环境**

当前 `launch` 启动 mocap 节点时，直接执行的是系统 `python3`，并不会自动 `source venv`。

### 1.2 `.desktop` 不能无脑直接复制，`.sh` 可以复用但通常仍需板卡侧重新 build

如果只是复制桌面图标文件 `.desktop` 到板卡，**通常不能直接用**，因为它里面有绝对路径。

如果复制的是 `.sh` 启动脚本，本身更容易迁移，但仍然有前提：

- 板卡上 ROS2 环境正常
- 工作区已重新 `colcon build`
- `install/setup.bash` 存在
- 路径、IP、设备配置已经按板卡环境改好

---

## 2. 推荐迁移方式

推荐做法不是拷贝桌面机的 `build/install/log`，而是：

1. 迁移源码、模型、launch、脚本、配置文件
2. 在板卡上重新安装 Python 依赖
3. 在板卡上重新检查设备 IP / mocap 网络配置
4. 在板卡上重新执行 `colcon build`
5. 再在板卡上启动 launch 或一键脚本

这样最稳，不容易踩到下面这些差异：

- 桌面机和板卡 CPU 架构不同
- 板卡上的 ROS2 版本不同
- 板卡上的 Python site-packages 不同
- `devices_pkg.msg` 需要重新生成
- 设备 IP / 上位机 IP 改变

---

## 3. 建议一起打包过去的内容

如果你的目标是把“当前这套手腕 + 五指 + mocap”完整迁移到板卡，建议至少带上：

### 3.1 工作区源码

优先直接迁移整个工作区：

```text
YQ_SDK_ROS2-master/
```

如果不方便迁整个工作区，至少保证这些目录完整：

```text
src/devices_pkg/
src/yhand_mapping/
```

尤其不要漏掉：

- `yhand_mapping/launch/`
- `yhand_mapping/config/`
- `yhand_mapping/scripts/`
- `devices_pkg/sdk/`
- `devices_pkg/msg/`
- `devices_pkg/src/Y_Hand_Node/`
- 各种模型文件 `.pt` / `.pth` / `.json`

### 3.2 mocap SDK 目录

必须一起带上：

```text
src/yhand_mapping/scripts/mocapapi_python-main/
```

其中包含：

- Python 脚本
- `requirements_ros2.txt`
- SDK 动态库 `librobotapi_arm64.so` / `librobotapi_x86-64.so`

### 3.3 启动脚本

如果板卡也要保留当前的一键启动方式，再带上：

- `scripts/start_wrist_five_finger_hybrid_fusion_desktop.sh`
- `scripts/start_wrist_five_finger_hybrid_fusion.desktop`

---

## 4. 板卡上必须检查和修改的配置

### 4.1 手部设备配置

这几个文件决定了手部驱动连接哪块板、哪台上位机、哪只手：

- `config/right_hand_device_config.json`
- `config/left_hand_device_config.json`
- `config/bimanual_hand_device_config.json`

迁移到板卡后，重点检查：

- `pc_ip`
- `sn`
- `boards[].ip`
- `boards[].port`
- `hand`
- `cmd_topic`
- `feedback_topic`

如果板卡网络环境变了，这里通常都要重新确认。

### 4.2 mocap 网络配置

mocap 的 UDP 地址当前写在：

```text
scripts/mocapapi_python-main/mocap_main_base.py
```

重点检查：

- `settings.SetSettingsUDPEx(...)`
- `settings.SetSettingsUDPServer(...)`

如果板卡 IP、PN-Link 主节点 IP、端口和桌面机不一样，这里必须同步修改。

### 4.3 当前左右手设备连接配置

如果你还会用到：

```bash
ros2 launch yhand_mapping yhand_pretighten_launch.py
```

或者还会跑老的一键流程：

```text
对指demo一键启动.sh
```

那还要额外检查：

```text
config/left_hand_device_config.json
config/right_hand_device_config.json
```

此辅助流程分别读取左右手 JSON 文件的 pc_ip、sn 和 boards 字段（包括可选 port）；官方设备 launch 保留上游默认配置。

---

## 5. 板卡上是否需要重新建 `venv`

### 5.1 什么情况下不需要

如果板卡满足下面条件，就不需要额外创建 `venv`：

1. `python3` 能直接导入所需依赖
2. `ros2 launch` 启动时使用的就是这个 `python3`
3. `source /opt/ros/.../setup.bash` 和 `source install/setup.bash` 后，`devices_pkg.msg` 可导入

这种情况下，直接安装系统级或用户级 Python 包即可。

### 5.2 什么情况下建议创建

下面情况建议单独给板卡建一个 `venv`：

1. 板卡系统 Python 太脏，版本和依赖不好控
2. `torch`、`numpy`、`docutils`、`pynput` 和现有系统包冲突
3. 同一块板卡上还要跑别的 Python 项目

但要注意：

**光建 `venv` 不够。**

因为当前 launch 里拉起 mocap 脚本用的是直接 `python3`，如果想真正使用 `venv`，需要满足任一条件：

1. 启动前先激活 `venv`，并确保后续 `ros2 launch` 继承该环境
2. 后续手动把 launch / 脚本改成显式使用 `venv/bin/python`

如果不做这一步，创建了 `venv` 也不会自动生效。

---

## 6. 板卡上推荐安装步骤

以下假设工作区落在：

```text
/home/<user>/YQ_SDK_ROS2-master
```

### 6.1 安装 Python 依赖

先进入工作区：

```bash
cd /home/<user>/YQ_SDK_ROS2-master
```

如果不用 `venv`，可以直接安装：

```bash
python3 -m pip install -r src/yhand_mapping/scripts/mocapapi_python-main/requirements_ros2.txt
python3 -m pip install numpy torch
```

如果要使用 `venv`，可参考：

```bash
cd /home/<user>/YQ_SDK_ROS2-master
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r src/yhand_mapping/scripts/mocapapi_python-main/requirements_ros2.txt
python3 -m pip install numpy torch
```

### 6.2 重新编译 ROS2 工作区

如果你没有把可用的 `install/` 一起带过去，或者桌面机和板卡架构不同，建议直接重编：

```bash
cd /home/<user>/YQ_SDK_ROS2-master
source /opt/ros/<distro>/setup.bash
colcon build --packages-up-to yhand_mapping
source install/setup.bash
```

说明：

- `devices_pkg.msg` 需要 build 后才能稳定导入
- `Y_Hand_Node` 也依赖这一步生成可执行文件
- 即使你把桌面机的 `install/` 一起复制到板卡，也不建议跨架构直接复用

---

## 7. 桌面一键启动能否直接迁移

### 7.1 `.sh` 脚本

这个脚本比 `.desktop` 更适合迁移：

```text
scripts/start_wrist_five_finger_hybrid_fusion_desktop.sh
```

它会做这些事：

1. 定位工作区
2. 自动 source ROS2
3. 检查 `install/setup.bash`
4. 调用：

```bash
ros2 launch yhand_mapping wrist_five_finger_hybrid_fusion_right_launch.py ...
```

但要注意两点：

1. 默认工作区路径有写死值  
   如果板卡路径不是原路径，建议启动前设置：

```bash
export YHAND_WS_ROOT=/你的板卡工作区/YQ_SDK_ROS2-master
```

2. 它依赖 `install/setup.bash`  
   也就是板卡上必须已经 build 过。

### 7.2 `.desktop` 文件

这个文件：

```text
scripts/start_wrist_five_finger_hybrid_fusion.desktop
```

不能直接默认复用，原因有：

1. `Exec=` 是绝对路径
2. `Path=` 是绝对路径
3. 板卡可能没有桌面环境
4. 板卡可能没有 `zenity`
5. 板卡可能没有 `gnome-terminal`

所以如果要在板卡上继续用 `.desktop`：

1. 先改 `Exec`
2. 再改 `Path`
3. 确认板卡确实运行图形桌面
4. 确认装了 `zenity` 和 `gnome-terminal`

如果板卡是纯命令行环境，建议不要走 `.desktop`，直接执行 `.sh` 或直接跑 `ros2 launch`。

---

## 8. 板卡上的推荐启动方式

### 8.1 直接用 launch

这是最稳的方式：

```bash
cd /home/<user>/YQ_SDK_ROS2-master
source /opt/ros/<distro>/setup.bash
source install/setup.bash
ros2 launch yhand_mapping wrist_five_finger_hybrid_fusion_right_launch.py
```

左手示例：

```bash
ros2 launch yhand_mapping wrist_five_finger_hybrid_fusion_right_launch.py target_hand:=left
```

不启动 mocap 节点示例：

```bash
ros2 launch yhand_mapping wrist_five_finger_hybrid_fusion_right_launch.py \
  target_hand:=left \
  start_mocap_node:=false
```

### 8.2 用 `.sh` 一键脚本

如果板卡有命令行环境，但不一定有 GUI，建议这样：

```bash
cd /home/<user>/YQ_SDK_ROS2-master
export YHAND_WS_ROOT=/home/<user>/YQ_SDK_ROS2-master
bash src/yhand_mapping/scripts/start_wrist_five_finger_hybrid_fusion_desktop.sh
```

说明：

- 有 `zenity` 时，会弹选择框
- 没有 `zenity` 时，会退化成终端输入
- 有 `gnome-terminal` 时，会新开终端
- 没有 `gnome-terminal` 时，会直接在当前终端执行

---

## 9. 板卡迁移后的最小验证流程

建议按下面顺序检查：

### 9.1 Python 依赖检查

```bash
python3 -c "import docutils, pynput, numpy"
python3 -c "import rclpy, tf2_ros, geometry_msgs"
```

如果使用模型推理，再补一条：

```bash
python3 -c "import torch; print(torch.__version__)"
```

### 9.2 ROS2 工作区检查

```bash
cd /home/<user>/YQ_SDK_ROS2-master
source /opt/ros/<distro>/setup.bash
source install/setup.bash
python3 -c "from devices_pkg.msg import YHandMsg"
```

### 9.3 mocap 动态库检查

确认板卡架构和动态库匹配：

- `aarch64/arm64` 使用 `librobotapi_arm64.so`
- `x86_64` 使用 `librobotapi_x86-64.so`

### 9.4 功能检查

先只启动主要链路：

```bash
ros2 launch yhand_mapping wrist_five_finger_hybrid_fusion_right_launch.py start_mocap_node:=false
```

确认手部驱动、手套链路、模型推理正常后，再打开 mocap：

```bash
ros2 launch yhand_mapping wrist_five_finger_hybrid_fusion_right_launch.py start_mocap_node:=true
```

这样更容易定位问题到底出在：

- 手部驱动
- glove 数据
- mocap 网络
- Python 依赖

---

## 10. 常见问题

### 10.1 复制过去后 `.desktop` 点不开

优先检查：

1. `Exec` 是否还是桌面机绝对路径
2. `Path` 是否还是桌面机绝对路径
3. 板卡是否真的有图形桌面
4. 是否安装了 `zenity`

### 10.2 一键启动提示找不到 `install/setup.bash`

说明板卡上当前工作区还没 build，或者路径不对。

先执行：

```bash
cd /home/<user>/YQ_SDK_ROS2-master
source /opt/ros/<distro>/setup.bash
colcon build --packages-up-to yhand_mapping
source install/setup.bash
```

### 10.3 明明建了 `venv`，但 launch 还是报缺依赖

原因通常是：

- 你建了 `venv`
- 但 `ros2 launch` 实际调用的不是 `venv` 里的 Python

这时要么先激活 `venv` 再启动，要么改脚本让它显式使用 `venv/bin/python`。

### 10.4 mocap 启动了但没数据

优先检查：

1. `mocap_main_base.py` 里的 UDP 本机 IP 是否是板卡当前 IP
2. PN-Link 主节点 IP / 端口是否正确
3. 板卡网口是否和 mocap 设备处于同网段

### 10.5 手部驱动起不来或反馈为空

优先检查：

1. `right_hand_device_config.json` / `left_hand_device_config.json`
2. `pc_ip`
3. `boards[].ip`
4. `sn`
5. 板卡网络是否能访问对应手部控制板

---

## 11. 最推荐的实际操作方案

如果你的目标是“把当前功能尽快稳定迁到板卡”，推荐直接照下面做：

1. 把整个 `YQ_SDK_ROS2-master` 打包到板卡
2. 在板卡安装 Python 依赖，不强制单独建 `venv`
3. 修改 `config/*.json` 里的设备网络参数
4. 修改 `mocap_main_base.py` 里的 UDP 地址
5. 板卡上重新 `colcon build --packages-up-to yhand_mapping`
6. 先用 `ros2 launch ... start_mocap_node:=false` 验证手部和手指
7. 再打开 `start_mocap_node:=true` 验证手腕 mocap
8. 最后再决定要不要继续保留 `.desktop` 桌面入口

这个顺序最稳，也最容易排查问题。
