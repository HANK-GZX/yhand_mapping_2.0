# README - Y-hand Wrist Distance Formulas and Control

当前电机编号见 [新腕带电机对应表](README_motor_mapping.md)。模型轴顺序按肌肉保留，不能按新编号排序。

本文件是腕部几何公式与运行控制的配套说明，基于原始公式文档整理，并补充了当前工程的启动与使用方法。

## 1. 目标

- 描述腕部双自由度（侧摆 `alpha`、弯曲 `beta`）到 28 根线长变化的解析公式。
- 给出线长变化到电机角度指令的换算关系。
- 说明当前工程中如何配置 SN/IP/Port 并运行控制。

## 2. 相关文件

- 公式与坐标表：`/home/ubuntu/yhand/wrist_mapping/Yhand_wrist_coordinates_and_distance_formulas_combined_LR_code_input_deg.xlsx`
- 本说明：`/home/ubuntu/yhand/wrist_mapping/README_Yhand_wrist_distance_and_control.md`
- 控制工程根目录：`src/yhand_mapping`
- 独立配置：`/home/ubuntu/yhand/wrist_mapping/YQ_SDK_ROS2-master/src/yhand_mapping/config/right_hand_device_config.json` 或 `left_hand_device_config.json`
- 交互控制脚本：`/home/ubuntu/yhand/wrist_mapping/YQ_SDK_ROS2-master/src/yhand_mapping/scripts/wrist_angle_motor_control.py`
- 手套数据发布脚本：`/home/ubuntu/yhand/wrist_mapping/YQ_SDK_ROS2-master/src/yhand_mapping/scripts/minimal_example_dual_stream_commented.py`
- 手套驱动控制脚本：`/home/ubuntu/yhand/wrist_mapping/YQ_SDK_ROS2-master/src/yhand_mapping/scripts/wrist_glove_motor_control.py`
- 手套驱动 launch：`/home/ubuntu/yhand/wrist_mapping/YQ_SDK_ROS2-master/src/yhand_mapping/launch/wrist_glove_control_launch.py`
- 手套驱动配置：`/home/ubuntu/yhand/wrist_mapping/YQ_SDK_ROS2-master/src/yhand_mapping/config/right_hand_device_config.json`

## 3. 坐标与通用模型

单位：长度 `mm`；公式角度单位 `rad`。  
坐标系：`X` 垂直底板向上；`Y` 左侧到右侧；`Z` 前侧到后侧。  
旋转中心：`C_flex=(23.997,0,0)`（绕 `Y` 轴），`C_side=(7.997,0,0)`（绕 `Z` 轴）。

令：

- `alpha`：侧摆角（绕 `Z` 轴）
- `beta`：弯曲角（绕 `Y` 轴）

对于腕带孔 `P_i=(31.007, y_i, z_i)`，底板孔 `B_i=(0, yb_i, zb_i)`：

```text
A_i(beta)=16.000+7.010*cos(beta)+z_i*sin(beta)
X_i=7.997+A_i(beta)*cos(alpha)-y_i*sin(alpha)
Y_i=A_i(beta)*sin(alpha)+y_i*cos(alpha)
Z_i=-7.010*sin(beta)+z_i*cos(beta)
L_i(alpha,beta)=sqrt(X_i^2+(Y_i-yb_i)^2+(Z_i-zb_i)^2)
```

## 4. 线长变化到电机位置指令

设初始线长为 `L0_i`，当前角度下线长为 `L_i`：

```text
DeltaL_i = L_i - L0_i
```

滚轮直径 `d=4mm`，周长 `pi*d`，电机位置角度指令（deg）：

```text
pos_i(deg) = DeltaL_i * 360 / (pi * 4)
```

符号约定：

- `DeltaL_i > 0` -> `pos_i > 0`（放线）
- `DeltaL_i < 0` -> `pos_i < 0`（收线）

## 5. 运行方式（两种模式）

### 5.1 修改设备配置

编辑：

`/home/ubuntu/yhand/wrist_mapping/YQ_SDK_ROS2-master/src/yhand_mapping/config/right_hand_device_config.json`

或左手配置：

`/home/ubuntu/yhand/wrist_mapping/YQ_SDK_ROS2-master/src/yhand_mapping/config/left_hand_device_config.json`

关键字段：

- `pc_ip`
- `sn`
- `boards[].id / ip / port`
- `cmd_topic / feedback_topic`
- `motor_position_offsets_deg`（长度为 28 的数组，单位 deg，按电机 1..28 顺序）
- `apply_timeout_sec`
- `controller_start_delay_sec`

### 5.2 编译与启动

```bash
cd /home/ubuntu/yhand/wrist_mapping/YQ_SDK_ROS2-master
colcon build --packages-up-to yhand_mapping
source install/setup.bash
python3 src/yhand_mapping/scripts/wrist_angle_motor_control.py --hand right
```

左手改为：

```bash
python3 src/yhand_mapping/scripts/wrist_angle_motor_control.py --hand left
```

### 5.3 交互输入说明

控制脚本支持连续输入，直到你主动退出。

- 输入：`<alpha_deg> <beta_deg>`，例如 `10 -8`
- 输入：`z`，执行全部回 0
- 输入：`x`，退出（退出前自动回 0）
- 输入：`h`，显示帮助

说明：最终发布的每个电机位置指令都会叠加 `motor_position_offsets_deg[i]`。  
例如第 1 电机设置 `+2.0`，则最终 `pos_cmd_1 = pos_base_1 + 2.0`。

### 5.4 手套实时控制（left/right 通用解算）

这一模式不需要手动输入角度，程序会持续读取手套 wrist 姿态并实时下发 28 个电机命令。

#### 5.4.1 角度定义与零位

- 输入数据来自话题：`/glove_joint_data_raw_skeleton`
- wrist 数据布局：`node 0 = [x, y, z, qx, qy, qz, qw]`
- 解算方式：相对四元数 -> 旋转矩阵直接提取 `Rz(alpha)*Ry(beta)`
- 控制映射：
  - `alpha`（侧摆）= 绕 `z` 轴角度
  - `beta`（弯曲）= 绕 `y` 轴角度
- Manus 与公式使用同一坐标系，不做坐标系映射/轴互换
- 启动后第一帧有效 `hand` 对应 wrist 数据作为零位，后续控制使用相对零位差值

#### 5.4.2 配置文件

编辑：

`/home/ubuntu/yhand/wrist_mapping/YQ_SDK_ROS2-master/src/yhand_mapping/config/right_hand_device_config.json`

或左手配置：

`/home/ubuntu/yhand/wrist_mapping/YQ_SDK_ROS2-master/src/yhand_mapping/config/left_hand_device_config.json`

常用字段：

- `hand`：`left` 或 `right`，默认建议右手配置写 `right`
- `pc_ip / sn / boards`：当前所选手的设备连接信息
- `cmd_topic / feedback_topic`：电机命令与反馈话题
- `glove_topic`：默认 `/glove_joint_data_raw_skeleton`
- `glove_side_code`：左手默认 `0.0`，右手默认 `1.0`
- `fallback_chunk_index`：当 `side_code` 不可用时，可强制取第几个 176 维块
- `motor_position_offsets_deg`：28 个电机独立 offset（deg）
- `alpha_scale / beta_scale`：侧摆和弯曲比例
- `alpha_offset_deg / beta_offset_deg`：整体角度偏置
- `alpha_limit_deg`：侧摆安全限位，默认 `±5deg`
- `beta_limit_deg`：弯曲安全限位，默认 `±30deg`
- `output_publish_rate_hz`：最终 `cmd_topic` 电机命令发布频率（真正下发到驱动）
- `deadband_deg`：手套角度死区，用于过滤极小抖动
- `zero_on_exit`：退出时是否执行完整回 0（含预放线 + 回零超时逻辑）

#### 5.4.3 启动步骤

1) 编译并 source：

```bash
cd /home/ubuntu/yhand/wrist_mapping/YQ_SDK_ROS2-master
colcon build --packages-up-to yhand_mapping
source install/setup.bash
```

2) 一条 launch 直接启动（默认同时拉起：手套发布 + 当前所选手驱动 + wrist 控制）：

```bash
ros2 launch yhand_mapping wrist_glove_control_launch.py
```

左手：

```bash
ros2 launch yhand_mapping wrist_glove_control_launch.py target_hand:=left
```

如果你只想禁用其中某个子进程，可用参数控制：

```bash
ros2 launch yhand_mapping wrist_glove_control_launch.py start_glove_node:=false
ros2 launch yhand_mapping wrist_glove_control_launch.py start_driver:=false
```

#### 5.4.4 运行与退出

- 运行中会持续实时控制，直到你主动退出
- 使用 `Ctrl+C` 退出
- 默认退出前会执行全部回 0（与手动角度控制一致）

#### 5.4.5 常见问题：收不到手套数据

1) 检查控制脚本日志里的 `pubs`：
   - `pubs=0`：说明 `/glove_joint_data_raw_skeleton` 没有发布者，确认你没有把 `start_glove_node:=false` 关闭，然后重新执行默认 launch。
2) 若日志出现 `side_codes=[-1.0]` 或 `Serial Number not found`：
   - 说明手套 SN 与 `minimal_example_dual_stream_commented.py` 里的 `left_glove_sn/right_glove_sn` 不匹配；
   - 这种情况下可先使用配置中的：
     - `allow_single_chunk_fallback=true`（单手数据时可继续用）
     - `fallback_chunk_index=1`（双手且 side_code 无法区分时，固定取第 2 块）

## 6. 附录：右手 28 电机独立公式（新腕带编号）

### M19 左前手腕
- A_19(beta)=16.000+7.010*cos(beta)-9.250*sin(beta)
- X_19=7.997+A_19(beta)*cos(alpha)+19.250*sin(alpha)
- Y_19=A_19(beta)*sin(alpha)-19.250*cos(alpha)
- Z_19=-7.010*sin(beta)-9.250*cos(beta)
- L_19(alpha,beta)=sqrt(X_19^2+(Y_19+15.441)^2+(Z_19+1.351)^2)

### M4 小指内收
- A_4(beta)=16.000+7.010*cos(beta)-17.000*sin(beta)
- X_4=7.997+A_4(beta)*cos(alpha)+10.000*sin(alpha)
- Y_4=A_4(beta)*sin(alpha)-10.000*cos(alpha)
- Z_4=-7.010*sin(beta)-17.000*cos(beta)
- L_4(alpha,beta)=sqrt(X_4^2+(Y_4+14.972)^2+(Z_4+4.012)^2)

### M27 小指屈肌
- A_27(beta)=16.000+7.010*cos(beta)-15.250*sin(beta)
- X_27=7.997+A_27(beta)*cos(alpha)+4.500*sin(alpha)
- Y_27=A_27(beta)*sin(alpha)-4.500*cos(alpha)
- Z_27=-7.010*sin(beta)-15.250*cos(beta)
- L_27(alpha,beta)=sqrt(X_27^2+(Y_27+14.048)^2+(Z_27+6.551)^2)

### M28 拇对掌肌
- A_28(beta)=16.000+7.010*cos(beta)-12.750*sin(beta)
- X_28=7.997+A_28(beta)*cos(alpha)+3.000*sin(alpha)
- Y_28=A_28(beta)*sin(alpha)-3.000*cos(alpha)
- Z_28=-7.010*sin(beta)-12.750*cos(beta)
- L_28(alpha,beta)=sqrt(X_28^2+(Y_28+13.295)^2+(Z_28+2.344)^2)

### M2 中指指浅屈肌
- A_2(beta)=16.000+7.010*cos(beta)-15.250*sin(beta)
- X_2=7.997+A_2(beta)*cos(alpha)+1.500*sin(alpha)
- Y_2=A_2(beta)*sin(alpha)-1.500*cos(alpha)
- Z_2=-7.010*sin(beta)-15.250*cos(beta)
- L_2(alpha,beta)=sqrt(X_2^2+(Y_2+12.697)^2+(Z_2+8.890)^2)

### M15 食指内收
- A_15(beta)=16.000+7.010*cos(beta)-12.750*sin(beta)
- X_15=7.997+A_15(beta)*cos(alpha)-0.000*sin(alpha)
- Y_15=A_15(beta)*sin(alpha)+0.000*cos(alpha)
- Z_15=-7.010*sin(beta)-12.750*cos(beta)
- L_15(alpha,beta)=sqrt(X_15^2+(Y_15+12.686)^2+(Z_15+4.617)^2)

### M3 无名指屈肌
- A_3(beta)=16.000+7.010*cos(beta)-17.750*sin(beta)
- X_3=7.997+A_3(beta)*cos(alpha)+3.000*sin(alpha)
- Y_3=A_3(beta)*sin(alpha)-3.000*cos(alpha)
- Z_3=-7.010*sin(beta)-17.750*cos(beta)
- L_3(alpha,beta)=sqrt(X_3^2+(Y_3+11.691)^2+(Z_3+6.750)^2)

### M17 食指指浅屈肌
- A_17(beta)=16.000+7.010*cos(beta)-17.750*sin(beta)
- X_17=7.997+A_17(beta)*cos(alpha)-0.000*sin(alpha)
- Y_17=A_17(beta)*sin(alpha)+0.000*cos(alpha)
- Z_17=-7.010*sin(beta)-17.750*cos(beta)
- L_17(alpha,beta)=sqrt(X_17^2+(Y_17-11.691)^2+(Z_17+6.750)^2)

### M16 中指指深屈肌
- A_16(beta)=16.000+7.010*cos(beta)-15.250*sin(beta)
- X_16=7.997+A_16(beta)*cos(alpha)-1.500*sin(alpha)
- Y_16=A_16(beta)*sin(alpha)+1.500*cos(alpha)
- Z_16=-7.010*sin(beta)-15.250*cos(beta)
- L_16(alpha,beta)=sqrt(X_16^2+(Y_16-12.686)^2+(Z_16+4.617)^2)

### M1 食指指深屈肌
- A_1(beta)=16.000+7.010*cos(beta)-17.750*sin(beta)
- X_1=7.997+A_1(beta)*cos(alpha)-3.000*sin(alpha)
- Y_1=A_1(beta)*sin(alpha)+3.000*cos(alpha)
- Z_1=-7.010*sin(beta)-17.750*cos(beta)
- L_1(alpha,beta)=sqrt(X_1^2+(Y_1-13.295)^2+(Z_1+2.344)^2)

### M14 食指外展
- A_14(beta)=16.000+7.010*cos(beta)-12.750*sin(beta)
- X_14=7.997+A_14(beta)*cos(alpha)-3.000*sin(alpha)
- Y_14=A_14(beta)*sin(alpha)+3.000*cos(alpha)
- Z_14=-7.010*sin(beta)-12.750*cos(beta)
- L_14(alpha,beta)=sqrt(X_14^2+(Y_14-12.697)^2+(Z_14+8.890)^2)

### M18 拇长屈肌
- A_18(beta)=16.000+7.010*cos(beta)-15.250*sin(beta)
- X_18=7.997+A_18(beta)*cos(alpha)-4.500*sin(alpha)
- Y_18=A_18(beta)*sin(alpha)+4.500*cos(alpha)
- Z_18=-7.010*sin(beta)-15.250*cos(beta)
- L_18(alpha,beta)=sqrt(X_18^2+(Y_18-14.048)^2+(Z_18+6.551)^2)

### M5 右前手腕
- A_5(beta)=16.000+7.010*cos(beta)-10.250*sin(beta)
- X_5=7.997+A_5(beta)*cos(alpha)-13.500*sin(alpha)
- Y_5=A_5(beta)*sin(alpha)+13.500*cos(alpha)
- Z_5=-7.010*sin(beta)-10.250*cos(beta)
- L_5(alpha,beta)=sqrt(X_5^2+(Y_5-14.972)^2+(Z_5+4.012)^2)

### M13 拇展肌
- A_13(beta)=16.000+7.010*cos(beta)-17.000*sin(beta)
- X_13=7.997+A_13(beta)*cos(alpha)-19.000*sin(alpha)
- Y_13=A_13(beta)*sin(alpha)+19.000*cos(alpha)
- Z_13=-7.010*sin(beta)-17.000*cos(beta)
- L_13(alpha,beta)=sqrt(X_13^2+(Y_13-15.441)^2+(Z_13+1.351)^2)

### M12 左后手腕
- A_12(beta)=16.000+7.010*cos(beta)+10.250*sin(beta)
- X_12=7.997+A_12(beta)*cos(alpha)+18.000*sin(alpha)
- Y_12=A_12(beta)*sin(alpha)-18.000*cos(alpha)
- Z_12=-7.010*sin(beta)+10.250*cos(beta)
- L_12(alpha,beta)=sqrt(X_12^2+(Y_12+14.972)^2+(Z_12-4.012)^2)

### M6 小指外展
- A_6(beta)=16.000+7.010*cos(beta)+11.326*sin(beta)
- X_6=7.997+A_6(beta)*cos(alpha)+15.380*sin(alpha)
- Y_6=A_6(beta)*sin(alpha)-15.380*cos(alpha)
- Z_6=-7.010*sin(beta)+11.326*cos(beta)
- L_6(alpha,beta)=sqrt(X_6^2+(Y_6+15.441)^2+(Z_6-1.351)^2)

### M25 小指伸肌
- A_25(beta)=16.000+7.010*cos(beta)+15.000*sin(beta)
- X_25=7.997+A_25(beta)*cos(alpha)+10.000*sin(alpha)
- Y_25=A_25(beta)*sin(alpha)-10.000*cos(alpha)
- Z_25=-7.010*sin(beta)+15.000*cos(beta)
- L_25(alpha,beta)=sqrt(X_25^2+(Y_25+12.697)^2+(Z_25-8.890)^2)

### M7 无名指外展
- A_7(beta)=16.000+7.010*cos(beta)+11.326*sin(beta)
- X_7=7.997+A_7(beta)*cos(alpha)+11.569*sin(alpha)
- Y_7=A_7(beta)*sin(alpha)-11.569*cos(alpha)
- Z_7=-7.010*sin(beta)+11.326*cos(beta)
- L_7(alpha,beta)=sqrt(X_7^2+(Y_7+14.048)^2+(Z_7-6.551)^2)

### M24 无名指伸肌
- A_24(beta)=16.000+7.010*cos(beta)+14.837*sin(beta)
- X_24=7.997+A_24(beta)*cos(alpha)+6.121*sin(alpha)
- Y_24=A_24(beta)*sin(alpha)-6.121*cos(alpha)
- Z_24=-7.010*sin(beta)+14.837*cos(beta)
- L_24(alpha,beta)=sqrt(X_24^2+(Y_24+13.295)^2+(Z_24-2.344)^2)

### M8 无名指内收
- A_8(beta)=16.000+7.010*cos(beta)+11.326*sin(beta)
- X_8=7.997+A_8(beta)*cos(alpha)+2.165*sin(alpha)
- Y_8=A_8(beta)*sin(alpha)-2.165*cos(alpha)
- Z_8=-7.010*sin(beta)+11.326*cos(beta)
- L_8(alpha,beta)=sqrt(X_8^2+(Y_8+12.686)^2+(Z_8-4.617)^2)

### M23 中指内收
- A_23(beta)=16.000+7.010*cos(beta)+11.326*sin(beta)
- X_23=7.997+A_23(beta)*cos(alpha)-1.835*sin(alpha)
- Y_23=A_23(beta)*sin(alpha)+1.835*cos(alpha)
- Z_23=-7.010*sin(beta)+11.326*cos(beta)
- L_23(alpha,beta)=sqrt(X_23^2+(Y_23+11.691)^2+(Z_23-6.750)^2)

### M9 中指伸肌
- A_9(beta)=16.000+7.010*cos(beta)+14.779*sin(beta)
- X_9=7.997+A_9(beta)*cos(alpha)-0.335*sin(alpha)
- Y_9=A_9(beta)*sin(alpha)+0.335*cos(alpha)
- Z_9=-7.010*sin(beta)+14.779*cos(beta)
- L_9(alpha,beta)=sqrt(X_9^2+(Y_9-11.691)^2+(Z_9-6.750)^2)

### M10 中指外展
- A_10(beta)=16.000+7.010*cos(beta)+11.326*sin(beta)
- X_10=7.997+A_10(beta)*cos(alpha)-5.610*sin(alpha)
- Y_10=A_10(beta)*sin(alpha)+5.610*cos(alpha)
- Z_10=-7.010*sin(beta)+11.326*cos(beta)
- L_10(alpha,beta)=sqrt(X_10^2+(Y_10-12.686)^2+(Z_10-4.617)^2)

### M11 食指伸肌
- A_11(beta)=16.000+7.010*cos(beta)+13.735*sin(beta)
- X_11=7.997+A_11(beta)*cos(alpha)-7.360*sin(alpha)
- Y_11=A_11(beta)*sin(alpha)+7.360*cos(alpha)
- Z_11=-7.010*sin(beta)+13.735*cos(beta)
- L_11(alpha,beta)=sqrt(X_11^2+(Y_11-12.697)^2+(Z_11-8.890)^2)

### M22 拇收肌
- A_22(beta)=16.000+7.010*cos(beta)+11.326*sin(beta)
- X_22=7.997+A_22(beta)*cos(alpha)-9.684*sin(alpha)
- Y_22=A_22(beta)*sin(alpha)+9.684*cos(alpha)
- Z_22=-7.010*sin(beta)+11.326*cos(beta)
- L_22(alpha,beta)=sqrt(X_22^2+(Y_22-13.295)^2+(Z_22-2.344)^2)

### M26 右后手腕
- A_26(beta)=16.000+7.010*cos(beta)+13.735*sin(beta)
- X_26=7.997+A_26(beta)*cos(alpha)-11.078*sin(alpha)
- Y_26=A_26(beta)*sin(alpha)+11.078*cos(alpha)
- Z_26=-7.010*sin(beta)+13.735*cos(beta)
- L_26(alpha,beta)=sqrt(X_26^2+(Y_26-14.048)^2+(Z_26-6.551)^2)

### M21 拇长伸肌
- A_21(beta)=16.000+7.010*cos(beta)+11.326*sin(beta)
- X_21=7.997+A_21(beta)*cos(alpha)-12.635*sin(alpha)
- Y_21=A_21(beta)*sin(alpha)+12.635*cos(alpha)
- Z_21=-7.010*sin(beta)+11.326*cos(beta)
- L_21(alpha,beta)=sqrt(X_21^2+(Y_21-14.972)^2+(Z_21-4.012)^2)

### M20 拇短伸肌
- A_20(beta)=16.000+7.010*cos(beta)+11.326*sin(beta)
- X_20=7.997+A_20(beta)*cos(alpha)-19.000*sin(alpha)
- Y_20=A_20(beta)*sin(alpha)+19.000*cos(alpha)
- Z_20=-7.010*sin(beta)+11.326*cos(beta)
- L_20(alpha,beta)=sqrt(X_20^2+(Y_20-15.441)^2+(Z_20-1.351)^2)
