# Y-Hand Manus Calibration 使用说明

当前电机编号见 [新腕带电机对应表](README_motor_mapping.md)。模型轴顺序按肌肉保留，不能按新编号排序。

本说明对应脚本：
- `src/yhand_mapping/scripts/manus_calibration.py`
- `src/yhand_mapping/launch/manus_calibration_launch.py`

当前版本特性：
- 支持 `action 0/1/2/3`。
- `action 2/3` 为持续循环动作，运行中可在脚本终端输入 `o` + `回车` 停止。

## 1. 前置条件

1. ROS2 工作空间已编译并 `source`。
2. `devices_pkg.msg` 可正常导入。
3. 使用 launch 方式时，系统已安装 `gnome-terminal`。
4. `config/left_hand_device_config.json` 中左手设备连接参数正确。

## 2. 启动方式

### 2.1 Y-Hand 电机标零

```bash
ros2 launch yhand_mapping yhand_pretighten_launch.py
```

说明：
- 运行期间以 `−0.05 A` 预紧；按 Ctrl+C 后停止收线、调用 SDK 设置全部 28 路当前位置为零点，再检查归零后的反馈并关闭驱动。
- 终端会打印 28 路位置；连续 3 帧全部满足 `|pos|≤1°` 且 `error=0` 才显示“零点反馈检查通过”。失败时会报告异常电机或缺失反馈。SDK 启动归零配置仍为 `false`。

### 2.2 启动手套校准launch：

```bash
ros2 launch yhand_mapping manus_calibration_launch.py
```

说明：
- 会拉起左手 `Y_Hand_Node` 驱动。
- 会新开一个 terminal 运行 `manus_calibration.py`。
- 可在 `manus_calibration_launch.py` 里设置 `ACTION_INDEX`：
  - `""`：交互菜单选择动作（推荐）
  - `"0"`, `"1"`, `"2"`, `"3"`：固定首个动作


## 3. 动作定义

### Action 0：回零（位置控制）

- 所有电机回到 0 位。
- 完成后保持在 0 位（`hold_zero_pos`）。

### Action 1：四指握拳（电流控制）

- 全部电机先以 `-0.01A` 张紧 `0.5s`，随后握拳 `5s`。
- 握拳阶段：食指深屈、中指深屈、无名指屈肌、小指屈肌使用 `-0.2A`；其余电机（包括食指浅屈）使用 `+0.05A` 放线，全部 `kp/kd=0`。
- 左手主动电机：`15/2/17/13`；右手主动电机：`1/16/3/27`。
- 动作结束后记录 28 路电机位置（供 Action 2 使用）。
- 动作结束后的原有小电流保持逻辑不变。

### Action 2：张开-闭合循环（位置控制）

- 前提：必须先做过 Action 1（要有记录位姿）。
- 逻辑：在“零位（张开）”与“Action 1 记录位姿（闭合）”之间循环切换。
- 控制方式：全部 28 路保留位置控制的 `kp/kd`，叠加以下四指电流前馈；未指定的通道沿用原参数，Action 1 本身不变。
- 回零 2s：全部目标位置为 `0`；四指伸肌使用 `-0.2A` 收线，四指屈肌（含食指/中指浅屈）使用 `+0.2A` 放线。
- 闭合 2s：全部目标位置为 Action 1 记录的 28 路位置；四指伸肌使用 `+0.2A` 放线，四指深屈/屈肌使用 `-0.15A` 收线。浅屈沿用原电流前馈。
- 左手伸肌：`25/23/10/11`；深屈/屈肌：`15/2/17/13`；浅屈：`3/16`。右手按对应肌肉使用相同电流。
- 当前节拍：`ACTION2_TOGGLE_INTERVAL_SEC`（默认 2s）。
- 停止方式：运行中输入 `o` 回车。
- 停止后行为：下发上述回零命令并持续保持，进入动作选择菜单，不等待实际到位。

### Action 3：对掌往复循环（混合控制）

- 启动时记录当前 28 路位置作为“初始位姿”。
- 左手对掌阶段（2.4s）：电机 `14` 对掌使用 `-0.25A`，电机 `27` 外展使用 `-0.2A`，电机 `4` 长屈使用 `-0.25A`；拇指其余电机 `6/7/8` 使用 `+0.1A` 放线，以上 6 路均 `kp/kd=0`。
- 对掌阶段其他四指的 18 路电机使用位置控制保持初始位姿；剩余 4 路腕部电机使用 `-0.01A` 张紧，`kp/kd=0`。
- 回位阶段（1.0s）：全部 28 路使用与 Action 0 一致的位控参数返回记录的初始位姿（保留原有电流前馈）。
- 右手对掌电流沿用原参数。
- 当前节拍：
  - `ACTION3_OPPOSITION_HOLD_SEC`（默认 2.4s）
  - `ACTION3_RETURN_HOLD_SEC`（默认 1.0s）
- 停止方式：运行中输入 `o` 回车。
- 停止后行为：下发初始位姿保持命令并进入动作选择菜单，不等待实际到位。

## 4. 运行时交互

交互菜单输入：
- `0/1/2/3`：执行对应动作
- `x`：退出

循环动作运行中（Action 2/3）：
- 输入 `o` + 回车：请求停止当前循环动作

## 5. 常用参数调节

可在 `manus_calibration.py` 顶部常量区调整：

- 通用：
  - `DEFAULT_HOLD_CURRENT`
  - `INIT_HOLD_CURRENT`
- Action 0：
  - `ACTION0_RETURN_KP`
  - `ACTION0_RETURN_KD`
  - `ACTION0_RETURN_TOR`
- Action 2：
  - `ACTION2_TOGGLE_INTERVAL_SEC`
  - `ACTION2_OPEN_EXT_CURRENT` / `ACTION2_OPEN_FLEX_CURRENT`
  - `ACTION2_CLOSE_EXT_CURRENT` / `ACTION2_CLOSE_FLEX_CURRENT`
  - `ACTION2_POS_CTRL_KP`
  - `ACTION2_POS_CTRL_KD`
- Action 3：
  - `ACTION3_OPPOSITION_CURRENT`
  - `ACTION3_ABDUCTION_CURRENT`
  - `ACTION3_LONG_FLEX_CURRENT`
  - `ACTION3_LEFT_OPPOSITION_CURRENT`
  - `ACTION3_LEFT_ABDUCTION_CURRENT`
  - `ACTION3_LEFT_LONG_FLEX_CURRENT`
  - `ACTION3_LEFT_RELEASE_CURRENT`
  - `ACTION3_OPPOSITION_HOLD_SEC`
  - `ACTION3_RETURN_HOLD_SEC`

## 6. 常见问题

1. 一直提示 waiting for feedback  
检查驱动是否已正常发布反馈 topic，检查手侧、topic 名称和网络配置。

2. Action 2 aborted: Action 1 positions not recorded  
先执行一次 Action 1，再执行 Action 2。

3. 输入 `o` 没反应  
请确认是在运行 `manus_calibration.py` 的那个交互终端里输入，并按回车。

4. launch 报 gnome-terminal not found  
安装 `gnome-terminal`，或手动分终端启动驱动与脚本。
