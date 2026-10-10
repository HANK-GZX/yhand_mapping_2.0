# Motor Position Test（简版）

当前电机编号见 [新腕带电机对应表](README_motor_mapping.md)。模型轴顺序按肌肉保留，不能按新编号排序。

对应文件：

- `launch/motor_position_test_launch.py`
- `scripts/motor_position_test.py`
- `scripts/muscle_adjustment_defaults.py`

---

## 1. 启动

左手：

```bash
ros2 launch yhand_mapping motor_position_test_launch.py target_hand:=left
```

右手：

```bash
ros2 launch yhand_mapping motor_position_test_launch.py target_hand:=right
```

可选关闭对抗肌辅助：

```bash
ros2 launch yhand_mapping motor_position_test_launch.py target_hand:=left assist_enable:=false
```

说明：

- 启动后会拉起目标手 `Y_Hand_Node`。
- 自动打开独立 `gnome-terminal` 运行交互脚本（可直接键盘输入）。

---

## 2. 操作键位

电机选择界面：

- `1-28`：选择电机
- `z`：全部回0
- `x`：退出程序（会先执行全部回0）

目标位置输入界面：

- 输入数字：作为目标位置 `pos`
- `m`：切换电机
- `z`：全部回0
- `x`：退出程序（会先执行全部回0）

---

## 3. 运行流程

1. 选择电机（1-28）。
2. 显示该电机默认参数：`vel / tor_active / kp / kd / idle_tor`。
3. 显示该电机当前反馈：`pos / vel / tor / kp / kd`。
4. 输入目标位置 `pos`，执行单电机运动。
5. 到位后保持当前位置并打印：
   - 目标位置
   - 位置误差
   - 停止后反馈
6. 继续输入下一个 `pos`，或 `m/z/x` 执行切换、回0、退出。

---

## 4. 控制规则（当前实现）

- 主动电机（你选中的电机）：
  - `pos`：终端输入
  - `vel / tor_active / kp / kd`：读取 `muscle_adjustment_defaults.py`
- 非主动电机：
  - 固定纯张紧：`[0, 0, tor, 0, 0]`，其中 `tor` 读取 defaults

对抗肌辅助（默认开启）：

- 若当前主动电机属于伸屈/侧摆映射，会在动作前短时对对抗肌施加放线扭矩
- 参数与 `muscle_adjustment.py` 对齐：
  - `ASSIST_RELEASE_TOR=0.5`
  - `ASSIST_RELEASE_SEC=0.4s`
- 支持范围：四指伸屈、四指侧摆、拇指伸屈与侧摆（双向映射）

判定与超时：

- 到位判定：`|error| <= 7` 且连续 `30` 帧
- 单电机超时：`5s`
- 全部回0超时：`6s`
