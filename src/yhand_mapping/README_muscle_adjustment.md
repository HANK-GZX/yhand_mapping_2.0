# Muscle Adjustment 功能说明（当前代码版）

当前电机编号见 [新腕带电机对应表](README_motor_mapping.md)。模型轴顺序按肌肉保留，不能按新编号排序。

本文档对应以下文件：

- `launch/muscle_adjustment_launch.py`
- `scripts/muscle_adjustment.py`
- `scripts/muscle_adjustment_defaults.py`

用于在**单手（左/右二选一）在线不断电**状态下，逐电机调试并保存默认参数。

---

## 1. 功能目标

1. 不依赖 `Test_Y_Hand_Node` 实现预紧与调参。
2. 支持交互式选电机并下发命令。
3. 未选中电机始终保持纯张紧（`kp=0,kd=0,tor=默认`）。
4. 支持保存默认参数到 Python 文件 + Excel。
5. 退出时执行全电机回零，再回纯张紧。

---

## 2. 与 yhand 配置关系

`muscle_adjustment_launch.py` 不写死 IP/SN，运行时从 `config/left_hand_device_config.json` / `config/right_hand_device_config.json` 读取：

- 左手：`config/left_hand_device_config.json`
- 右手：`config/right_hand_device_config.json`

所以设备连接参数由本包的左右手 JSON 配置文件 提供。

---

## 3. 启动方式

### 3.1 选择左右手

编辑 `launch/muscle_adjustment_launch.py`：

```python
TARGET_HAND = "left"   # 或 "right"
```

### 3.2 启动命令

```bash
ros2 launch yhand_mapping muscle_adjustment_launch.py
```

启动后会：

1. 只启动目标手 `Y_Hand_Node`。
2. 延时后自动拉起独立 `gnome-terminal` 运行 `muscle_adjustment.py`（可正常键盘交互）。
3. 不启动 `test_y_hand_node`。

### 3.3 话题映射

- 左手：`/yhand_left_cmd` + `/yhand_left_feedback`
- 右手：`/yhand_right_cmd` + `/yhand_right_feedback`

---

## 4. 指令模式（当前实现）

进入电机后支持 3 种模式：

1. 位置模式（原 5 维）

```text
pos vel tor kp kd
```

示例：

```text
-500 0 -0.03 1000 10
```

2. 电流模式（定时）

```text
tor <tor> <sec>
```

示例：

```text
tor -0.05 2
```

3. 速度模式（定时）

```text
vel <vel> <sec> [tor] [kp] [kd]
```

示例：

```text
vel -20 1.5
vel -20 1.5 -0.03 500 10
```

说明：

- 位置模式：按位置误差判定到位。
- 电流/速度模式：按持续时间结束，不做位置到位判定。

---

## 5. 关键运行逻辑

### 5.1 空闲状态

全电机纯张紧：

- `pos=0`
- `vel=0`
- `tor=defaults[hand]["tor"][i]`（或全局预紧值）
- `kp=0`
- `kd=0`

### 5.2 到位后保持（仅位置模式）

当位置模式判定 `reached` 后，默认会保持当前电机目标：

- 由 `HOLD_SELECTED_AFTER_REACHED = True` 控制。

### 5.3 超时参数（已分离）

- 单电机动作超时：`SINGLE_MOTOR_TIMEOUT_SEC`
- 回零流程超时：`RETURN_ZERO_TIMEOUT_SEC`

默认当前为：单电机 `5.0s`，回零 `6.0s`，可独立修改。

### 5.4 全局预紧

在电机选择界面：

- `p`：开启全局预紧（全部电机 `tor=-0.05A`，`kp=0,kd=0`）
- `n`：恢复默认张紧（按 defaults 每电机 tor）

### 5.5 伸屈/侧摆辅助（当前实现）

1. 仅在映射到 `EXTENSOR_ASSIST_MAP` 的相关电机上生效（双向：A 主动会放 B，B 主动也会放 A）。  
2. 主动电机执行前，会对“对抗肌”短时施加放线扭矩：  
   - 模式：`release_tor`  
   - 放线扭矩：`ASSIST_RELEASE_TOR=0.5`  
   - 放线时长：`ASSIST_RELEASE_SEC=0.5s`  
3. 放线阶段结束后，自动恢复到标准单电机命令（非主动电机回 defaults 小张紧）。  
4. 电机选择界面和执行后界面都可按 `a` 切换辅助偏好（`off` / `release_tor`）。
5. 当前包含的对抗关系：
   - 四指伸屈（食/中含浅屈+深屈）
   - 四指侧摆（外展/内收）
   - 拇指伸屈与侧摆（外展/内收）
6. 当前不包含“拇指对掌肌”的单独辅助放线映射。

### 5.6 切换电机时缓释

输入 `m` 切换电机时，上一电机会先执行短时零电流缓释（`SWITCH_RELEASE_SEC=0.8s`），然后回到 defaults 小张紧，最后打印全部电机反馈与当前下发命令。

---

## 6. 交互按键语义

### 6.1 电机选择界面

- 输入 `1-28`：进入该电机
- `p`：开启全局预紧
- `n`：恢复默认张紧
- `a`：切换辅助偏好（`off/release_tor`）
- `x`：退出程序（回零+张紧）

### 6.2 指令输入界面

- `q`：回到电机选择（回零+张紧）
- `m`：切换电机（不退出程序）
- `a`：切换辅助偏好（`off/release_tor`）
- `x`：退出程序（回零+张紧）

### 6.3 执行后“下一步”界面

- `c`：继续调当前电机
- `s`：保存默认参数
- `m`：切换电机
- `a`：切换辅助偏好（`off/release_tor`）
- `q`：回到电机选择（回零+张紧）
- `x`：退出程序（回零+张紧）

注意：`c` 只能在“下一步”界面输入，不能在“输入指令”提示下输入。

---

## 7. 保存逻辑（干净版）

按 `s` 保存时按模式处理：

1. 位置模式：保存 `vel/KP/KD/tor`
2. 电流模式：只保存 `tor`
3. 速度模式：保存 `vel/KP/KD/tor`

保存目标：

1. `scripts/muscle_adjustment_defaults.py`
2. `scripts/motor_test_logs/muscle_adjustment_<left|right>.xlsx`
   - sheet: `<hand>_defaults`
   - 字段：`motor_id, vel_default, kp, kd, tor_default, tor_active_default, updated_at`

为保证文件干净，Excel 不保留历史交互日志 sheet。

---

## 8. 回零与退出

退出流程：

1. 全电机从当前位姿插值回零。
2. 判定阈值：`|pos| <= ARRIVAL_POS_TOL`（当前默认 `7`）。
3. 若超时则停止回零，直接切回纯张紧退出。

---

## 9. 常用可调参数（scripts/muscle_adjustment.py）

- `ARRIVAL_POS_TOL`
- `ARRIVAL_CONFIRM_FRAMES`
- `SINGLE_MOTOR_TIMEOUT_SEC`
- `RETURN_ZERO_TIMEOUT_SEC`
- `GLOBAL_PRETIGHTEN_TOR`
- `HOLD_SELECTED_AFTER_REACHED`
- `FEEDBACK_STALE_SEC`
- `ASSIST_RELEASE_TOR`
- `ASSIST_RELEASE_SEC`
- `SWITCH_RELEASE_SEC`

---

## 10. 常见问题速查

1. 提示“未收到反馈数据”
   - 检查设备在线、`config_dev` 正确、反馈话题是否有发布。

2. 提示命令话题冲突（publisher>1）
   - 有其他节点在同一 `cmd_topic` 发命令，先停掉。

3. 输入 `tor` 感觉“不动”
   - `tor` 是力/电流前馈，不保证目标位姿；看 `pos` 是否有变化，并可增大 `sec` 或改用 `vel` 模式验证。

4. 频繁 `feedback_stale`
   - 反馈中断，程序会自动回纯张紧；检查网络与驱动进程。

5. `Cannot import devices_pkg.msg`
   - 先 source 工作空间。

6. `openpyxl is required`
   - 安装 `openpyxl` 后再保存 Excel。
