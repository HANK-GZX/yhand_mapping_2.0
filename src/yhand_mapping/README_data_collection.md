# Data Collection 功能说明（当前实现，2026-09）

当前电机编号见 [新腕带电机对应表](README_motor_mapping.md)。模型轴顺序按肌肉保留，不能按新编号排序。

本文档对应当前代码：

- `launch/data_collection_left_independent_launch.py`
- `launch/data_collection_right_independent_launch.py`
- `launch/data_collection_launch.py`（legacy）
- `scripts/data_collection_main.py`
- `scripts/data_collection_config.py`
- `scripts/data_collection_health_monitor.py`

并依赖：

- `scripts/muscle_adjustment_defaults.py`（默认 `vel/tor/kp/kd`）
- `scripts/motor_test_logs/workspace_boundaries/<hand>/boundary_models.json`（组合边界）
- `scripts/minimal_example_dual_stream_commented.py`（手套数据发布）

---

## 1. 总体目标

数据采集在边界约束下，为每个手指按组合采集 `电机命令 -> 电机反馈 -> 手套角度` 样本，输出可用于映射/学习的结构化数据集。

当前实现由三部分组成：

1. `controller`（主控制终端）
2. `worker`（实际执行采样）
3. 可选 `health monitor`（只读状态，不控制）

---

## 2. 推荐启动方式

### 2.1 左手独立采集（推荐）

```bash
cd /home/yhand-demo/Yhand_Mapping_2.0
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch yhand_mapping data_collection_left_independent_launch.py target_finger:=all
```

需要在自动采集结束后继续人工补采时，建议显式保留交互终端和补采开关：

```bash
ros2 launch yhand_mapping data_collection_left_independent_launch.py \
  target_finger:=all \
  collect_in_terminal:=true \
  manual_supplement_after_run:=true
```

### 2.2 右手独立采集（推荐）

```bash
ros2 launch yhand_mapping data_collection_right_independent_launch.py target_finger:=all
```

### 2.3 单手指采集（示例：左手无名指）

```bash
ros2 launch yhand_mapping data_collection_left_independent_launch.py target_finger:=ring
```

多根手指按输入顺序依次采集，例如中指 → 无名指 → 小指：

```bash
ros2 launch yhand_mapping data_collection_left_independent_launch.py target_finger:=middle,ring,little
```

同一次运行复用驱动和手套节点；重复手指只采集一次。右手替换为 `data_collection_right_independent_launch.py`。

### 2.4 legacy 启动（不建议双手并行时使用）

```bash
ros2 launch yhand_mapping data_collection_launch.py target_hand:=left target_finger:=all
```

---

## 3. 独立 launch 的硬隔离内容

左右手独立 launch 中已经固定：

- Driver 节点名（左/右不同）
- cmd topic（`/yhand_left_cmd` / `/yhand_right_cmd`）
- feedback topic（`/yhand_left_feedback` / `/yhand_right_feedback`）
- 设备 SN / Board IP / Port 分别读取 `config/left_hand_device_config.json` 和 `config/right_hand_device_config.json`

这样可以做到：左手异常退出不直接影响右手进程。

---

## 4. launch 参数（独立版本）

常用参数：

- `target_finger`：`all|thumb|index|middle|ring|little`，也支持逗号分隔列表，按输入顺序采集
- `start_glove_node`：是否启动手套发布（默认 `true`）
- `disable_glove_required`：手套不就绪是否允许继续启动（默认 `true`）
- `collect_in_terminal`：`true|false|auto`（controller 是否在独立 `gnome-terminal` 中运行）
- `manual_supplement_after_run`：自动阶段结束后是否询问进入人工补点（默认 `true`）
- `resume`：是否从断点恢复（默认 `false`）
- `resume_file`：断点文件路径（为空时用默认路径）
- `live_status_file`：实时状态文件路径（为空时用默认路径）
- `start_health_monitor`：是否开启健康监控窗口（默认 `false`）
- `health_monitor_in_terminal`：健康监控是否独立终端
- `health_refresh_sec`：健康窗口刷新周期

---

## 5. 运行模式（`data_collection_main.py`）

脚本支持：

- `--run-mode default`：传统单进程模式
- `--run-mode controller`：主菜单，只负责发命令/拉起 worker
- `--run-mode worker`：只执行采集

独立 launch 默认以 `controller` 拉起，并由 controller 再拉起 worker。

默认 `collect_in_terminal:=true` 时，会看到两个控制相关终端：

1. `data_collection_left/right`：controller 主控制终端。
2. `data_collection_left/right_worker`：实际运动、采样和人工补点终端。

controller 启动 worker 时会继续传递 `manual_supplement_after_run`。因此该参数设为
`true` 时，自动采集完成后 worker 不会直接跳过人工补点入口。

> 注意：当前 `a` 键已改为“仅启动 worker”，不再用于 worker 内部暂停/恢复。

---

## 6. 主控制终端（controller）按键

启动后主终端提示：

- `a`：启动 worker（若已运行，仅提示，不重复启动）
- `p`：发送“暂停+回0”给 worker
- `f <finger>`：切换采集优先手指
- `r`：恢复上次被中断手指
- `z`：执行回0
- `t`：全手张紧电流 `-0.1A`
- `i`：全手电流置 `0A`
- `m`：进入人工补点
- `e`：急停（`0A`）
- `s`：状态
- `k`：强制停止 worker
- `q`：退出 controller（并尝试安全关闭 worker）
- `h`：帮助

控制命令通过 `control_file` 写入，worker 轮询读取。

在独立 launch 的常规用法中，应在 controller 终端输入命令；人工补点的目标值则在
worker 终端输入。不要关闭 controller 终端，否则会触发整个 launch 退出。

---

## 7. worker 的三个交互层

### 7.1 auto-menu（自动采集开始前）

这一层主要用于直接运行脚本且 stdin 为交互终端的情况。通过独立 launch 的
controller 启动时，controller 的 `a` 已代表启动自动采集，worker 通常直接进入自动流程。

可用命令：

- `a`：开始自动采集
- `f <finger>`：调整 finger 优先级顺序
- `m`：先进入人工补点
- `z`：回0校准
- `t`：全手 `-0.1A`
- `i`：全手 `0A`
- `e`：急停 `0A`
- `q`：不开始自动，安全退出

### 7.2 auto-control（自动采集运行中）

可用命令：

- `p`：暂停并回0，再保持
- `f <finger>`：请求切换手指
- `r`：恢复被中断手指
- `z`：回0
- `t`：全手 `-0.1A` 并暂停
- `i`：全手 `0A` 并暂停
- `m`：进入人工补点
- `e`：急停 `0A` 并暂停
- `q`：安全退出
- `h`：帮助

### 7.3 manual-supplement（人工补点）

可用命令：

- 直接输入：`m17:-120 m22:5`（补 1 点）
- `n <k> m17:-120 m22:5`（连续补 k 点）
- `f <finger>`：切手指
- `c <combo_id>`：指定组合（空则恢复 auto 选 combo）
- `m <mode>`：指定 active_mode（空则 auto）
- `z`：回0校准
- `s`：显示状态
- `q`：退出人工补点
- `h`：帮助

### 7.4 人工补点的两种入口

自动运行中临时补点：

1. 在 controller 终端输入 `m`。
2. worker 进入人工补点，不再二次确认。
3. 在 worker 终端输入目标，例如 `m17:-120 m22:5`。
4. 在 worker 终端输入 `q`，只退出人工补点并恢复自动流程。

自动阶段全部完成后补点：

1. 启动参数需为 `manual_supplement_after_run:=true`。
2. worker 终端显示 `自动采集已结束，进入手动补采模式？(y/N):`。
3. 输入 `y` 后进入人工补点；直接回车或输入 `n` 会跳过。
4. 人工补点中输入 `q` 后，本次 worker 任务结束，其终端可以正常关闭。

人工补点依赖交互式 stdin。使用 `collect_in_terminal:=false` 或无 TTY 环境运行时，
程序会打印 `stdin is not interactive` 并跳过人工补点。

### 7.5 终端退出与 launch 生命周期

独立 launch 的关系可以简化为：

```text
ros2 launch
├── Y_Hand_Node
├── glove publisher（可选）
└── controller 终端
    └── worker 终端
```

- 进入人工补点不会导致 controller 或 launch 退出。
- worker 正常完成或单独退出时，controller 仍保持运行，可再次输入 `a` 或 `m`。
- 人工补点中输入 `q` 只退出当前人工补点层。
- controller 中输入 `q` 或关闭 controller 终端，会触发整个独立 launch 关闭。
- `Y_Hand_Node` 退出也会触发整个独立 launch 关闭。
- controller 中输入 `k` 只停止当前 worker，不退出 controller。

---

## 8. 自动采集完整流程（每个手指）

固定手指顺序：

`thumb -> index -> middle -> ring -> little`

每个手指的总体阶段是：

1. `preuniform`（单电机预均匀采样）
2. `random`（组合随机采样）
3. `edge_enhance`（边界增强采样）

### 8.1 quota 和 mode 计划

- 四指每个 combo 的名义目标：`FOUR_FINGER_QUOTA_BASE_PER_COMBO=1000`
- 拇指每个 combo 的基础配额：`QUOTA_BASE_PER_COMBO=2000`，再按 `opp_on/off` 比例分配
- 每个 combo 上下浮动：`QUOTA_FLOAT_RATIO`
- thumb 有 `opp_on/off` 比例约束：`THUMB_OPPOSITION_ON_RATIO`
- active_mode quota：
  - thumb: `full/single/double`
  - 四指: `xy/x_only/y_only`

当前策略下随机阶段会禁掉单电机模式：

- 四指：禁 `x_only/y_only`，仅 `xy`
- thumb：禁 `single/double`，仅 `full`

由 `DISABLE_RANDOM_SINGLE_MOTOR_MODES=True` 控制。

四指保留 ±10% 浮动，每个 combo 配额范围为 900–1100 点。完整组合下，
食指和中指各 6 个 combo，名义目标各 6000 点；无名指和小指各 4 个 combo，名义目标各 4000 点。
预均匀采样计入配额，每指额外 500 点的屈曲边界增强仍保留，因此最终总点数可能超过基础配额。
恢复旧断点时保留已采点数，并采用新配额重新判断完成状态。

### 8.2 preuniform（单电机预均匀）

`PRE_RANDOM_UNIFORM_ENABLE=True` 时执行，目的是先覆盖单轴基础样本：

- 非伸肌电机：从 0 向“收紧边界”按固定步长均匀采点
  - 屈曲：`PRE_RANDOM_FLEX_STEP`
  - 侧摆：`PRE_RANDOM_SWAY_STEP`
  - 其他：`PRE_RANDOM_OTHER_STEP`
- 伸肌电机：先把屈肌拉到收紧侧，再让伸肌向 0 方向均匀采点
  - 步长：`PRE_RANDOM_EXTENSOR_STEP`

预均匀采样运动阶段使用以下辅助电流（左右手及拇指按肌肉名称匹配）：

- 屈肌主动时：本指伸肌以 **+0.05 A** 放线 **0.5 秒**，`kp/kd=0`；随后伸肌保持 **`kp=kd=I=0`** 至屈肌到位或超时。其他电机保持默认电流。
- 内收/外展主动时：本指全部伸肌和屈肌始终设置 **`kp=kd=I=0`**，另一侧侧摆以 **+0.05 A** 放线 **0.5 秒**，`kp/kd=0`；随后该对抗侧摆也保持 **`kp=kd=I=0`** 至主动侧摆到位或超时。其他电机保持默认电流。
- 伸肌主动时：本指全部屈肌和两侧侧摆以 **+0.05 A** 放线 **1 秒**，`kp/kd=0`；随后屈肌保持 **`kp=kd=I=0`**，两路侧摆以 **−0.03 A** 纯电流张紧、`kp/kd=0`，持续至伸肌到位或超时。伸肌持续位控到目标。

主动电机立即开始位控，放线与其运动同时执行。主动电机目标误差在 **±14°** 内连续 **1 秒**确认到位，运动 **3 秒**超时。
屈肌和侧摆预采不再使用按位移计算的短暂放线及 `−0.1 A` 回收时序。
随后全部恢复默认纯电流张紧，沿用第 1–2 秒的结果采集窗口。
参数为 `PRE_RANDOM_FLEX_EXTENSOR_RELEASE_TOR_A`、`PRE_RANDOM_SWAY_OPPOSITE_RELEASE_TOR_A` 和 `PRE_RANDOM_PASSIVE_RELEASE_SEC`，
伸肌预采使用 `PRE_RANDOM_EXTENSOR_PASSIVE_RELEASE_TOR_A`、`PRE_RANDOM_EXTENSOR_PASSIVE_RELEASE_SEC` 和 `PRE_RANDOM_EXTENSOR_SWAY_TIGHTEN_TOR_A`，
受 `ANTAGONIST_RELEASE_ENABLE` 控制。该调整仅用于 `preuniform`；随机组合及人工补点沿用原有策略。

若过程被中断，可恢复后继续。

### 8.3 random（组合随机）

1. 在可用 combo 中加权选 combo：
   - 距离 `combo_min` 越远，权重越高
   - 剩余容量越大，权重越高
   - 覆盖缺口越大，权重越高
2. 在该 combo 边界内按 coverage 驱动采样目标点
3. 检查步长与方向约束：
   - `ENFORCE_ACTIVE_TARGET_DECREASE=True`（主动电机目标需更小）
   - 最小步长限制 `MIN_STEP_ABS`
4. 执行点控制 `_execute_point`
5. 记录 process/result 与统计

### 8.4 combo 卡死保护（stall）

当某 combo 连续 `target=None` 达到阈值：

- 窗口长度：`COMBO_STALL_FAIL_TRIGGER=50`
- 若该 combo 当前完成率 `>= COMBO_STALL_COMPLETE_RATIO(0.70)`：
  - 计一次 stall-window
  - 达到 `COMBO_STALL_COMPLETE_HIT_WINDOWS=3` 后冻结该 combo（min=max=count）
- 若完成率 `<0.70`：
  - 先回0重试，最多 `COMBO_STALL_MAX_ZERO_RETRY=10` 次
  - 重试耗尽后冻结该 combo

### 8.5 finger 级解冻保护

如果 finger 还没达到 `FINGER_COMPLETE_MIN_RATIO=0.80`，但所有 combo 都不可继续：

- 执行回0
- 解冻 combo cap（恢复到 base min/max）并继续随机
- 最多 `FINGER_UNFREEZE_MAX_ROUNDS=10` 轮
- 若连续无进展轮数达到 `FINGER_UNFREEZE_MAX_NO_PROGRESS_ROUNDS=3`，停止该 finger，防死循环

### 8.6 edge_enhance（边界增强）

随机阶段后追加边界点：

- thumb 外展：`THUMB_ABDUCTION_EXTRA_POINTS=500`
- thumb 屈曲：`THUMB_FLEXION_EXTRA_POINTS=500`
- 四指屈曲（每个手指）：`FOUR_FINGER_FLEXION_EXTRA_POINTS_PER_FINGER=500`

并用 edge bias 提高“边界附近 20% 区域”采样概率。

---

## 9. 单点执行内部阶段（`_execute_point`）

每个点在终端里会看到阶段流转：

- 含主动伸肌的点：`extensor_first` → `remaining_targets`（有其他目标时）
- 其他点：`move`、`release+rewind` / `rewind` / `target_hold`（按被动电机时间线）
- `arrived`（达到阈值并稳定 `ARRIVAL_CONFIRM_SEC`）
- `stable`（稳定采样阶段）

### 9.1 被动电机策略

当点的主动电机中包含本指伸肌时（左右手均支持，包括拇指短伸、长伸及伸肌单电机预采点），默认采用分阶段运动：

1. 伸肌立即位控到目标；本指全部屈肌和两路侧摆同时持续以 **+0.08 A** 放线，`kp/kd=0`，包括该 combo 的主动侧摆。不设置独立的放线等待时间。其他待运动目标（例如拇指对掌）暂时位控保持点开始时的位置。
2. 伸肌反馈在 **±14°** 内连续保持 **1 秒**后，保持伸肌位置目标，并让剩余主动电机位控到各自目标。本指未参与目标位控的屈肌和侧摆继续以 **+0.08 A** 放线。
3. 全部主动电机达到同样的到位条件后，全部 28 路恢复默认纯电流张紧，再按原来的稳定结果窗口采集。该策略不执行运动期间的 `−0.1 A` 回收。

伸肌阶段最多等待 **3 秒**，其后的组合到位阶段另有 **3 秒**。伸肌超时不会开始侧摆目标运动；任何阶段超时都会恢复默认张紧并沿用原来的超时点记录方式，`reached=false`。缺失、过期或非有限的位置反馈不能完成到位确认。

参数位于 `data_collection_config.py`：`EXTENSOR_FIRST_ENABLE`、`EXTENSOR_FIRST_RELEASE_CURRENT_A`、`EXTENSOR_FIRST_TIMEOUT_SEC`。该策略还受 `ANTAGONIST_RELEASE_ENABLE` 控制；关闭辅助时保持原来的同时位控方式。点的 JSON 元数据增加 `motion_policy`、`extensor_stage_reached`、`extensor_stage_elapsed_sec`、`timeout_stage`，用于区分执行方式和失败阶段。

其他点中，同手指非主动电机和其他手指都保持“纯电流张紧”行：`[0,0,tor,0,0]`。

主动电机集合的对侧电机会按 `active->target` 映射执行：

- 放线：`DATA_COLLECTION_RELEASE_TOR_*`
- 收线：`REWIND_TO_ZERO_TOR=-0.1`
- 再切回默认张紧 tor

### 9.2 到位与超时

- 到位判定阈值：`POINT_ARRIVAL_TOL`
- 到位确认时间：`ARRIVAL_CONFIRM_SEC`
- 普通点最长运动时间：`MOVE_TIMEOUT_SEC`
- 含伸肌点：伸肌阶段使用 `EXTENSOR_FIRST_TIMEOUT_SEC`，其后组合到位阶段使用 `MOVE_TIMEOUT_SEC`

### 9.3 稳定结果窗口

无论 reached 或 timeout，都进入稳定阶段采样。

- 稳定采样时长：`max(STABLE_HOLD_SEC, RESULT_WINDOW_END_SEC)`
- 最终 result 使用窗口：`[RESULT_WINDOW_START_SEC, RESULT_WINDOW_END_SEC]`

---

## 10. 回0校准逻辑

### 10.1 周期校准

- 每 `ZERO_CALIBRATE_EVERY_POINTS` 个全局点触发一次。

### 10.2 手动触发

- `z` 命令可随时触发。

### 10.3 校准流程

1. 仅四指的 **6 路屈肌和 8 路侧摆**，共 **14 路**，以 **+0.08 A** 纯电流放线 **1 秒**，`kp/kd=0`。其余 14 路（含四指伸肌、拇指和腕部）位控保持该阶段开始的位置；无有效反馈时使用默认纯电流。
2. 放线结束后，**四指伸肌先位控回 `pos=0`**，前馈电流固定 **−0.05 A**（`ZERO_EXTENSOR_TOR_A`），`kp/kd/vel` 沿用默认值。左手食指/中指/无名指/小指电机为 **25/23/10/11**，右手为 **11/9/24/25**。四指屈肌和侧摆 14 路切换为纯电流 **−0.03 A** 张紧（`ZERO_FLEX_SWAY_TIGHTEN_TOR_A`），**`kp/kd=0`**，允许被动位置变化；拇指和腕部共 10 路保持放线结束的位置。
3. 四指伸肌反馈均在 **±14°** 内持续 **2 秒**后，其他电机才位控回零。最终全部 **28 路 `pos=0`**，伸肌继续使用 **−0.05 A** 前馈，其他电机沿用默认 `kp/kd/vel/tor`。拇指和腕部在此阶段按原有方式回零。
4. 每阶段最多等待 **5 秒**，单次事件最多 **3 轮**尝试。全手成功也需全部反馈在 **±14°** 内持续 **2 秒**。反馈缺失、过期或非有限值不计入到位确认；伸肌失败不会进入其他电机回零阶段，失败后保持实测位置，无有效反馈则下发零电流。

每次事件都会写入 `zero_calibration_events.csv`。

周期、菜单 `z`、预采启动以及停滞恢复的回0都使用该流程。这里只返回已有零位，不调用 SDK `Set_Zero`；含伸肌 combo 运动时的 **+0.08 A** 放线策略独立保持。

---

## 11. 人工补点详细规则

1. 仅允许当前手指的电机 ID。
2. 主动电机数量限制：
   - thumb 最多 3
   - 四指最多 2
3. `combo` 自动匹配优先级：
   - active_set 完全匹配
   - active_set 为 combo active_set 子集
   - active_set 交集最大
4. 手动点与自动点完全走同一 `_execute_point`，所以：
   - 对侧放线/收线逻辑一致
   - process/result 保存格式一致
   - 计数和 timeout 统计一致

退出人工补点后：

- 如果是自动运行中进入：回到自动继续
- 如果是自动全部结束后进入：退出后本次 worker 任务结束，controller 继续运行

---

## 12. 断点续采与实时状态

### 12.1 resume

- 开启：`resume:=true`
- 文件：`resume_file`（默认 `motor_test_logs/data_collection/<hand>_resume_latest.json`）

保存内容包含：

- 全局点计数/超时计数
- 当前 phase/reason/stage
- 每个 finger 的 combo 计数、mode 计数、coverage、冻结状态
- startup first frame 文件路径

### 12.2 live status

- 文件：`live_status_file`（默认 `<hand>_live_status.json`）
- 可由 `data_collection_health_monitor.py` 读取并渲染

健康监控只读，不会影响控制流程。

---

## 13. 输出目录与文件

根目录：

- `scripts/motor_test_logs/data_collection/<timestamp>_<hand>_collection/`

关键文件：

- `<hand>_hand/startup_first_frame.json`
- `<hand>_hand/startup_first_frame_glove48.csv`
- `<hand>_hand/startup_first_frame_motor_pos.csv`
- `<hand>_hand/zero_calibration_events.csv`
- `<hand>_hand/runtime_health.json`
- `<hand>_hand/run_summary.json`
- `<hand>_hand/<finger>/point_log.csv`
- `<hand>_hand/<finger>/finger_summary.json`
- `<hand>_hand/<finger>/process/*`
- `<hand>_hand/<finger>/result/*`

`run_summary.json` 里会单独统计：

- `manual_supplement_points`
- `manual_supplement_timeouts`

---

## 14. 常见问题

1. `Startup not ready`
- 如果只是不接手套调试，可设置 `disable_glove_required:=true`。

2. 为什么某个 finger 未达到 nominal_total 就结束
- 可能触发了 combo stall 冻结和 finger 防死循环保护（见第 8.4/8.5 节）。

3. `manual supplement` 没出现
- 确认启动参数为 `manual_supplement_after_run:=true`。
- 自动结束后需要在提示处输入 `y`；直接回车默认跳过。
- 当前终端不是交互 TTY 时会自动跳过，建议使用 `collect_in_terminal:=true`。
- controller 按 `m` 后，应切换到 worker 终端输入人工目标。

4. `p` 后为什么不是立即继续
- `p` 定义是“暂停+回0+保持”；要继续请用 `r` 或 `f <finger>`。

5. `a` 为什么不再能暂停/恢复
- 已改为 controller 侧“仅启动 worker”，worker 运行中不接受 `a` 切换暂停。

6. 为什么 worker 终端关闭了，但 launch 还在运行
- worker 完成后退出是正常行为；controller 才是 launch 管理的主控制进程。
- 回到 controller 终端输入 `s` 可查看状态，输入 `a` 可启动新的 worker。

7. 为什么整个 launch 突然退出
- 检查是否关闭了 controller 终端，或在 controller 中输入了 `q`。
- 检查 `Y_Hand_Node` 是否异常退出；驱动和 controller 都配置了 `on_exit=Shutdown`。

---

## 15. 最小操作建议（常用）

1. 启动独立采集 launch（左或右），人工补采建议使用：
   ```bash
   ros2 launch yhand_mapping data_collection_left_independent_launch.py \
     collect_in_terminal:=true \
     manual_supplement_after_run:=true
   ```
2. 在 controller 终端按 `a` 启动 worker。
3. 运行中如需插手：
   - `p` 暂停+回0
   - `m` 人工补点
   - `r` 恢复
4. 自动结束后，在 worker 的人工补点提示处输入 `y`；不需要补点则输入 `n`。
5. 完成后查看：
   - `run_summary.json`
   - 每手指 `finger_summary.json`
   - `zero_calibration_events.csv`

---

## 16. CSV 整手位控数采（新脚本）

对应脚本：

- `scripts/data_collection_from_motor_feedback_csv.py`

用途：

- 保留当前数采的采样与落盘逻辑（`process/result`、`point_log.csv`、`run_summary.json`）。
- 不再使用自动随机采样/单电机采点。
- 改为逐行读取 `five_finger_delta_mapping_main.py` 录制的 `motor_feedback` CSV。
- 每一行作为一个“整手 24 电机组合位控点”执行采集。
- 默认关闭原先 `_execute_point` 的对侧放线/回收流程（纯组合位控）；如需兼容旧流程可手动开启参数。

### 16.1 输入 CSV 格式

支持列名：

- `motor_<id>_pos`（推荐，five-finger 录制默认）
- `m<id>_pos`（兼容）

可带 `frame_idx` 列；若缺失则自动用 CSV 行号。

到位判定说明（CSV 模式）：

- 以 CSV 表头检测到的控制电机全集作为到位判定集合（常见为 24 电机）。
- 若某一行缺少该全集中的任一电机目标值，该行会被跳过（不作为采点）。
- 到位条件是：该全集内所有电机 `|feedback_pos-target_pos| <= point_arrival_tol`，并连续保持 `arrival_confirm_sec`（默认 `1.0s`）。
- 到位后执行预紧收线等待：默认先保持 `post_reach_pretighten_sec=2.0s`，再进入结果记录窗口（默认 `result_window_sec=1.0s`）。
- `process` 记录范围：从该点 cmd 发布开始，一直记录到结果窗口结束（包含到位后预紧等待阶段）。
- `result` 记录范围：仅取到位后预紧等待之后的结果窗口。

### 16.2 运行示例

推荐（launch）：

```bash
cd ~/yhand_mapping/YQ_SDK_ROS2-master
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch yhand_mapping data_collection_from_motor_feedback_csv_launch.py \
  target_hand:=right \
  cmd_csv_path:=src/yhand_mapping/scripts/motor_test_logs/motor_feedback_frames/right_motor_feedback_xxx.csv \
  stride:=3 \
  max_points:=500
```

脚本直启（等价）：

```bash
cd ~/yhand_mapping/YQ_SDK_ROS2-master
source /opt/ros/jazzy/setup.bash
source install/setup.bash

python3 src/yhand_mapping/scripts/data_collection_from_motor_feedback_csv.py \
  --hand right \
  --cmd-csv-path src/yhand_mapping/scripts/motor_test_logs/motor_feedback_frames/right_motor_feedback_xxx.csv \
  --stride 3 \
  --max-points 500
```

### 16.3 常用参数

- launch 参数：
  - `target_hand`：`left|right`
  - `cmd_csv_path`：输入命令 CSV（必填）
  - `start_driver_node`：是否在 launch 内启动手驱动
  - `start_glove_node`：是否在 launch 内启动手套发布
  - `disable_glove_required`：手套不就绪时是否继续
  - `start_row` / `stride` / `max_points`
  - `collection_name`
  - `point_arrival_tol`（默认 `3.0`）
  - `arrival_confirm_sec`：连续到位判定时长（默认 `1.0`）
  - `post_reach_pretighten_sec`：到位后预紧收线等待时长（默认 `2.0`）
  - `result_window_sec`：预紧后结果窗口记录时长（默认 `1.0`）
  - `zero_calibrate_every_points` / `zero_calibrate_at_start`
  - `use_antagonist_release`：是否启用旧的放线/收线时序（默认 `false`）
  - `collect_in_terminal`：`true|false|auto`

- `--cmd-csv-path`：输入命令 CSV（必填）
- `--start-row`：从第几行开始（0-based，默认 0）
- `--stride`：每隔 N 行取 1 行（默认 1）
- `--max-points`：最多执行点数（0 表示全部）
- `--point-arrival-tol`：到位阈值（默认 `3.0`）
- `--arrival-confirm-sec`：连续到位判定时长（默认 `1.0`）
- `--post-reach-pretighten-sec`：到位后预紧收线等待时长（默认 `2.0`）
- `--result-window-sec`：预紧后结果窗口记录时长（默认 `1.0`）
- `--zero-calibrate-every-points`：周期回0间隔（<=0 关闭）
- `--zero-calibrate-at-start`：是否启动即回0（true/false）
- `--use-antagonist-release`：是否启用旧的放线/收线时序（默认 false）
- `--disable-glove-required`：手套不就绪时是否继续

输出目录仍在：

- `scripts/motor_test_logs/data_collection/<timestamp>_<hand>_collection/`
