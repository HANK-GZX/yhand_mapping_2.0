# 左拇指 combo 对指控制

这层用于“外展绕弧→内收对掌”的任务路径，与现有 `linear/mlp` 均可组合。它在同一个 28 电机控制节点内修改拇指六行，四指继续使用所选模型。默认不启用。

| 阶段 | 采集 combo 名称 | 主动位置控制 | 被动侧 |
|---|---|---|---|
| 外展绕弧 | `thumb_flex_long_out_opp_on` | M4 长屈、M27 外展、M14 对掌 | M6/M7 伸肌、M8 内收 |
| 靠近小指 | `thumb_flex_long_in_opp_on` | M4 长屈、M8 内收、M14 对掌 | M6/M7 伸肌、M27 外展 |

`in/out` 表示肌肉角色，不是直接取某个手套角的正负号。两段可以各采多个关键点。返回时使用同一关键点路径反向查询；这是位置插值路径，尚不包含单独的主动伸展 combo，实际返回能力也需要验证。

## 控制逻辑

1. 以操作手相对零位的四角 `Δq`，投影到**实测操作手姿态关键点**组成的四维折线，得到路径段和段内进度。它是任务方向标定，不把操作手四角直接当作机器人四角。
2. 在该段插值实测的机器人电机位置，作为姿态相关的肌腱长度目标。M4/M14 的补偿是这个张紧位置目标与原模型输出之差，允许不同姿态有不同补偿。不是固定负偏置，也不做全局 `m<=0`。
3. 通过相对中立的路径距离、离路径的距离平滑混入 combo 控制。外展与内收 combo 之间连续交接 M27/M8 的主动权重。M4/M14 在两种 combo 中都保留主动位置控制。
4. 被动侧位置目标过渡到当前反馈位置，KP/KD 平滑降为零，前馈平滑降为零。这里是**零电流被动释放**，不是正电流主动放线脉冲。目标位置始终限速，避免原模型与任务路径交接时直接跳变。
5. 启用本策略时，拇指六电机均使用标定文件中的行程、位置变化速度和反馈电流上限。超出反馈行程/电流上限会沿现有异常退出流程停止所选全部手指。调度卡顿不会允许累积成一个很大的位置步长。

离路径很远时，combo 权重为零，回到原模型目标，但仍受该文件的拇指行程和位置变化速度约束。**这不保证域外目标具有正确张紧状态**；初期应在标定的任务路径附近操作。中立附近逐渐恢复原位置控制。默认不传标定文件时，现有控制目标与参数保持原逻辑。

这是**经过张紧标定的位置补偿**，不是张力闭环。电流用于记录和上限检查，未把电机电流直接当作肌腱张力，也未实现低电流自动累积收线。静摩擦、运动方向和预紧变化仍会影响实际张力。需要动态防松时，应先校准电流/张紧关系，再增加有界反馈补偿。

## 为什么不直接从旧日志填写位置

20261026_150435_968260 日志中 M4/M14 已经松线，而 M7 负载较大。那些反馈位置不能作为“正确对指且张紧”的监督标签。全局模型的训练数据又缺少操作手姿态到实际正确对指姿态的配对。因此不能用 +59°、+12° 或观测到的 M7 停滞位置直接生成新控制标定。

仓库提供 `config/left_thumb_combo_template.json`。其中未知的行程、速度、电流上限保留 `null`，关键点为空。模板会被控制器明确拒绝，避免把示例数值当作已标定参数。测试中的 `SYNTHETIC_TEST_ONLY` 数据仅验证算法，不能用于机器手。

## 1. 采集方向与张紧位置

先通过已有机器人调试方式，准备正确的机器人姿态与适当预紧；同时让操作手摆成希望对应的姿态。确保只有一个电机命令发布源；不要同时运行摇操控制与另一个电机调试控制器。

记录器只订阅 `/yhand_left_feedback`、手套四角及左手 raw 在线信号，**不启动 SDK、不发布电机命令、不做预紧、不改变电机零位**。它可以与唯一的调试控制源同时运行。需要调试控制源及手套发布器已经在线。

```bash
cd /home/yhand-demo/Yhand_Mapping_2.0
source install/setup.bash
python3 src/yhand_mapping/scripts/thumb_combo_control/collect_thumb_combo_calibration.py capture \
  --motor-zero-id left_thumb_20261026_taut_zero \
  --output src/yhand_mapping/scripts/motor_test_logs/thumb_combo_calibration/recording.json
```

终端依次输入：

```text
neutral
out arc_1
out arc_2
in approach
in opposition
save
quit
```

每次输入都表示你已经确认机器人姿态正确、主动肌腱适当张紧，且操作手姿态匹配。至少需要 neutral、out、in 三点；多采一点能减少线性插值误差。首点必须是原机械预紧零位附近，不能用松线姿态作为中立点。

默认取最近 1 秒均值，至少 10 帧，四角和六个电机位置各自跨度不超过 3°，并检查数据新鲜度。运动或数据中断会拒绝记录。用 `undo` 撤销最后一点；`save` 保存，`quit` 不自动保存。采集期间不要重新机械归零。录制文件包含电流，需结合实际张紧情况判断。

## 2. 填写约束并生成标定文件

```bash
cp src/yhand_mapping/config/left_thumb_combo_template.json /tmp/left_thumb_combo_settings.json
```

将 `/tmp/left_thumb_combo_settings.json` 中这些字段填写为该机器手已经验证的值：

- `motor_position_limits_deg`：六电机允许位置范围，使用当前电机零位的坐标，并包含零位。
- `max_position_rate_deg_sec`：六电机允许的目标位置变化速度。
- `max_abs_current_A`：六电机反馈电流绝对值上限。
- `active_feedforward_A`：可保持 0；如需前馈，应填经过验证的值。不要直接照抄采集时的 −0.1 A。

`corridor_inner_deg/outer_deg` 是匹配操作手任务路径的距离范围，`neutral_blend_deg` 是中立附近的混入距离，均使用 `q_metric_weights` 定义的四角加权欧氏距离。模板 5/15/8° 和过渡时间 0.4 秒只是软件初始设置，需要随实际操作调整。

```bash
python3 src/yhand_mapping/scripts/thumb_combo_control/collect_thumb_combo_calibration.py build \
  --recording src/yhand_mapping/scripts/motor_test_logs/thumb_combo_calibration/recording.json \
  --settings /tmp/left_thumb_combo_settings.json \
  --output src/yhand_mapping/config/left_thumb_combo_calibrated.json
```

生成器检查：路径为外展→内收；关键点不重复且在允许行程内；原电机零位不被悄悄平移；位置、速度、电流均有有效参数。它记录原始采集文件 SHA256。静态参数校验不代表已经完成实机验证。

## 3. 预览与启动

先编译安装新脚本：

```bash
colcon build --packages-select yhand_mapping --symlink-install
source install/setup.bash
```

启动时电机零位和操作手中立姿态必须与标定一致；机械重新预紧归零后要检查标定是否仍适用，`motor_zero_id` 用来追溯，软件不能仅凭编码器数值自动识别肌腱零位变化。

```bash
ros2 launch yhand_mapping left_hand_q_to_motor_launch.py \
  finger:=five input_mode:=2 model:=mlp \
  thumb_combo_calibration:=/home/yhand-demo/Yhand_Mapping_2.0/src/yhand_mapping/config/left_thumb_combo_calibrated.json
```

已有手套发布器时追加 `start_glove_node:=false`。仅拇指测试可使用 `finger:=thumb`。不传 `thumb_combo_calibration` 就使用原控制。

离线预览用控制脚本 `--thumb-combo-calibration <文件> --dry-run --q ...`。预览的 MIT 矩阵表示从假定零反馈启动的**第一个限速周期**，不会直接跳到最终目标；`thumb_combo.motor_target_deg` 显示当前期望目标。

## 日志

原 `m*_prediction`、`m*_target` 继续记录模型预测和原裁剪目标；拇指 `m*_command` 记录实际发布的位置（纯电流模式为空）。新增：

- `thumb_combo[_from/_to/_weight/_distance_deg/_segment]`：组合、交接权重、路径距离与段号。
- `m*_combo_target`：限幅后的组合层目标。
- `m*_combo_correction`：实际发布的位置相对原模型目标的总修正，包含路径补偿、限幅和限速影响。
- `m*_combo_gain_scale`：位置增益比例。
- `m*_command_kp/kd`：最终发布增益；原有电流、位置反馈继续记录。

metadata 保存完整标定文件及哈希；q_cmd 遥测增加 `thumb_combo`，电机命令仍取实际发布矩阵。`q_cmd` 继续表示操作手请求，并不等于组合路径下机器人实际关节角。M6/M7 纯被动时不能把其位置列直接解释成正在跟踪某个关节目标。
