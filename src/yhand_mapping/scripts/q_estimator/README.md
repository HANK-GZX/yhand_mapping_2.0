# 电机反馈 → 关节角状态估计

- `left_hand_motor_to_q_estimator.py`：五指估计入口，支持拟合、实时估计和离线预测。
- `left_thumb_motor_to_q_estimator.py`：单拇指入口及共用估计逻辑。
- `motor_to_q_logging.py`：记录 q_cmd、手套角、电机反馈及估计角。
- `analyze_motor_to_q_log.py`：日志对比分析和历史遥操日志回放。
- `fit_four_finger_motor_to_q_mlp.py`：使用四指采集配对数据训练独立q0/q123分支MLP。
- `mlp_motor_to_q.py`：分支MLP的纯Python推理和输入肌群权重定义。

ROS 启动命令保持不变：

```bash
ros2 launch yhand_mapping left_hand_motor_to_q_estimator_launch.py model:=linear
```

模型仍在 `config/`，日志仍在 `scripts/motor_test_logs/motor_to_q_estimator/`。
详细用法见[五指估计器说明](../../README_left_hand_motor_to_q_estimator.md)
和[拇指估计器说明](../../README_left_thumb_motor_to_q_estimator.md)。

## 四指分支MLP

每根手指有两个独立网络：全部电机位置→q0，以及全部电机位置→q1/q2/q3。
隐藏层均为32×32 tanh。输入先按训练折标准化，再按肌群加权：

| 手指 | 侧摆电机 | 伸屈电机 |
| --- | --- | --- |
| 食指 | M1/M28 | M25/M3/M15 |
| 中指 | M9/M24 | M23/M16/M2 |
| 无名指 | M22/M21 | M10/M17 |
| 小指 | M18/M20 | M11/M13 |

q0分支：侧摆权重1、伸屈0.25；q123分支：伸屈权重1、侧摆0.25。
训练增加0.001×`mean(W_first²×(1/gain²−1))`惩罚，降低弱输入路径通过放大第一层权重抵消增益的倾向。
这是软优先级，不保证输出导数严格为4:1；模型保存训练分布上的标准化局部敏感度供审查。
输出保持实测绝对手套角，未施加F(0)=参考角约束，也未自动重设编码器坐标。

在工作空间根目录训练：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 \
  src/yhand_mapping/scripts/q_estimator/fit_four_finger_motor_to_q_mlp.py
```

读取`data_collection/index_data、middle_data、ring_data、little_data`的正式xy点，
点数5606/5889/2670/2227，包含超时点。无名指使用全部2670点。
不使用q_cmd或合成标签；不修改原线性/二次模型。
模型写入`config/motor_to_q_mlp/left_<finger>_motor_to_q_estimator.json`。
训练输出在`scripts/motor_test_logs/motor_to_q_mlp/时间戳/`，含报告、模型快照、逐点验证预测、精确内外层折分和折模型。

验证包含四角每维2°分箱的姿态分组五折、留一动作组合，线性/二次基线使用完全相同的外层训练集。
每折标准化只使用训练数据；两个分支独立在内层早停，选轮数后重训全部外层训练数据。
最终轮数取五个姿态折内层所选轮数的中位数，再训练全量数据。外层测试不用于选轮数。
相邻姿态箱仍可能接近；同批次采集均值验证不代表跨批次、动态或外载精度。

只运行四指估计，输出16角到`/left_four_estimated_q`等话题，并保留逐指话题：

```bash
source install/setup.bash
ros2 launch yhand_mapping left_hand_motor_to_q_estimator_launch.py \
  finger:=four model:=mlp
```

旁路记录五指遥操时需指定`q_cmd_topic:=/left_five_q_command_state`。
`finger:=five model:=mlp`使用四指MLP，拇指保留原多项式默认模型；单独`finger:=thumb model:=mlp`会报错，拇指尚未训练。
原`model:=auto/linear/quadratic`默认行为保持。
运行时输入仍为真实电机反馈减显式`motor_offset`；训练坐标与当前编码器坐标需要一致。
日志默认开启，超界有效性规则保持，新的模型及类型会记录在日志快照中。
新增脚本和配置需先执行`colcon build --packages-select yhand_mapping --symlink-install`安装。

不启动ROS的单指预览：

```bash
python3 src/yhand_mapping/scripts/q_estimator/left_hand_motor_to_q_estimator.py \
  predict --finger index --model mlp --positions 0 0 0 0 0
```

这里的全0仅用于观察模型截距，可能超出采集范围，不是新零位标定。
