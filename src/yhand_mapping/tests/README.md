# 测试目录

测试注册集中在本目录的 `CMakeLists.txt`，包根目录通过 `add_subdirectory(tests)` 引入。按控制与采集、当前q→m模型、m→q估计及记录分组，保留原有环境设置。遥操测试包含54项使用模拟SDK的DDS集成检查，全组超时由120秒调整为300秒。

拇指以 `test_thumb_coordination_mlp.py` 覆盖当前版本15共享网络：三头共享特征、屈伸输出保持、q0/q1/q2/q3方向、训练集标准化、零位与绝对/相对角协议、分组权重加载，以及版本14兼容。旧独立对掌、独立侧摆、三组耦合和阶段性屈伸训练的独立测试脚本已移除，关键运行检查并入当前测试。

四指当前结构、通用推理与坐标契约、控制停机、数据采集、边界及逆向估计测试保留。无名指与中指按手指集中注册：

- `test_ring_mlp.py`：保留共享伸屈、M10输出层微调、M17深弯收线和M22侧摆弯曲补偿的4个测试类、18项用例。
- `test_middle_mlp.py`：保留M2深屈肌及M23/M16伸屈调整的2个测试类、10项用例。

原阶段脚本已合并删除，测试类、断言和训练夹具保持。两份合并测试分别注册为`ring_mlp`与`middle_mlp`，超时均为30秒。

`test_left_little_q_to_motor_test.py` 还是其他控制测试的模拟SDK辅助入口，因此保留。

`test_motor_mapping.py`保留编号、侧摆选择、配置偏移、腕部补偿和旧CSV兼容检查；清理已删除`mp`预处理模块的测试，移除冻结旧增益/边界电流的快照断言，配置检查只校验编号相关字段，不再绑定设备序列号等部署设置。

在工作空间根目录运行当前拇指和通用MLP测试：

```bash
source /opt/ros/jazzy/setup.bash
python3 -m pytest -q \
  src/yhand_mapping/tests/test_thumb_coordination_mlp.py \
  src/yhand_mapping/tests/test_left_hand_q_to_motor_mlp.py
```

构建并通过CMake运行全部注册测试：

```bash
colcon build --packages-select yhand_mapping --symlink-install
colcon test --packages-select yhand_mapping --event-handlers console_direct+
colcon test-result --verbose
```

`scripts/q_2_m linear mapping/left_little_q_to_motor_test.py` 是实际遥操控制入口，名称中的test不表示它是可删除的单元测试。历史训练脚本仍用于旧权重复现和验证折重建，入口索引见[q→m MLP说明](../scripts/q_2_m%20MLP%20mapping/README.md)。
