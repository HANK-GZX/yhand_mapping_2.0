# 测试目录

测试注册集中在本目录的 `CMakeLists.txt`，包根目录通过 `add_subdirectory(tests)` 引入。按控制与采集、当前q→m模型、m→q估计及记录分组，保留原有环境设置。通用遥操测试包含56项使用模拟SDK的DDS集成检查，全组超时由120秒调整为300秒。

`test_joint_coupling.py`集中覆盖手动输入的日志配对约束：q0/q1邻域、q2/q3交叉配对拒绝、区间空缺、绝对/相对角转换、多指整组拒绝、有限值检查、真实参考的零位与分布外示例，以及两角/三角只查询不发目标。历史补充测试验证原点保持、整段验证集隔离、手动目标不入库，以及过期、非有限、坐标错误、重复目标和疑似角度分支筛选。通用遥操测试另含2项单食指／五指手动话题集成检查，验证配对拒绝后原目标保持及停机，不连接真实电机。

采集数据补充检查仍集中在`test_joint_coupling.py`：使用当前模型参考转换Δq、同姿态格整体留出、到位超时保留观测角、有限值及sample_count校验、缓存快照过期拒绝，以及不把留出点加入参考、不替换原点和不重复补充。

拇指以 `test_thumb_coordination_mlp.py` 覆盖当前版本15共享网络：三头共享特征、屈伸输出保持、q0/q1/q2/q3方向、训练集标准化、零位与绝对/相对角协议、分组权重加载，以及版本14兼容。旧独立对掌、独立侧摆、三组耦合和阶段性屈伸训练的独立测试脚本已移除，关键运行检查并入当前测试。

四指当前结构、通用推理与坐标契约、控制停机、数据采集、边界及逆向估计测试保留。无名指与中指按手指集中注册：

- `test_ring_mlp.py`：保留共享伸屈、M10输出层微调、M17深弯收线和M22侧摆弯曲补偿的4个测试类、18项用例。
- `test_middle_mlp.py`：保留M2深屈肌及M23/M16伸屈调整的2个测试类、10项用例。

原阶段脚本已合并删除，测试类、断言和训练夹具保持。两份合并测试分别注册为`ring_mlp`与`middle_mlp`，超时均为30秒。

`test_left_little_q_to_motor_test.py` 还是其他控制测试的模拟SDK辅助入口，因此保留。

`test_left_thumb_q_to_motor.py`集中保留拇指linear模型及协调候选的28项检查：基础拟合与零位、v3完整四角侧摆补偿与q2过渡回归、对掌q1优先、M14首次运动触发／启动交接／实时跟随、五指版本选择及其余四指输出保持、参数校验，以及2项单拇指／五指v3假SDK完整控制与停机测试。假SDK测试只连接隔离DDS域中的模拟驱动。

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
