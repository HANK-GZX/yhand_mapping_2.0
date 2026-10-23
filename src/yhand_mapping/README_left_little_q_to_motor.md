# 左手五指：四个q计算单指电机位置

新增`--finger thumb`：M4长屈、M7长伸固定使用四q一次模型；M6短伸、M8内收、M27外展、M14对掌固定使用包含平方项和交叉项的四q完整二次模型。两种`--model`选项对拇指均选择这套混合关系，其他手指继续原有选项。

```bash
python3 src/yhand_mapping/scripts/left_little_q_to_motor.py --finger thumb --q 0 0 0 0
```

Python使用`FingerMapper(finger='thumb').predict([0,0,0,0])`，六路返回0。模型文件为`config/left_thumb_q_to_motor_models.json`，已按用户要求恢复扩展前的完整备份，来自8149个正式full采样点。参考角恢复为采集启动帧`[8.491,38.262,26.028,-26.219]°`，自由截距拟合后展开为`f(q_reference+Δq)-f(q_reference)`。`scripts/fit_left_thumb_q_to_motor.py`保存可复算流程，系数文件保留标准化系数与验证指标。该参考不是精确电机零位标定。

纯映射和遥操入口均恢复原输入幅度与完整多项式。撤回此次新零位约束重拟合、CMC 1.15倍增益、框外切线延伸和新增拇指输出窗口，保留既有一次／二次混合模型及五指遥操。恢复后需要重启原控制入口。

实机入口新增`finger:=thumb`单拇指和`finger:=five`五指统一遥操，见[遥操说明](README_left_little_q_to_motor_test.md)。

脚本：`scripts/left_little_q_to_motor.py`。计算位置并打印结果，不发布电机命令。只依赖Python标准库，无需启动ROS。

使用`--finger index`计算食指M25伸肌、M3浅屈、M15深屈、M1内收、M28外展，五路全部使用四个Δq的一次模型。配置为`config/left_index_q_to_motor_models.json`，来源为已分析的5606个正式采样点。采用增量关系`Δm=BΔq`，相对q全0时五路输出0；linear/quadratic两个选项在食指中均为同一套一次关系。

```bash
python3 src/yhand_mapping/scripts/left_little_q_to_motor.py --finger index --q -5 10 20 5
```

Python接口为`FingerMapper(finger='index').predict([-5,10,20,5])`。训练启动参考不是精确电机零位q，源模型的带截距验证误差不能当作增量控制误差。[实机测试说明](README_left_little_q_to_motor_test.md)包含食指单独控制及`finger:=four`四指统一控制的指令。

使用`--finger middle`计算中指五台电机：M23伸肌、M16浅屈、M2深屈、M9内收、M24外展，均使用四个Δq的一次关系。系数来自20261016批次5889个正式xy点，配置位于`config/left_middle_q_to_motor_models.json`。

```bash
python3 src/yhand_mapping/scripts/left_little_q_to_motor.py --finger middle --q -5 10 20 5
```

Python接口：`FingerMapper(finger='middle').predict([-5,10,20,5])`。中指使用增量输出`Δm=BΔq`，相对四q全0时五路输出全0；两种model选项在中指中均使用一次关系。训练启动参考不是精确电机零位，原坐标带截距验证指标仅保留为来源信息。中指控制入口、系数及不添加电机限位的配置见[实机测试说明](README_left_little_q_to_motor_test.md)。

默认选择小指。添加`--finger ring`使用无名指，自动加载`config/left_ring_q_to_motor_models.json`，输出M10伸肌、M17屈肌、M22内收、M21外展。无名指四台电机均使用四个Δq：伸屈保留原线性/二次模型，侧摆在两种模型选项下均使用新批次的四q一次补偿。机器人参考q为`[-2,-6,0,-12]°`，Manus角度取`[20:24]`。

无名指侧摆来自`20261021_171709_left_collection`中排除预采和末尾500个密集点后的2170个正式点，使用带截距拟合的斜率，并扣除零输入预测，保持Δq=0时电机输出0：

```text
M22 = -1.27305 Δq0 +0.65828 Δq1 -0.07069 Δq2 -0.08362 Δq3
M21 = +1.36390 Δq0 -0.34749 Δq1 +0.04464 Δq2 -0.00860 Δq3
```

SDK控制时M10/M17/M22/M21均下发位控目标，侧摆不再执行启动放线。零位对齐后的控制精度需要实机验证；原始带截距拟合的留出误差不能直接用于评价此控制坐标。下文小指模型不变。

```bash
python3 src/yhand_mapping/scripts/left_little_q_to_motor.py --finger ring --q 0 10 20 5
```

Python调用无名指：`FingerMapper(finger='ring').predict([0,10,20,5])`；原`LittleFingerMapper`接口保留。[四q侧摆离线分析](scripts/motor_test_logs/data_collection/20261021_171709_left_collection/analysis/ring_distribution/SIDE_FOUR_Q.md)记录补偿前后的同批留出结果。以下公式和默认示例均为小指。

输入顺序固定为小指的`q0 q1 q2 q3`，单位为度；输出为实际feedback位置坐标，单位也为度。

| 电机 | 肌肉 | 使用的角度 |
|---|---|---|
| M11 | 伸肌 | q0、q1、q2、q3 |
| M13 | 屈肌 | q0、q1、q2、q3 |
| M18 | 内收 | 仅q0 |
| M20 | 外展 | 仅q0 |

## 直接输入

在工作空间根目录运行：

```bash
python3 src/yhand_mapping/scripts/left_little_q_to_motor.py \
  --q 0 0 0 0
```

默认输入为相对零位的角度，`0 0 0 0`表示零位，不会再次减去零位偏移。默认使用线性模型。原始左手小指glove角度对应完整glove48的数组`[24:28]`，即第25～28列。

上述输入的线性输出为：

```text
M11（伸肌）: 0.000000°
M13（屈肌）: 0.000000°
M18（内收）: 0.000000°
M20（外展）: 0.000000°
```

省略`--q`即可反复输入；支持空格、逗号、分号分隔，`quit`退出：

```bash
python3 src/yhand_mapping/scripts/left_little_q_to_motor.py
```

## 相对角度与二次模型

输入相对角度的定义为：

```text
Δq = q_glove − [-5, -9, 7, -10]°
```

该值是用户提供的电机零位对应手套角度。线性和二次模型都重新拟合并施加原点约束：Δq=0时四台电机位置均为0。默认直接使用输入的Δq；`--relative`可保留，但无需添加。只有显式使用`--absolute`时才输入原始手套角度并由程序减去零位：

```bash
python3 src/yhand_mapping/scripts/left_little_q_to_motor.py \
  --absolute --q -5 -9 7 -10
```

这与上面的相对角度零位示例结果相同。不要把使用其他零位定义的角度直接当作这里的相对输入。

选择二次模型并以JSON输出：

```bash
python3 src/yhand_mapping/scripts/left_little_q_to_motor.py \
  --relative --q 0 10 20 5 --model quadratic --json
```

二次模型中，M11/M13包含四q的平方项和交叉项，M18/M20仅包含q0与q0²。默认线性式为：

```text
M11 = +0.484942958·Δq0 +1.921249408·Δq1 +1.842011823·Δq2 +2.290227667·Δq3
M13 = -0.461330233·Δq0 -1.590409566·Δq1 -1.990482480·Δq2 -2.359760673·Δq3
M18 = -1.487356669·Δq0
M20 = +3.648489925·Δq0
```

## Python调用

从脚本目录可导入：

```python
from left_little_q_to_motor import LittleFingerMapper

mapper = LittleFingerMapper()  # 启动时加载一次系数
positions = mapper.predict([0, 0, 0, 0])  # 默认相对角度
# {'M11': 0.0, 'M13': 0.0, 'M18': 0.0, 'M20': 0.0}
positions = mapper.predict([0, 10, 20, 5], relative=True, model='quadratic')
positions = mapper.predict([-5, -9, 7, -10], relative=False)  # 显式原始角度
```

系数文件为`config/left_little_q_to_motor_models.json`，来自`20261020_165513_left_collection/left_hand/little`的2,227个正式xy点，包含超时点。原始电机标签保持不变，通过无截距最小二乘重新拟合；新零位来自用户，脚本未进行物理标定。配置保存输入范围、拟合误差和按combo分层的五折随机交叉验证误差。

原点约束与旧侧摆数据存在冲突：线性模型M18五折MAE为6.04°，M20为15.08°（电机位置误差），M20的R²为−0.251；二次模型M20也为13.24°。因此这次修正了零位对应关系，不能认定侧摆精度改善。伸屈线性MAE约2.77°/2.72°，二次约0.98°/1.10°。详细对比见[重拟合报告](scripts/motor_test_logs/data_collection/20261020_165513_left_collection/analysis/little_zero_constrained/ANALYSIS.md)。

输出不自动裁剪到电机范围；输入NaN/Inf、错误数量或计算溢出会报错。小指侧摆模型仅使用q0，无名指侧摆加入四q补偿；数值输出不能替代实机跟踪验证。

本页的纯计算脚本返回原始模型预测。SDK控制入口在发送前将M11/M13/M18/M20分别裁剪到`[-5,450]`、`[-400,20]`、`[-15,50]`、`[-80,30]`°，不限制输入关节角。实机控制入口和启动方法见[小指SDK实机测试](README_left_little_q_to_motor_test.md)。
