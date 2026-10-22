# 左手小指/无名指：四个q计算四台电机位置

脚本：`scripts/left_little_q_to_motor.py`。计算位置并打印结果，不发布电机命令。只依赖Python标准库，无需启动ROS。

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
