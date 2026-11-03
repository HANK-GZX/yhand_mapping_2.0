"""Portable grouped motor-feedback -> q MLP evaluation; no Torch/ROS dependency."""
import math

FOUR_FINGERS = ('index', 'middle', 'ring', 'little')
SIDE_MOTORS = {'index': (1, 28), 'middle': (9, 24), 'ring': (22, 21), 'little': (18, 20)}


def finite_vector(values, size, label):
    result = [float(v) for v in values]
    if len(result) != size or not all(math.isfinite(v) for v in result):
        raise ValueError(f'{label}需要{size}个有限数值')
    return result


def input_gains(finger, motor_ids, secondary=.25):
    if finger not in SIDE_MOTORS or not math.isfinite(secondary) or not 0 < secondary <= 1:
        raise ValueError('需要四指之一和(0,1]内的次要输入权重')
    side = SIDE_MOTORS[finger]
    return [[1.0 if mid in side else secondary for mid in motor_ids],
            [secondary if mid in side else 1.0 for mid in motor_ids]]


class GroupedMLP:
    def __init__(self, fit, inputs):
        if fit.get('architecture') != 'independent_q0_q123' or fit.get('activation') != 'tanh':
            raise ValueError('需要q0/q123独立tanh网络')
        self.output_center = finite_vector(fit['output_center'], 4, '输出中心')
        self.output_scale = finite_vector(fit['output_scale'], 4, '输出尺度')
        if min(self.output_scale) <= 0:
            raise ValueError('输出尺度必须大于0')
        branches = fit['branches']
        if len(branches) != 2 or [b['output_indices'] for b in branches] != [[0], [1, 2, 3]]:
            raise ValueError('需要互不重叠的q0与q1/q2/q3分支')
        self.branches = []
        for branch in branches:
            gains = finite_vector(branch['input_weights'], inputs, '输入权重')
            if any(not 0 < g <= 1 for g in gains):
                raise ValueError('输入权重必须在(0,1]内')
            layers = []
            previous = inputs
            if len(branch['layers']) != 3:
                raise ValueError('分支需要两个隐藏层和一个输出层')
            for i, layer in enumerate(branch['layers']):
                activation = 'identity' if i == 2 else 'tanh'
                if layer['activation'] != activation or not layer['bias']:
                    raise ValueError('网络层激活或宽度错误')
                bias = finite_vector(layer['bias'], len(layer['bias']), '网络偏置')
                weight = [finite_vector(row, previous, '网络权重') for row in layer['weight']]
                if len(weight) != len(bias):
                    raise ValueError('网络权重和偏置形状不一致')
                layers.append((weight, bias, activation))
                previous = len(bias)
            if previous != len(branch['output_indices']):
                raise ValueError('网络输出宽度与关节索引不一致')
            self.branches.append((branch['output_indices'], gains, layers))

    def predict_standardized(self, x):
        output = [0.0]*4
        for indices, gains, layers in self.branches:
            values = [v*g for v, g in zip(x, gains)]
            for weight, bias, activation in layers:
                values = [b+math.fsum(w*v for w, v in zip(row, values)) for row, b in zip(weight, bias)]
                if activation == 'tanh':
                    values = [math.tanh(v) for v in values]
            for j, value in zip(indices, values):
                output[j] = value*self.output_scale[j]+self.output_center[j]
        return finite_vector(output, 4, 'MLP估计角')
