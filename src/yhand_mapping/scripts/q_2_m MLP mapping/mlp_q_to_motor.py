#!/usr/bin/env python3
"""NumPy-only inference for independently fitted left-finger q -> motor MLPs."""
import json
from pathlib import Path
import sys

import numpy as np

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE/'scripts/q_2_m linear mapping'))
from left_little_q_to_motor import FINGER_SPECS, finite_values

MODEL_DIR = PACKAGE/'config/q_to_motor_mlp'
INPUT_COORDINATES = 'delta_glove_angles'
ALIGNMENT = 'f(delta_q) - f(0)'
LEGACY_ALIGNMENT = 'f(q_reference + delta_q) - f(q_reference)'
SIDE_ALIGNMENT = 'f(delta_q)-f(0); side=q0_anchor+limited_bending_correction'
SIDE_WINDING_LIMIT_DEG = 2.0
SIDE_MOTORS = {'index': (1, 28), 'middle': (9, 24), 'ring': (22, 21), 'little': (18, 20)}
SIDE_DIRECTIONS = {'middle': (1, -1), 'ring': (-1, 1), 'little': (-1, 1)}


def side_bending_policy(finger, limit=SIDE_WINDING_LIMIT_DEG):
    if finger not in SIDE_MOTORS:
        return None
    return {'type': 'q0_anchor_bounded_winding', 'motor_ids': list(SIDE_MOTORS[finger]),
            'max_extra_winding_motor_deg': float(limit),
            'anchor_delta_bending_angles': [0.0, 0.0, 0.0],
            'positive_payout_correction': 'unchanged',
            'negative_correction': 'C*tanh(correction/C)'}


def validate_side_bending_policy(config):
    policy = config.get('side_bending_policy')
    if config['schema_version'] not in (4, 5, 6, 7, 8, 9, 10, 11):
        if policy is not None:
            raise ValueError('侧摆弯曲收线约束需要版本4至11模型')
        return
    expected = SIDE_MOTORS.get(config['finger'])
    if (config['model_type'] != 'mlp' or not expected or not isinstance(policy, dict)
            or policy.get('type') != 'q0_anchor_bounded_winding'
            or policy.get('motor_ids') != list(expected)
            or policy.get('anchor_delta_bending_angles') != [0.0]*3
            or policy.get('positive_payout_correction') != 'unchanged'
            or policy.get('negative_correction') != 'C*tanh(correction/C)'):
        raise ValueError('侧摆约束结构或电机定义错误')
    limit = finite_values([policy.get('max_extra_winding_motor_deg')], 1, '侧摆额外收线上限')[0]
    if not 0 < limit <= 20:
        raise ValueError('侧摆额外收线上限必须在(0,20]电机度内')
    policy['max_extra_winding_motor_deg'] = limit


def limit_side_bending(config, positions, anchor_positions):
    """Bound only winding caused by bending; preserve q0 control and payout."""
    policy = config.get('side_bending_policy')
    if policy is None:
        return positions
    positions = np.asarray(positions, dtype=float)
    anchor_positions = np.asarray(anchor_positions, dtype=float)
    indices = [config['motor_order'].index(mid) for mid in policy['motor_ids']]
    correction = positions[..., indices]-anchor_positions[..., indices]
    limit = policy['max_extra_winding_motor_deg']
    limited = anchor_positions[..., indices]+limit*np.tanh(np.minimum(correction, 0)/limit)
    result = np.array(positions, copy=True)
    result[..., indices] = np.where(correction < 0, limited, positions[..., indices])
    return result


def coordinate_contract(config, alignment, legacy_alignment):
    """Reject mixed coordinate metadata; keep explicitly tagged v1 models readable."""
    return (config.get('schema_version'), config.get('input_coordinates'), config.get('output_alignment')) in (
        (15, INPUT_COORDINATES, alignment),
        (14, INPUT_COORDINATES, alignment),
        (13, INPUT_COORDINATES, alignment),
        (12, INPUT_COORDINATES, alignment),
        (3, INPUT_COORDINATES, alignment),
        (2, INPUT_COORDINATES, alignment),
        (1, 'absolute_glove_angles', legacy_alignment),
    )


def network_input(config, delta_q):
    """New models consume delta directly; only legacy models add the reference."""
    delta = np.asarray(delta_q, dtype=float)
    if config['input_coordinates'] == INPUT_COORDINATES:
        return delta
    return delta + np.asarray(config['q_reference'])


def validate_network(n, outputs):
    for key, size in [('input_center', 4), ('input_scale', 4),
                      ('output_center', outputs), ('output_scale', outputs)]:
        n[key] = finite_values(n[key], size, key)
    if min(n['input_scale']+n['output_scale']) <= 0:
        raise ValueError('标准化尺度必须大于0')
    if 'branches' in n:
        if n.get('architecture') != 'role_weighted_branches' or 'layers' in n:
            raise ValueError('加权分支网络结构标记错误')
        branches = n['branches']
        if not isinstance(branches, list) or not 1 <= len(branches) <= outputs:
            raise ValueError('加权分支数量错误')
        covered = []
        for branch in branches:
            indices = branch['output_indices']
            if not indices or any(type(i) is not int or not 0 <= i < outputs for i in indices):
                raise ValueError('分支输出索引错误')
            gains = finite_values(branch['input_weights'], 4, '分支输入权重')
            if not all(0 < value <= 1 for value in gains):
                raise ValueError('分支输入权重必须在(0,1]内')
            branch['input_weights'] = gains
            if branch.get('shared_thumb_coordination'):
                validate_thumb_coordination(branch)
                covered.extend(indices)
                continue
            monotonic = branch.get('monotonic_inputs')
            direction = branch.get('monotonic_direction', 1)
            shared = branch.get('shared_antagonistic_pair', False)
            thumb_shared = branch.get('shared_thumb_flex_ext',False)
            thumb_side = branch.get('shared_thumb_side',False)
            thumb_opp = branch.get('thumb_monotonic_opposition',False)
            if thumb_opp:
                if (thumb_opp is not True or shared or thumb_shared or thumb_side or indices not in([0],[4])
                        or monotonic!=[1]or direction!=1 or type(direction)is not int
                        or 'output_directions'in branch or 'bending_layers'in branch or 'side_decomposition'in branch):
                    raise ValueError('拇指对掌单调MLP需要M14单输出和q1正向单调路径')
            elif thumb_side:
                if (thumb_side is not True or shared or thumb_shared or indices not in([0,1],[3,5])
                        or monotonic!=[0]or branch.get('output_directions')!=[1,-1]
                        or any(type(d)is not int for d in branch.get('output_directions',[]))
                        or 'monotonic_direction'in branch or 'bending_layers'in branch or 'side_decomposition'in branch):
                    raise ValueError('拇指共享侧摆需要q0单调、M8放线/M27收线方向及完整四角输入')
            elif thumb_shared:
                if (thumb_shared is not True or shared or indices!=[0,1,2]or monotonic!=[2,3]
                        or branch.get('output_directions')!=[-1,1,1]
                        or any(type(d)is not int for d in branch.get('output_directions',[]))
                        or 'monotonic_direction'in branch):
                    raise ValueError('拇指共享屈伸需要M4/M6/M7、q2/q3单调及[-1,+1,+1]方向')
            elif shared:
                if (shared is not True or indices != [0,1] or monotonic != [1,2,3]
                        or branch.get('output_directions') != [1,-1]
                        or any(type(d)is not int for d in branch.get('output_directions',[]))
                        or 'monotonic_direction' in branch):
                    raise ValueError('联合伸屈分支必须共享隐藏层并以[+1,-1]输出两个通道')
            elif 'output_directions' in branch:
                raise ValueError('逐输出方向只适用于声明的共享肌群网络')
            if monotonic is not None:
                if (monotonic not in ([1, 2, 3], [1, 2], [0], [2,3], [1]) or (len(indices) != 1 and not shared and not thumb_shared and not thumb_side)
                        or (monotonic==[2,3]and not thumb_shared)
                        or (monotonic==[1]and not thumb_opp)
                        or type(direction) is not int or direction not in (-1, 1)):
                    raise ValueError('单调分支的输入轴、共享输出组或方向标记不合法')
            elif 'monotonic_direction' in branch:
                raise ValueError('单调方向必须与单调输入约束一起声明')
            hidden_activation = branch.get('hidden_activation', 'mixed_softplus')
            if hidden_activation not in ('mixed_softplus', 'mixed_softplus_75', 'mixed_softplus_tanh', 'softplus'):
                raise ValueError('未知单调隐藏层激活')
            if monotonic is None and 'hidden_activation' in branch:
                raise ValueError('只有单调分支允许声明单调隐藏层激活')
            validate_layers(branch['layers'], len(indices), monotonic=monotonic is not None,
                            monotonic_inputs=monotonic, hidden_activation=hidden_activation)
            convex_axes=branch.get('convex_bend_inputs')
            if convex_axes is not None:
                if (not isinstance(convex_axes,list)or not convex_axes or len(set(convex_axes))!=len(convex_axes)
                        or any(type(i)is not int or i not in(1,2)for i in convex_axes)
                        or not branch.get('shared_antagonistic_pair')or hidden_activation!='mixed_softplus'
                        or len(branch['layers'])!=3):
                    raise ValueError('弯曲曲率连接约束需要共享混合Softplus网络及q1/q2输入')
                ls=branch['layers'];split=len(ls[0]['bias'])//2
                if len(ls[0]['bias'])!=len(ls[1]['bias']):raise ValueError('曲率共享层宽度必须相同')
                if np.any(np.asarray(ls[0]['weight'])[split:,convex_axes]!=0):
                    raise ValueError('曲率约束的凹特征不得读取指定弯曲角')
                if 'flexor_convex_readout'in branch and type(branch['flexor_convex_readout'])is not bool:
                    raise ValueError('屈肌曲率读出标记必须是布尔值')
                if branch.get('flexor_convex_readout'):
                    if np.any(np.asarray(ls[-1]['weight'])[1,split:]!=0):
                        raise ValueError('屈肌曲率输出不得读取第二隐藏层凹特征')
                elif np.any(np.asarray(ls[1]['weight'])[split:,:split]!=0):
                    raise ValueError('共享曲率结构的凹层不得读取凸特征')
            elif 'flexor_convex_readout'in branch:
                raise ValueError('曲率读出缺少对应的弯曲输入')
            if monotonic == [0]and not thumb_side:
                if branch.get('side_decomposition') != 'q0_plus_bending' or 'bending_layers' not in branch:
                    raise ValueError('单调侧摆需要独立q0主分支和弯曲补偿分支')
                if np.any(np.asarray(branch['layers'][0]['weight'])[:, 1:] != 0):
                    raise ValueError('侧摆q0主分支不得读取弯曲角')
                bend_axes=branch.get('bending_monotonic_inputs')
                bend_activation=branch.get('bending_hidden_activation','tanh')
                if bend_axes is not None and (not isinstance(bend_axes,list)or len(bend_axes)!=1
                        or type(bend_axes[0])is not int or bend_axes[0]!=1 or bend_activation!='mixed_softplus_tanh'):
                    raise ValueError('侧摆弯曲单调分支需要q1与Softplus/tanh激活')
                if bend_axes is None and bend_activation!='tanh':
                    raise ValueError('新侧摆弯曲激活必须声明单调输入')
                validate_layers(branch['bending_layers'],len(indices),monotonic=bend_axes is not None,
                    monotonic_inputs=bend_axes,hidden_activation=bend_activation)
                if np.any(np.asarray(branch['bending_layers'][0]['weight'])[:, 0] != 0):
                    raise ValueError('侧摆弯曲补偿分支不得读取q0')
            elif 'bending_layers' in branch or 'side_decomposition' in branch:
                raise ValueError('只有单调侧摆分支允许独立弯曲补偿')
            covered.extend(indices)
        if sorted(covered) != list(range(outputs)):
            raise ValueError('分支输出必须完整覆盖且不得重复')
        return
    validate_layers(n['layers'], outputs)


def validate_layers(layers, outputs, *, monotonic=False, monotonic_inputs=None,
                    hidden_activation='mixed_softplus', input_width=4):
    if not 2 <= len(layers) <= 5:
        raise ValueError('MLP需要1至4个隐藏层')
    width = input_width
    for i, layer in enumerate(layers):
        out = len(layer['bias'])
        activation = 'identity' if i == len(layers)-1 else (hidden_activation if monotonic else 'tanh')
        if (layer.get('activation') != activation or not 1 <= out <= 256
                or len(layer['weight']) != out):
            raise ValueError('MLP层形状或激活函数错误')
        layer['bias'] = finite_values(layer['bias'], out, 'bias')
        layer['weight'] = [finite_values(row, width, 'weight') for row in layer['weight']]
        if monotonic:
            axes = [1, 2, 3] if monotonic_inputs is None else monotonic_inputs
            constrained = np.asarray(layer['weight'])[:, axes] if i == 0 else np.asarray(layer['weight'])
            if np.any(constrained < 0):
                raise ValueError('单调分支的指定输入路径权重不得为负')
        width = out
    if width != outputs:
        raise ValueError('MLP输出数与电机数不一致')


def validate_thumb_coordination(branch):
    """Validate shared neural features, output heads and signed connected paths."""
    if (branch.get('shared_thumb_coordination') is not True
            or branch.get('output_indices') not in ([0, 1, 2], [3, 4, 5])
            or branch.get('motor_ids') != [8, 14, 27]
            or any(k in branch for k in ('layers', 'monotonic_inputs', 'bending_layers',
                'output_directions', 'shared_thumb_side', 'thumb_monotonic_opposition',
                'shared_thumb_flex_ext', 'shared_antagonistic_pair', 'monotonic_direction', 'side_decomposition'))):
        raise ValueError('共享对掌侧摆需要M8/M14/M27三个输出头')
    shared = branch.get('shared_layers')
    if not isinstance(shared, list) or len(shared) != 2:
        raise ValueError('共享姿态特征需要两个隐藏层')
    width = 4
    for i, layer in enumerate(shared):
        out = len(layer['bias'])
        if not 1 <= out <= 256 or len(layer['weight']) != out or layer.get('activation') != 'mixed_softplus':
            raise ValueError('共享姿态特征形状或激活错误')
        layer['bias'] = finite_values(layer['bias'], out, '共享bias')
        layer['weight'] = [finite_values(row, width, '共享weight') for row in layer['weight']]
        if i and np.any(np.asarray(layer['weight']) < 0):
            raise ValueError('共享第二层需要非负权重以保留方向约束')
        width = out
    heads = branch.get('heads')
    if not isinstance(heads, list) or len(heads) != 3:
        raise ValueError('共享姿态特征需要恰好三个输出头')
    connected = []
    for head, mid, direction in zip(heads, (8, 14, 27), (1, -1, -1)):
        if head.get('motor_id') != mid or type(head.get('direction')) is not int or head['direction'] != direction:
            raise ValueError('共享输出头电机顺序或读出符号错误')
        validate_layers(head['layers'], 1, monotonic=True, monotonic_inputs=list(range(width)),
                        input_width=width, hidden_activation='mixed_softplus')
        reach = np.ones(1, dtype=bool)
        for layer in reversed(head['layers']):
            reach = np.any(np.asarray(layer['weight'])[reach] > 0, axis=0)
        if not reach.any():
            raise ValueError('输出头不得与共享特征断开')
        connected.append(reach)
        ancestors = np.any(np.asarray(shared[1]['weight'])[reach] > 0, axis=0)
        axis, sign = (1, -1) if mid == 14 else (0, 1)
        if not ancestors.any() or np.any(np.asarray(shared[0]['weight'])[ancestors, axis]*sign < 0):
            raise ValueError('共享特征的连接破坏侧摆q0或对掌q1单调方向')
    if not np.logical_and.reduce(connected).any():
        raise ValueError('三路必须实际读取共同的姿态特征')


def validate_model_network(config, outputs):
    network = config['network']
    validate_network(network, outputs)
    branched = 'branches' in network
    monotonic = [b for b in network.get('branches', []) if 'monotonic_inputs' in b]
    directions = {index:(tuple(b['monotonic_inputs']), direction)
                  for b in monotonic for index,direction in zip(b['output_indices'],
                      b.get('output_directions',[b.get('monotonic_direction',1)]))}
    expected = {}
    if config['schema_version'] in (5, 6, 7):
        expected[0] = ((1, 2, 3), 1)
    if config['schema_version'] in (6, 7):
        expected[1] = ((1, 2, 3), -1)
    if config['schema_version'] == 7:
        expected.update({2: ((0,), -1), 3: ((0,), 1)})
        if outputs != 4:
            raise ValueError('版本7仅支持四电机伸屈加侧摆结构')
    if config['schema_version'] in (8, 9, 10):
        if config['finger'] != 'middle' or outputs != 5:
            raise ValueError('版本8/9/10仅支持中指五电机结构')
        expected = {0: ((1,2,3), 1), 2: ((1,2,3), -1)}
        if config['schema_version'] in (9, 10):
            expected.update({3: ((0,), 1), 4: ((0,), -1)})
        if config['schema_version'] == 10:
            expected[1] = ((1, 2), -1)
    shared=[b for b in network.get('branches',[])if b.get('shared_antagonistic_pair')]
    thumb_shared=[b for b in network.get('branches',[])if b.get('shared_thumb_flex_ext')]
    thumb_sides=[b for b in network.get('branches',[])if b.get('shared_thumb_side')]
    thumb_opps=[b for b in network.get('branches',[])if b.get('thumb_monotonic_opposition')]
    coordination=[b for b in network.get('branches',[])if b.get('shared_thumb_coordination')]
    if config['schema_version']==15:
        if (config['finger']!='thumb' or outputs!=6 or len(thumb_shared)!=1
                or len(coordination)!=1 or thumb_sides or thumb_opps
                or sorted(tuple(b['output_indices'])for b in network['branches'])!=[(0,1,2),(3,4,5)]):
            raise ValueError('版本15需要拇指共享屈伸及三头共享对掌侧摆两组网络')
        expected={0:((2,3),-1),1:((2,3),1),2:((2,3),1)}
    elif config['schema_version']in(12,13,14):
        if (config['finger']!='thumb'or outputs!=6 or len(thumb_shared)!=1
                or sorted(tuple(b['output_indices'])for b in network.get('branches',[]))!=[(0,1,2),(3,5),(4,)]):
            raise ValueError('版本12/13/14要求拇指屈伸三输出、侧摆双输出和对掌单输出三组MLP')
        expected={0:((2,3),-1),1:((2,3),1),2:((2,3),1)}
    elif thumb_shared:
        raise ValueError('拇指共享单调屈伸需要版本12/13/14')
    if coordination and config['schema_version']!=15:
        raise ValueError('共享对掌侧摆输出头需要版本15')
    if config['schema_version']in(13,14):
        if len(thumb_sides)!=1 or thumb_sides[0]['output_indices']!=[3,5]:
            raise ValueError('版本13/14要求M8/M27共享q0单调侧摆网络')
        expected.update({3:((0,),1),5:((0,),-1)})
    elif thumb_sides:
        raise ValueError('拇指共享q0单调侧摆需要版本13/14')
    if config['schema_version']==14:
        if len(thumb_opps)!=1 or thumb_opps[0]['output_indices']!=[4]:
            raise ValueError('版本14要求M14对q1正向单调')
        expected[4]=((1,),1)
    elif thumb_opps:
        raise ValueError('拇指q1单调对掌需要版本14')
    if config['schema_version']==11:
        if (config['finger']!='ring' or outputs!=4 or len(shared)!=1
                or shared[0]['output_indices']!=[0,1]):
            raise ValueError('版本11要求无名指M10/M17共用一个双输出MLP')
        expected={0:((1,2,3),1),1:((1,2,3),-1),2:((0,),-1),3:((0,),1)}
    elif shared:
        raise ValueError('共享伸屈网络需要版本11')
    if directions != expected:
        raise ValueError('单调分支的输入、方向或数量与模型版本不一致')
    if ((config['schema_version'] == 3 and not branched)
            or (config['schema_version'] in (1, 2) and branched)):
        raise ValueError('版本3必须使用加权分支结构，版本1/2必须使用原全连接结构')
    if branched:
        for branch in network['branches']:
            if 'convex_bend_inputs'in branch and not(config['schema_version']==11 and branch.get('shared_antagonistic_pair')):
                raise ValueError('弯曲曲率连接约束仅适用于版本11共享伸屈网络')
            if ('bending_monotonic_inputs'in branch and
                    not(config['schema_version']==11 and config['finger']=='ring'and branch['output_indices']==[2])):
                raise ValueError('q1单调弯曲补偿当前仅适用于版本11无名指M22')
            if (branch.get('hidden_activation', 'mixed_softplus') != 'mixed_softplus'
                    and not (config['schema_version']==10 and branch['output_indices'] in ([0],[1]))
                    and not (config['schema_version']==11 and branch.get('shared_antagonistic_pair'))):
                if not(config['schema_version']in(12,13,14,15)and(branch.get('shared_thumb_flex_ext')or branch.get('shared_thumb_side')or branch.get('thumb_monotonic_opposition'))):
                    raise ValueError('新激活仅适用于声明的中指、无名指或拇指单调肌群分支')
        policy = config.get('input_priority', {})
        if policy.get('applied_after_standardization') is not True:
            raise ValueError('分支输入权重必须声明在标准化之后应用')
        for branch in network['branches']:
            for index in branch['output_indices']:
                declared = policy.get('per_motor_weights', {}).get(str(config['motor_order'][index]))
                if declared != branch['input_weights']:
                    raise ValueError('电机输入优先级与网络分支权重不一致')


def forward(network, q):
    """Evaluate in the network's declared input coordinates, returning motor degrees."""
    x = (np.asarray(q, dtype=float)-np.asarray(network['input_center']))/network['input_scale']
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        def layers(value, sequence):
            for layer in sequence:
                value = value @ np.asarray(layer['weight']).T + layer['bias']
                if layer['activation'] == 'tanh':
                    value = np.tanh(value)
                elif layer['activation'] == 'softplus':
                    value = np.logaddexp(0, value)
                elif layer['activation'] == 'mixed_softplus_tanh':
                    split = value.shape[-1]//2
                    value = np.concatenate((np.logaddexp(0,value[..., :split]),
                                            np.tanh(value[..., split:])),axis=-1)
                elif layer['activation'] in ('mixed_softplus', 'mixed_softplus_75'):
                    split = (3*value.shape[-1]//4 if layer['activation'] == 'mixed_softplus_75'
                             else value.shape[-1]//2)
                    value = np.concatenate((np.logaddexp(0, value[..., :split]),
                                            -np.logaddexp(0, -value[..., split:])), axis=-1)
            return value
        if 'branches' in network:
            prediction = np.empty((*x.shape[:-1], len(network['output_center'])))
            for branch in network['branches']:
                if branch.get('shared_thumb_coordination'):
                    features = layers(x*np.asarray(branch['input_weights']), branch['shared_layers'])
                    for index, head in zip(branch['output_indices'], branch['heads']):
                        prediction[..., index] = layers(features, head['layers'])[..., 0]*head['direction']
                    continue
                value = layers(x*np.asarray(branch['input_weights']), branch['layers'])
                value = value*np.asarray(branch.get('output_directions',branch.get('monotonic_direction',1)))
                if 'bending_layers' in branch:
                    value = value+layers(x*np.asarray(branch['input_weights']), branch['bending_layers'])
                prediction[..., branch['output_indices']] = value
        else:
            prediction = layers(x, network['layers'])
        return prediction*np.asarray(network['output_scale']) + network['output_center']


def motor_forward(config, delta_q, *, neutral=None):
    """Vectorized controller mapping, including neutral and optional side policy."""
    delta = np.asarray(delta_q, dtype=float)
    if neutral is None:
        neutral = forward(config['network'], network_input(config, np.zeros(4)))
    result = forward(config['network'], network_input(config, delta))-neutral
    if config.get('side_bending_policy') is None:
        return result
    anchor_q = np.zeros_like(delta)
    anchor_q[..., 0] = delta[..., 0]
    anchor = forward(config['network'], network_input(config, anchor_q))-neutral
    return limit_side_bending(config, result, anchor)


class MLPFingerMapper:
    """Match FingerMapper's controller interface; no ROS, Torch or device access."""

    def __init__(self, model_path=None, *, finger='little'):
        if finger not in FINGER_SPECS:
            raise ValueError('未知手指')
        self.finger = finger
        spec = FINGER_SPECS[finger]
        self.finger_label = spec['label']
        self.motor_ids = tuple(spec['motor_inputs'])
        self.motor_inputs = {mid: [0, 1, 2, 3] for mid in self.motor_ids}
        self.glove_slice = slice(*spec['glove_slice'])
        self.motor_roles = dict(zip(self.motor_ids, spec.get('motor_roles', ['伸肌', '屈肌', '内收', '外展'])))
        self.model_path = Path(model_path) if model_path is not None else MODEL_DIR/f'left_{finger}_q_to_motor_mlp.json'
        self.config = c = json.loads(self.model_path.read_text(encoding='utf-8'))
        v4 = (c.get('schema_version') in (4, 5, 6, 7, 8, 9, 10, 11) and c.get('input_coordinates') == INPUT_COORDINATES
              and c.get('output_alignment') == SIDE_ALIGNMENT)
        if (not (v4 or coordinate_contract(c, ALIGNMENT, LEGACY_ALIGNMENT)) or c.get('model_type') != 'mlp'
                or c.get('hand') != 'left' or c.get('finger') != finger
                or c.get('mapping') != 'q_to_motor' or c.get('motor_order') != list(self.motor_ids)
                or c.get('q_units') != 'degrees' or c.get('motor_position_units') != 'degrees'):
            raise ValueError('需要该手指的q→m MLP及匹配的版本、输入和零位增量坐标')
        validate_side_bending_policy(c)
        self.q_reference = tuple(finite_values(c['q_reference'], 4, 'q参考'))
        self.network = n = c['network']
        validate_model_network(c, len(self.motor_ids))
        self._neutral = forward(n, network_input(c, np.zeros(4)))
        if not np.isfinite(self._neutral).all():
            raise ValueError('MLP零位输出不是有限值')

    def effective_models(self, model='mlp'):
        if model != 'mlp':
            raise ValueError('MLP映射器只接受model=mlp')
        return dict.fromkeys(self.motor_ids, 'mlp')

    def effective_degrees(self, model='mlp'):
        self.effective_models(model)
        return dict.fromkeys(map(str, self.motor_ids), None)

    def predict(self, q, *, relative=True, model='mlp'):
        self.effective_models(model)
        angles = np.array(finite_values(q, 4, 'q'))
        try:
            with np.errstate(over='raise', invalid='raise'):
                delta = angles if relative else angles-np.asarray(self.q_reference)
                result = motor_forward(self.config, delta, neutral=self._neutral)
            if not np.isfinite(result).all():
                raise ValueError('MLP预测不是有限值')
        except (OverflowError, FloatingPointError) as exc:
            raise ValueError('MLP输入过大或计算溢出') from exc
        return {f'M{mid}': float(value) for mid, value in zip(self.motor_ids, result)}
