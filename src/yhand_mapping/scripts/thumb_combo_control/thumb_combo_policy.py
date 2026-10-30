#!/usr/bin/env python3
"""Measured thumb task path, tendon-length compensation and passive coordination.

No ROS or model training dependency. Coordinates use the existing motor zero;
this module never changes motor zero or infers a safe stroke from a stalled log.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

MOTORS = (4, 6, 7, 8, 14, 27)
COMBOS = {'thumb_flex_long_out_opp_on': (4, 27, 14),
          'thumb_flex_long_in_opp_on': (4, 8, 14)}


def vector(value, count, name):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f'{name}需要{count}个数值')
    try:
        result = [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}需要实测数值，不能留空') from exc
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f'{name}必须为有限数值')
    return result


def positive(value, name):
    result = vector([value], 1, name)[0]
    if result <= 0:
        raise ValueError(f'{name}必须大于0')
    return result


def ramp(previous, target, step):
    return previous + max(-step, min(step, target-previous))


def smooth(x):
    x = max(0., min(1., x))
    return x*x*(3.-2.*x)


class ThumbComboPolicy:
    def __init__(self, path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.config = c = json.loads(raw)
        if (c.get('schema_version') != 1 or c.get('kind') != 'thumb_combo_path'
                or c.get('hand') != 'left' or c.get('motor_order') != list(MOTORS)):
            raise ValueError('需要左拇指版本1 combo 标定文件')
        if c.get('calibration_status') != 'measured':
            raise ValueError('combo文件尚无实测标定；请先采集零位、绕弧、对指关键点')
        if not isinstance(c.get('motor_zero_id'), str) or not c['motor_zero_id'].strip():
            raise ValueError('需要本次机械预紧零位的motor_zero_id')
        vector(c['operator_zero_q_deg'], 4, '操作手零位')
        self.metric = vector(c['q_metric_weights'], 4, '角度距离权重')
        if min(self.metric) <= 0:
            raise ValueError('角度距离权重必须大于0')
        self.inner = positive(c['corridor_inner_deg'], 'corridor_inner_deg')
        self.outer = positive(c['corridor_outer_deg'], 'corridor_outer_deg')
        self.neutral_blend = positive(c['neutral_blend_deg'], 'neutral_blend_deg')
        self.transition = positive(c['transition_sec'], 'transition_sec')
        if self.outer <= self.inner:
            raise ValueError('outer必须大于inner')
        self.limits, self.rates, self.max_current, self.active_ff = {}, {}, {}, {}
        for mid in MOTORS:
            key = str(mid)
            lo, hi = vector(c['motor_position_limits_deg'][key], 2, f'M{mid}行程')
            if not lo < hi or not lo <= 0 <= hi:
                raise ValueError(f'M{mid}行程必须递增并包含现有电机零位')
            self.limits[mid] = (lo, hi)
            self.rates[mid] = positive(c['max_position_rate_deg_sec'][key], f'M{mid}速度')
            self.max_current[mid] = positive(c['max_abs_current_A'][key], f'M{mid}最大电流')
            self.active_ff[mid] = vector([c['active_feedforward_A'][key]], 1, f'M{mid}前馈')[0]
            if abs(self.active_ff[mid]) > self.max_current[mid]:
                raise ValueError(f'M{mid}前馈超过电流约束')
        self.frames = c['keyframes']
        if not isinstance(self.frames, list) or len(self.frames) < 3:
            raise ValueError('至少需要中立、外展绕弧、内收对掌三个关键点')
        self.arcs = [0.]
        combos = []
        for i, frame in enumerate(self.frames):
            q = frame['q_relative_deg'] = vector(frame['q_relative_deg'], 4, '关键点q')
            p = frame['motor_positions_deg'] = vector(frame['motor_positions_deg'], 6, '关键点电机位置')
            combo = frame['combo_id']
            if i == 0:
                if combo != 'neutral' or any(abs(x) > 1e-8 for x in q+p):
                    raise ValueError('首关键点必须是现有电机零位与操作手相对零位')
            else:
                if combo not in COMBOS:
                    raise ValueError('仅支持长屈＋外展/内收＋对掌 combo')
                combos.append(combo)
                distance = self.distance(q, self.frames[i-1]['q_relative_deg'])
                if distance < 1e-6:
                    raise ValueError('相邻关键点q不能重复')
                self.arcs.append(self.arcs[-1]+distance)
            for mid, value in zip(MOTORS, p):
                if not self.limits[mid][0] <= value <= self.limits[mid][1]:
                    raise ValueError(f'关键点M{mid}超出标定行程')
        # The accepted task is an outward arc followed by adduction/opposition.
        first_in = next((i for i, x in enumerate(combos) if x.endswith('_in_opp_on')), None)
        if (not combos or not combos[0].endswith('_out_opp_on') or first_in is None
                or any(x.endswith('_out_opp_on') for x in combos[first_in:])):
            raise ValueError('路径顺序应为外展绕弧→内收对掌；返回时沿原路径反向')
        self.last_time = None
        self.positions = None
        self.gain_scales = dict.fromkeys(MOTORS, 1.)
        self.currents = None
        self.last_state = {'enabled': True, 'weight': 0., 'combo': 'neutral'}

    def distance(self, x, y):
        return math.sqrt(sum(w*(a-b)**2 for w, a, b in zip(self.metric, x, y)))

    def project(self, delta_q):
        """Project operator-relative q onto the measured task polyline, no q clipping."""
        q = vector(delta_q, 4, '拇指相对角')
        candidates = []
        for i, (left, right) in enumerate(zip(self.frames[:-1], self.frames[1:])):
            x, y = left['q_relative_deg'], right['q_relative_deg']
            v = [b-a for a, b in zip(x, y)]
            u = sum(w*(p-a)*d for w, p, a, d in zip(self.metric, q, x, v)) / sum(w*d*d for w, d in zip(self.metric, v))
            u = max(0., min(1., u))
            point = [a+u*d for a, d in zip(x, v)]
            candidates.append((self.distance(q, point), i, u, point))
        distance, i, u, point = min(candidates, key=lambda item: item[:2])
        left, right = self.frames[i:i+2]
        arc = self.arcs[i]+u*(self.arcs[i+1]-self.arcs[i])
        weight = smooth(arc/self.neutral_blend)*(1.-smooth((distance-self.inner)/(self.outer-self.inner)))
        positions = {mid: a+u*(b-a) for mid, a, b in zip(MOTORS, left['motor_positions_deg'], right['motor_positions_deg'])}
        # Neutral gains blend through the same scalar task weight, not through
        # an artificial neutral muscle combo that would keep extensors active.
        combo_left = right['combo_id'] if i == 0 else left['combo_id']
        combo_right = right['combo_id']
        active = {mid: (1.-u)*float(mid in COMBOS[combo_left])+u*float(mid in COMBOS[combo_right]) for mid in MOTORS}
        return {'weight': weight, 'distance_deg': distance, 'arc_deg': arc,
                'segment': i, 'segment_fraction': u, 'projected_q_relative_deg': point,
                'combo': combo_right if u >= .5 else combo_left,
                'combo_from': combo_left, 'combo_to': combo_right,
                'positions': positions, 'active_weights': active}

    def reset(self):
        self.last_time = None
        self.positions = None
        self.currents = None
        self.gain_scales = dict.fromkeys(MOTORS, 1.)
        self.last_state = {'enabled': True, 'weight': 0., 'combo': 'inactive'}

    def apply(self, matrix, delta_q, feedback, now):
        """Return one 28-motor matrix; only the six thumb rows can change.

        feedback follows controller order [position, velocity, current, error].
        Calibrated path positions replace extrapolated positions inside the
        corridor. Passive muscles fade to KP/KD/I=0, with no forced payout pulse.
        """
        now = vector([now], 1, '时间')[0]
        if self.last_time is not None and now < self.last_time:
            raise ValueError('时间不能倒退')
        rows = {mid: vector(feedback[mid-1][:4], 4, f'M{mid}反馈') for mid in MOTORS}
        for mid, row in rows.items():
            if row[3] != 0:
                raise RuntimeError(f'combo M{mid}反馈错误')
            if abs(row[2]) > self.max_current[mid]:
                raise RuntimeError(f'combo M{mid}电流{row[2]:.4f}A超过标定上限{self.max_current[mid]:.4f}A')
            if not self.limits[mid][0] <= row[0] <= self.limits[mid][1]:
                raise RuntimeError(f'combo M{mid}反馈位置超出标定行程')
        desired = self.project(delta_q)
        if self.positions is None:
            self.positions = {mid: rows[mid][0] for mid in MOTORS}
            self.currents = {mid: float(matrix[mid-1][2]) for mid in MOTORS}
            dt = 0.
        else:
            # Avoid a large permitted step after a delayed scheduling cycle.
            dt = min(now-self.last_time, .05)
        self.last_time = now
        result = copy.deepcopy(matrix)
        weight = desired['weight']
        corrections, targets, limited, passive = {}, {}, [], []
        for mid in MOTORS:
            original = vector(matrix[mid-1], 5, f'M{mid}MIT')
            active = desired['active_weights'][mid]
            path_target = active*desired['positions'][mid]+(1.-active)*rows[mid][0]
            target = (1.-weight)*original[0]+weight*path_target
            lo, hi = self.limits[mid]
            bounded = max(lo, min(hi, target))
            if bounded != target:
                limited.append(mid)
            self.positions[mid] = ramp(self.positions[mid], bounded, self.rates[mid]*dt)
            scale = 1.-weight*(1.-active)
            self.gain_scales[mid] = ramp(self.gain_scales[mid], scale, dt/self.transition)
            # Zero-current passive side; active FF is explicitly calibrated,
            # never copied blindly from the old -0.1 A collection condition.
            ff = (1.-weight)*original[2]+weight*active*self.active_ff[mid]
            self.currents[mid] = ramp(self.currents[mid], ff, self.max_current[mid]*dt/self.transition)
            result[mid-1] = [self.positions[mid], 0., self.currents[mid],
                             original[3]*self.gain_scales[mid], original[4]*self.gain_scales[mid]]
            corrections[str(mid)] = self.positions[mid]-original[0]
            targets[str(mid)] = bounded
            if self.gain_scales[mid] == 0 and self.currents[mid] == 0:
                passive.append(mid)
        self.last_state = {k: v for k, v in desired.items() if k not in ('positions', 'active_weights')}
        self.last_state.update(enabled=True, motor_target_deg=targets, correction_deg=corrections,
                               gain_scale={str(k): v for k, v in self.gain_scales.items()},
                               passive_motor_ids=passive, limited_motor_ids=limited)
        return result
