"""Manual-input support from observed delta-q tuples, not mechanical limits."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_REFERENCE = Path(__file__).resolve().parents[1] / 'config/manual_joint_coupling.json'


class JointCouplingError(ValueError):
    def __init__(self, results):
        self.results = results
        super().__init__('；'.join(JointCouplingReference.format_result(result)
                                 for result in results if not result['accepted']))


def merged_intervals(values, tolerance):
    intervals = []
    for value in sorted(values):
        lo, hi = float(value - tolerance), float(value + tolerance)
        if intervals and lo <= intervals[-1][1]:
            intervals[-1][1] = max(hi, intervals[-1][1])
        else:
            intervals.append([lo, hi])
    return intervals


def interval_text(intervals):
    return ' / '.join(f'[{lo:.1f}, {hi:.1f}]' for lo, hi in intervals) or '无记录支持'


class JointCouplingReference:
    def __init__(self, path=DEFAULT_REFERENCE):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.data = json.loads(raw)
        if self.data.get('schema_version') != 1 or self.data.get('coordinates') != 'delta_q_deg':
            raise ValueError('关节耦合参考必须为 schema_version=1 的 delta_q_deg 数据')
        self.tolerance = np.asarray(self.data['tolerance_deg'], dtype=float)
        if self.tolerance.shape != (4,) or not np.isfinite(self.tolerance).all() or (self.tolerance <= 0).any():
            raise ValueError('关节耦合容差必须是四个正有限数')
        self.clouds = {}
        for finger, profile in self.data['fingers'].items():
            values = np.asarray(profile['observed_delta_q_deg'], dtype=float)
            if values.ndim != 2 or values.shape[1] != 4 or not len(values) or not np.isfinite(values).all():
                raise ValueError(f'{finger}缺少有效的四角配对数据')
            self.clouds[finger] = values

    def condition(self, finger, q01, q2=None):
        if finger not in self.clouds:
            raise ValueError(f'耦合参考缺少{finger}数据')
        values = np.asarray(q01, dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError('q0/q1必须是两个有限数')
        if q2 is not None and not math.isfinite(q2):
            raise ValueError('q2必须为有限数')
        cloud = self.clouds[finger]
        distance = np.max(abs(cloud[:, :2] - values) / self.tolerance[:2], axis=1)
        local = cloud[distance <= 1 + 1e-9]
        nearest = cloud[int(np.argmin(distance))]
        matching_q2 = (local[abs(local[:, 2] - q2) <= self.tolerance[2] + 1e-9]
                       if q2 is not None else local)
        return dict(finger=finger, q01_deg=values.tolist(), condition_count=len(local),
                    tolerance_deg=self.tolerance.tolist(),
                    nearest_condition_recorded_delta_q_deg=nearest.tolist(),
                    q2_intervals_deg=merged_intervals(local[:, 2], self.tolerance[2]),
                    q3_intervals_deg=merged_intervals(matching_q2[:, 3], self.tolerance[3]),
                    q2_condition_deg=q2,
                    paired_examples_deg=self._examples(matching_q2[:, 2:], limit=4))

    @staticmethod
    def _examples(pairs, limit=4):
        if not len(pairs):
            return []
        ordered = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        return ordered[np.unique(np.linspace(0, len(ordered) - 1, min(limit, len(ordered)), dtype=int))].tolist()

    def check(self, finger, delta_q):
        q = np.asarray(delta_q, dtype=float)
        if q.shape != (4,) or not np.isfinite(q).all():
            raise ValueError('耦合检查需要四个有限Δq')
        result = self.condition(finger, q[:2], float(q[2]))
        cloud = self.clouds[finger]
        mask = np.all(abs(cloud[:, :2] - q[:2]) <= self.tolerance[:2] + 1e-9, axis=1)
        local = cloud[mask]
        accepted = False
        if len(local):
            distances = np.max(abs(local[:, 2:] - q[2:]) / self.tolerance[2:], axis=1)
            nearest = local[int(np.argmin(distances))]
            accepted = bool(distances.min() <= 1 + 1e-9)
        else:
            nearest = np.asarray(result['nearest_condition_recorded_delta_q_deg'])
        result.update(accepted=accepted, requested_delta_q_deg=q.tolist(),
                      reason='supported' if accepted else ('q23_pair_unsupported' if len(local) else 'q01_unobserved'),
                      nearest_recorded_delta_q_deg=nearest.tolist())
        return result

    def require(self, fingers, delta_q):
        values = list(delta_q)
        if len(values) != 4 * len(fingers):
            raise ValueError('耦合检查角度数与手指数量不一致')
        results = [self.check(finger, values[4*i:4*i+4]) for i, finger in enumerate(fingers)]
        rejected = [result for result in results if not result['accepted']]
        if rejected:
            raise JointCouplingError(results)
        return results

    @staticmethod
    def format_result(result):
        prefix = f"{result['finger']}："
        if result['condition_count'] == 0 and 'reason' not in result:
            return prefix + 'q0/q1附近缺少记录；最近记录Δq=' + str(
                [round(value, 1) for value in result['nearest_condition_recorded_delta_q_deg']])
        if result.get('reason') == 'q01_unobserved':
            return prefix + 'q0/q1附近缺少记录，本次目标未执行；最近记录Δq=' + str(
                [round(value, 1) for value in result['nearest_recorded_delta_q_deg']])
        if result.get('accepted') is False:
            return (prefix + 'q2/q3配对缺少记录，本次目标未执行；'
                    + 'q2支持区间=' + interval_text(result['q2_intervals_deg'])
                    + '°；当前q2下q3支持区间=' + interval_text(result['q3_intervals_deg'])
                    + '°；附近已记录(q2,q3)=' + str(
                        [round(v, 1) for v in result['nearest_recorded_delta_q_deg'][2:]]))
        return (prefix + f"q0/q1附近有{result['condition_count']}个姿态代表点；"
                + 'q2支持区间=' + interval_text(result['q2_intervals_deg'])
                + '°；' + ('当前q2下' if result['q2_condition_deg'] is not None else '全部配对的')
                + 'q3支持区间=' + interval_text(result['q3_intervals_deg'])
                + '°；配对示例=' + str([[round(v, 1) for v in pair] for pair in result['paired_examples_deg']]))
