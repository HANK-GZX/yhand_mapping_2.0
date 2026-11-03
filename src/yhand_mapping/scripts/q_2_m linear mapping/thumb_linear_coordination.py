"""Optional axis-priority coordination for an explicit thumb linear candidate.

This is a control adjustment of the existing polynomial, not a learned model
or a physical calibration. The default model has no such profile.
"""
import math


class ThumbLinearCoordination:
    def __init__(self, config):
        if (not isinstance(config, dict) or isinstance(config.get('schema_version'), bool)
                or config.get('schema_version') != 2 or config.get('side_mode') != 'full'):
            raise ValueError('拇指协调只支持v3完整四角侧摆配置')
        if set(config) != {'schema_version', 'side_mode', 'opposition_secondary_weight'}:
            raise ValueError('拇指协调配置包含缺失或不支持的参数')
        self.config = dict(config)
        self.side_mode = 'full'
        try:
            value = config['opposition_secondary_weight']
            self.opp_weight = float(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('拇指角度协调参数不完整') from exc
        if isinstance(value, bool) or not math.isfinite(self.opp_weight) or not 0 <= self.opp_weight <= 1:
            raise ValueError('对掌协调权重必须是[0,1]内的有限数值')
        self.last = None

    def apply(self, q, raw, evaluate):
        opposition = evaluate([0., q[1], 0., 0.])
        result = dict(raw)
        result['M14'] = opposition['M14']+self.opp_weight*(raw['M14']-opposition['M14'])
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError('拇指协调输出不是有限数值')
        self.last = dict(side_mode=self.side_mode, side_secondary_weight=1., raw=dict(raw),
                         side_anchor=None, opposition_anchor=opposition)
        return result
