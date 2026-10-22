#!/usr/bin/env python3
"""Calculate left little/ring motor positions from four relative glove angles."""

import argparse
import json
import math
from pathlib import Path
import re
import sys


FINGER_SPECS = {
    'little': {'label':'小指','motor_inputs':{11:[0,1,2,3],13:[0,1,2,3],18:[0],20:[0]},'glove_slice':(24,28)},
    'ring': {'label':'无名指','motor_inputs':{10:[0,1,2,3],17:[0,1,2,3],22:[0,1,2,3],21:[0,1,2,3]},'glove_slice':(20,24)},
}
DEFAULT_MODEL_PATHS = {finger:Path(__file__).resolve().parents[1]/f'config/left_{finger}_q_to_motor_models.json'
                       for finger in FINGER_SPECS}
DEFAULT_MODEL_PATH = DEFAULT_MODEL_PATHS['little']
MOTOR_INPUTS = FINGER_SPECS['little']['motor_inputs']
MOTOR_ROLES = dict(zip(MOTOR_INPUTS,['伸肌','屈肌','内收','外展']))


def finite_values(values, size, name):
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name}必须是{size}个有限数值') from exc
    if len(result) != size or not all(math.isfinite(value) for value in result):
        raise ValueError(f'{name}必须是{size}个有限数值')
    return result


class FingerMapper:
    """Load coefficients once, then call predict() repeatedly without file I/O.

    Little side motors use q0; ring motors all use four q.
    Inputs and outputs are degrees. No motor commands are published.
    """

    def __init__(self, model_path=None, *, finger='little'):
        if finger not in FINGER_SPECS:
            raise ValueError('finger必须为little或ring')
        self.finger = finger
        spec = FINGER_SPECS[finger]
        self.finger_label = spec['label']
        self.motor_inputs = spec['motor_inputs']
        self.motor_ids = tuple(self.motor_inputs)
        self.motor_roles = dict(zip(self.motor_ids,['伸肌','屈肌','内收','外展']))
        self.glove_slice = slice(*spec['glove_slice'])
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATHS[finger]
        with self.model_path.open(encoding='utf-8') as stream:
            self.config = json.load(stream)
        if (self.config.get('schema_version') != 1
                or self.config.get('hand') != 'left'
                or self.config.get('finger') != finger
                or self.config.get('motor_order') != list(self.motor_ids)
                or self.config.get('q_units') != 'degrees'
                or self.config.get('motor_position_units') != 'degrees'):
            raise ValueError(f'模型必须是左手{self.finger_label}、角度单位为度的版本1模型')
        self.q_reference = tuple(finite_values(self.config['q_reference'], 4, '角度参考'))
        self.fits = {}
        for mid, indices in self.motor_inputs.items():
            target = self.config['motors'][str(mid)]
            if target['input_q_indices'] != indices:
                raise ValueError(f'M{mid}输入角度与要求不一致')
            for name, degree in [('linear', 1), ('quadratic', 2)]:
                fit = target['fits'][name]
                count = len(indices)
                terms = 1 + count + (count * (count + 1) // 2 if degree == 2 else 0)
                if fit['degree'] != degree:
                    raise ValueError(f'M{mid}的{name}阶数不一致')
                center = finite_values(fit['input_center'], count, '输入中心')
                scale = finite_values(fit['input_scale'], count, '输入尺度')
                if any(value <= 0 for value in scale):
                    raise ValueError('输入尺度必须大于0')
                coefficients = finite_values(fit['coefficients'], terms, '模型系数')
                self.fits[(mid, name)] = (center, scale, coefficients)

    def predict(self, q, *, relative=True, model='linear'):
        """Return a position for each selected finger motor, keyed by M<number>.

        By default q is relative [q0,q1,q2,q3], with the neutral pose at zero.
        Set relative=False explicitly for original glove angles.
        Results are the fitted feedback-position coordinates, without clipping.
        """
        if model not in ('linear', 'quadratic'):
            raise ValueError('model必须为linear或quadratic')
        angles = finite_values(q, 4, 'q')
        if not relative:
            angles = [value - offset for value, offset in zip(angles, self.q_reference)]
        result = {}
        try:
            for mid, indices in self.motor_inputs.items():
                center, scale, coefficients = self.fits[(mid, model)]
                x = [(angles[j] - center[k]) / scale[k] for k, j in enumerate(indices)]
                features = [1.0, *x]
                if model == 'quadratic':
                    features.extend(x[j] * x[k] for j in range(len(x)) for k in range(j, len(x)))
                value = math.fsum(a * b for a, b in zip(coefficients, features))
                if not math.isfinite(value):
                    raise ValueError('输入过大，计算结果不是有限数值')
                result[f'M{mid}'] = value
        except (OverflowError, FloatingPointError) as exc:
            raise ValueError('输入过大，计算溢出') from exc
        return result


class LittleFingerMapper(FingerMapper):
    """Compatibility entry point for existing little-finger callers."""
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        super().__init__(model_path,finger='little')


def _print_positions(positions, as_json, roles=MOTOR_ROLES):
    if as_json:
        print(json.dumps(positions, ensure_ascii=False, allow_nan=False))
    else:
        for mid, role in roles.items():
            print(f'M{mid}（{role}）: {positions[f"M{mid}"]:.6f}°')


def _interactive(mapper, args):
    coordinate = '已减本模型参考值的相对q' if args.relative else '原始glove q'
    print(f'输入4个{coordinate}（单位°，空格/逗号/分号分隔）；输入quit退出。', file=sys.stderr)
    while True:
        try:
            if sys.stdin.isatty():
                print('q0 q1 q2 q3 > ', end='', file=sys.stderr, flush=True)
            line = sys.stdin.readline()
            if not line:
                return 0
            line = line.strip()
            if line.lower() in ('q', 'quit', 'exit'):
                return 0
            if not line:
                continue
            parts = re.split(r'[\s,，;；]+', line.strip('[]【】()（） '))
            positions = mapper.predict(parts, relative=args.relative, model=args.model)
            _print_positions(positions, args.json, mapper.motor_roles)
        except ValueError as exc:
            print(f'输入错误：{exc}', file=sys.stderr)
        except KeyboardInterrupt:
            print(file=sys.stderr)
            return 130


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finger',choices=list(FINGER_SPECS),default='little')
    parser.add_argument('--q', type=float, nargs=4, metavar=('Q0', 'Q1', 'Q2', 'Q3'),
                        help='相对零位的四角度（°）；省略时进入交互输入')
    coordinates = parser.add_mutually_exclusive_group()
    coordinates.add_argument('--relative', dest='relative', action='store_true',
                             help='输入相对q（默认），不再减零位')
    coordinates.add_argument('--absolute', dest='relative', action='store_false',
                             help='显式输入原始glove q，程序减去零位')
    parser.set_defaults(relative=True)
    parser.add_argument('--model', choices=['linear', 'quadratic'], default='linear',
                        help='默认linear；小指侧摆用q0，无名指侧摆始终用四q一次补偿')
    parser.add_argument('--model-file', type=Path,
                        help='模型系数JSON文件')
    parser.add_argument('--json', action='store_true', help='以JSON输出四台电机位置')
    args = parser.parse_args(argv)
    try:
        mapper = FingerMapper(args.model_file,finger=args.finger)
        if args.q is None:
            return _interactive(mapper, args)
        _print_positions(mapper.predict(args.q, relative=args.relative, model=args.model), args.json, mapper.motor_roles)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
