#!/usr/bin/env python3
"""Convert existing v1 absolute-angle models to equivalent v2 delta-angle models."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from mlp_q_to_motor import (ALIGNMENT, INPUT_COORDINATES, MODEL_DIR, PACKAGE,
                            MLPFingerMapper, forward)


def convert(config):
    """Only translate input statistics/support; weights and motor labels stay fixed."""
    if config.get('schema_version') != 1 or config.get('input_coordinates') != 'absolute_glove_angles':
        raise ValueError('转换只接受版本1绝对角度模型')
    result = deepcopy(config)
    reference = np.asarray(config['q_reference'])
    result['network']['input_center'] = (np.asarray(config['network']['input_center'])-reference).tolist()
    if config['model_type'] == 'mlp':
        result['output_alignment'] = ALIGNMENT
    else:
        raise ValueError('未知神经模型类型')
    bounds = np.asarray(result['q_absolute_training_bounds_deg'])
    result.update(schema_version=2, input_coordinates=INPUT_COORDINATES,
        training_input_definition='delta_q = recorded_glove_angles - q_reference; before standardization',
        training_target_definition='Unchanged recorded motor feedback positions in degrees',
        q_delta_training_bounds_deg=(bounds-reference[:, None]).tolist(),
        coordinate_migration={
            'method': 'Exact input translation: input_center -= q_reference; weights, scales and motor labels unchanged',
            'validation_note': 'Existing validation scores and folds retained from the original absolute-angle training; predictions are equivalent. Future training groups delta angles.',
            'retrained': False,
        })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mlp-dir', type=Path, default=MODEL_DIR)
    parser.add_argument('--backup-dir', type=Path,
                        default=PACKAGE/'scripts/motor_test_logs/q_to_motor_delta_migration'/datetime.now().strftime('%Y%m%d_%H%M%S'))
    args = parser.parse_args(argv)
    pending, checks = [], []
    rng = np.random.default_rng(20261027)
    for folder, model_type, mapper_type in [(args.mlp_dir, 'mlp', MLPFingerMapper)]:
        for path in sorted(folder.glob(f'left_*_q_to_motor_{model_type}.json')):
            original = json.loads(path.read_text())
            if original.get('schema_version') in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
                continue
            # Validate the source contract before translating anything.
            old = mapper_type(path, finger=original['finger'])
            updated = convert(original)
            pending.append((path, original, updated, mapper_type, old))
    if not pending:
        print('No version-1 models to migrate.')
        return 0
    args.backup_dir.mkdir(parents=True, exist_ok=False)
    prepared = []
    for path, original, updated, mapper_type, old in pending:
        backup = args.backup_dir/path.parent.name/path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(path.read_bytes())
        temporary = backup.with_suffix('.delta.json')
        temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        new = mapper_type(temporary, finger=original['finger'])
        bounds = np.asarray(updated['q_delta_training_bounds_deg'])
        delta = np.vstack([np.zeros(4), rng.uniform(bounds[:, 0]-10, bounds[:, 1]+10, (512, 4))])
        absolute = delta+old.q_reference
        np.testing.assert_allclose(forward(old.network, absolute), forward(new.network, delta), atol=1e-9, rtol=1e-10)
        differences = []
        for d, a in zip(delta, absolute):
            previous = np.array(list(old.predict(d).values()))
            current = np.array(list(new.predict(d).values()))
            np.testing.assert_allclose(previous, current, atol=1e-9, rtol=1e-10)
            np.testing.assert_allclose(current, list(new.predict(a, relative=False).values()), atol=1e-9, rtol=1e-10)
            differences.append(float(abs(previous-current).max()))
        assert not any(new.predict([0]*4).values())
        checks.append({'model': str(path), 'poses_checked': len(delta),
                       'maximum_motor_difference_deg': max(differences)})
        prepared.append((path, temporary))
    # All candidates have passed before replacing any selected model.
    for path, candidate in prepared:
        target = path.with_suffix('.json.tmp')
        target.write_bytes(candidate.read_bytes())
        target.replace(path)
    (args.backup_dir/'verification.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(f'Migrated {len(prepared)} models; originals and verification: {args.backup_dir}')
    print(f'Maximum motor output difference: {max(row["maximum_motor_difference_deg"] for row in checks):.3g} deg')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
