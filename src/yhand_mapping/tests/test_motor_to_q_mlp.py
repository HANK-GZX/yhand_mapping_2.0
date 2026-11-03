"""Four-finger inverse MLP portability, branch priorities and state contracts."""
from copy import deepcopy
import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'scripts/q_estimator'))
import left_hand_motor_to_q_estimator as hand
from left_thumb_motor_to_q_estimator import MotorToQEstimator
from mlp_motor_to_q import FOUR_FINGERS, SIDE_MOTORS


def models():
    return hand.load_models(PACKAGE/'config', FOUR_FINGERS, 'mlp')


def numpy_prediction(model, positions):
    f = model.config['fits']['mlp']
    x = (np.asarray(positions)-f['input_center'])/f['input_scale']
    result = np.empty(4)
    for b in f['branches']:
        y = x*np.asarray(b['input_weights'])
        for layer in b['layers']:
            y = np.asarray(layer['weight']) @ y+layer['bias']
            if layer['activation'] == 'tanh':
                y = np.tanh(y)
        result[b['output_indices']] = y
    return result*f['output_scale']+f['output_center']


def test_four_artifacts_have_independent_branches_and_muscle_priorities():
    counts = [5606, 5889, 2670, 2227]
    for (finger, model), count in zip(models().items(), counts):
        c = model.config
        assert c['schema_version'] == 2
        assert c['training_points'] == count
        assert c['input_coordinates'] == 'recorded_motor_feedback_positions'
        assert c['output_coordinates'] == 'absolute_glove_angles'
        assert c['source_sha256']['mixed_session_training'] is False
        branches = c['fits']['mlp']['branches']
        assert [b['output_indices'] for b in branches] == [[0], [1, 2, 3]]
        for k, b in enumerate(branches):
            assert len(b['layers']) == 3
            assert len(b['layers'][-1]['bias']) == (1 if k == 0 else 3)
            for mid, gain in zip(model.motor_ids, b['input_weights']):
                primary = (mid in SIDE_MOTORS[finger]) == (k == 0)
                assert gain == (1 if primary else .25)


def test_portable_inference_matches_numpy_for_recorded_bounds_and_extrapolation():
    rng = np.random.default_rng(20261030)
    for model in models().values():
        bounds = np.array(model.bounds)
        probes = np.vstack([np.zeros(len(bounds)), model.center,
                            rng.uniform(bounds[:, 0]-10, bounds[:, 1]+10, (12, len(bounds)))])
        for position in probes:
            np.testing.assert_allclose(model.predict(position), numpy_prediction(model, position), atol=1e-10)
        assert model.predict([0]*len(bounds)) != model.q_zero


def test_changing_q0_branch_leaves_q123_identical():
    model = models()['index']
    modified = deepcopy(model.config)
    modified['fits']['mlp']['branches'][0]['layers'][-1]['bias'][0] += 1
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'model.json'
        path.write_text(json.dumps(modified))
        changed = MotorToQEstimator(path, 'mlp', finger='index')
        a, b = model.predict(model.center), changed.predict(model.center)
    assert a[1:] == b[1:]
    assert b[0]-a[0] == pytest.approx(model.config['fits']['mlp']['output_scale'][0])


@pytest.mark.parametrize('mutation', ['coordinates', 'scale', 'outputs', 'weight_shape', 'schema'])
def test_invalid_artifacts_are_rejected(mutation):
    c = deepcopy(models()['index'].config)
    fit = c['fits']['mlp']
    if mutation == 'coordinates':
        c['input_coordinates'] = 'delta_motor_commands'
    elif mutation == 'scale':
        fit['input_scale'][0] = 0
    elif mutation == 'outputs':
        fit['branches'][1]['output_indices'] = [0, 1, 2]
    elif mutation == 'weight_shape':
        fit['branches'][0]['layers'][0]['weight'][0].pop()
    elif mutation == 'schema':
        c['schema_version'] = 1
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'model.json'
        path.write_text(json.dumps(c))
        with pytest.raises(ValueError):
            MotorToQEstimator(path, 'mlp', finger='index')


def test_four_state_offsets_references_timeout_and_finger_failure():
    fitted = models()
    msg = SimpleNamespace(motors=[SimpleNamespace(pos=0., error=0) for _ in range(28)])
    offsets = list(range(18))
    ids = [mid for m in fitted.values() for mid in m.motor_ids]
    centers = [v for m in fitted.values() for v in m.center]
    for mid, center, offset in zip(ids, centers, offsets):
        msg.motors[mid-1].pos = center+offset
    state = hand.FiveFingerState(fitted, motor_offset=offsets, q_zero=list(range(16)))
    state.update(msg, 1)
    status = state.snapshot(1)
    assert status['finger_order'] == list(FOUR_FINGERS)
    expected = [v for m in fitted.values() for v in m.predict(m.center)]
    np.testing.assert_allclose(status['q_absolute_deg'], expected)
    np.testing.assert_allclose(status['q_relative_deg'], np.array(expected)-np.arange(16))
    assert len(state.motor_ids) == 18
    assert len(status['q_absolute_deg']) == 16
    assert status['valid']
    msg.motors[21].error = 1
    state.update(msg, 1.1)
    failed = state.snapshot(1.1)
    assert not failed['publishable']
    assert failed['fingers']['index']['valid']
    assert failed['fingers']['ring']['state'] == 'motor_error'
    assert state.snapshot(1.7)['fingers']['index']['state'] == 'stale_feedback'


def test_five_mlp_keeps_existing_thumb_and_explicit_thumb_mlp_is_rejected():
    fitted = hand.load_models(PACKAGE/'config', model='mlp')
    assert fitted['thumb'].model == fitted['thumb'].config['default_model']
    assert all(fitted[f].model == 'mlp' for f in FOUR_FINGERS)
    with pytest.raises(ValueError):
        hand.load_models(PACKAGE/'config', ('thumb',), 'mlp')
    # A direct custom model directory works as well as config/motor_to_q_mlp.
    direct = hand.load_models(PACKAGE/'config/motor_to_q_mlp', FOUR_FINGERS, 'mlp')
    assert list(direct) == list(FOUR_FINGERS)


def test_training_fold_scaling_comes_only_from_training_rows():
    from argparse import Namespace
    import torch
    from fit_four_finger_motor_to_q_mlp import train
    torch.set_num_threads(1)
    rng = np.random.default_rng(123)
    m = rng.normal(size=(30, 4))
    q = rng.normal(size=(30, 4))
    args = Namespace(epochs=3, width=4, lr=.006, patience=2,
                     secondary_input_weight=.25, input_priority_penalty=.001)
    fit, epochs = train(m, q, 'little', args, seed=7, validation=(m+1e3, q+1e3))
    np.testing.assert_allclose(fit['input_center'], m.mean(0))
    np.testing.assert_allclose(fit['input_scale'], m.std(0))
    np.testing.assert_allclose(fit['output_center'], q.mean(0))
    assert len(epochs) == 2 and all(1 <= e <= 3 for e in epochs)


def test_four_predict_cli_and_logging_keep_command_estimate_and_model_snapshots():
    from motor_to_q_logging import EstimatorLog
    fitted = models()
    positions = [v for model in fitted.values() for v in model.center]
    output = io.StringIO()
    with redirect_stdout(output):
        assert hand.main(['predict', '--finger', 'four', '--model', 'mlp',
                          '--positions', *map(str, positions)]) == 0
    status = json.loads(output.getvalue())
    assert status['finger_order'] == list(FOUR_FINGERS)
    assert len(status['q_absolute_deg']) == 16
    msg = SimpleNamespace(motors=[SimpleNamespace(pos=0., vel=0., tor=0., error=0) for _ in range(28)])
    for model in fitted.values():
        for mid, position in zip(model.motor_ids, model.center):
            msg.motors[mid-1].pos = position
    reference = [v for m in fitted.values() for v in m.q_zero]
    with tempfile.TemporaryDirectory() as directory:
        args = SimpleNamespace(log_dir=Path(directory), log_max_age=.1,
            feedback_topic='/test_feedback', q_cmd_topic='/left_five_q_command_state',
            glove_topic='/glove_joint_data', glove_raw_topic='', robot_glove_topic='',
            feedback_timeout=.5, q_zero=None, motor_offset=None)
        log = EstimatorLog(args, fitted)
        packet = {'schema_version': 1, 'hand': 'left', 'fingers': list(FOUR_FINGERS),
                  'active': True, 'control_state': 'position', 'q_reference_deg': reference,
                  'q_cmd_absolute_deg': reference, 'q_cmd_relative_deg': [0.]*16,
                  'motor_position_command_deg': [0.]*28}
        log.q_command(packet, log.started+.001)
        log.feedback(msg, status, log.started+.002)
        log.close()
        with (Path(directory)/'samples.csv').open() as stream:
            row = next(csv.DictReader(stream))
        assert row['q_cmd_aligned'] == '1'
        assert 'q_est_abs_thumb_q0' not in row
        assert float(row['q_est_abs_index_q0']) == pytest.approx(status['q_absolute_deg'][0])
        assert float(row['q_cmd_abs_index_q0']) == reference[0]
        metadata = json.loads((Path(directory)/'metadata.json').read_text())
        assert metadata['finger_order'] == list(FOUR_FINGERS)
        assert metadata['closed_cleanly']
        for finger, snapshot in metadata['model_snapshots'].items():
            assert snapshot['selected_model'] == 'mlp'
            restored = MotorToQEstimator(Path(directory)/snapshot['file'], finger=finger)
            assert restored.predict(restored.center) == fitted[finger].predict(fitted[finger].center)
