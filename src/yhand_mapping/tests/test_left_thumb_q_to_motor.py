"""Thumb polynomial provenance and five-finger command composition, without hardware."""
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'scripts'))
import fit_left_thumb_q_to_motor as fitting
import left_little_q_to_motor_test as control


class ThumbModelTests(unittest.TestCase):
    def setUp(self):
        self.mapper = control.create_mapper('thumb')

    def test_mixed_selection_four_q_columns_and_neutral_for_both_global_options(self):
        self.assertEqual(self.mapper.motor_ids, (4, 6, 7, 8, 14, 27))
        self.assertEqual(self.mapper.q_reference,(8.491,38.262,26.028,-26.219))
        self.assertFalse(self.mapper.config['reference_is_exact_zero_calibration'])
        self.assertEqual(control.left_manus_q(list(range(48)), 'thumb'), list(range(8, 12)))
        self.assertEqual(self.mapper.effective_degrees(), {'4':1, '6':2, '7':1, '8':2, '14':2, '27':2})
        for option in ['linear', 'quadratic']:
            self.assertEqual(self.mapper.predict([0]*4, model=option), dict.fromkeys(['M4','M6','M7','M8','M14','M27'], 0))
            self.assertEqual(self.mapper.predict(self.mapper.q_reference, relative=False, model=option),
                             self.mapper.predict([0]*4))
            self.assertEqual(self.mapper.predict([1, 5, 8, 2], model=option), self.mapper.predict([1, 5, 8, 2]))
        for mid in self.mapper.motor_ids:
            self.assertEqual(self.mapper.motor_inputs[mid], [0, 1, 2, 3])

    def test_runtime_predictions_match_original_standardized_source_polynomial_differences(self):
        # Independent evaluation in source coordinates catches lost cross terms,
        # wrong normalization and incorrect neutral subtraction during export.
        q = np.random.default_rng(71).uniform(-80,120,(30,4))
        for mid, selected in self.mapper.effective_models().items():
            source = self.mapper.config['motors'][str(mid)]['fits'][selected]['source_fit']
            center, scale = np.array(source['input_center']), np.array(source['input_scale'])
            beta = np.array(source['coefficients'])
            def original(values):
                x = (values-center)/scale
                terms = [np.ones(len(x)), *x.T]
                if selected == 'quadratic':
                    terms += [x[:, i]*x[:, j] for i in range(4) for j in range(i, 4)]
                return np.column_stack(terms)@beta
            expected = original(q)-original(np.zeros((1, 4)))
            actual = np.array([self.mapper.predict(row)[f'M{mid}'] for row in q])
            np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-10)

    def test_quadratic_channels_include_all_squares_and_six_cross_terms(self):
        for mid in [6, 8, 14, 27]:
            model = self.mapper.config['motors'][str(mid)]['fits']['quadratic']
            self.assertEqual(len(model['coefficients']), 15)
            self.assertEqual(model['feature_order'][5:], [f'x{i}*x{j}' for i in range(4) for j in range(i, 4)])
            self.assertTrue(all(abs(c)>1e-12 for c in model['coefficients'][5:]))
        for mid in [4, 7]:
            a = self.mapper.predict([1, 2, 3, 4])[f'M{mid}']
            b = self.mapper.predict([2, 4, 6, 8])[f'M{mid}']
            self.assertAlmostEqual(b, 2*a)

    def test_invalid_inputs_and_wrong_finger_model_rejected(self):
        for q in [[0]*3, [0]*5, [math.nan]*4, [math.inf]*4, [1e308]*4]:
            with self.assertRaises(ValueError):
                control.prepare_target(self.mapper, q)
        with self.assertRaises(ValueError):
            control.FingerMapper(PACKAGE/'config/left_little_q_to_motor_models.json', finger='thumb')

    def test_single_thumb_commands_six_channels_and_preserves_other22_tension(self):
        defaults = control.load_defaults(PACKAGE/'scripts/muscle_adjustment_defaults.py')
        target = control.prepare_target(self.mapper, [1, 3, 5, 2])
        frame = control.position_matrix(defaults, target['positions'], control.default_feedforward('thumb'), 'thumb')
        idle = control.current_matrix(defaults, 'thumb')
        for mid in range(1, 29):
            if mid in self.mapper.motor_ids:
                self.assertEqual(frame[mid-1], [target['positions'][mid], 0, 0, defaults['kp'][mid-1], defaults['kd'][mid-1]])
                self.assertEqual(idle[mid-1], [0]*5)
            else:
                self.assertEqual(frame[mid-1], [0, 0, defaults['tor'][mid-1], 0, 0])
                self.assertEqual(frame[mid-1], idle[mid-1])

    def test_restored_controller_keeps_inputs_and_thumb_predictions_unchanged(self):
        for q in [[5,-10,3,4],[-56.8,40.4,108.7,-41.2]]:
            target=control.prepare_target(self.mapper,q)
            self.assertEqual(target['q_relative'],q)
            self.assertEqual(target['clipped_motors'],[])
            self.assertEqual(target['positions'],
                             {mid:self.mapper.predict(q)[f'M{mid}'] for mid in self.mapper.motor_ids})
        self.assertEqual(control.FINGER_MOTOR_POSITION_LIMITS_DEG['thumb'],{})
        self.assertNotIn('teleop_cmc_gain',self.mapper.config)
        self.assertNotIn('extrapolation_policy',self.mapper.config)


class FiveFingerTests(unittest.TestCase):
    def test_five_independent_inputs_correct_slices_and_24_unique_motors(self):
        mapper = control.create_mapper('five')
        self.assertEqual(control.left_manus_q(list(range(48)), 'five'), list(range(8, 28)))
        self.assertEqual(len(mapper.motor_ids), 24)
        self.assertEqual(len(set(mapper.motor_ids)), 24)
        self.assertEqual(set(range(1, 29))-set(mapper.motor_ids), {5, 12, 19, 26})
        q = [1, 3, 5, 2, -5, 10, 20, 5, 2, 15, 25, 8, -3, 10, 15, 4, 1, 8, 12, 2]
        for option in ['linear', 'quadratic']:
            expected = {}
            for i, finger in enumerate(['thumb','index','middle','ring','little']):
                expected.update(control.create_mapper(finger).predict(q[4*i:4*i+4], model=option))
            self.assertEqual(mapper.predict(q, model=option), expected)
        changed = list(q)
        changed[0] += 3
        for mid in set(mapper.motor_ids)-set(control.create_mapper('thumb').motor_ids):
            self.assertEqual(mapper.predict(q)[f'M{mid}'], mapper.predict(changed)[f'M{mid}'])
        self.assertEqual(mapper.motor_inputs[4], [0, 1, 2, 3])
        self.assertEqual(mapper.motor_inputs[25], [4, 5, 6, 7])
        self.assertEqual(mapper.motor_inputs[23], [8, 9, 10, 11])
        self.assertEqual(mapper.motor_inputs[22], [12, 13, 14, 15])
        self.assertEqual(mapper.motor_inputs[18], [16])

    def test_five_frame_matches_separate_finger_frames_and_stop_zeros_all24(self):
        mapper = control.create_mapper('five')
        defaults = control.load_defaults(PACKAGE/'scripts/muscle_adjustment_defaults.py')
        q = [1, 3, 5, 2]*5
        plan = control.prepare_target(mapper, q)
        frame = control.position_matrix(defaults, plan['positions'], control.default_feedforward('five'),
                                        'five', feedback_positions=dict.fromkeys(mapper.motor_ids, 0))
        for i, finger in enumerate(mapper.fingers):
            single = control.create_mapper(finger)
            positions = control.prepare_target(single, q[4*i:4*i+4])['positions']
            expected = control.position_matrix(defaults, positions, control.default_feedforward(finger), finger,
                                               feedback_positions=dict.fromkeys(single.motor_ids, 0))
            for mid in single.motor_ids:
                self.assertEqual(frame[mid-1], expected[mid-1])
        stop = control.current_matrix(defaults, 'five')
        for mid in range(1, 29):
            self.assertEqual(stop[mid-1], [0]*5 if mid in mapper.motor_ids else [0, 0, defaults['tor'][mid-1], 0, 0])

    def test_thumb_and_five_cli_run_without_ros_and_report_actual_mixed_models(self):
        for finger, size, motors in [('thumb', 4, 6), ('five', 20, 24)]:
            command = [sys.executable, str(PACKAGE/'scripts/left_little_q_to_motor_test.py'),
                       '--finger', finger, '--dry-run', '--q', *(['0']*size)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(len(data['motor_positions']), motors)
            self.assertEqual(set(data['motor_positions'].values()), {0})
            self.assertEqual({mid:data['model_effective_degrees'][mid] for mid in ['4','6','7','8','14','27']},
                             {'4':1,'6':2,'7':1,'8':2,'14':2,'27':2})
            invalid = subprocess.run(command[:-size]+['0']*(size-1), capture_output=True, timeout=10)
            self.assertNotEqual(invalid.returncode, 0)

    def test_all_twenty_angles_must_stabilize_before_latching_zero(self):
        for moving_index in [0, 3, 7, 11, 15, 19]:
            source = control.ManusRelativeInput(1, max_range_deg=3, size=20)
            source.update([0]*20, 0)
            moving = [0]*20
            moving[moving_index] = 10
            source.update(moving, .5)
            self.assertIsNone(source.update([0]*20, 1))
            self.assertIsNone(source.reference)
            self.assertEqual(source.update([0]*20, 2), [0]*20)

    def test_five_uses_original_input_amplitudes_for_all_fingers(self):
        mapper=control.create_mapper('five');q=[5,-10,3,4]*5
        plan=control.prepare_target(mapper,q)
        self.assertEqual(plan['q_relative'],q)
        for i,finger in enumerate(mapper.fingers):
            single=control.prepare_target(control.create_mapper(finger),q[4*i:4*i+4])
            for mid,value in single['positions'].items():
                self.assertEqual(plan['positions'][mid],value)


class PolynomialExportTests(unittest.TestCase):
    def test_zero_constrained_fit_honors_the_calibration_and_predicts_known_polynomial(self):
        rng=np.random.default_rng(23);q=rng.uniform(-20,40,(100,4))
        coefficients=rng.normal(size=(15,6));coefficients[0]=0
        fitted=fitting.fit(q,fitting.features(q,2)@coefficients,2,zero_constrained=True)
        np.testing.assert_array_equal(fitting.predict(np.zeros((1,4)),fitted,2),np.zeros((1,6)))
        np.testing.assert_allclose(fitting.predict(q,fitted,2),fitting.features(q,2)@coefficients,atol=1e-9)

    def test_standardized_fit_reexpands_known_polynomial_with_nonzero_reference(self):
        rng = np.random.default_rng(18)
        q = rng.uniform(-20, 40, (100, 4))
        coefficients = rng.normal(size=(15, 6))
        # All outputs have distinct squares/cross terms and intercepts.
        m = fitting.features(q, 2)@coefficients
        fitted = fitting.fit(q, m, 2)
        rebased = fitting.rebase_coefficients(fitted, 2)
        np.testing.assert_allclose(rebased[1:], coefficients[1:], atol=1e-10)
        np.testing.assert_array_equal(rebased[0], np.zeros(6))


if __name__ == '__main__':
    unittest.main()
