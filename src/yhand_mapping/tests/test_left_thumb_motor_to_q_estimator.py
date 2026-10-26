"""Independent inverse regression, feedback quality and optional isolated ROS I/O."""
import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'scripts'))
sys.path.insert(0, str(PACKAGE/'scripts/q_estimator'))
import left_thumb_motor_to_q_estimator as inverse


def message(positions):
    motors = [SimpleNamespace(pos=999.0, error=0) for _ in range(28)]
    for mid, value in zip(inverse.MOTOR_IDS, positions):
        motors[mid-1].pos = value
    return SimpleNamespace(motors=motors)


class InverseEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.model = inverse.MotorToQEstimator()
        self.m = self.model.center

    def test_fitted_artifact_keeps_absolute_coordinates_and_complete_features(self):
        c = self.model.config
        self.assertEqual(c['training_points'], 8149)
        self.assertEqual(c['training_timeouts'], 3126)
        self.assertEqual(c['output_coordinates'], 'absolute_glove_angles')
        self.assertEqual(c['fits']['quadratic']['design_rank'], 28)
        self.assertEqual(c['fits']['quadratic']['feature_order'][7:],
                         [f'x{i}*x{j}' for i in range(6) for j in range(i, 6)])
        self.assertNotEqual(self.model.predict([0.0]*6), list(inverse.Q_ZERO))
        for name in ['stratified_5fold', 'leave_one_combo_out']:
            linear = c['fits']['linear']['validation'][name]['mae_deg']
            quadratic = c['fits']['quadratic']['validation'][name]['mae_deg']
            self.assertTrue(all(q < l for q, l in zip(quadratic, linear)))

    def test_standardized_inference_matches_independent_polynomial_evaluation(self):
        rng = np.random.default_rng(66)
        for name in ['linear', 'quadratic']:
            model = inverse.MotorToQEstimator(model=name)
            f = model.config['fits'][name]
            for x in rng.uniform(-1.5, 1.5, (20, 6)):
                positions = x*np.array(f['input_scale'])+f['input_center']
                beta = np.array(f['coefficients'])
                expected = beta[0]+x@beta[1:7]
                if name == 'quadratic':
                    k = 7
                    for i in range(6):
                        for j in range(i, 6):
                            expected += x[i]*x[j]*beta[k]
                            k += 1
                np.testing.assert_allclose(model.predict(positions), expected, atol=1e-10)

    def test_feedback_selects_correct_motors_and_does_not_use_other_motor_data(self):
        state = inverse.FeedbackState(self.model)
        msg = message(self.m)
        msg.motors[0].pos = math.nan
        msg.motors[0].error = 7
        state.update(msg, 10)
        status = state.snapshot(10.1)
        self.assertTrue(status['valid'])
        np.testing.assert_allclose(status['q_absolute_deg'], self.model.predict(self.m))
        np.testing.assert_allclose(status['q_relative_deg'], np.array(status['q_absolute_deg'])-inverse.Q_ZERO)

    def test_no_data_timeout_and_recovery_never_treat_old_output_as_fresh(self):
        state = inverse.FeedbackState(self.model, timeout=.5)
        self.assertEqual(state.snapshot(5)['state'], 'waiting_for_feedback')
        state.update(message(self.m), 10)
        self.assertTrue(state.snapshot(10.4)['publishable'])
        stale = state.snapshot(10.51)
        self.assertEqual(stale['state'], 'stale_feedback')
        self.assertFalse(stale['valid'])
        self.assertFalse(stale['publishable'])
        self.assertNotIn('q_absolute_deg', stale)
        state.update(message(self.m), 11)
        self.assertTrue(state.snapshot(11)['valid'])
        self.assertEqual(state.snapshot(11)['sequence'], 2)

    def test_bad_frames_immediately_invalidate_previous_good_frame(self):
        state = inverse.FeedbackState(self.model)
        for bad, expected in [(message([math.nan]*6), 'invalid_feedback'),
                              (message([math.inf]*6), 'invalid_feedback'),
                              (SimpleNamespace(motors=[]), 'invalid_feedback')]:
            state.update(message(self.m), 0)
            state.update(bad, .1)
            status = state.snapshot(.2)
            self.assertEqual(status['state'], expected)
            self.assertFalse(status['publishable'])
        bad = message(self.m)
        bad.motors[13].error = 9
        state.update(bad, 1)
        self.assertEqual(state.snapshot(1)['state'], 'motor_error')
        self.assertEqual(state.snapshot(1)['error_motors'], [14])

    def test_outside_bounds_is_flagged_without_clipping_and_diagnostic_mode_stays_invalid(self):
        m = list(self.m)
        m[0] = self.model.bounds[0][0]-20
        for allow in [False, True]:
            state = inverse.FeedbackState(self.model, allow_extrapolation=allow)
            state.update(message(m), 0)
            status = state.snapshot(.1)
            self.assertEqual(status['state'], 'outside_training_bounds')
            self.assertEqual(status['outside_motor_ids'], [4])
            self.assertFalse(status['valid'])
            self.assertEqual(status['publishable'], allow)
            self.assertEqual(status['model_motor_positions_deg'], m)
            self.assertEqual(status['q_absolute_deg'], self.model.predict(m))

    def test_explicit_encoder_offset_and_relative_reference_do_not_modify_absolute_model(self):
        offset = [10, 20, 30, 40, 50, 60]
        state = inverse.FeedbackState(self.model, motor_offset=offset, q_zero=[1, 2, 3, 4])
        state.update(message([v+o for v, o in zip(self.m, offset)]), 0)
        status = state.snapshot(0)
        self.assertTrue(status['valid'])
        np.testing.assert_allclose(status['q_absolute_deg'], self.model.predict(self.m))
        np.testing.assert_allclose(status['q_relative_deg'], np.array(self.model.predict(self.m))-[1, 2, 3, 4])

    def test_unseen_motor_combination_with_outside_angle_is_not_valid_feedback(self):
        m = [-100, 30, 100, 40, -80, 10]
        self.assertEqual(self.model.outside_motors(m), [])
        self.assertIn(2, self.model.outside_q(self.model.predict(m)))
        for allow in [False, True]:
            state = inverse.FeedbackState(self.model, allow_extrapolation=allow)
            state.update(message(m), 0)
            status = state.snapshot(0)
            self.assertFalse(status['valid'])
            self.assertEqual(status['publishable'], allow)
            self.assertIn(2, status['outside_q_indices'])

    def test_invalid_config_inputs_and_nonfinite_options_rejected(self):
        for values in [[0]*5, [math.nan]*6, [math.inf]*6, [1e308]*6]:
            with self.assertRaises(ValueError):
                self.model.predict(values)
        for options in [{'timeout': 0}, {'timeout': math.nan}, {'q_zero': [0]*3},
                        {'motor_offset': [math.inf]*6}]:
            with self.assertRaises(ValueError):
                inverse.FeedbackState(self.model, **options)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.json'
            c = dict(self.model.config, mapping='q_to_motor')
            path.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                inverse.MotorToQEstimator(path)

    def test_supervised_fit_recovers_known_polynomial_on_unseen_positions(self):
        import fit_left_thumb_q_to_motor as forward_fit
        rng = np.random.default_rng(72)
        m = rng.uniform(-3, 4, (180, 6))
        beta = rng.normal(size=(28, 4))
        q = np.array([inverse.feature_row(row, 2) for row in m])@beta
        meta = [{'combo_id': f'group{i%6}', 'timeout': '0'} for i in range(len(m))]
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(collection_dir=Path(directory), refresh_cache=False,
                                      output=Path(directory)/'inverse.json')
            with patch.object(forward_fit, 'load', return_value=(q, m, meta, {})), redirect_stdout(io.StringIO()):
                inverse.fit_model(args)
            model = inverse.MotorToQEstimator(args.output)
            test = rng.uniform(-3, 4, (10, 6))
            expected = np.array([inverse.feature_row(row, 2) for row in test])@beta
            np.testing.assert_allclose([model.predict(row) for row in test], expected, atol=1e-10)
            self.assertTrue(all(e < 1e-10 for e in model.config['fits']['quadratic']['validation']['leave_one_combo_out']['mae_deg']))

    def test_predict_cli_works_without_starting_ros_and_reports_units(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = inverse.main(['predict', '--positions', *map(str, self.m)])
        self.assertEqual(code, 0)
        data = json.loads(output.getvalue())
        self.assertEqual(data['outside_motor_ids'], [])
        np.testing.assert_allclose(data['q_absolute_deg'], self.model.predict(self.m))

    def test_launch_starts_only_estimator_with_feedback_topic_and_optional_extrapolation(self):
        from launch import LaunchContext
        from launch.actions import ExecuteProcess
        from launch.utilities import perform_substitutions
        path = PACKAGE/'launch/left_thumb_motor_to_q_estimator_launch.py'
        spec = importlib.util.spec_from_file_location('inverse_launch', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        context = LaunchContext()
        names = ['feedback_topic', 'model_file', 'model', 'q_topic', 'relative_q_topic',
                 'joint_state_topic', 'valid_topic', 'status_topic', 'publish_hz', 'feedback_timeout']
        context.launch_configurations.update({k: f'test_{k}' for k in names})
        context.launch_configurations['allow_extrapolation'] = 'false'
        actions = module.setup(context)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], ExecuteProcess)
        command = [perform_substitutions(context, item) for item in actions[0].cmd]
        self.assertEqual(Path(command[1]).parts[-3:],
                         ('scripts', 'q_estimator', 'left_thumb_motor_to_q_estimator.py'))
        self.assertEqual(command[2], 'run')
        self.assertNotIn('--allow-extrapolation', command)
        self.assertEqual(command[command.index('--feedback-topic')+1], 'test_feedback_topic')
        context.launch_configurations['allow_extrapolation'] = 'true'
        command = [perform_substitutions(context, item) for item in module.setup(context)[0].cmd]
        self.assertIn('--allow-extrapolation', command)


@unittest.skipUnless(os.environ.get('YHAND_ESTIMATOR_ROS_TEST') == '1', 'Opt-in isolated DDS test')
class ROSIntegrationTests(unittest.TestCase):
    def test_fake_feedback_publishes_absolute_relative_radians_and_invalidates_on_dropout(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from devices_pkg.msg import YHandMsg
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool, Float64MultiArray

        root = '/estimator_test_'+uuid.uuid4().hex
        args = argparse.Namespace(feedback_timeout=.15, q_zero=inverse.Q_ZERO,
                                  motor_offset=[0]*6, allow_extrapolation=False, publish_hz=50)
        for key in ['feedback_topic', 'q_topic', 'relative_q_topic', 'joint_state_topic', 'valid_topic', 'status_topic']:
            setattr(args, key, root+'/'+key)
        model = inverse.MotorToQEstimator()
        rclpy.init(args=[])
        node = harness = executor = None
        try:
            node = inverse.create_estimator_node(args, model)
            harness = rclpy.create_node('inverse_estimator_test')
            pub = harness.create_publisher(YHandMsg, args.feedback_topic, 10)
            q, relative, joints, valid = [], [], [], []
            harness.create_subscription(Float64MultiArray, args.q_topic, lambda msg: q.append(msg), 10)
            harness.create_subscription(Float64MultiArray, args.relative_q_topic, lambda msg: relative.append(msg), 10)
            harness.create_subscription(JointState, args.joint_state_topic, lambda msg: joints.append(msg), 10)
            harness.create_subscription(Bool, args.valid_topic, lambda msg: valid.append(msg.data), 10)
            executor = SingleThreadedExecutor()
            executor.add_node(node)
            executor.add_node(harness)

            def spin_until(condition, timeout=4, feedback=None):
                end = time.monotonic()+timeout
                while time.monotonic() < end:
                    if feedback is not None:
                        pub.publish(feedback)
                    executor.spin_once(timeout_sec=.01)
                    if condition():
                        return
                self.fail('DDS condition did not arrive')

            msg = YHandMsg()
            for mid, value in zip(inverse.MOTOR_IDS, model.center):
                msg.motors[mid-1].pos = value
            actual_m = [msg.motors[mid-1].pos for mid in inverse.MOTOR_IDS]
            expected = model.predict(actual_m)
            spin_until(lambda: q and relative and joints and valid and valid[-1], feedback=msg)
            np.testing.assert_allclose(q[-1].data, expected, atol=1e-8)
            np.testing.assert_allclose(relative[-1].data, np.array(expected)-inverse.Q_ZERO, atol=1e-8)
            np.testing.assert_allclose(joints[-1].position, np.radians(expected), atol=1e-8)
            self.assertEqual(joints[-1].name, list(inverse.Q_NAMES))
            publishers = node.get_publisher_names_and_types_by_node(node.get_name(), node.get_namespace())
            self.assertFalse(any('devices_pkg/msg/YHandMsg' in types for _, types in publishers))
            spin_until(lambda: valid and not valid[-1])
            count = len(q)
            stop = time.monotonic()+.2
            while time.monotonic() < stop:
                executor.spin_once(timeout_sec=.01)
            self.assertEqual(len(q), count)
            spin_until(lambda: len(q) > count and valid[-1], feedback=msg)
            msg.motors[13].error = 7
            spin_until(lambda: valid and not valid[-1], feedback=msg)
            self.assertEqual(node.state.snapshot(time.monotonic())['state'], 'motor_error')
        finally:
            if executor is not None:
                executor.shutdown()
            if node is not None:
                node.destroy_node()
            if harness is not None:
                harness.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    unittest.main()
