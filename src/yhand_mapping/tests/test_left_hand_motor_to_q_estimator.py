"""Five-finger inverse-model composition and isolated read-only ROS integration."""
import argparse
from contextlib import redirect_stdout
import csv
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
import uuid

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'scripts'))
sys.path.insert(0, str(PACKAGE/'scripts/q_estimator'))
import left_hand_motor_to_q_estimator as hand
from left_thumb_motor_to_q_estimator import FeedbackState, MotorToQEstimator


def fake_feedback(models):
    msg = SimpleNamespace(motors=[SimpleNamespace(pos=0.0, error=0) for _ in range(28)])
    for model in models.values():
        for mid, value in zip(model.motor_ids, model.center):
            msg.motors[mid-1].pos = value
    return msg


class FiveFingerTests(unittest.TestCase):
    def setUp(self):
        self.models = hand.load_models(PACKAGE/'config')

    def test_five_models_preserve_finger_order_columns_unique_motor_ids_and_real_counts(self):
        self.assertEqual(tuple(self.models), hand.FINGERS)
        ids = [mid for model in self.models.values() for mid in model.motor_ids]
        self.assertEqual(len(ids), 24)
        self.assertEqual(len(set(ids)), 24)
        self.assertEqual(set(range(1, 29))-set(ids), {5, 12, 19, 26})
        for i, (finger, model) in enumerate(self.models.items()):
            self.assertEqual(model.config['glove_q_slice'], [8+4*i, 12+4*i])
            self.assertEqual(model.q_names, tuple(f'left_{finger}_q{j}' for j in range(4)))
            self.assertEqual(model.config['training_points'], [8149, 5606, 5889, 2670, 2227][i])
            self.assertEqual(len(model.coefficients), 1+len(model.motor_ids)+len(model.motor_ids)*(len(model.motor_ids)+1)//2)
        with self.assertRaises(ValueError):
            MotorToQEstimator(PACKAGE/'config/left_ring_motor_to_q_estimator.json', finger='index')

    def test_complete_vector_matches_independent_per_finger_absolute_and_relative_outputs(self):
        state = hand.FiveFingerState(self.models)
        msg = fake_feedback(self.models)
        state.update(msg, 10)
        status = state.snapshot(10.1)
        self.assertTrue(status['valid'])
        self.assertTrue(status['publishable'])
        self.assertEqual(len(status['q_absolute_deg']), 20)
        for i, (finger, model) in enumerate(self.models.items()):
            single = FeedbackState(model)
            single.update(msg, 10)
            s = single.snapshot(10.1)
            self.assertEqual(status['q_absolute_deg'][4*i:4*i+4], s['q_absolute_deg'])
            self.assertEqual(status['q_relative_deg'][4*i:4*i+4], s['q_relative_deg'])
        changed = fake_feedback(self.models)
        changed.motors[24].pos += .01  # M25 belongs to index, not other fingers.
        state.update(changed, 11)
        newer = state.snapshot(11)
        self.assertNotEqual(newer['fingers']['index']['q_absolute_deg'], status['fingers']['index']['q_absolute_deg'])
        for finger in ['thumb', 'middle', 'ring', 'little']:
            self.assertEqual(newer['fingers'][finger]['q_absolute_deg'], status['fingers'][finger]['q_absolute_deg'])

    def test_one_motor_error_invalidates_only_its_finger_and_complete_vector(self):
        state = hand.FiveFingerState(self.models)
        msg = fake_feedback(self.models)
        msg.motors[21].error = 4  # M22 ring
        state.update(msg, 0)
        s = state.snapshot(0)
        self.assertFalse(s['valid'])
        self.assertFalse(s['publishable'])
        self.assertNotIn('q_absolute_deg', s)
        self.assertEqual(s['fingers']['ring']['state'], 'motor_error')
        self.assertTrue(all(s['fingers'][f]['publishable'] for f in hand.FINGERS if f != 'ring'))
        msg.motors[21].error = 0
        state.update(msg, .1)
        self.assertTrue(state.snapshot(.1)['valid'])

    def test_nonfinite_single_motor_outside_domain_and_wrist_are_isolated(self):
        for value, expected in [(math.nan, 'invalid_feedback'), (1e4, 'outside_training_bounds')]:
            state = hand.FiveFingerState(self.models)
            msg = fake_feedback(self.models)
            msg.motors[2].pos = value  # M3 index
            state.update(msg, 0)
            s = state.snapshot(0)
            self.assertEqual(s['fingers']['index']['state'], expected)
            self.assertFalse(s['publishable'])
            self.assertTrue(all(s['fingers'][f]['valid'] for f in hand.FINGERS if f != 'index'))
        state = hand.FiveFingerState(self.models)
        msg = fake_feedback(self.models)
        msg.motors[4].pos = math.nan
        msg.motors[4].error = 1
        state.update(msg, 0)
        self.assertTrue(state.snapshot(0)['valid'])

    def test_waiting_whole_frame_failure_timeout_and_no_stale_filling(self):
        state = hand.FiveFingerState(self.models, timeout=.5)
        self.assertFalse(state.snapshot(0)['publishable'])
        state.update(fake_feedback(self.models), 10)
        s = state.snapshot(10.6)
        self.assertNotIn('q_absolute_deg', s)
        self.assertTrue(all(v['state'] == 'stale_feedback' for v in s['fingers'].values()))
        state.update(SimpleNamespace(motors=[]), 11)
        self.assertTrue(all(v['state'] == 'invalid_feedback' for v in state.snapshot(11)['fingers'].values()))

    def test_explicit_24_motor_offsets_and_20_angle_references_are_ordered(self):
        msg = fake_feedback(self.models)
        offsets = list(range(24))
        ids = [mid for model in self.models.values() for mid in model.motor_ids]
        for mid, offset in zip(ids, offsets):
            msg.motors[mid-1].pos += offset
        zero = list(range(20))
        state = hand.FiveFingerState(self.models, q_zero=zero, motor_offset=offsets)
        state.update(msg, 0)
        s = state.snapshot(0)
        self.assertTrue(s['valid'])
        expected = [v for model in self.models.values() for v in model.predict(model.center)]
        np.testing.assert_allclose(s['q_absolute_deg'], expected)
        np.testing.assert_allclose(s['q_relative_deg'], np.array(expected)-zero)
        for options in [{'q_zero': [0]*4}, {'motor_offset': [0]*23}]:
            with self.assertRaises(ValueError):
                hand.FiveFingerState(self.models, **options)

    def test_extrapolation_remains_invalid_and_never_bypasses_motor_errors(self):
        state = hand.FiveFingerState(self.models, allow_extrapolation=True)
        msg = fake_feedback(self.models)
        msg.motors[3].pos = 1000
        state.update(msg, 0)
        s = state.snapshot(0)
        self.assertFalse(s['valid'])
        self.assertTrue(s['publishable'])
        self.assertEqual(len(s['q_absolute_deg']), 20)
        msg.motors[3].error = 7
        state.update(msg, .1)
        self.assertFalse(state.snapshot(.1)['publishable'])

    def test_predict_cli_has_20_angles_and_rejects_truncated_input_without_ros(self):
        positions = [v for model in self.models.values() for v in model.center]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(hand.main(['predict', '--positions', *map(str, positions)]), 0)
        s = json.loads(output.getvalue())
        self.assertEqual(len(s['q_absolute_deg']), 20)
        self.assertTrue(s['valid'])
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            hand.main(['predict', '--positions', *map(str, positions[:-1])])

    def test_source_cache_checks_metadata_and_uses_correct_glove_motor_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'left_hand/index'
            folder.mkdir(parents=True)
            log = folder/'point_log.csv'
            rows = [{'point_id': str(i+1), 'finger': 'index', 'active_mode': 'xy' if i >= 4 else 'x_only',
                     'combo_id': 'combo', 'timeout': '0'} for i in range(40)]
            with log.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            cache = root/'analysis/motor_to_q_estimator/index/data_cache.npz'
            cache.parent.mkdir(parents=True)
            glove = np.arange(40*48).reshape(40, 48)
            feedback = np.arange(40*28*5).reshape(40, 28, 5)
            np.savez(cache, point_ids=np.arange(1, 41), glove=glove, feedback=feedback)
            snapshot = cache.parent/'point_log_snapshot.csv'
            snapshot.write_bytes(log.read_bytes())
            q, m, meta, provenance = hand.load_finger_data(root, 'index')
            np.testing.assert_array_equal(q, glove[4:, 12:16])
            np.testing.assert_array_equal(m, feedback[4:][:, np.array([25, 3, 15, 1, 28])-1, 0])
            self.assertEqual(len(meta), 36)
            self.assertFalse(provenance['mixed_session_training'])
            log.write_text(log.read_text().replace('combo', 'changed'))
            with self.assertRaisesRegex(ValueError, '日志已变化'):
                hand.load_finger_data(root, 'index')

    def test_launch_keeps_driver_external_and_routes_single_or_five_parameters(self):
        from launch import LaunchContext
        from launch.utilities import perform_substitutions
        spec = importlib.util.spec_from_file_location('hand_inverse_launch', PACKAGE/'launch/left_hand_motor_to_q_estimator_launch.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        context = LaunchContext()
        values = dict(finger='five', feedback_topic='/test_feedback', model_dir='/models', model='auto',
            publish_hz='50', feedback_timeout='.5', allow_extrapolation='false', q_topic='', relative_q_topic='',
            joint_state_topic='', valid_topic='', status_topic='', q_zero='', motor_offset='')
        context.launch_configurations.update(values)
        actions = module.setup(context)
        self.assertEqual(len(actions), 1)
        command = [perform_substitutions(context, value) for value in actions[0].cmd]
        self.assertEqual(Path(command[1]).parts[-3:],
                         ('scripts', 'q_estimator', 'left_hand_motor_to_q_estimator.py'))
        self.assertEqual(command[2], 'run')
        self.assertEqual(command[command.index('--finger')+1], 'five')
        self.assertNotIn('--q-topic', command)
        context.launch_configurations.update(finger='index', allow_extrapolation='true', q_zero='1,2,3,4')
        command = [perform_substitutions(context, value) for value in module.setup(context)[0].cmd]
        self.assertIn('--allow-extrapolation', command)
        self.assertEqual(command[command.index('--q-zero')+1:command.index('--q-zero')+5], ['1', '2', '3', '4'])


@unittest.skipUnless(os.environ.get('YHAND_ESTIMATOR_ROS_TEST') == '1', 'Opt-in isolated DDS test')
class ROSFiveFingerTests(unittest.TestCase):
    def test_combined_outputs_per_finger_fault_isolation_recovery_and_dropout(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from devices_pkg.msg import YHandMsg
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool, Float64MultiArray

        models = hand.load_models(PACKAGE/'config')
        root = '/five_inverse_'+uuid.uuid4().hex
        args = argparse.Namespace(feedback_timeout=.2, q_zero=None, motor_offset=None,
                                  allow_extrapolation=False, publish_hz=50)
        for key in ['feedback_topic', 'q_topic', 'relative_q_topic', 'joint_state_topic', 'valid_topic', 'status_topic']:
            setattr(args, key, root+'/'+key)
        rclpy.init(args=[])
        node = harness = executor = None
        try:
            node = hand.create_five_node(args, models)
            harness = rclpy.create_node('five_inverse_harness')
            pub = harness.create_publisher(YHandMsg, args.feedback_topic, 10)
            q, relative, joints, valid, ring, index = [], [], [], [], [], []
            for typ, topic, target in [(Float64MultiArray, args.q_topic, q),
                                      (Float64MultiArray, args.relative_q_topic, relative),
                                      (JointState, args.joint_state_topic, joints), (Bool, args.valid_topic, valid),
                                      (Float64MultiArray, '/left_ring_estimated_q', ring),
                                      (Float64MultiArray, '/left_index_estimated_q', index)]:
                harness.create_subscription(typ, topic, lambda msg, target=target: target.append(msg), 10)
            executor = SingleThreadedExecutor()
            executor.add_node(node)
            executor.add_node(harness)

            def spin_until(condition, feedback=None, timeout=4):
                end = time.monotonic()+timeout
                while time.monotonic() < end:
                    if feedback is not None:
                        pub.publish(feedback)
                    executor.spin_once(timeout_sec=.01)
                    if condition():
                        return
                self.fail('DDS condition did not arrive')

            msg = YHandMsg()
            for model in models.values():
                for mid, value in zip(model.motor_ids, model.center):
                    msg.motors[mid-1].pos = value
            expected = [q for model in models.values() for q in model.predict(
                [float(np.float32(msg.motors[mid-1].pos)) for mid in model.motor_ids])]
            zeros = [q for model in models.values() for q in model.q_zero]
            spin_until(lambda: q and relative and joints and ring and index and valid and valid[-1].data, msg)
            np.testing.assert_allclose(q[-1].data, expected)
            np.testing.assert_allclose(relative[-1].data, np.array(expected)-zeros)
            np.testing.assert_allclose(joints[-1].position, np.radians(expected))
            self.assertEqual(len(joints[-1].name), 20)
            pubs = node.get_publisher_names_and_types_by_node(node.get_name(), node.get_namespace())
            self.assertFalse(any('devices_pkg/msg/YHandMsg' in types for _, types in pubs))
            msg.motors[21].error = 9
            spin_until(lambda: valid and not valid[-1].data and node.state.snapshot(time.monotonic())['fingers']['ring']['state'] == 'motor_error', msg)
            # Drain already queued good outputs before counting fault-period publications.
            until = time.monotonic()+.05
            while time.monotonic() < until:
                executor.spin_once(timeout_sec=.005)
            old_ring, old_q, old_index = len(ring), len(q), len(index)
            spin_until(lambda: len(index) > old_index+2, msg)
            self.assertEqual(len(ring), old_ring)
            self.assertEqual(len(q), old_q)
            msg.motors[21].error = 0
            spin_until(lambda: len(q) > old_q and valid[-1].data, msg)
            spin_until(lambda: valid and not valid[-1].data)
            self.assertTrue(all(s['state'] == 'stale_feedback' for s in node.state.snapshot(time.monotonic())['fingers'].values()))
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
