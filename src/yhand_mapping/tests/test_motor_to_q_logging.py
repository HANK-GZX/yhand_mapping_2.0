"""Command telemetry, reception-time alignment, diagnostics and log analysis."""
import argparse
import csv
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
import left_little_q_to_motor_test as control
from motor_to_q_logging import EstimatorLog


def packet(fingers=hand.FINGERS):
    n = 4*len(fingers)
    return {'schema_version': 1, 'hand': 'left', 'fingers': list(fingers), 'active': True,
            'control_state': 'position', 'q_reference_deg': list(range(n)),
            'q_cmd_absolute_deg': [30+i for i in range(n)], 'q_cmd_relative_deg': [30]*n,
            'motor_position_command_deg': list(range(28))}


def feedback(models):
    msg = SimpleNamespace(motors=[SimpleNamespace(pos=0.0, vel=0.0, tor=-.1, error=0) for _ in range(28)])
    for m in models.values():
        for mid, value in zip(m.motor_ids, m.center):
            msg.motors[mid-1].pos = value
    return msg


def options(directory):
    return argparse.Namespace(log_dir=directory, log_max_age=.1, feedback_topic='/feedback',
        q_cmd_topic='/cmd_state', glove_topic='/operator_glove', glove_raw_topic='/operator_raw',
        robot_glove_topic='', feedback_timeout=.5, q_zero=None, motor_offset=None)


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.models = hand.load_models(PACKAGE/'config', model='linear')

    def test_controller_telemetry_is_requested_q_and_final_motor_frame_without_mutation(self):
        mapper = control.create_mapper('five')
        target = control.prepare_target(mapper, [1, 3, 5, 2]*5)
        defaults = control.load_defaults(PACKAGE/'scripts/muscle_adjustment_defaults.py')
        matrix = control.position_matrix(defaults, target['positions'], control.default_feedforward('five'), 'five',
                                         feedback_positions=dict.fromkeys(mapper.motor_ids, 0))
        before = json.dumps(target, sort_keys=True)
        telemetry = control.q_command_snapshot(mapper, matrix, target, control_state='position',
            sequence=5, input_mode='manus', operator_zero=[0]*20, emitted_unix_sec=123)
        self.assertTrue(telemetry['active'])
        self.assertEqual(telemetry['q_cmd_absolute_deg'], target['q_glove'])
        self.assertEqual(telemetry['q_cmd_relative_deg'], target['q_relative'])
        self.assertEqual(telemetry['q_reference_deg'], list(mapper.q_reference))
        for mid in mapper.motor_ids:
            self.assertEqual(telemetry['motor_position_command_deg'][mid-1], matrix[mid-1][0])
        self.assertIsNone(telemetry['motor_position_command_deg'][4])
        self.assertEqual(json.dumps(target, sort_keys=True), before)
        idle = control.q_command_snapshot(mapper, control.current_matrix(defaults, 'five'), target,
            control_state='stopping', sequence=6, input_mode='manus')
        self.assertFalse(idle['active'])
        self.assertIsNone(idle['q_cmd_absolute_deg'])

    def test_log_alignment_age_invalid_diagnostics_model_snapshot_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            args = options(Path(directory)/'run')
            log = EstimatorLog(args, self.models)
            t = log.started
            state = hand.FiveFingerState(self.models)
            msg = feedback(self.models)
            log.q_command(packet(), t+.01)
            log.glove(list(range(48)), t+.02)
            log.glove_raw([0.0]*176, t+.02)
            state.update(msg, t+.03)
            log.feedback(msg, state.snapshot(t+.03), t+.03)
            # Keep the invalid estimate frame, with stale side inputs and NaN recorded as missing.
            msg.motors[3].pos = math.nan
            state.update(msg, t+.5)
            log.feedback(msg, state.snapshot(t+.5), t+.5)
            log.close()
            rows = list(csv.DictReader((args.log_dir/'samples.csv').open()))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['q_cmd_aligned'], '1')
            self.assertEqual(rows[0]['glove_aligned'], '1')
            self.assertEqual(float(rows[0]['q_cmd_abs_index_q0']), 34)
            self.assertEqual(float(rows[0]['operator_glove_index_q0']), 12)
            self.assertEqual(float(rows[0]['m25_position']), self.models['index'].center[0])
            self.assertEqual(rows[1]['q_cmd_aligned'], '0')
            self.assertEqual(rows[1]['thumb_valid'], '0')
            self.assertEqual(rows[1]['m4_position'], '')
            self.assertEqual(rows[1]['q_est_abs_thumb_q0'], '')
            events = [json.loads(line) for line in (args.log_dir/'events.jsonl').read_text().splitlines()]
            self.assertIsNone(events[-1]['data']['motors'][3]['position'])
            metadata = json.loads((args.log_dir/'metadata.json').read_text())
            self.assertTrue(metadata['closed_cleanly'])
            self.assertEqual(metadata['rows'], 2)
            self.assertEqual(metadata['glove_role'], 'operator_input')
            for f, snap in metadata['model_snapshots'].items():
                self.assertTrue((args.log_dir/snap['file']).is_file())
                self.assertEqual(snap['selected_model'], 'linear')
            with self.assertRaises(FileExistsError):
                EstimatorLog(args, self.models)

    def test_missing_or_malformed_qcmd_never_reuses_a_previous_target(self):
        with tempfile.TemporaryDirectory() as directory:
            log = EstimatorLog(options(Path(directory)/'run'), self.models)
            t = log.started
            log.q_command(packet(), t)
            for invalid in [[], {}, dict(packet(), q_cmd_absolute_deg=[1])]:
                log.q_command(invalid, t+.01)
                self.assertNotIn('q_cmd', log.latest)
            state = hand.FiveFingerState(self.models)
            msg = feedback(self.models)
            state.update(msg, t+.02)
            log.feedback(msg, state.snapshot(t+.02), t+.02)
            log.close()
            row = next(csv.DictReader((log.directory/'samples.csv').open()))
            self.assertEqual(row['q_cmd_abs_thumb_q0'], '')
            self.assertEqual(row['q_cmd_aligned'], '0')
            self.assertEqual(row['q_cmd_age_sec'], '')

    def test_snapshot_uses_loaded_model_if_source_is_refitted_later(self):
        from left_thumb_motor_to_q_estimator import MotorToQEstimator
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.json'
            source.write_text(json.dumps(self.models['thumb'].config))
            model = MotorToQEstimator(source, 'linear')
            expected = model.predict(model.center)
            changed = json.loads(source.read_text())
            changed['fits']['linear']['coefficients'][0][0] += 1000
            source.write_text(json.dumps(changed))
            log = EstimatorLog(options(Path(directory)/'run'), {'thumb': model})
            log.close()
            snapshot = MotorToQEstimator(log.directory/'models/left_thumb_motor_to_q_estimator.json', 'linear')
            self.assertEqual(snapshot.predict(model.center), expected)
            self.assertNotEqual(MotorToQEstimator(source, 'linear').predict(model.center), expected)

    def test_analysis_uses_robot_glove_only_for_ground_truth_and_command_deviation_separately(self):
        from analyze_motor_to_q_log import analyze
        with tempfile.TemporaryDirectory() as directory:
            args = options(Path(directory)/'run')
            args.robot_glove_topic = '/independent_robot_glove'
            models = {'thumb': self.models['thumb']}
            log = EstimatorLog(args, models)
            t = log.started
            msg = feedback(models)
            from left_thumb_motor_to_q_estimator import FeedbackState
            state = FeedbackState(models['thumb'])
            for dt in [.02, .05]:
                state.update(msg, t+dt)
                s = state.snapshot(t+dt)
                operator, robot = [0]*48, [0]*48
                operator[8:12] = [100]*4
                robot[8:12] = [v+2 for v in s['q_absolute_deg']]
                log.q_command(packet(('thumb',)), t+dt)
                log.glove(operator, t+dt)
                log.glove_raw([0.0]*176, t+dt)
                log.glove(robot, t+dt, robot=True)
                log.feedback(msg, s, t+dt)
            log.close()
            rows = list(csv.DictReader((log.directory/'samples.csv').open()))
            metadata = json.loads((log.directory/'metadata.json').read_text())
            metadata['_source_root'] = str(log.directory)
            summary = analyze(rows, metadata, log.directory/'analysis')
            for metric in summary['fingers']['thumb']['robot_glove_estimation_error']:
                self.assertAlmostEqual(metric['mae_deg'], 2)
            self.assertGreater(summary['fingers']['thumb']['q_command_deviation'][0]['mae_deg'], 10)
            self.assertTrue((log.directory/'analysis/thumb_angles.png').is_file())
            metadata['robot_glove_available'] = False
            summary = analyze(rows, metadata, log.directory/'without_robot')
            self.assertEqual(summary['fingers']['thumb']['robot_glove_estimation_error'], [])


@unittest.skipUnless(os.environ.get('YHAND_ESTIMATOR_ROS_TEST') == '1', 'Opt-in isolated DDS test')
class ROSLoggingTests(unittest.TestCase):
    def test_receives_qcmd_glove_and_feedback_and_flushes_on_node_shutdown(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from devices_pkg.msg import YHandMsg
        from std_msgs.msg import Float64MultiArray, String
        root = '/inverse_log_'+uuid.uuid4().hex
        with tempfile.TemporaryDirectory() as directory:
            args = options(Path(directory)/'run')
            args.feedback_timeout, args.publish_hz = .5, 50
            args.log_enabled, args.allow_extrapolation = True, False
            for key in ['feedback_topic', 'q_topic', 'relative_q_topic', 'joint_state_topic', 'valid_topic',
                        'status_topic', 'q_cmd_topic', 'glove_topic', 'glove_raw_topic']:
                setattr(args, key, root+'/'+key)
            models = hand.load_models(PACKAGE/'config', model='linear')
            rclpy.init(args=[])
            node = harness = executor = None
            try:
                node = hand.create_five_node(args, models)
                harness = rclpy.create_node('inverse_log_test')
                cmd = harness.create_publisher(String, args.q_cmd_topic, 10)
                glove = harness.create_publisher(Float64MultiArray, args.glove_topic, 10)
                raw = harness.create_publisher(Float64MultiArray, args.glove_raw_topic, 10)
                fb = harness.create_publisher(YHandMsg, args.feedback_topic, 10)
                msg = YHandMsg()
                for model in models.values():
                    for mid, value in zip(model.motor_ids, model.center):
                        msg.motors[mid-1].pos = value
                executor = SingleThreadedExecutor()
                executor.add_node(node)
                executor.add_node(harness)
                end = time.monotonic()+3
                while time.monotonic() < end:
                    cmd.publish(String(data=json.dumps(packet())))
                    glove.publish(Float64MultiArray(data=list(map(float, range(48)))))
                    raw.publish(Float64MultiArray(data=[0.0]*176))
                    fb.publish(msg)
                    executor.spin_once(timeout_sec=.01)
                    if node.diagnostic_log.rows > 4 and node.diagnostic_log.left_at is not None:
                        break
                self.assertGreater(node.diagnostic_log.rows, 4)
                executor.remove_node(node)
                node.destroy_node()
                node = None
                rows = list(csv.DictReader((args.log_dir/'samples.csv').open()))
                aligned = [r for r in rows if r['q_cmd_aligned'] == '1' and r['glove_aligned'] == '1']
                self.assertTrue(aligned)
                self.assertEqual(float(aligned[-1]['q_cmd_abs_thumb_q0']), 30)
                self.assertEqual(float(aligned[-1]['operator_glove_thumb_q0']), 8)
                self.assertEqual(aligned[-1]['robot_glove_thumb_q0'], '')
                self.assertTrue(json.loads((args.log_dir/'metadata.json').read_text())['closed_cleanly'])
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
