"""Task-path compensation, continuous muscle handoff and control integration."""
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE/'scripts/thumb_combo_control'),
                str(PACKAGE/'scripts/q_2_m linear mapping')]
from thumb_combo_policy import COMBOS, MOTORS, ThumbComboPolicy
from collect_thumb_combo_calibration import accepted_frame, build_profile
import left_little_q_to_motor_test as control


def fixture():
    """Synthetic positions/limits for tests ONLY; never a hardware profile."""
    c = json.loads((PACKAGE/'config/left_thumb_combo_template.json').read_text())
    c.update(calibration_status='measured', motor_zero_id='SYNTHETIC_TEST_ONLY')
    for mid in MOTORS:
        key = str(mid)
        c['motor_position_limits_deg'][key] = [-120., 80.]
        c['max_position_rate_deg_sec'][key] = 100.
        c['max_abs_current_A'][key] = .3
    c['keyframes'] = [
        {'label': 'neutral', 'combo_id': 'neutral', 'q_relative_deg': [0., 0., 0., 0.],
         'motor_positions_deg': [0.]*6},
        {'label': 'arc', 'combo_id': 'thumb_flex_long_out_opp_on', 'q_relative_deg': [20., -5., 10., 5.],
         'motor_positions_deg': [-20., 0., 0., 0., -25., -30.]},
        {'label': 'opp', 'combo_id': 'thumb_flex_long_in_opp_on', 'q_relative_deg': [-5., 5., 0., -30.],
         'motor_positions_deg': [-60., 0., 0., -35., -55., 0.]}]
    return c


class ComboTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'profile.json'
        self.path.write_text(json.dumps(fixture()))
        self.policy = ThumbComboPolicy(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_path_projection_retains_measured_motor_targets_and_side_handoff(self):
        c = fixture()
        for i in (1, 2):
            frame = c['keyframes'][i]
            result = self.policy.project(frame['q_relative_deg'])
            self.assertAlmostEqual(result['weight'], 1.)
            for mid, value in zip(MOTORS, frame['motor_positions_deg']):
                self.assertAlmostEqual(result['positions'][mid], value)
            self.assertEqual({mid for mid, value in result['active_weights'].items() if value == 1},
                             set(COMBOS[frame['combo_id']]))
        middle = [(a+b)/2 for a, b in zip(c['keyframes'][1]['q_relative_deg'], c['keyframes'][2]['q_relative_deg'])]
        result = self.policy.project(middle)
        self.assertAlmostEqual(result['active_weights'][8], .5)
        self.assertAlmostEqual(result['active_weights'][27], .5)
        self.assertEqual(result['active_weights'][4], 1.)
        self.assertEqual(result['active_weights'][14], 1.)
        for mid in (6, 7):
            self.assertEqual(result['active_weights'][mid], 0.)

    def test_endpoint_applies_calibrated_targets_and_releases_extensors_without_affecting_four_fingers(self):
        mapper = control.create_mapper('five', model='mlp')
        q = fixture()['keyframes'][-1]['q_relative_deg'] + [-2., 5., 10., 3.]*4
        target = control.prepare_target(mapper, q, model='mlp')
        defaults = control.load_defaults(PACKAGE/'scripts/muscle_adjustment_defaults.py')
        original = control.position_matrix(defaults, target['positions'], control.default_feedforward('five'),
                                           'five', feedback_positions=dict.fromkeys(mapper.motor_ids, 0.))
        self.assertGreater(original[3][0], 0.)
        self.assertGreater(original[13][0], fixture()['keyframes'][-1]['motor_positions_deg'][4])
        fb = [[0.]*4 for _ in range(28)]
        previous = None
        for tick in range(101):
            matrix = self.policy.apply(original, q[:4], fb, tick*.02)
            if previous is not None:
                for mid in MOTORS:
                    self.assertLessEqual(abs(matrix[mid-1][0]-previous[mid-1][0]), 2.+1e-8)
                    self.assertLessEqual(abs(matrix[mid-1][3]-previous[mid-1][3]), defaults['kp'][mid-1]*.05+1e-7)
            for mid in MOTORS:
                fb[mid-1][0] = matrix[mid-1][0]
            for mid in set(range(1, 29))-set(MOTORS):
                self.assertEqual(matrix[mid-1], original[mid-1])
            previous = matrix
        self.assertAlmostEqual(matrix[3][0], -60.)
        self.assertAlmostEqual(matrix[13][0], -55.)
        for mid in (6, 7, 27):
            self.assertEqual(matrix[mid-1][2:], [0., 0., 0.])
        for mid in (4, 8, 14):
            self.assertGreater(matrix[mid-1][3], 0.)

    def test_neutral_outside_corridor_return_and_stop_reset(self):
        self.assertEqual(self.policy.project([0.]*4)['weight'], 0.)
        self.assertEqual(self.policy.project([1000., -1000., 1000., -1000.])['weight'], 0.)
        # Continuous falloff at the corridor's outer edge along a perpendicular.
        end = fixture()['keyframes'][-1]['q_relative_deg']
        d = [end[0]-20., end[1]+5., end[2]-10., end[3]-5.]
        n = [d[1], -d[0], 0., 0.]
        norm = sum(x*x for x in n)**.5
        weights = [self.policy.project([x+distance*v/norm for x, v in zip(end, n)])['weight']
                   for distance in [14.99, 15., 15.01]]
        self.assertLess(max(weights), 1e-5)
        matrix = [[0., 0., 0., 800., 50.] for _ in range(28)]
        self.policy.apply(matrix, end, [[0.]*4 for _ in range(28)], 10.)
        self.policy.reset()
        self.assertIsNone(self.policy.positions)
        self.assertIsNone(self.policy.last_time)
        self.assertEqual(self.policy.last_state['combo'], 'inactive')

    def test_calibrated_limits_current_and_feedback_errors_fail_before_output(self):
        matrix = [[0., 0., 0., 800., 50.] for _ in range(28)]
        for channel, value in [(0, -121.), (2, -.31), (3, 1.)]:
            with self.subTest(channel=channel):
                fb = [[0.]*4 for _ in range(28)]
                fb[6][channel] = value
                with self.assertRaises(RuntimeError):
                    self.policy.apply(matrix, [0.]*4, fb, 0.)
        # Candidate model positions are bounded even outside the task corridor.
        matrix[3][0] = 10000.
        fb = [[0.]*4 for _ in range(28)]
        for tick in range(101):
            out = self.policy.apply(matrix, [1000.]*4, fb, tick*.02)
        self.assertEqual(out[3][0], 80.)
        self.assertIn(4, self.policy.last_state['limited_motor_ids'])

    def test_template_bad_geometry_and_wrong_finger_rejected(self):
        with self.assertRaisesRegex(ValueError, '实测标定'):
            ThumbComboPolicy(PACKAGE/'config/left_thumb_combo_template.json')
        for change in ['zero', 'sequence', 'duplicate', 'limits']:
            with self.subTest(change=change):
                c = fixture()
                if change == 'zero':
                    c['keyframes'][0]['motor_positions_deg'][0] = 3.
                elif change == 'sequence':
                    c['keyframes'][1]['combo_id'] = 'thumb_flex_long_in_opp_on'
                elif change == 'duplicate':
                    c['keyframes'][1]['q_relative_deg'] = [0.]*4
                else:
                    c['motor_position_limits_deg']['4'] = [0., -120.]
                self.path.write_text(json.dumps(c))
                with self.assertRaises(ValueError):
                    ThumbComboPolicy(self.path)
        with self.assertRaisesRegex(ValueError, 'thumb/five'):
            control.load_thumb_combo_policy(self.path, 'four')

    def test_capture_rejects_stale_or_moving_samples_and_builder_preserves_motor_zero(self):
        records = [{'time': i*.02, 'operator_q_deg': [2., 30., 0., -10.],
                    'motor_positions_deg': [0.]*6, 'motor_current_A': [-.03]*6}
                   for i in range(51)]
        accepted = accepted_frame(records, 'neutral', 'n', now=1., window_sec=1.)
        with self.assertRaises(ValueError):
            accepted_frame(records, 'neutral', 'n', now=2., window_sec=1.)
        moving = copy.deepcopy(records)
        moving[20]['operator_q_deg'][0] += 4.
        with self.assertRaises(ValueError):
            accepted_frame(moving, 'neutral', 'n', now=1., window_sec=1.)
        recording = {'hand': 'left', 'motor_order': list(MOTORS), 'motor_zero_id': 'TEST', 'frames': [accepted]}
        for frame in fixture()['keyframes'][1:]:
            recording['frames'].append({'label': frame['label'], 'combo_id': frame['combo_id'],
                'operator_q_deg': [a+b for a, b in zip(frame['q_relative_deg'], accepted['operator_q_deg'])],
                'motor_positions_deg': list(frame['motor_positions_deg']),
                'motor_current_A': [-.03]*6, 'accepted_taut_pose': True, 'samples': 51})
        result = build_profile(recording, fixture())
        self.assertEqual(result['keyframes'][-1]['motor_positions_deg'], fixture()['keyframes'][-1]['motor_positions_deg'])
        self.assertEqual(result['keyframes'][-1]['q_relative_deg'], fixture()['keyframes'][-1]['q_relative_deg'])
        recording['frames'][0]['motor_positions_deg'][0] = 20.
        with self.assertRaisesRegex(ValueError, '原零位'):
            build_profile(recording, fixture())

    def test_cli_dry_run_and_launch_route_profile_to_single_controller(self):
        script = PACKAGE/'scripts/q_2_m linear mapping/left_little_q_to_motor_test.py'
        result = subprocess.run([sys.executable, str(script), '--finger', 'five', '--model', 'mlp',
            '--q', *map(str, fixture()['keyframes'][-1]['q_relative_deg']+[0.]*16),
            '--thumb-combo-calibration', str(self.path), '--dry-run'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        packet = json.loads(result.stdout)
        self.assertEqual(packet['thumb_combo']['motor_target_deg']['4'], -60.)
        self.assertEqual(packet['mit_matrix'][3][0], 0.)  # First ramped frame.
        from launch import LaunchContext
        from launch.actions import ExecuteProcess
        from launch.utilities import perform_substitutions
        spec = importlib.util.spec_from_file_location('combo_launch', PACKAGE/'launch/left_hand_q_to_motor_launch.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ctx = LaunchContext()
        ctx.launch_configurations.update(finger='five', input_mode='2', relative='true', model='mlp',
            publish_hz='50', current='', start_driver_node='false', start_angle_input_terminal='false',
            start_glove_node='false', manus_topic='/glove', manus_raw_topic='/raw', manus_zero_seconds='1',
            manus_timeout='.5', thumb_combo_calibration=str(self.path))
        processes = [a for a in module._launch_setup(ctx) if isinstance(a, ExecuteProcess)]
        self.assertEqual(len(processes), 1)
        command = [perform_substitutions(ctx, item) for item in processes[0].cmd]
        self.assertEqual(command[command.index('--thumb-combo-calibration')+1], str(self.path))


class FakeSDKComboTests(unittest.TestCase):
    def exercise(self, mode='normal'):
        try:
            import rclpy
            from devices_pkg.msg import YHandMsg
        except ImportError:
            self.skipTest('source install/setup.bash for FakeSDK tests')
        helper = PACKAGE/'tests/test_left_little_q_to_motor_test.py'
        with tempfile.TemporaryDirectory(prefix='thumb-combo-fake-') as directory:
            root = Path(directory)
            profile = root/'synthetic.json'
            config = fixture()
            if mode.startswith('manus'):
                config['keyframes'][1]['q_relative_deg'] = [8., -5., 2., 4.]
                config['keyframes'][2]['q_relative_deg'] = [1., 10., 12., 3.]
            profile.write_text(json.dumps(config))
            fake = root/'fake_sdk'
            fake.write_text(f'#!{sys.executable}\nimport runpy\nrunpy.run_path({str(helper)!r},run_name="fake_little_sdk")\n')
            fake.chmod(0o755)
            topic = f'/thumb_combo_test_{os.getpid()}'
            env = dict(os.environ, ROS_LOCALHOST_ONLY='1', ROS_DOMAIN_ID=str(100+os.getpid()%100),
                ROS_LOG_DIR=str(root/'roslog'), LITTLE_TEST_EVENTS=str(root/'events.jsonl'),
                LITTLE_TEST_GLOVE_TOPIC=topic+'/glove', LITTLE_TEST_TARGET_TOPIC=topic+'/target',
                LITTLE_TEST_RAW_TOPIC=topic+'/raw', LITTLE_TEST_MODE=mode, LITTLE_TEST_FINGER='five',
                PYTHONDONTWRITEBYTECODE='1')
            cmd = [sys.executable, str(PACKAGE/'scripts/q_2_m linear mapping/left_little_q_to_motor_test.py'),
                '--finger', 'five', '--model', 'mlp', '--q',
                *map(str, fixture()['keyframes'][-1]['q_relative_deg']+[0.]*16),
                '--thumb-combo-calibration', str(profile), '--driver-executable', str(fake),
                '--cmd-topic', topic+'/cmd', '--feedback-topic', topic+'/feedback',
                '--glove-topic', topic+'/glove', '--output-dir', str(root/'output'),
                '--hold-seconds', '1.2' if mode == 'normal' else '10',
                '--feedback-timeout', '.25', '--ready-timeout', '8']
            if mode.startswith('manus'):
                index = cmd.index('--q')
                del cmd[index:index+21]
                cmd += ['--input-mode', 'manus', '--manus-topic', topic+'/glove',
                        '--manus-raw-topic', topic+'/raw', '--manus-zero-seconds', '.1',
                        '--manus-timeout', '.2']
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=20)
            self.assertEqual(result.returncode, 0 if mode in ('normal','manus_long_run') else 1, result.stdout+'\n'+result.stderr)
            events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
            matrices = [e['matrix'] for e in events if e['event'] == 'command']
            self.assertTrue(matrices)
            self.assertTrue(all(len(matrix) == 28 for matrix in matrices))
            self.assertTrue(all(not e.get('pretighten', False) for e in events if e['event'] == 'command'))
            self.assertTrue(all(row[2:] == [0., 0., 0.] for mid in control.create_mapper('five').motor_ids
                                for row in [matrices[-1][mid-1]]))
            with (root/'output/samples.csv').open() as f:
                rows = list(csv.DictReader(f))
            active = [row for row in rows if row['thumb_combo_weight'] and float(row['thumb_combo_weight']) > .99]
            self.assertTrue(active)
            for row in active:
                self.assertAlmostEqual(float(row['m4_command'])-float(row['m4_prediction']), float(row['m4_combo_correction']))
                self.assertEqual(float(row['m25_command']), float(row['m25_target']))
            if mode in ('normal', 'manus_long_run'):
                settled = [row for row in active if row['m7_control_mode'] == 'zero_current']
                self.assertTrue(settled)
                self.assertAlmostEqual(float(settled[-1]['m4_command']), -60., places=4)
                self.assertAlmostEqual(float(settled[-1]['m14_command']), -55., places=4)
                self.assertEqual(float(settled[-1]['m7_command_kp']), 0.)

    def test_combo_targets_and_passive_extensors_with_one_driver(self):
        self.exercise()

    def test_feedback_dropout_stops_combo_and_four_fingers(self):
        self.exercise('dropout')

    def test_manus_direction_drives_combo_with_single_command_source(self):
        self.exercise('manus_long_run')

    def test_manus_dropout_stops_combo_even_with_live_motor_feedback(self):
        self.exercise('manus_dropout')


if __name__ == '__main__':
    unittest.main()
