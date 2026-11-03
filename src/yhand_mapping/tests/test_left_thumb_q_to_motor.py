"""Thumb polynomial provenance and five-finger command composition, without hardware."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'scripts'))
sys.path.insert(0, str(PACKAGE/'scripts/q_2_m linear mapping'))
import fit_left_thumb_q_to_motor as fitting
import left_little_q_to_motor_test as control
from motor_trajectory import MotorTrajectoryLimiter, mapper_motion_limits


MOTION_LIMITS = {'14': dict(max_velocity_deg_sec=120., max_acceleration_deg_sec2=480.)}
V3_MODEL = PACKAGE/'config/experiments/left_thumb_linear_coordination_v3.json'


class ThumbFullCompensationTests(unittest.TestCase):
    def setUp(self):
        self.mapper = control.create_mapper('thumb', V3_MODEL)

    def test_all_four_angles_and_original_other_muscles_are_preserved(self):
        for q in ([0.]*4, [-8.62, 8.10, -2.54, -44.40], [8.81, 7.31, -11.68, -20.89],
                  [10., -10., 10., 0.], [10., -10., 30., 0.], [20., 0., 60., 20.]):
            raw = self.mapper._evaluate(q, self.mapper.effective_models())
            result = self.mapper.predict(q)
            for mid in (4, 6, 7, 8, 27):
                self.assertEqual(result[f'M{mid}'], raw[f'M{mid}'])
            anchor = self.mapper._evaluate([0., q[1], 0., 0.], self.mapper.effective_models())['M14']
            self.assertAlmostEqual(result['M14'], anchor+.25*(raw['M14']-anchor))
            self.assertEqual(self.mapper.coordination.last['side_secondary_weight'], 1.)
            self.assertIsNone(self.mapper.coordination.last['side_anchor'])
        self.assertEqual(self.mapper.effective_degrees()['8'], 2)
        self.assertEqual(self.mapper.effective_degrees()['27'], 2)
        self.assertEqual(set(self.mapper.predict([0.]*4).values()), {0.})

    def test_logged_q2_transition_no_longer_amplifies_side_commands(self):
        before = [7.7668045977011495, -26.924827586206895, 20.478540229885056, 13.292091954022988]
        after = [7.95580459770115, -24.6608275862069, 28.082540229885055, 12.488091954022988]
        positions = [self.mapper.predict(q) for q in (before, after)]
        self.assertAlmostEqual(positions[1]['M8']-positions[0]['M8'], 23.29638957879539)
        # The problematic inward pose restores both stronger adduction winding
        # and more antagonistic abduction payout, rather than just one channel.
        result = self.mapper.predict([-8.62, 8.1, -2.54, -44.4])
        self.assertLess(result['M8'], -55.)
        self.assertGreater(result['M27'], 60.)

    def test_full_side_profile_rejects_legacy_gate_and_invalid_weights(self):
        from thumb_linear_coordination import ThumbLinearCoordination
        base = self.mapper.config['thumb_linear_coordination']
        for patch in ({'schema_version':1}, {'side_mode':'q0_only'}, {'bend_blend_start_deg':10.},
                      {'side_secondary_weight':1.}, {'opposition_secondary_weight':True},
                      {'opposition_secondary_weight':float('nan')}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                ThumbLinearCoordination(dict(base, **patch))




    def test_q1_primary_response_and_absolute_relative_coordinates(self):
        base = control.create_mapper('thumb')
        for q1 in (-25., -10., 0., 10.):
            self.assertAlmostEqual(self.mapper.predict([0., q1, 0., 0.])['M14'],
                                   base.predict([0., q1, 0., 0.])['M14'])
        q = np.array([9., 7., -12., -20.])
        np.testing.assert_allclose(list(self.mapper.predict(q).values()),
            list(self.mapper.predict(q+np.array(self.mapper.q_reference), relative=False).values()), atol=1e-10)
        group = type('Group', (), {'mappers': (self.mapper, control.create_mapper('index'))})()
        self.assertEqual(mapper_motion_limits(group), {14: self.mapper.config['teleop_motion_limits']['14']})

    def test_five_v3_changes_only_thumb_and_preserves_twenty_angle_contract(self):
        original = control.create_mapper('five')
        current = control.create_mapper('five', thumb_version='v3')
        self.assertEqual(current.motor_ids, original.motor_ids)
        self.assertEqual(len(current.motor_ids), 24)
        self.assertEqual(current.q_reference, original.q_reference)
        self.assertEqual(current.model_paths['thumb'], str(V3_MODEL))
        self.assertEqual(mapper_motion_limits(current), {14: self.mapper.config['teleop_motion_limits']['14']})
        self.assertEqual(control.selected_thumb_version(current), 'v3')
        self.assertEqual(control.selected_thumb_version(original), 'default')
        self.assertFalse(mapper_motion_limits(original))
        for finger in ('index', 'middle', 'ring', 'little'):
            self.assertEqual(current.model_paths[finger], original.model_paths[finger])
        for q in np.random.default_rng(83).uniform(-10, 30, (12, 20)):
            before, after = original.predict(q), current.predict(q)
            thumb = self.mapper.predict(q[:4])
            for mid in current.motor_ids:
                self.assertEqual(after[f'M{mid}'], thumb[f'M{mid}'] if mid in self.mapper.motor_ids
                                 else before[f'M{mid}'])
        self.assertEqual(set(current.predict([0.]*20).values()), {0.})

    def test_v3_selector_rejects_conflicting_fingers_models_and_files(self):
        for kwargs in ({'finger':'four'}, {'finger':'thumb', 'model':'mlp'},
                       {'finger':'five', 'model':'mlp'}, {'finger':'thumb', 'model_path':V3_MODEL}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                control.create_mapper(thumb_version='v3', **kwargs)
        with self.assertRaises(ValueError):
            control.create_mapper('five', thumb_version='unknown')

    def test_five_v3_cli_dry_run_selects_thumb_model_and_only_limits_m14(self):
        command = [sys.executable, str(PACKAGE/'scripts/q_2_m linear mapping/left_little_q_to_motor_test.py'),
                   '--finger', 'five', '--model', 'linear', '--thumb-version', 'v3',
                   '--dry-run', '--q', *map(str, [10., -10., 20., 20.]*5)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['thumb_version'], 'v3')
        self.assertEqual(data['model_files']['thumb'], str(V3_MODEL))
        self.assertEqual(set(data['motor_trajectory_limits']), {'14'})
        self.assertEqual(data['mit_matrix'][13][0], 0.)
        self.assertNotEqual(data['motor_positions']['M14'], 0.)


class MotorTrajectoryTests(unittest.TestCase):
    def fixture(self):
        limiter = MotorTrajectoryLimiter({14: MOTION_LIMITS['14']})
        matrix = [[0., 0., 0., 0., 0.] for _ in range(28)]
        matrix[13] = [-200., 0., 0., 800., 50.]
        feedback = [[0., 0., 0., 0.] for _ in range(28)]
        return limiter, matrix, feedback

    def test_step_and_reversal_obey_speed_acceleration_and_settle(self):
        limiter, matrix, feedback = self.fixture()
        positions, velocities = [], []
        for i in range(600):
            matrix[13][0] = -200. if i < 200 else 100.
            out = limiter.apply(matrix, feedback, i*.02)
            positions.append(out[13][0])
            velocities.append(limiter.states[14][1])
            feedback[13][0] = out[13][0]
            for mid in range(28):
                if mid != 13:
                    self.assertEqual(out[mid], matrix[mid])
        self.assertLessEqual(max(abs(np.diff(positions)/.02)), 120.+1e-8)
        self.assertLessEqual(max(abs(np.diff(velocities)/.02)), 480.+1e-8)
        self.assertAlmostEqual(positions[-1], 100., places=8)
        self.assertEqual(velocities[-1], 0.)

    def test_initialization_from_feedback_reset_and_late_tick(self):
        limiter, matrix, feedback = self.fixture()
        feedback[13][0] = 42.
        self.assertEqual(limiter.apply(matrix, feedback, 0.)[13][0], 42.)
        previous = limiter.apply(matrix, feedback, .02)[13][0]
        later = limiter.apply(matrix, feedback, 10.)[13][0]
        self.assertLessEqual(abs(later-previous), 120*.05)
        matrix[13] = [0.]*5
        self.assertEqual(limiter.apply(matrix, feedback, 10.02)[13], [0.]*5)
        self.assertNotIn(14, limiter.states)
        matrix[13] = [-200., 0., 0., 800., 50.]
        feedback[13][0] = 9.
        self.assertEqual(limiter.apply(matrix, feedback, 10.04)[13][0], 9.)
        with self.assertRaises(ValueError):
            limiter.apply(matrix, feedback, 1.)
        feedback[13][0] = float('nan')
        with self.assertRaises(ValueError):
            limiter.apply(matrix, feedback, 10.06)


class StartupTrajectoryTests(unittest.TestCase):
    def fixture(self):
        config = json.loads(V3_MODEL.read_text())['teleop_motion_limits']['14']
        limiter = MotorTrajectoryLimiter({14: config})
        matrix = [[0., 0., 0., 0., 0.] for _ in range(28)]
        matrix[13] = [2., 0., 0., 800., 50.]
        feedback = [[0., 0., 0., 0.] for _ in range(28)]
        return limiter, matrix, feedback

    def test_waits_for_first_motion_then_follows_live_targets_without_rearming(self):
        limiter, matrix, feedback = self.fixture()
        for i in range(251):
            result = limiter.apply(matrix, feedback, i*.02)
        self.assertAlmostEqual(result[13][0], 2.)  # Small inputs still respond.
        self.assertEqual(limiter.last[14]['phase'], 'waiting')
        matrix[13][0] = -400.
        phases, soft_velocities = [], []
        for i in range(251, 401):
            result = limiter.apply(matrix, feedback, i*.02)
            phase = limiter.last[14]['phase']
            phases.append(phase)
            if phase == 'startup':
                soft_velocities.append(limiter.states[14][1])
            for mid in range(28):
                if mid != 13:
                    self.assertEqual(result[mid], matrix[mid])
        self.assertEqual(set(phases), {'startup', 'handoff', 'follow'})
        self.assertLessEqual(max(map(abs, soft_velocities)), 120.+1e-8)
        self.assertLessEqual(max(abs(np.diff(soft_velocities)/.02)), 480.+1e-8)
        self.assertEqual(result[13][0], -400.)
        matrix[13][0] = 180.
        self.assertEqual(limiter.apply(matrix, feedback, 8.02)[13][0], 180.)
        self.assertEqual(limiter.last[14]['phase'], 'follow')

    def test_handoff_does_not_dump_accumulated_error_at_a_time_boundary(self):
        limiter, matrix, feedback = self.fixture()
        matrix[13][0] = 400.
        positions, phases = [], []
        for i in range(2011):
            positions.append(limiter.apply(matrix, feedback, i*.001)[13][0])
            phases.append(limiter.last[14]['phase'])
        self.assertLess(max(abs(np.diff(positions))), 1.)
        self.assertEqual(phases[-1], 'follow')
        self.assertEqual(positions[-1], 400.)
        self.assertLess(abs(positions[2001]-positions[1999]), .001)

    def test_inactive_reset_rearms_and_delayed_tick_does_not_skip_startup(self):
        limiter, matrix, feedback = self.fixture()
        matrix[13][0] = -400.
        feedback[13][0] = 42.
        self.assertEqual(limiter.apply(matrix, feedback, 0.)[13][0], 42.)
        later = limiter.apply(matrix, feedback, 20.)[13][0]
        self.assertLessEqual(abs(later-42.), 120.*.05)
        self.assertEqual(limiter.last[14]['phase'], 'startup')
        self.assertAlmostEqual(limiter.last[14]['startup_elapsed_sec'], .05)
        matrix[13] = [0.]*5
        self.assertEqual(limiter.apply(matrix, feedback, 20.02)[13], [0.]*5)
        self.assertNotIn(14, limiter.startup_states)
        self.assertNotIn(14, limiter.last)
        matrix[13] = [-400., 0., 0., 800., 50.]
        feedback[13][0] = 9.
        self.assertEqual(limiter.apply(matrix, feedback, 20.04)[13][0], 9.)
        limiter.reset()
        self.assertFalse(limiter.states or limiter.startup_states or limiter.last)

    def test_invalid_startup_profiles_are_rejected(self):
        config = json.loads(V3_MODEL.read_text())['teleop_motion_limits']['14']
        for patch in ({'mode':'unknown'}, {'startup_duration_sec':0.},
                      {'handoff_duration_sec':float('inf')}, {'activation_threshold_deg':True},
                      {'activation_threshold_deg':-1.}, {'max_velocity_deg_sec':0.},
                      {'max_velocity_deg_sec':True}, {'max_acceleration_deg_sec2':float('nan')}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                MotorTrajectoryLimiter({14: dict(config, **patch)})


class FakeSDKThumbCoordinationTests(unittest.TestCase):
    def test_v3_restores_full_sides_and_hands_over_to_direct_following(self):
        self.exercise('thumb')

    def test_five_v3_controls_24_motors_and_preserves_four_finger_outputs(self):
        self.exercise('five')

    def exercise(self, finger):
        config = json.loads(V3_MODEL.read_text())
        long_frames = 180
        try:
            from devices_pkg.msg import YHandMsg
        except ImportError:
            self.skipTest('source install/setup.bash for isolated DDS integration')
        with tempfile.TemporaryDirectory(prefix='thumb-coordination-sdk-') as directory:
            root = Path(directory)
            model = root/'candidate.json'
            model.write_text(json.dumps(config))
            helper = PACKAGE/'tests/test_left_little_q_to_motor_test.py'
            driver = root/'fake_sdk'
            driver.write_text(f'#!{sys.executable}\nimport runpy\n'
                              f'runpy.run_path({str(helper)!r},run_name="fake_little_sdk")\n')
            driver.chmod(0o755)
            topic = f'/thumb_coordination_{os.getpid()}'
            env = dict(os.environ, ROS_LOCALHOST_ONLY='1', ROS_DOMAIN_ID=str(100+os.getpid()%100),
                       ROS_LOG_DIR=str(root/'roslog'), PYTHONDONTWRITEBYTECODE='1',
                       LITTLE_TEST_EVENTS=str(root/'events.jsonl'), LITTLE_TEST_FINGER=finger,
                       LITTLE_TEST_MODE='manus_long_run', LITTLE_TEST_GLOVE_TOPIC=topic+'/glove',
                       LITTLE_TEST_MANUS_LONG_FRAMES=str(long_frames),
                       LITTLE_TEST_RAW_TOPIC=topic+'/raw', LITTLE_TEST_TARGET_TOPIC=topic+'/target')
            command = [sys.executable, str(PACKAGE/'scripts/q_2_m linear mapping/left_little_q_to_motor_test.py'),
                       '--finger', finger, '--model', 'linear',
                       '--input-mode', 'manus', '--driver-executable', str(driver),
                       '--cmd-topic', topic+'/cmd', '--feedback-topic', topic+'/feedback',
                       '--manus-topic', topic+'/glove', '--manus-raw-topic', topic+'/raw',
                       '--manus-zero-seconds', '.1', '--ready-timeout', '5',
                       '--output-dir', str(root/'output')]
            command += ['--thumb-version', 'v3'] if finger == 'five' else ['--model-file', str(model)]
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, start_new_session=True)
            try:
                stdout, stderr = process.communicate(timeout=15)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.communicate(timeout=8)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
            self.assertEqual(process.returncode, 0, stdout+'\n'+stderr)
            metadata = json.loads((root/'output/metadata.json').read_text())
            self.assertEqual(metadata['thumb_version'], 'v3')
            self.assertEqual(metadata['finger'], finger)
            self.assertEqual(metadata['position_target_mode'], 'model_with_startup_smoothing')
            self.assertEqual(metadata['motor_trajectory_limits'], config['teleop_motion_limits'])
            self.assertEqual(metadata['thumb_linear_coordination'], config['thumb_linear_coordination'])
            self.assertEqual(json.loads((root/'output/models/thumb.json').read_text()), config)
            import csv
            with (root/'output/samples.csv').open() as stream:
                records = [r for r in csv.DictReader(stream) if r['state'] == 'position']
            phases = [r['m14_trajectory_phase'] for r in records]
            self.assertTrue({'startup', 'handoff', 'follow'}.issubset(phases))
            followed = [r for r in records if r['m14_trajectory_phase'] == 'follow']
            self.assertGreater(len(followed), 10)
            for r in followed:
                self.assertEqual(float(r['m14_target']), float(r['m14_command']))
            for r in records:
                for mid in (8, 27):
                    self.assertEqual(float(r[f'm{mid}_command']), float(r[f'm{mid}_uncoordinated_prediction']))
                self.assertEqual(float(r['thumb_side_secondary_weight']), 1.)
            if finger == 'five':
                baseline = control.create_mapper('five')
                self.assertEqual(set(metadata['model_files']), set(baseline.fingers))
                self.assertEqual(len(metadata['motor_ids']), 24)
                self.assertEqual(set(p.stem for p in (root/'output/models').glob('*.json')),
                                 set(baseline.fingers))
                for r in records:
                    q = [float(r[f'target_relative_q{j}']) for j in range(20)]
                    expected = control.prepare_target(baseline, q)['positions']
                    for mid in baseline.motor_ids:
                        if mid not in (4, 6, 7, 8, 14, 27):
                            self.assertAlmostEqual(float(r[f'm{mid}_command']), expected[mid], places=8)
            self.assertTrue(any(abs(float(r['m14_target'])-float(r['m14_command'])) > 5 for r in records))
            self.assertTrue(all(r['m8_uncoordinated_prediction'] and r['m27_uncoordinated_prediction'] for r in records))
            events = [json.loads(r) for r in (root/'events.jsonl').read_text().splitlines()]
            commands = [e for e in events if e['event'] == 'command']
            self.assertEqual(events[-1]['event'], 'driver_stopped')
            for mid in control.CONTROL_FINGER_SPECS[finger]['motor_inputs']:
                self.assertEqual(commands[-1]['matrix'][mid-1], [0]*5)
            defaults = control.load_defaults(PACKAGE/'scripts/muscle_adjustment_defaults.py')
            unselected = set(range(1, 29))-set(metadata['motor_ids'])
            for entry in commands:
                for mid in unselected:
                    row = entry['matrix'][mid-1]
                    self.assertEqual(row[:2]+row[3:], [0.]*4)
                    self.assertAlmostEqual(row[2], defaults['tor'][mid-1], places=6)


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
        for option in ['linear', 'mlp']:
            selected = control.create_mapper('five',model=option)
            expected = {}
            for i, finger in enumerate(['thumb','index','middle','ring','little']):
                expected.update(control.create_mapper(finger,model=option).predict(q[4*i:4*i+4], model=option))
            self.assertEqual(selected.predict(q, model=option), expected)
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
            command = [sys.executable, str(PACKAGE/'scripts/q_2_m linear mapping/left_little_q_to_motor_test.py'),
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
