"""Observed-pair constraints, conditional hints and manual control wiring."""
import importlib.util
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'scripts'))
sys.path.insert(0, str(PACKAGE / 'scripts/q_2_m linear mapping'))
from joint_coupling import DEFAULT_REFERENCE, JointCouplingError, JointCouplingReference

SCRIPT = PACKAGE / 'scripts/q_2_m linear mapping/left_little_q_to_motor_test.py'
spec = importlib.util.spec_from_file_location('coupling_control', SCRIPT)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class JointCouplingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'reference.json'
        self.data = dict(schema_version=1, coordinates='delta_q_deg', tolerance_deg=[5, 10, 10, 10],
                         source={}, fingers={finger: dict(observed_delta_q_deg=[
                             [0, 0, 0, 0], [0, 0, 100, 80], [20, 50, 30, 60]])
                             for finger in ['thumb', 'index', 'middle', 'ring', 'little']})
        self.path.write_text(json.dumps(self.data))
        self.reference = JointCouplingReference(self.path)

    def test_rejects_crossed_pair_even_when_both_angles_have_individual_support(self):
        self.assertTrue(self.reference.check('index', [0, 0, 0, 0])['accepted'])
        self.assertTrue(self.reference.check('index', [0, 0, 100, 80])['accepted'])
        for q in ([0, 0, 0, 80], [0, 0, 100, 0], [0, 0, 50, 40]):
            self.assertFalse(self.reference.check('index', q)['accepted'])

    def test_only_pairs_with_matching_q0_q1_are_allowed(self):
        self.assertTrue(self.reference.check('index', [20, 50, 30, 60])['accepted'])
        self.assertFalse(self.reference.check('index', [0, 0, 30, 60])['accepted'])
        result = self.reference.check('index', [100, 100, 0, 0])
        self.assertEqual(result['reason'], 'q01_unobserved')

    def test_q3_hint_depends_on_q2_and_preserves_gaps(self):
        hints = self.reference.condition('index', [0, 0])
        self.assertEqual(hints['q2_intervals_deg'], [[-10, 10], [90, 110]])
        self.assertEqual(self.reference.condition('index', [0, 0], 0)['q3_intervals_deg'], [[-10, 10]])
        self.assertEqual(self.reference.condition('index', [0, 0], 100)['q3_intervals_deg'], [[70, 90]])
        self.assertEqual(self.reference.condition('index', [0, 0], 50)['q3_intervals_deg'], [])

    def test_five_finger_rejection_is_atomic_and_does_not_call_mapper(self):
        mapper = control.create_mapper('five')
        q = [0.] * 20
        q[6:8] = [0, 80]
        from unittest.mock import patch
        with patch.object(mapper, 'predict') as predict:
            with self.assertRaises(JointCouplingError) as raised:
                control.prepare_target(mapper, q, coupling=self.reference)
            predict.assert_not_called()
        self.assertEqual(len(raised.exception.results), 5)
        self.assertEqual([r['finger'] for r in raised.exception.results if not r['accepted']], ['index'])

    def test_absolute_and_relative_coordinates_match_without_changing_q(self):
        mapper = control.create_mapper('index')
        q = [20, 50, 30, 60]
        absolute = np.array(mapper.q_reference) + q
        a = control.prepare_target(mapper, q, coupling=self.reference)
        b = control.prepare_target(mapper, absolute, relative=False, coupling=self.reference)
        np.testing.assert_allclose(a['q_relative'], q)
        np.testing.assert_allclose(b['q_relative'], q)
        self.assertEqual(a['positions'], b['positions'])
        self.assertEqual(a['positions'], control.prepare_target(mapper, q)['positions'])

    def test_rejects_invalid_reference_and_nonfinite_queries(self):
        for q in ([0, 0, float('nan'), 0], [0, 0, 0, float('inf')], [0, 0]):
            with self.assertRaises(ValueError):
                self.reference.check('index', q)
        self.data['coordinates'] = 'absolute'
        self.path.write_text(json.dumps(self.data))
        with self.assertRaises(ValueError):
            JointCouplingReference(self.path)

    def test_deployed_reference_allows_neutral_and_rejects_original_example(self):
        reference = JointCouplingReference(DEFAULT_REFERENCE)
        source = reference.data['source']
        self.assertEqual(source['kind'], 'historical_and_collection_union')
        self.assertEqual(source['anchor']['log_name'], '20261102_161407_627992')
        self.assertTrue(set(source['held_out_logs']).isdisjoint(log['log'] for log in source['additional_logs']))
        self.assertGreaterEqual(reference.data['fingers']['middle']['excluded_angle_branch_samples'], 35)
        for finger, batch in source['collection_supplement']['fingers'].items():
            self.assertTrue(set(batch['training_point_ids']).isdisjoint(batch['held_out_point_ids']))
            self.assertEqual(batch['valid_points'],len(batch['training_point_ids'])+len(batch['held_out_point_ids']))
        for finger in reference.clouds:
            self.assertTrue(reference.check(finger, [0, 0, 0, 0])['accepted'], finger)
            self.assertFalse(reference.check(finger, [20, 90, 0, 90])['accepted'], finger)

    def test_controller_cli_rejects_before_ros_and_off_preserves_mapping(self):
        base = [sys.executable, str(SCRIPT), '--finger', 'index', '--dry-run']
        reject = subprocess.run(base + ['--manual-coupling', 'reject', '--q', '20', '90', '0', '90'],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(reject.returncode, 2)
        self.assertIn('缺少记录', reject.stderr)
        off = subprocess.run(base + ['--manual-coupling', 'off', '--q', '20', '90', '0', '90'],
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(off.returncode, 0, off.stderr)
        self.assertEqual(json.loads(off.stdout)['q_relative'], [20, 90, 0, 90])

    def test_manus_mode_does_not_load_manual_reference(self):
        from unittest.mock import patch
        with patch.object(control, 'run', return_value=0) as run:
            self.assertEqual(control.main(['--input-mode', 'manus', '--manual-coupling', 'reject',
                                          '--coupling-reference', '/nonexistent/manual_reference.json']), 0)
        self.assertIsNone(run.call_args.kwargs['coupling'])


class TerminalInputTests(unittest.TestCase):
    def test_conditional_queries_never_return_targets_and_invalid_pair_is_rejected(self):
        import joint_angle_terminal_input as terminal
        reference = JointCouplingReference(DEFAULT_REFERENCE)
        for text in ['0 0', '0 0 0']:
            target, hint = terminal.input_target(text, 'index', coupling=reference)
            self.assertIsNone(target)
            self.assertIn('仅查询', hint)
        with self.assertRaisesRegex(ValueError, '缺少记录'):
            terminal.input_target('20 90 0 90', 'index', coupling=reference)
        self.assertEqual(terminal.input_target('zero', 'index', coupling=reference)[0], [0] * 4)
        for text in ['0 0 nan 0', '0 0 inf 0', '0 0 0']:
            with self.assertRaises(ValueError):
                terminal.input_target(text, 'index')

    def test_absolute_terminal_input_is_deferred_to_controller_conversion(self):
        import joint_angle_terminal_input as terminal
        q = [3.627, -6.019, .028, -2.929]
        result, _ = terminal.input_target(' '.join(map(str, q)), 'index',
            coupling=JointCouplingReference(DEFAULT_REFERENCE), relative=False)
        self.assertEqual(result, q)


class HistoryBuilderTests(unittest.TestCase):
    def test_bracketed_sign_flip_is_removed_without_rejecting_all_negative_angles(self):
        from build_manual_joint_coupling import suspect_angle_branches
        q = np.array([[0,0,98,40],[60,20,-115,20],[61,21,-108,18],[0,0,90,41],
                      [0,0,-5,0],[0,0,-65,0],[0,0,-70,0],[0,0,-50,0]],dtype=float)
        mask, known, bracketed = suspect_angle_branches(q)
        self.assertEqual(mask.tolist(),[False,True,True,False,False,False,False,False])
        self.assertEqual((known,bracketed),(0,2))

    def test_augmentation_preserves_original_points_and_excludes_holdout_manual_and_bad_rows(self):
        from build_manual_joint_coupling import augment_history, digest, write_reference
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root/'logs'
            reference_q = [-5., -9., 7., -10.]

            def make_log(name, mode, records):
                path = logs/'little_q_to_motor_test'/name
                path.mkdir(parents=True)
                meta = dict(hand='left', finger='little', fingers=['little'], model='linear', input_mode=mode,
                            manus_zero_q=[11,12,13,14] if mode=='manus' else None,
                            q_reference=reference_q, finger_q_slices={'little':[0,4]})
                (path/'metadata.json').write_text(json.dumps(meta))
                rows = []
                for identity, q, age, mismatch in records:
                    row = dict(state='position', target_id=identity, elapsed_sec=float(identity),
                               feedback_age_sec=.01, manus_age_sec=age, manus_left_raw_age_sec=.01)
                    row.update({f'target_relative_q{i}':v for i,v in enumerate(q)})
                    row.update({f'target_q{i}':v+reference_q[i]+mismatch for i,v in enumerate(q)})
                    rows.append(row)
                with (path/'samples.csv').open('w') as stream:
                    writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                return path

            anchor = make_log('20260101_anchor','manus',[(1,[0,0,0,0],.01,0)])
            train = make_log('20260102_train','manus',[
                (1,[0,30,40,20],.01,0), (1,[0,30,40,20],.01,0),
                (2,[0,30,float('nan'),20],.01,0), (3,[0,0,160,160],2.,0),
                (4,[0,0,170,170],.01,1), (5,[0,125,-100,-100],.01,0)])
            held = make_log('20260103_held','manus',[(1,[0,0,200,200],.01,0)])
            make_log('20260104_manual','manual',[(1,[0,0,180,180],.01,0)])
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(dict(selected_logs=[{'log':str(held.relative_to(logs))}])))
            baseline = root/'baseline.json'
            original = [0.01,0.02,0.03,0.04]
            data = dict(schema_version=1, coordinates='delta_q_deg', tolerance_deg=[5,10,10,10],
                        source=dict(log_name=anchor.name, samples_sha256=digest(anchor/'samples.csv'),
                                    metadata_sha256=digest(anchor/'metadata.json'),fresh_unique_target_updates=1),
                        fingers={'little':dict(observed_delta_q_deg=[original], source_samples=1,
                                               excluded_angle_branch_samples=0, operator_zero_q_deg=[11,12,13,14])})
            write_reference(data,baseline)
            output = root/'candidate.json'
            augmented = augment_history(logs,manifest,baseline,output)
            ref = JointCouplingReference(output)
            np.testing.assert_array_equal(ref.clouds['little'][0], original)
            self.assertTrue(ref.check('little',[0,30,40,20])['accepted'])
            self.assertFalse(ref.check('little',[0,0,180,180])['accepted'])
            self.assertFalse(ref.check('little',[0,0,200,200])['accepted'])
            self.assertEqual(augmented['fingers']['little']['additional_representative_count'],1)
            self.assertEqual(augmented['fingers']['little']['excluded_angle_branch_samples'],1)
            self.assertEqual(len(augmented['source']['additional_logs']),1)
            self.assertEqual(augmented['source']['additional_logs'][0]['log'],str(train.relative_to(logs)))
            quality = augmented['source']['additional_logs'][0]['quality_counts']
            for key in ['nonfinite','stale_or_missing_age','coordinate_mismatch','duplicate_target_id']:
                self.assertEqual(quality[key],1)


class CollectionBuilderTests(unittest.TestCase):
    def test_collection_uses_mapping_reference_groups_duplicates_and_keeps_measured_timeouts(self):
        from build_manual_joint_coupling import collection_finger,COLLECTION_CACHES
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            folder=root/'index_data/index'
            folder.mkdir(parents=True)
            cache=root/'index_data'/COLLECTION_CACHES['index']
            cache.parent.mkdir(parents=True)
            ids=np.arange(1,13)
            reference=np.array([3.,4.,5.,6.])
            q=np.array([[0.,j*8.,j*9.,j*4.] for j in range(12)])
            q[1]=q[0]+.1
            glove=np.zeros((12,48))
            glove[:,12:16]=q+reference
            glove[10,12]=np.nan
            sample_count=np.full(12,30)
            sample_count[11]=0
            np.savez(cache,point_ids=ids,glove=glove,feedback=np.zeros((12,28,5)),sample_count=sample_count)
            rows=[dict(point_id=i,finger='index',reached='0',timeout='1',zero_calib_state='failed_after_3_attempts',
                       active_mode='x_only') for i in ids]
            log=folder/'point_log.csv'
            with log.open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (cache.parent/'point_log_snapshot.csv').write_bytes(log.read_bytes())
            with (folder.parent/'startup_first_frame_glove48.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=[f'q_{i}' for i in range(48)])
                writer.writeheader()
                writer.writerow({f'q_{i}':float(reference[i-12]+10) if 12<=i<16 else 0 for i in range(48)})
            batch=collection_finger(root,'index',reference,verify_raw=False)
            np.testing.assert_allclose(batch['delta'][:10],q[:10])
            np.testing.assert_allclose(batch['info']['startup_minus_mapping_reference_deg'],[10]*4)
            self.assertEqual(batch['info']['valid_points'],10)
            self.assertEqual(batch['info']['invalid_finite_points'],1)
            self.assertEqual(batch['info']['invalid_sample_count_points'],1)
            self.assertEqual(batch['held'][0],batch['held'][1])
            self.assertFalse(np.any(batch['held']&batch['train']))
            self.assertEqual(sum(batch['info']['training_timeout_flags'].values()),batch['info']['training_points'])
            # Snapshot mismatch must fail rather than silently accepting stale cached q.
            (cache.parent/'point_log_snapshot.csv').write_text(log.read_text().replace('x_only','xy',1))
            with self.assertRaisesRegex(ValueError,'快照已过期'):
                collection_finger(root,'index',reference,verify_raw=False)

    def test_collection_augmentation_never_adds_held_out_points_or_replaces_parent(self):
        from build_manual_joint_coupling import augment_collection,write_reference
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            baseline=root/'baseline.json'
            write_reference(dict(schema_version=1,coordinates='delta_q_deg',tolerance_deg=[5,10,10,10],
                source=dict(kind='historical_log_union'),fingers={'index':dict(q_reference_deg=[0]*4,
                    observed_delta_q_deg=[[.1,.2,.3,.4]])}),baseline)
            delta=np.array([[0,0,0,0],[0,50,60,30],[0,80,180,180]],dtype=float)
            fake=dict(delta=delta,train=np.array([True,True,False]),held=np.array([False,False,True]),
                      info=dict(training_point_ids=[1,2],held_out_point_ids=[3]))
            output=root/'candidate.json'
            with patch('build_manual_joint_coupling.collection_finger',return_value=fake):
                augment_collection(root,baseline,output)
            ref=JointCouplingReference(output)
            np.testing.assert_array_equal(ref.clouds['index'][0],[.1,.2,.3,.4])
            self.assertTrue(ref.check('index',delta[1])['accepted'])
            self.assertFalse(ref.check('index',delta[2])['accepted'])
            with self.assertRaisesRegex(ValueError,'重复入库'):
                augment_collection(root,output,root/'again.json')


if __name__ == '__main__':
    unittest.main()
