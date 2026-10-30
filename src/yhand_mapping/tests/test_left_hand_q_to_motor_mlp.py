"""Portable nonlinear inference, leakage-resistant splits and shared control routing."""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
SCRIPTS = PACKAGE/'scripts'
sys.path.insert(0, str(SCRIPTS/'q_2_m MLP mapping'))
sys.path.insert(0, str(SCRIPTS/'q_2_m linear mapping'))
import left_little_q_to_motor_test as control
from mlp_q_to_motor import (MLPFingerMapper, LEGACY_ALIGNMENT, SIDE_ALIGNMENT, forward, network_input,
                            motor_forward, side_bending_policy, limit_side_bending)
from migrate_delta_inputs import convert
import fit_left_hand_q_to_motor_mlp as fit
from role_weighted_training import role_input_weights, build_network, export_network, input_priority_penalty


class MLPTests(unittest.TestCase):
    def test_control_and_cli_accept_only_linear_and_mlp(self):
        script=SCRIPTS/'q_2_m linear mapping/left_little_q_to_motor_test.py'
        for option in ('linear','mlp'):
            mapper=control.create_mapper('five',model=option)
            self.assertEqual(len(mapper.predict([0]*20,model=option)),24)
        for option in ('quadratic','residual_mlp'):
            for finger in ('little','five'):
                with self.assertRaises(ValueError):control.create_mapper(finger,model=option)
            with self.assertRaises(ValueError):control.prepare_target(control.create_mapper('little'),[0]*4,model=option)
            with self.assertRaises(ValueError):control.create_mapper('five').predict([0]*20,model=option)
            result=subprocess.run([sys.executable,str(script),'--model',option,'--dry-run','--q','0','0','0','0'],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,2)
            self.assertIn('invalid choice',result.stderr)
        spec=importlib.util.spec_from_file_location('model_options_launch',PACKAGE/'launch/left_hand_q_to_motor_launch.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from launch.actions import DeclareLaunchArgument
        option=next(a for a in module.generate_launch_description().entities if isinstance(a,DeclareLaunchArgument)and a.name=='model')
        self.assertEqual(set(option.choices),{'linear','mlp'})

    def test_side_winding_limit_preserves_bend_outputs_q0_control_and_payout(self):
        rng=np.random.default_rng(74)
        for finger in fit.FINGERS[1:]:
            current=MLPFingerMapper(finger=finger)
            c=json.loads(current.model_path.read_text())
            c.update(schema_version=c['schema_version'] if c['schema_version'] in (5,6,7,8,9,10,11) else 4,output_alignment=SIDE_ALIGNMENT,side_bending_policy=side_bending_policy(finger))
            delta=rng.uniform([-40,-50,-30,-40],[40,150,180,120],size=(100,4))
            anchor_q=np.zeros_like(delta);anchor_q[:,0]=delta[:,0]
            neutral=forward(c['network'],np.zeros(4))
            raw=forward(c['network'],delta)-neutral
            anchor=forward(c['network'],anchor_q)-neutral
            constrained=motor_forward(c,delta)
            np.testing.assert_array_equal(constrained[:,:-2],raw[:,:-2])
            self.assertTrue(np.all(constrained[:,-2:]>=anchor[:,-2:]-2-1e-12))
            self.assertTrue(np.all(constrained[:,-2:]>=raw[:,-2:]-1e-12))
            positive=raw[:,-2:]>=anchor[:,-2:]
            np.testing.assert_array_equal(constrained[:,-2:][positive],raw[:,-2:][positive])
            np.testing.assert_array_equal(motor_forward(c,anchor_q),anchor)
            np.testing.assert_array_equal(motor_forward(c,np.zeros(4)),np.zeros(len(c['motor_order'])))
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'model.json';path.write_text(json.dumps(c))
                mapper=MLPFingerMapper(path,finger=finger)
                np.testing.assert_allclose(list(mapper.predict(delta[0]).values()),constrained[0],atol=1e-10)
                np.testing.assert_allclose(list(mapper.predict(delta[0]+mapper.q_reference,relative=False).values()),constrained[0],atol=1e-10)

    def test_side_policy_is_smooth_and_bad_or_untagged_policies_are_rejected(self):
        mapper=MLPFingerMapper(finger='index')
        c=json.loads(mapper.model_path.read_text())
        c.update(schema_version=c['schema_version'] if c['schema_version'] in (5,6,7) else 4,output_alignment=SIDE_ALIGNMENT,side_bending_policy=side_bending_policy('index'))
        anchor=np.zeros(5)
        for value in [-1e6,-1e-6,0,1e-6,100]:
            position=np.array([4.,5.,6.,value,value])
            actual=limit_side_bending(c,position,anchor)
            np.testing.assert_array_equal(actual[:3],position[:3])
            self.assertGreaterEqual(actual[-1],-2)
            if value>=0:self.assertEqual(actual[-1],value)
            if abs(value)<1e-5:self.assertAlmostEqual(actual[-1],value,places=12)
        edits=[lambda x:x['side_bending_policy'].__setitem__('max_extra_winding_motor_deg',-1),
               lambda x:x['side_bending_policy'].__setitem__('max_extra_winding_motor_deg',float('nan')),
               lambda x:x['side_bending_policy'].__setitem__('motor_ids',[25,3]),
               lambda x:x.pop('side_bending_policy'),
               lambda x:x.__setitem__('schema_version',3)]
        for edit in edits:
            bad=json.loads(json.dumps(c));edit(bad)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'bad.json';path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):MLPFingerMapper(path,finger='index')

    def test_legacy_model_conversion_preserves_raw_feedback_and_control_predictions(self):
        current = MLPFingerMapper(finger='thumb')
        legacy = json.loads(current.model_path.read_text())
        # Coordinate migration must remain covered after the live thumb becomes
        # branched; use an explicit dense legacy fixture independent of weights.
        rng = np.random.default_rng(91)
        legacy.update(schema_version=2, network=dict(input_center=[1.,2.,3.,4.],input_scale=[2.,3.,4.,5.],
            output_center=[0.]*6,output_scale=[3.]*6,layers=[
                dict(weight=rng.normal(0,.1,(b,a)).tolist(),bias=rng.normal(0,.1,b).tolist(),activation=activation)
                for a,b,activation in[(4,8,'tanh'),(8,8,'tanh'),(8,6,'identity')]]))
        legacy.pop('input_priority',None)
        self.assertEqual(legacy['schema_version'], 2)
        reference = np.asarray(legacy['q_reference'])
        legacy.update(schema_version=1, input_coordinates='absolute_glove_angles', output_alignment=LEGACY_ALIGNMENT)
        legacy['network']['input_center'] = (np.asarray(legacy['network']['input_center'])+reference).tolist()
        updated = convert(legacy)
        delta = np.array([[0, 0, 0, 0], [2, 10, 30, 5], [-3, 20, 45, 12]])
        np.testing.assert_allclose(forward(legacy['network'], delta+reference), forward(updated['network'], delta), atol=1e-10)
        self.assertEqual(legacy['network']['layers'], updated['network']['layers'])
        self.assertEqual(legacy['network']['output_center'], updated['network']['output_center'])
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/name for name in ['legacy.json', 'delta.json']]
            for path, config in zip(paths, [legacy, updated]):
                path.write_text(json.dumps(config))
            old, new = [MLPFingerMapper(path, finger='thumb') for path in paths]
            for d in delta:
                np.testing.assert_allclose(list(old.predict(d).values()), list(new.predict(d).values()), atol=1e-10)
            bad = dict(updated, input_coordinates='absolute_glove_angles')
            paths[1].write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                MLPFingerMapper(paths[1], finger='thumb')

    def test_role_priorities_are_applied_after_standardization_and_survive_export(self):
        import torch
        gains = np.array([[1, .25, .25, .25], [.25, 1, 1, 1]])
        net = build_network(2, 4, gains)
        with torch.no_grad():
            for branch in net.branches:
                for layer in branch:
                    if isinstance(layer, torch.nn.Linear):
                        layer.weight.fill_(.1)
                        layer.bias.zero_()
        center, scale = np.array([10., -30., 8., 5.]), np.array([2., 7., 9., 11.])
        network = export_network(net, center, scale, np.zeros(2), np.ones(2))
        eps = 1e-4
        q = center+eps*np.eye(4)*scale
        response = forward(network, q)-forward(network, center)
        self.assertAlmostEqual(response[0, 0]/response[1, 0], 4, places=6)
        self.assertAlmostEqual(response[1, 1]/response[0, 1], 4, places=6)
        with torch.no_grad():
            np.testing.assert_allclose(forward(network,q), net(torch.tensor((q-center)/scale,dtype=torch.float32)).numpy(), atol=1e-10)
        before = float(input_priority_penalty(net).detach())
        with torch.no_grad():
            for i, branch in enumerate(net.branches):
                branch[0].weight[:, net.input_weights[i] < 1] *= 4
        self.assertAlmostEqual(float(input_priority_penalty(net).detach())/before, 16, places=5)
        for finger in fit.FINGERS[1:]:
            ids = list(control.FINGER_SPECS[finger]['motor_inputs'])
            priorities = role_input_weights(finger, ids)
            np.testing.assert_array_equal(priorities[-2:], [[1,.25,.25,.25]]*2)
            np.testing.assert_array_equal(priorities[:-2], [[.25,1,1,1]]*(len(ids)-2))

    def test_malformed_weighted_branch_outputs_and_weights_are_rejected(self):
        from mlp_q_to_motor import validate_network
        net = build_network(2, 4, [[1,.25,.25,.25],[.25,1,1,1]])
        valid = export_network(net,np.zeros(4),np.ones(4),np.zeros(2),np.ones(2))
        for mutate in [lambda n:n['branches'][0]['input_weights'].__setitem__(0,0),
                       lambda n:n['branches'][0]['input_weights'].__setitem__(0,float('nan')),
                       lambda n:n['branches'][1].__setitem__('output_indices',n['branches'][0]['output_indices']),
                       lambda n:n['branches'][0]['layers'][0]['weight'][0].pop()]:
            bad=json.loads(json.dumps(valid));mutate(bad)
            with self.assertRaises(ValueError):
                validate_network(bad,2)

    def test_training_entry_uses_delta_inputs_and_unchanged_feedback_for_each_finger(self):
        rng = np.random.default_rng(27)
        delta = rng.normal(size=(24, 4))*3+[2, 4, 6, 8]
        for finger in fit.FINGERS:
            reference = np.asarray(json.loads((PACKAGE/f'config/left_{finger}_q_to_motor_models.json').read_text())['q_reference'])
            absolute = delta+reference
            y = rng.normal(size=(len(delta), len(control.FINGER_SPECS[finger]['motor_inputs'])))*10+80
            original_labels = y.copy()
            meta = [dict(point_id=str(i+1), timeout='0', combo_id='test') for i in range(len(delta))]

            def validate(q, labels, combo, **kwargs):
                np.testing.assert_allclose(q, delta, atol=1e-13)
                self.assertIs(labels, y)
                np.testing.assert_array_equal(kwargs['input_weights'],role_input_weights(finger,list(control.FINGER_SPECS[finger]['motor_inputs'])))
                return {}, {}, 2, np.arange(len(q)), np.arange(len(q)) % 5

            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                args = SimpleNamespace(data_root=folder, refresh_cache=False, epochs=2, width=32,
                                       seed=27, torch_version='test', output_dir=folder/'models')
                with patch.object(fit, 'load_data', return_value=(absolute, y, meta, {})), \
                        patch.object(fit, 'validate', side_effect=validate):
                    c = fit.fit_finger(finger, args, folder/'report')
                mapper = MLPFingerMapper(args.output_dir/f'left_{finger}_q_to_motor_mlp.json', finger=finger)
                self.assertEqual(c['input_coordinates'], 'delta_glove_angles')
                np.testing.assert_allclose(mapper.network['input_center'], delta.mean(0), atol=1e-13)
                np.testing.assert_array_equal(mapper.network['output_center'], y.mean(0))
                np.testing.assert_array_equal(y, original_labels)
                saved = np.load(folder/'report'/finger/'validation_predictions.npz')
                np.testing.assert_allclose(saved['q_delta'], delta, atol=1e-13)
                np.testing.assert_array_equal(saved['q_absolute'], absolute)
                np.testing.assert_array_equal(saved['motor_feedback'], original_labels)
                self.assertEqual(set(mapper.predict([0]*4).values()), {0})

    def test_held_out_predictions_match_serialized_models_and_zero_alignment(self):
        for finger in fit.FINGERS:
            mapper = control.create_mapper(finger, model='mlp')
            self.assertEqual(list(mapper.predict([0]*4).values()), [0]*len(mapper.motor_ids))
            q = np.asarray(mapper.q_reference)+[1, 2, 3, 4]
            expected = motor_forward(mapper.config, [1,2,3,4])
            actual = list(mapper.predict([1, 2, 3, 4]).values())
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            np.testing.assert_allclose(actual, list(mapper.predict(q, relative=False).values()), atol=1e-12)
            self.assertEqual(mapper.motor_inputs, dict.fromkeys(mapper.motor_ids, [0, 1, 2, 3]))
            self.assertEqual(set(mapper.effective_degrees().values()), {None})

    def test_wrong_finger_malformed_network_nonfinite_and_invalid_modes_fail(self):
        mapper = MLPFingerMapper(finger='thumb')
        with self.assertRaises(ValueError):
            MLPFingerMapper(mapper.model_path, finger='little')
        for q in [[0]*3, [0]*5, [math.nan]*4, [math.inf]*4]:
            with self.assertRaises(ValueError):
                mapper.predict(q)
        def first_layer(config):
            network=config['network']
            return (network['layers'] if 'layers' in network else network['branches'][0]['layers'])[0]
        for mutate in [lambda c: c['network']['input_scale'].__setitem__(0, 0),
                       lambda c: first_layer(c)['weight'][0].pop(),
                       lambda c: first_layer(c)['bias'].__setitem__(0, math.nan),
                       lambda c: c.__setitem__('output_alignment', 'absolute motor target')]:
            c = json.loads(mapper.model_path.read_text())
            mutate(c)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'bad.json'
                path.write_text(json.dumps(c))
                with self.assertRaises(ValueError):
                    MLPFingerMapper(path, finger='thumb')
        with self.assertRaises(ValueError):
            mapper.predict([0]*4, model='linear')
        with self.assertRaises(ValueError):
            control.create_mapper('five', model='linear', model_dir=Path('/tmp'))

    def test_all20_angles_route_to24_motors_and_preserve_existing_clipping_and_wrist(self):
        mapper = control.create_mapper('five', model='mlp')
        q = [2, 8, 12, 3]*5
        raw = mapper.predict(q)
        target = control.prepare_target(mapper, q, model='mlp')
        for i, finger in enumerate(fit.FINGERS):
            one = control.create_mapper(finger, model='mlp')
            for mid, value in one.predict(q[4*i:4*i+4]).items():
                self.assertEqual(raw[mid], value)
        self.assertEqual(len(raw), 24)
        self.assertEqual(target['q_relative'], q)
        for mid, indices in mapper.motor_inputs.items():
            i = next(i for i, one in enumerate(mapper.mappers) if mid in one.motor_ids)
            self.assertEqual(indices, list(range(4*i, 4*i+4)))
        defaults = control.load_defaults(SCRIPTS/'muscle_adjustment_defaults.py')
        # Large but finite input exercises the inherited ring/little clipping;
        # this path never publishes or starts a driver.
        positions = {mid: 1e4 for mid in mapper.motor_ids}
        clipped = control.clamp_motor_positions(positions, 'five')
        for mid, (_, hi) in control.FINGER_MOTOR_POSITION_LIMITS_DEG['five'].items():
            self.assertEqual(clipped[mid], hi)
        frame = control.position_matrix(defaults, target['positions'], control.default_feedforward('five'),
                                        'five', feedback_positions=dict.fromkeys(mapper.motor_ids, 0))
        for mid in [5, 12, 19, 26]:
            self.assertEqual(frame[mid-1], [0, 0, defaults['tor'][mid-1], 0, 0])
        stop = control.current_matrix(defaults, 'five')
        self.assertTrue(all(stop[mid-1] == [0]*5 for mid in mapper.motor_ids))

    def test_cli_mlp_dry_run_never_imports_torch_or_ros(self):
        # Invoke the actual CLI in a fresh interpreter; block training/ROS modules
        # to establish that portable runtime inference has no hidden dependency.
        code = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'rclpy', 'devices_pkg'):
            raise ImportError('offline runtime must not import '+fullname)
sys.meta_path.insert(0, Block())
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parent))
import runpy
script = sys.argv.pop(1)
runpy.run_path(script, run_name='__main__')
"""
        cmd = [sys.executable, '-c', code, str(SCRIPTS/'q_2_m linear mapping/left_little_q_to_motor_test.py'),
               '--finger', 'five', '--model', 'mlp', '--dry-run', '--q', *(['0']*20)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(len(output['motor_positions']), 24)
        self.assertEqual(set(output['motor_positions'].values()), {0})
        self.assertEqual(set(output['model_effective_models'].values()), {'mlp'})
        self.assertEqual(set(output['model_effective_degrees'].values()), {None})

    def test_grouping_and_ols_baseline_avoid_repeat_leakage(self):
        rng = np.random.default_rng(91)
        q = rng.normal(size=(100, 4))*10
        repeated = np.repeat(q, 3, axis=0)
        groups = fit.pose_groups(repeated)
        folds = fit.grouped_folds(groups)
        self.assertTrue(np.all(folds.reshape(-1, 3) == folds.reshape(-1, 3)[:, :1]))
        for fold in np.unique(folds):
            self.assertFalse(set(groups[folds == fold]) & set(groups[folds != fold]))
        beta = rng.normal(size=(5, 6))
        y = np.column_stack([np.ones(len(q)), q])@beta
        np.testing.assert_allclose(fit.polynomial(q[:80], y[:80], q[80:], 1), y[80:], atol=1e-10)

    def test_launch_accepts_mlp_and_custom_model_directory(self):
        from launch import LaunchContext
        from launch.actions import ExecuteProcess
        from launch.utilities import perform_substitutions
        spec = importlib.util.spec_from_file_location('mlp_launch', PACKAGE/'launch/left_hand_q_to_motor_launch.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        context = LaunchContext()
        context.launch_configurations.update(finger='five', input_mode='2', relative='true', model='mlp',
            mlp_model_dir='/tmp/mlp custom models', start_driver_node='false', start_glove_node='false',
            start_angle_input_terminal='false', publish_hz='50', current='', manus_topic='/glove',
            manus_raw_topic='/raw', manus_zero_seconds='1', manus_timeout='.5')
        process = next(a for a in module._launch_setup(context) if isinstance(a, ExecuteProcess))
        command = [perform_substitutions(context, part) for part in process.cmd]
        self.assertEqual(command[command.index('--model')+1], 'mlp')
        self.assertEqual(command[command.index('--mlp-model-dir')+1], '/tmp/mlp custom models')
        self.assertIn('--no-start-driver', command)

    def test_actual_training_data_sources_counts_and_finger_targets(self):
        data_root = SCRIPTS/'motor_test_logs/data_collection'
        if not (data_root/'thumb_data/left_hand/thumb/point_log.csv').is_file():
            self.skipTest('Real collection recordings are local, untracked analysis inputs')
        counts = dict(thumb=8149, index=5606, middle=5889, ring=2170, little=2227)
        for finger, count in counts.items():
            q, m, meta, source = fit.load_data(data_root, finger)
            self.assertEqual(q.shape, (count, 4))
            self.assertEqual(m.shape, (count, len(control.FINGER_SPECS[finger]['motor_inputs'])))
            self.assertEqual(len(meta), count)
            self.assertTrue(all(r['finger'] == finger for r in meta))
            model = MLPFingerMapper(finger=finger)
            if finger == 'thumb' and 'excluded_collection_combos' in model.config:
                from data_collection_config import THUMB_EXCLUDED_COMBO_IDS
                import hashlib
                selected = [int(r['point_id']) for r in meta if r['combo_id'] not in THUMB_EXCLUDED_COMBO_IDS]
                self.assertEqual(model.config['raw_source_points'], count)
                self.assertEqual(model.config['training_points'], len(selected))
                self.assertEqual(model.config['excluded_collection_combos'], sorted(THUMB_EXCLUDED_COMBO_IDS))
                self.assertEqual(model.config['selected_point_ids_sha256'], hashlib.sha256(json.dumps(selected).encode()).hexdigest())
                for result in model.config['validation'].values():
                    self.assertEqual(result['n_test'], len(selected))
            else:
                self.assertEqual(model.config['training_points'], count)
            self.assertEqual(model.config['source'], source)


if __name__ == '__main__':
    unittest.main()
