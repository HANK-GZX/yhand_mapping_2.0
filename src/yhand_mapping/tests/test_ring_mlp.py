"""Ring MLP regression: shared pair, M10 readout, M17 contraction and M22 bending."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'scripts/q_2_m MLP mapping'))

from fit_monotonic_extensor import replace_motor_head
from fit_ring_joint_mlp import assemble
from joint_antagonistic_training import train_pair
from mlp_q_to_motor import MLPFingerMapper, forward, motor_forward
from ring_extensor_readout_training import features, pair_network, replace_readout, tune_readout
from ring_side_bend_training import fit_bending, replace_bending, side_network


class RingJointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.old=MLPFingerMapper(finger='ring').config
        n=cls.old['network']
        signs=np.array([[1,1,1,1],[-1,1,-1,1],[1,-1,-1,1],[-1,-1,1,1],
                        [1,1,1,-1],[-1,1,-1,-1],[1,-1,-1,-1],[-1,-1,1,-1]])
        cls.q=np.asarray(n['input_center'])+np.asarray(n['input_scale'])*signs
        y=np.column_stack([20+cls.q@np.array([.1,2,1.5,1]),-30-cls.q@np.array([.1,2.2,1.7,1.1])])
        saved=y.copy();cls.head,_=train_pair(cls.q,y,epochs=140,width=8)
        np.testing.assert_array_equal(y,saved)
        np.testing.assert_allclose(cls.head['output_center'],saved.mean(0),atol=1e-12)
        np.testing.assert_allclose(cls.head['output_scale'],saved.std(0),atol=1e-12)
        cls.config=assemble(cls.old,cls.head)

    def mapper(self,config):
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'model.json';path.write_text(json.dumps(config))
            return MLPFingerMapper(path,finger='ring')

    def test_exactly_one_trunk_and_one_two_neuron_output_layer(self):
        n=self.config['network'];self.assertEqual(len(n['branches']),3)
        pair=n['branches'][0]
        self.assertEqual(pair['output_indices'],[0,1])
        self.assertEqual([(len(l['weight'][0]),len(l['weight']))for l in pair['layers']],[(4,8),(8,8),(8,2)])
        self.assertNotIn('bending_layers',pair)
        self.assertTrue(pair['shared_antagonistic_pair'])
        # A shared hidden parameter affects both outputs in opposite directions.
        modified=deepcopy(self.config);modified['network']['branches'][0]['layers'][1]['bias'][0]+=.2
        q=np.array([0,20,70,20]);change=motor_forward(modified,q)-motor_forward(self.config,q)
        # Neutral alignment can cancel constant offsets; inspect raw output here.
        raw=forward(modified['network'],q)-forward(self.config['network'],q)
        self.assertGreater(raw[0],0);self.assertLess(raw[1],0)
        self.assertTrue(np.isfinite(change).all())

    def test_opposite_monotonic_directions_including_outside_data_range(self):
        q=np.random.default_rng(84).uniform([-40,-40,-30,-40],[40,120,160,120],(1000,4))
        for axis in(1,2,3):
            more=q.copy();more[:,axis]+=3
            change=motor_forward(self.config,more)-motor_forward(self.config,q)
            self.assertGreaterEqual(float(change[:,0].min()),-1e-10)
            self.assertLessEqual(float(change[:,1].max()),1e-10)

    def test_sides_neutral_reference_and_portable_batch_are_preserved(self):
        q=np.random.default_rng(83).uniform([-40,-40,-30,-40],[40,120,160,120],(500,4))
        np.testing.assert_allclose(motor_forward(self.config,q)[:,2:],motor_forward(self.old,q)[:,2:],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(motor_forward(self.config,np.zeros(4)),np.zeros(4))
        self.assertEqual(self.config['q_reference'],self.old['q_reference'])
        self.assertEqual(self.config['side_bending_policy'],self.old['side_bending_policy'])
        mapper=self.mapper(self.config)
        np.testing.assert_allclose(motor_forward(mapper.config,q[:4]),
            [list(mapper.predict(x).values())for x in q[:4]],atol=1e-12,rtol=0)
        np.testing.assert_allclose(list(mapper.predict(q[0]).values()),
            list(mapper.predict(q[0]+mapper.q_reference,relative=False).values()),atol=1e-12)

    def test_schema_directions_weights_and_single_output_refits_are_rejected(self):
        edits=[lambda c:c.__setitem__('schema_version',7),
            lambda c:c['network']['branches'][0].__setitem__('output_directions',[1,1]),
            lambda c:c['network']['branches'][0].__setitem__('output_directions',[True,-1]),
            lambda c:c['network']['branches'][0].pop('shared_antagonistic_pair'),
            lambda c:c['network']['branches'][0]['layers'][0]['weight'][0].__setitem__(1,-.1),
            lambda c:c['network']['branches'][0]['layers'][-1]['weight'][0].__setitem__(0,-.1),
            lambda c:c['network']['branches'][0]['layers'][-1]['weight'].pop()]
        for edit in edits:
            config=deepcopy(self.config);edit(config)
            with self.assertRaises(ValueError):self.mapper(config)
        for index in(0,1):
            with self.assertRaises(ValueError):replace_motor_head(self.config,self.head,index)


class RingExtensorReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.source=MLPFingerMapper(finger='ring').config
        cls.network=pair_network(cls.source)
        rng=np.random.default_rng(221)
        cls.q=rng.uniform([-8,0,0,0],[4,25,100,40],(96,4))
        # Synthetic unit-test labels only. Production always uses measured CSV feedback.
        y=forward(cls.network,cls.q)
        y[:,0]+=.2*(cls.q[:,1]-12.)
        before=y.copy()
        cls.head,_=tune_readout(cls.network,cls.q,y,epochs=120,coordination_penalty=0,boundary_weight=4)
        np.testing.assert_array_equal(y,before)
        cls.config=replace_readout(cls.source,cls.head)

    def test_only_m10_output_weights_and_bias_can_change(self):
        restored=deepcopy(self.head)
        for key in('weight','bias'):
            restored['branches'][0]['layers'][-1][key][0]=deepcopy(self.network['branches'][0]['layers'][-1][key][0])
        self.assertEqual(restored,self.network)
        self.assertNotEqual(self.head['branches'][0]['layers'][-1]['weight'][0],
                            self.network['branches'][0]['layers'][-1]['weight'][0])
        self.assertEqual(len(features(self.network,self.q)[0]),32)

    def test_m17_sides_and_normalization_are_exactly_preserved(self):
        q=np.random.default_rng(222).uniform([-40,-60,-30,-50],[40,120,170,130],(1000,4))
        np.testing.assert_array_equal(motor_forward(self.config,q)[:,1:],motor_forward(self.source,q)[:,1:])
        np.testing.assert_array_equal(motor_forward(self.config,np.zeros(4)),np.zeros(4))
        for key in('input_center','input_scale','output_center','output_scale'):
            self.assertEqual(self.config['network'][key],self.source['network'][key])
        self.assertEqual(self.config['q_reference'],self.source['q_reference'])
        for axis in(1,2,3):
            more=q.copy();more[:,axis]+=1
            self.assertGreaterEqual(float((motor_forward(self.config,more)[:,0]-motor_forward(self.config,q)[:,0]).min()),-1e-10)

    def test_shared_single_network_and_portable_predictions_are_preserved(self):
        pair=self.config['network']['branches'][0]
        self.assertEqual(pair['output_indices'],[0,1])
        self.assertTrue(pair['shared_antagonistic_pair'])
        self.assertEqual([(len(l['weight'][0]),len(l['weight']))for l in pair['layers']],[(4,32),(32,32),(32,2)])
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'model.json';path.write_text(json.dumps(self.config))
            mapper=MLPFingerMapper(path,finger='ring')
            np.testing.assert_allclose(list(mapper.predict(self.q[0]).values()),motor_forward(self.config,self.q[0]),atol=1e-12)

    def test_updates_to_any_frozen_parameter_are_rejected(self):
        edits=[lambda n:n['branches'][0]['layers'][0]['bias'].__setitem__(0,99.),
               lambda n:n['branches'][0]['layers'][-1]['weight'][1].__setitem__(0,.1),
               lambda n:n['output_scale'].__setitem__(0,1.)]
        for edit in edits:
            head=deepcopy(self.head);edit(head)
            with self.assertRaises(ValueError):replace_readout(self.source,head)

    def test_full_pair_refit_drops_obsolete_readout_claims(self):
        from fit_ring_joint_mlp import assemble
        source=deepcopy(self.config)
        source['extensor_readout_training']={'epochs':1500,'frozen_motor_ids':[17,22,21]}
        source['extensor_readout_validation']={'old':True}
        config=assemble(source,self.network)
        self.assertNotIn('extensor_readout_training',config)
        self.assertNotIn('extensor_readout_validation',config)
        self.assertEqual(config['previous_extensor_readout_metadata']['extensor_readout_training']['epochs'],1500)


class RingFlexorSharedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.old=MLPFingerMapper(finger='ring').config;n=cls.old['network']
        signs=np.array([[1,1,1,1],[-1,1,-1,1],[1,-1,-1,1],[-1,-1,1,1],
                        [1,1,1,-1],[-1,1,-1,-1],[1,-1,-1,-1],[-1,-1,1,-1]])
        q=np.array(n['input_center'])+np.array(n['input_scale'])*signs
        # Synthetic unit-test fixture; production training uses original motor feedback.
        y=np.column_stack([10+q@np.array([.1,2,1.5,1]),-20-q@np.array([.1,2.5,1.6,1])])
        saved=y.copy()
        cls.head,_=train_pair(q,y,width=8,epochs=120,convex_bend_inputs=(1,),flexor_convex_readout=True,coordination_penalty=4.)
        np.testing.assert_array_equal(y,saved)
        np.testing.assert_allclose(cls.head['output_center'],saved.mean(0),atol=1e-12)
        np.testing.assert_allclose(cls.head['output_scale'],saved.std(0),atol=1e-12)
        cls.config=assemble(cls.old,cls.head)

    def test_one_shared_network_and_structural_connection_masks(self):
        pair=self.config['network']['branches'][0]
        self.assertEqual(pair['output_indices'],[0,1]);self.assertTrue(pair['shared_antagonistic_pair'])
        self.assertEqual([(len(l['weight'][0]),len(l['weight']))for l in pair['layers']],[(4,8),(8,8),(8,2)])
        np.testing.assert_array_equal(np.array(pair['layers'][0]['weight'])[4:,1],np.zeros(4))
        np.testing.assert_array_equal(np.array(pair['layers'][-1]['weight'])[1,4:],np.zeros(4))
        self.assertNotIn('bending_layers',pair)

    def test_winding_slope_does_not_fade_in_q1(self):
        q=np.random.default_rng(141).uniform([-40,-50,-30,-50],[40,120,160,130],(1000,4))
        current=motor_forward(self.config,q)
        for axis in(1,2,3):
            more=q.copy();more[:,axis]+=1
            diff=motor_forward(self.config,more)-current
            self.assertGreaterEqual(float(diff[:,0].min()),-1e-10)
            self.assertLessEqual(float(diff[:,1].max()),1e-10)
        more=q.copy();less=q.copy();more[:,1]+=1;less[:,1]-=1
        second=motor_forward(self.config,more)[:,1]+motor_forward(self.config,less)[:,1]-2*current[:,1]
        self.assertLessEqual(float(second.max()),1e-9)

    def test_sides_reference_zero_and_scalar_batch_are_preserved(self):
        q=np.random.default_rng(142).uniform([-40,-50,-30,-50],[40,120,160,130],(500,4))
        np.testing.assert_array_equal(motor_forward(self.config,q)[:,2:],motor_forward(self.old,q)[:,2:])
        np.testing.assert_array_equal(motor_forward(self.config,np.zeros(4)),np.zeros(4))
        self.assertEqual(self.old['q_reference'],self.config['q_reference'])
        self.assertEqual(self.old['side_bending_policy'],self.config['side_bending_policy'])
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'model.json';path.write_text(json.dumps(self.config));mapper=MLPFingerMapper(path,finger='ring')
            np.testing.assert_allclose([list(mapper.predict(v).values())for v in q[:4]],motor_forward(self.config,q[:4]),atol=1e-12,rtol=0)
            np.testing.assert_allclose(list(mapper.predict(q[0]).values()),list(mapper.predict(q[0]+mapper.q_reference,relative=False).values()),atol=1e-12,rtol=0)

    def test_invalid_connection_masks_or_metadata_are_rejected(self):
        edits=[lambda b:b['layers'][0]['weight'][4].__setitem__(1,.1),
               lambda b:b['layers'][-1]['weight'][1].__setitem__(4,.1),
               lambda b:b.__setitem__('convex_bend_inputs',[0]),
               lambda b:b.__setitem__('convex_bend_inputs',[True]),
               lambda b:b.__setitem__('flexor_convex_readout',1),
               lambda b:b.pop('convex_bend_inputs')]
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'model.json'
            for edit in edits:
                c=deepcopy(self.config);edit(c['network']['branches'][0]);path.write_text(json.dumps(c))
                with self.assertRaises(ValueError):MLPFingerMapper(path,finger='ring')


class RingSideBendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.old=MLPFingerMapper(finger='ring').config
        cls.q=np.random.default_rng(31).uniform([-10,0,0,0],[4,25,100,40],(64,4))
        # Unit-test synthetic fixture only; production labels are measured CSV feedback.
        y=-1.2*cls.q[:,0]+.4*cls.q[:,1]-.1*cls.q[:,2]+2
        before=y.copy();cls.head=fit_bending(side_network(cls.old),cls.q,y,epochs=90,width=4)
        np.testing.assert_array_equal(y,before)
        cls.config=replace_bending(cls.old,cls.head)

    def test_only_m22_bending_parameters_change(self):
        before,after=self.old['network']['branches'],self.config['network']['branches']
        self.assertEqual(before[0],after[0]);self.assertEqual(before[2],after[2])
        self.assertEqual(before[1]['layers'],after[1]['layers'])
        for key in('input_center','input_scale','output_center','output_scale'):
            self.assertEqual(self.old['network'][key],self.config['network'][key])
        q=np.random.default_rng(32).uniform([-40,-40,-20,-40],[40,100,150,120],(500,4))
        np.testing.assert_array_equal(motor_forward(self.old,q)[:,[0,1,3]],motor_forward(self.config,q)[:,[0,1,3]])

    def test_q0_motion_and_bend_q1_direction(self):
        q=np.random.default_rng(33).uniform([-40,-40,-20,-40],[40,100,150,120],(500,4))
        base=motor_forward(self.config,q)
        more=q.copy();more[:,1]+=1
        self.assertGreaterEqual(float((motor_forward(self.config,more)[:,2]-base[:,2]).min()),-1e-10)
        more=q.copy();more[:,0]+=1
        self.assertLessEqual(float((motor_forward(self.config,more)[:,2]-base[:,2]).max()),1e-10)
        q[:,1:]=0
        np.testing.assert_allclose(motor_forward(self.old,q),motor_forward(self.config,q),atol=1e-12,rtol=0)
        np.testing.assert_array_equal(motor_forward(self.config,np.zeros(4)),np.zeros(4))

    def test_portable_scalar_batch_and_absolute_coordinates(self):
        with tempfile.TemporaryDirectory()as folder:
            p=Path(folder)/'model.json';p.write_text(json.dumps(self.config))
            mapper=MLPFingerMapper(p,finger='ring')
            np.testing.assert_allclose([list(mapper.predict(q).values())for q in self.q[:4]],motor_forward(self.config,self.q[:4]),atol=1e-12,rtol=0)
            np.testing.assert_allclose(list(mapper.predict(self.q[0]).values()),list(mapper.predict(self.q[0]+mapper.q_reference,relative=False).values()),atol=1e-12,rtol=0)

    def test_invalid_monotonic_metadata_and_weights_are_rejected(self):
        edits=[lambda b:b['bending_layers'][0]['weight'][0].__setitem__(0,.1),
               lambda b:b['bending_layers'][0]['weight'][0].__setitem__(1,-.1),
               lambda b:b['bending_layers'][1]['weight'][0].__setitem__(0,-.1),
               lambda b:b.__setitem__('bending_monotonic_inputs',[2]),
               lambda b:b.__setitem__('bending_monotonic_inputs',[True]),
               lambda b:b.__setitem__('bending_hidden_activation','tanh')]
        with tempfile.TemporaryDirectory()as folder:
            p=Path(folder)/'model.json'
            for edit in edits:
                c=deepcopy(self.config);edit(c['network']['branches'][1]);p.write_text(json.dumps(c))
                with self.assertRaises(ValueError):MLPFingerMapper(p,finger='ring')

    def test_frozen_q0_or_scale_changes_are_rejected(self):
        for edit in(lambda n:n['branches'][0]['layers'][0]['bias'].__setitem__(0,99),
                    lambda n:n['output_scale'].__setitem__(0,1)):
            head=deepcopy(self.head);edit(head)
            with self.assertRaises(ValueError):replace_bending(self.old,head)


if __name__ == '__main__':
    unittest.main()
