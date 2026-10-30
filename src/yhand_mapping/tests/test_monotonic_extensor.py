"""Behavioral checks for monotonicity, serialized constraints and frozen outputs."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'scripts/q_2_m MLP mapping'))
from monotonic_extensor_training import train_extensor
from fit_monotonic_extensor import replace_extensor, replace_motor_head
from mlp_q_to_motor import MLPFingerMapper, forward, motor_forward


class MonotonicExtensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.original = MLPFingerMapper(finger='little').config
        # Use identical input normalization to the source, as the real refit does.
        center = np.asarray(cls.original['network']['input_center'])
        scale = np.asarray(cls.original['network']['input_scale'])
        # Symmetric samples have the required exact per-axis means/stds.
        cls.q = center+scale*np.array([[1,1,1,1],[-1,1,-1,1],[1,-1,-1,1],[-1,-1,1,1],
                                       [1,1,1,-1],[-1,1,-1,-1],[1,-1,-1,-1],[-1,-1,1,-1]])
        y = 40+cls.q@np.array([.2, 2., 1.5, 2.2])
        cls.head, _ = train_extensor(cls.q, y, epochs=120, width=8)
        cls.config = replace_extensor(cls.original, cls.head)

    def test_monotonicity_in_and_well_outside_training_range(self):
        rng = np.random.default_rng(7)
        q = rng.uniform([-40,-60,-60,-60], [40,180,180,180], size=(1200,4))
        for axis in (1, 2, 3):
            more = q.copy(); more[:,axis] += rng.uniform(.1, 20, len(q))
            self.assertGreaterEqual(float((motor_forward(self.config, more)[:,0]-motor_forward(self.config, q)[:,0]).min()), -1e-10)
        # Batching and scalar evaluation must use the same mixed activations.
        np.testing.assert_allclose(motor_forward(self.config,q[:5]),
                                   [motor_forward(self.config,row) for row in q[:5]],atol=1e-12)

    def test_other_motors_side_policy_coordinates_and_zero_are_preserved(self):
        q = np.array([[0,0,0,0], [-8,50,113,72], [10,0,0,0], [-20,-30,-20,-40]])
        np.testing.assert_allclose(motor_forward(self.config,q)[:,1:],
                                   motor_forward(self.original,q)[:,1:],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(motor_forward(self.config,np.zeros(4)),np.zeros(4))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.json';path.write_text(json.dumps(self.config))
            mapper=MLPFingerMapper(path,finger='little')
            np.testing.assert_allclose(list(mapper.predict(q[1]).values()),
                list(mapper.predict(q[1]+mapper.q_reference,relative=False).values()),atol=1e-12)

    def test_constraint_tampering_and_schema_downgrade_are_rejected(self):
        changes = [lambda c:c.__setitem__('schema_version',4),
            lambda c:c['network']['branches'][0].__setitem__('monotonic_inputs',[2,3]),
            lambda c:c['network']['branches'][0]['layers'][0]['weight'][0].__setitem__(1,-.1),
            lambda c:c['network']['branches'][0]['layers'][1]['weight'][0].__setitem__(0,-.1),
            lambda c:c['network']['branches'][0]['layers'][-1]['weight'][0].__setitem__(0,-.1),
            lambda c:c['network']['branches'][0]['layers'][0].__setitem__('activation','tanh')]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.json'
            for change in changes:
                config=deepcopy(self.config);change(config);path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):MLPFingerMapper(path,finger='little')

    def test_mixed_softplus_is_finite_for_large_inputs(self):
        q=np.array([[0,1e4,1e4,1e4], [0,-1e4,-1e4,-1e4]])
        self.assertTrue(np.isfinite(forward(self.head,q)).all())


class MonotonicFlexorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.original=MLPFingerMapper(finger='little').config
        center=np.asarray(cls.original['network']['input_center'])
        scale=np.asarray(cls.original['network']['input_scale'])
        cls.q=center+scale*np.array([[1,1,1,1],[-1,1,-1,1],[1,-1,-1,1],[-1,-1,1,1],
                                     [1,1,1,-1],[-1,1,-1,-1],[1,-1,-1,-1],[-1,-1,1,-1]])
        cls.labels=-40-cls.q@np.array([.2,2.,1.5,2.2])
        cls.saved_labels=cls.labels.copy()
        cls.head,_=train_extensor(cls.q,cls.labels,epochs=120,width=8,direction=-1)
        cls.config=replace_motor_head(cls.original,cls.head,1)

    def test_flexor_is_nonincreasing_in_and_outside_training_range(self):
        rng=np.random.default_rng(8)
        q=rng.uniform([-40,-60,-60,-60],[40,180,180,180],size=(1200,4))
        for axis in (1,2,3):
            more=q.copy();more[:,axis]+=rng.uniform(.1,20,len(q))
            self.assertLessEqual(float((motor_forward(self.config,more)[:,1]-motor_forward(self.config,q)[:,1]).max()),1e-10)
        np.testing.assert_array_equal(self.labels,self.saved_labels)
        np.testing.assert_allclose(self.head['output_center'],[self.labels.mean()],atol=1e-12)
        np.testing.assert_allclose(self.head['output_scale'],[self.labels.std()],atol=1e-12)

    def test_m11_side_policy_and_zero_are_preserved(self):
        q=np.array([[0,0,0,0],[-6.245,36.252,103.546,58.482],[7,46,95,18]])
        np.testing.assert_allclose(motor_forward(self.config,q)[:,[0,2,3]],
            motor_forward(self.original,q)[:,[0,2,3]],atol=1e-12,rtol=0)
        self.assertEqual(self.config['side_bending_policy'],self.original['side_bending_policy'])
        np.testing.assert_array_equal(motor_forward(self.config,np.zeros(4)),np.zeros(4))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.json';path.write_text(json.dumps(self.config))
            mapper=MLPFingerMapper(path,finger='little')
            np.testing.assert_allclose(list(mapper.predict(q[1]).values()),
                list(mapper.predict(q[1]+mapper.q_reference,relative=False).values()),atol=1e-12)

    def test_wrong_direction_schema_and_direction_without_constraints_are_rejected(self):
        changes=[lambda c:c.__setitem__('schema_version',5),
            lambda c:c['network']['branches'][1].__setitem__('monotonic_direction',1),
            lambda c:c['network']['branches'][1].__setitem__('monotonic_direction',0),
            lambda c:c['network']['branches'][1].__setitem__('monotonic_direction',True),
            lambda c:c['network']['branches'][1].pop('monotonic_inputs'),
            lambda c:c['network']['branches'][1]['layers'][-1]['weight'][0].__setitem__(0,-.1)]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.json'
            for change in changes:
                c=deepcopy(self.config);change(c);path.write_text(json.dumps(c))
                with self.assertRaises(ValueError):MLPFingerMapper(path,finger='little')


if __name__=='__main__':unittest.main()
