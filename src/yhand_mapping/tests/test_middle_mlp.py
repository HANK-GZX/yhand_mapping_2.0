"""Middle MLP regression: M2 deep flexion and M23/M16 bending contracts."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'scripts/q_2_m MLP mapping'))

from fit_middle_bend import assemble
from fit_monotonic_extensor import replace_motor_head
from middle_bend_training import train_bend
from mlp_q_to_motor import MLPFingerMapper, forward, motor_forward
from monotonic_extensor_training import train_extensor


class MiddleDeepFlexorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.previous=MLPFingerMapper(finger='middle').config
        n=cls.previous['network'];center=np.asarray(n['input_center']);scale=np.asarray(n['input_scale'])
        q=center+scale*np.array([[1,1,1,1],[-1,1,-1,1],[1,-1,-1,1],[-1,-1,1,1],
                                 [1,1,1,-1],[-1,1,-1,-1],[1,-1,-1,-1],[-1,-1,1,-1]])
        cls.head,_=train_extensor(q,-30-2*q[:,1]-1.8*q[:,2]-1.2*q[:,3],width=8,epochs=120,direction=-1)
        cls.config=replace_motor_head(cls.previous,cls.head,2)

    def test_only_m2_output_changes_and_zero_is_preserved(self):
        q=np.random.default_rng(14).uniform([-30,-40,-20,-30],[45,100,150,120],(1000,4))
        np.testing.assert_allclose(motor_forward(self.config,q)[:,[0,1,3,4]],motor_forward(self.previous,q)[:,[0,1,3,4]],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(motor_forward(self.config,np.zeros(4)),np.zeros(5))
        with self.assertRaises(ValueError):replace_motor_head(self.previous,self.head,1)

    def test_deep_flexor_has_correct_direction_and_portable_metadata(self):
        q=np.random.default_rng(15).uniform([-30,-40,-20,-30],[45,100,150,120],(500,4))
        for axis in (1,2,3):
            more=q.copy();more[:,axis]+=3
            self.assertLessEqual(float((motor_forward(self.config,more)[:,2]-motor_forward(self.config,q)[:,2]).max()),1e-10)
        with tempfile.TemporaryDirectory()as directory:
            p=Path(directory)/'model.json';p.write_text(json.dumps(self.config))
            mapper=MLPFingerMapper(p,finger='middle')
            self.assertEqual(mapper.motor_ids,(23,16,2,9,24))
            self.assertIn(mapper.config['schema_version'],(8,9,10))

    def test_wrong_five_motor_direction_and_version_are_rejected(self):
        changes=[lambda c:c.__setitem__('schema_version',6),
            lambda c:c['network']['branches'][2].__setitem__('monotonic_direction',1)]
        with tempfile.TemporaryDirectory()as directory:
            p=Path(directory)/'model.json'
            for change in changes:
                c=deepcopy(self.config);change(c);p.write_text(json.dumps(c))
                with self.assertRaises(ValueError):MLPFingerMapper(p,finger='middle')


class MiddleBendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.source = MLPFingerMapper(finger='middle').config
        n = cls.source['network']
        signs = np.array([[1,1,1,1],[-1,1,-1,1],[1,-1,-1,1],[-1,-1,1,1],
                          [1,1,1,-1],[-1,1,-1,-1],[1,-1,-1,-1],[-1,-1,1,-1]])
        cls.q = np.asarray(n['input_center'])+np.asarray(n['input_scale'])*signs
        heads = {}
        for mid, direction, activation, slopes in [
                (23, 1, 'mixed_softplus_tanh', [.2,2,1.5,1.]),
                (16, -1, 'mixed_softplus_tanh', [.2,-2,-1.5,.1])]:
            y = -20+cls.q@np.array(slopes)
            saved = y.copy()
            heads[mid], _ = train_bend(cls.q, y, epochs=120, width=8, direction=direction,
                                       activation=activation)
            np.testing.assert_array_equal(y, saved)
            np.testing.assert_allclose(heads[mid]['output_center'], [saved.mean()], atol=1e-12)
        cls.heads = heads
        cls.config = assemble(cls.source, heads)

    def load_config(self, config):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.json'
            path.write_text(json.dumps(config))
            return MLPFingerMapper(path, finger='middle')

    def test_direction_inside_and_outside_the_training_range(self):
        q = np.random.default_rng(109).uniform([-50,-70,-120,-120],[75,160,180,140],(1000,4))
        base = motor_forward(self.config, q)
        for axis in (1,2,3):
            more = q.copy()
            more[:,axis] += 5
            diff = motor_forward(self.config, more)-base
            self.assertGreaterEqual(float(diff[:,0].min()), -1e-10)
            if axis in (1,2):
                self.assertLessEqual(float(diff[:,1].max()), 1e-10)

    def test_q3_is_not_forced_to_the_deep_flexor_direction(self):
        config = deepcopy(self.config)
        # Negative effective q3 weights permit shallow-flexor payout with q3;
        # the same negative weights in q1/q2 must be rejected below.
        branch = config['network']['branches'][1]
        self.assertEqual(branch['monotonic_inputs'], [1,2])
        for row in branch['layers'][0]['weight']:
            row[3] = -abs(row[3])-.01
        mapper = self.load_config(config)
        q = np.array([0,70,100,40.])
        more = q.copy()
        more[3] += 20
        self.assertGreater(motor_forward(mapper.config, more)[1]-motor_forward(mapper.config,q)[1], 0)

    def test_m2_sides_reference_and_neutral_are_preserved(self):
        q = np.random.default_rng(110).uniform([-50,-70,-120,-120],[75,160,180,140],(500,4))
        np.testing.assert_allclose(motor_forward(self.config,q)[:,2:],
                                   motor_forward(self.source,q)[:,2:],atol=1e-12,rtol=0)
        self.assertEqual(self.config['side_bending_policy'],self.source['side_bending_policy'])
        self.assertEqual(self.config['q_reference'],self.source['q_reference'])
        np.testing.assert_array_equal(motor_forward(self.config,np.zeros(4)),np.zeros(5))

    def test_large_negative_q1_tails_converge_instead_of_unbounded_reverse_commands(self):
        q=np.array([[0,-1e5,0,0],[0,-1e6,0,0]])
        for mid in (23,16):
            result=forward(self.heads[mid],q)
            self.assertTrue(np.isfinite(result).all())
            np.testing.assert_allclose(result[0],result[1],atol=1e-10,rtol=0)

    def test_portable_roundtrip_and_batch_coordinates(self):
        mapper = self.load_config(self.config)
        q = np.array([[-20,76,101,90.],[0,0,0,0],[20,-40,-100,-110]])
        np.testing.assert_allclose(motor_forward(mapper.config,q),
            [list(mapper.predict(row).values()) for row in q],atol=1e-12,rtol=0)
        np.testing.assert_allclose(list(mapper.predict(q[0]).values()),
            list(mapper.predict(q[0]+mapper.q_reference,relative=False).values()),atol=1e-12,rtol=0)
        self.assertTrue(np.isfinite(forward(self.heads[23],[[0,1e4,1e4,1e4]])).all())

    def test_invalid_direction_activation_weights_and_schema_are_rejected(self):
        edits = [lambda c:c.__setitem__('schema_version',9),
            lambda c:c['network']['branches'][1].__setitem__('monotonic_direction',1),
            lambda c:c['network']['branches'][1]['layers'][0]['weight'][0].__setitem__(1,-.1),
            lambda c:c['network']['branches'][0]['layers'][0].__setitem__('activation','tanh'),
            lambda c:c['network']['branches'][0].__setitem__('hidden_activation','unknown'),
            lambda c:c['input_priority']['per_motor_weights']['23'].__setitem__(3,1.)]
        for edit in edits:
            config=deepcopy(self.config)
            edit(config)
            with self.assertRaises(ValueError):
                self.load_config(config)

    def test_existing_m2_refit_keeps_version10_and_the_new_heads(self):
        head,_=train_extensor(self.q,-30-self.q[:,1]-self.q[:,2]-self.q[:,3],
                              epochs=100,width=8,direction=-1)
        config=replace_motor_head(self.config,head,2)
        mapper=self.load_config(config)
        self.assertEqual(mapper.config['schema_version'],10)
        q=np.array([[-20,76,101,90.],[0,0,0,0]])
        np.testing.assert_allclose(motor_forward(config,q)[:,[0,1,3,4]],
                                   motor_forward(self.config,q)[:,[0,1,3,4]],atol=1e-12,rtol=0)


if __name__ == '__main__':
    unittest.main()
