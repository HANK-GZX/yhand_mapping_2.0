"""Side direction, additive independence, preserved bend motors and portability."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'scripts/q_2_m MLP mapping'))
from mlp_q_to_motor import MLPFingerMapper,motor_forward,SIDE_DIRECTIONS
from monotonic_side_training import train_side
from fit_monotonic_side import assemble_side_model
from monotonic_extensor_training import train_extensor
from fit_monotonic_extensor import replace_motor_head


class MonotonicSideTests(unittest.TestCase):
    finger='little'
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.original=MLPFingerMapper(finger=cls.finger).config
        n=cls.original['network'];center=np.asarray(n['input_center']);scale=np.asarray(n['input_scale'])
        q=center+scale*np.array([[1,1,1,1],[-1,1,-1,1],[1,-1,-1,1],[-1,-1,1,1],
                                 [1,1,1,-1],[-1,1,-1,-1],[1,-1,-1,-1],[-1,-1,1,-1]])
        if cls.original['schema_version']==5:
            flexor,_=train_extensor(q,-40-2*q[:,1]-1.5*q[:,2]-2.2*q[:,3],width=8,epochs=120,direction=-1)
            cls.original=replace_motor_head(cls.original,flexor,2 if cls.finger=='middle'else 1)
        side_ids=cls.original['motor_order'][-2:]
        directions=SIDE_DIRECTIONS[cls.finger]
        labels={side_ids[0]:-10+directions[0]*1.5*q[:,0]+.1*q[:,1],side_ids[1]:-15+directions[1]*3*q[:,0]-.1*q[:,2]}
        heads={}
        for mid,direction in zip(side_ids,directions):
            saved=labels[mid].copy()
            heads[mid],_=train_side(q,labels[mid],width=8,epochs=120,direction=direction)
            np.testing.assert_array_equal(saved,labels[mid])
            np.testing.assert_allclose(heads[mid]['output_center'],[saved.mean()],atol=1e-12)
        cls.config=assemble_side_model(cls.original,heads,{mid:dict(training={},validation={})for mid in heads})

    def test_q0_directions_hold_after_bending_policy(self):
        rng=np.random.default_rng(12)
        q=rng.uniform([-50,-60,-40,-60],[40,160,180,120],(1000,4))
        more=q.copy();more[:,0]+=rng.uniform(.1,20,len(q))
        change=motor_forward(self.config,more)-motor_forward(self.config,q)
        for index,direction in zip((-2,-1),SIDE_DIRECTIONS[self.finger]):
            self.assertGreaterEqual(float((direction*change[:,index]).min()),-1e-10)
        # A q0-only change must not change the bend correction itself.
        anchor=np.zeros_like(q);anchor[:,0]=q[:,0]
        more_anchor=np.zeros_like(q);more_anchor[:,0]=more[:,0]
        extra=motor_forward(self.config,q)[:,-2:]-motor_forward(self.config,anchor)[:,-2:]
        more_extra=motor_forward(self.config,more)[:,-2:]-motor_forward(self.config,more_anchor)[:,-2:]
        np.testing.assert_allclose(extra,more_extra,atol=1e-12,rtol=0)
        self.assertGreaterEqual(float(extra.min()),-2-1e-10)

    def test_extensor_flexor_zero_and_coordinates_preserved(self):
        q=np.random.default_rng(4).uniform([-40,-60,-40,-60],[40,160,180,120],(300,4))
        np.testing.assert_allclose(motor_forward(self.config,q)[:,:-2],motor_forward(self.original,q)[:,:-2],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(motor_forward(self.config,np.zeros(4)),np.zeros(len(self.original['motor_order'])))
        self.assertEqual(self.config['side_bending_policy'],self.original['side_bending_policy'])
        with tempfile.TemporaryDirectory()as directory:
            p=Path(directory)/'model.json';p.write_text(json.dumps(self.config))
            mapper=MLPFingerMapper(p,finger=self.finger)
            np.testing.assert_allclose(list(mapper.predict(q[0]).values()),list(mapper.predict(q[0]+mapper.q_reference,relative=False).values()),atol=1e-12)

    def test_leaking_inputs_wrong_signs_and_schema_are_rejected(self):
        edits=[lambda c:c.__setitem__('schema_version',6),
            lambda c:c['network']['branches'][-2]['layers'][0]['weight'][0].__setitem__(0,-.1),
            lambda c:c['network']['branches'][-2]['layers'][0]['weight'][0].__setitem__(1,.1),
            lambda c:c['network']['branches'][-1]['bending_layers'][0]['weight'][0].__setitem__(0,.1),
            lambda c:c['network']['branches'][-1].__setitem__('monotonic_direction',-SIDE_DIRECTIONS[self.finger][1]),
            lambda c:c['network']['branches'][-1].pop('side_decomposition')]
        with tempfile.TemporaryDirectory()as directory:
            p=Path(directory)/'model.json'
            for edit in edits:
                c=deepcopy(self.config);edit(c);p.write_text(json.dumps(c))
                with self.assertRaises(ValueError):MLPFingerMapper(p,finger=self.finger)


class RingMonotonicSideTests(MonotonicSideTests):
    finger='ring'

    def test_motor_metadata_matches_ring_instead_of_little(self):
        self.assertEqual(self.config['motor_order'],[10,17,22,21])
        self.assertEqual(self.config['side_structure']['motor_directions'],{'22':-1,'21':1})
        self.assertEqual(self.config['training']['updated_motor_ids'],[22,21])


class MiddleMonotonicSideTests(MonotonicSideTests):
    finger='middle'

    def test_five_motors_and_reversed_side_directions(self):
        self.assertEqual(self.config['schema_version'],10 if self.original['schema_version']==10 else 9)
        self.assertEqual(self.config['motor_order'],[23,16,2,9,24])
        self.assertEqual(self.config['side_structure']['motor_directions'],{'9':1,'24':-1})
        self.assertEqual(self.config['training']['updated_motor_ids'],[9,24])
        self.assertEqual(self.config['network']['branches'][1].get('monotonic_inputs'),
                         self.original['network']['branches'][1].get('monotonic_inputs'))



if __name__=='__main__':unittest.main()
