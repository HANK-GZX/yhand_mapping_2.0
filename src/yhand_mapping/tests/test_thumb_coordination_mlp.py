"""Shared feature coupling, per-output monotonic guarantees and schema15 inference."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P/'scripts/q_2_m MLP mapping'))
from mlp_q_to_motor import MLPFingerMapper, forward, motor_forward
from thumb_coordination_training import train_coordination, replace_coordination, quadratic_reference_jacobian
from migrate_delta_inputs import main as migrate_main
from thumb_fold_models import load_thumb_fold


class ThumbCoordinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.q = np.random.default_rng(915).normal(size=(128, 4))
        # Isolated synthetic fixture; production labels remain real feedback.
        cls.coefficients = np.array([[.7,-.5,-.7],[-.2,.8,.2],[.2,-.3,-.1],[-.1,.1,.2]])
        cls.y = cls.q@cls.coefficients
        ref = quadratic_reference_jacobian(cls.q, cls.y, cls.q[::4])
        saved = cls.y.copy()
        cls.head, _ = train_coordination(cls.q, cls.y, probe_q=cls.q[::4], reference_jacobian=ref,
            width=12, head_widths=(4,8,4), epochs=300)
        np.testing.assert_array_equal(saved, cls.y)
        cls.source = deepcopy(MLPFingerMapper(finger='thumb').config)
        for k in ('input_center','input_scale'):cls.source['network'][k] = cls.head[k][:]
        for k in ('output_center','output_scale'):cls.source['network'][k][3:] = cls.head[k][:]
        cls.config = replace_coordination(cls.source, cls.head)

    def mapper(self, c):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'thumb.json';p.write_text(json.dumps(c))
            return MLPFingerMapper(p, finger='thumb')

    def test_output_directions_and_preserved_flex_ext_inside_and_outside(self):
        q = np.random.default_rng(916).uniform(-100,150,(1000,4))
        before, after = motor_forward(self.source,q),motor_forward(self.config,q)
        np.testing.assert_array_equal(after[:,:3],before[:,:3])
        more=q.copy();more[:,0]+=1;d=motor_forward(self.config,more)-after
        self.assertGreaterEqual(d[:,3].min(),-1e-9);self.assertLessEqual(d[:,5].max(),1e-9)
        more=q.copy();more[:,1]+=1
        self.assertGreaterEqual((motor_forward(self.config,more)-after)[:,4].min(),-1e-9)
        for axis in (2, 3):
            more = q.copy()
            more[:, axis] += 1
            delta = motor_forward(self.config, more)-after
            self.assertLessEqual(delta[:, 0].max(), 1e-9)
            self.assertGreaterEqual(delta[:, 1:3].min(), -1e-9)
        self.assertEqual(self.config['network']['branches'][0],self.source['network']['branches'][0])

    def test_common_feature_perturbation_changes_all_three_outputs(self):
        c = deepcopy(self.config);b=c['network']['branches'][1]
        base=forward(c['network'],[0,0,0,0]);index=b['feature_groups']['common'][0]
        b['shared_layers'][1]['bias'][index]+=.2
        change=forward(c['network'],[0,0,0,0])-base
        np.testing.assert_array_equal(change[:3],np.zeros(3))
        self.assertGreater(change[3],0);self.assertLess(change[4],0);self.assertLess(change[5],0)

    def test_new_format_zero_batch_and_absolute_input_roundtrip(self):
        m=self.mapper(self.config);q=np.array([10,-5,30,40.])
        np.testing.assert_array_equal(list(m.predict(np.zeros(4)).values()),np.zeros(6))
        np.testing.assert_allclose(list(m.predict(q).values()),motor_forward(self.config,q),atol=1e-10,rtol=0)
        np.testing.assert_allclose(list(m.predict(q).values()),list(m.predict(q+m.q_reference,relative=False).values()),atol=1e-10)
        self.assertTrue(np.isfinite(motor_forward(self.config,[1e4,-1e4,1e4,1e4])).all())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'left_thumb_q_to_motor_mlp.json';p.write_text(json.dumps(self.config));before=p.read_bytes()
            self.assertEqual(migrate_main(['--mlp-dir',d,'--backup-dir',str(Path(d)/'unused')]),0)
            self.assertEqual(before,p.read_bytes())

    def test_connected_path_tampering_is_rejected(self):
        common=self.config['network']['branches'][1]['feature_groups']['common'][0]
        edits=[lambda c:c.__setitem__('schema_version',14),
            lambda c:c['network']['branches'][1]['shared_layers'][0]['weight'][common].__setitem__(0,-.1),
            lambda c:c['network']['branches'][1]['shared_layers'][0]['weight'][common].__setitem__(1,.1),
            lambda c:c['network']['branches'][1]['heads'][1].__setitem__('motor_id',27),
            lambda c:c['network']['branches'][1]['heads'][0]['layers'][1]['weight'][0].__setitem__(0,-.1)]
        for edit in edits:
            c=deepcopy(self.config);edit(c)
            with self.assertRaises(ValueError):self.mapper(c)

    def test_training_fold_reference_derivative_and_bad_data(self):
        np.testing.assert_allclose(self.head['input_center'], self.q.mean(0), atol=1e-12)
        np.testing.assert_allclose(self.head['input_scale'], self.q.std(0), atol=1e-12)
        np.testing.assert_allclose(self.head['output_center'], self.y.mean(0), atol=1e-12)
        np.testing.assert_allclose(self.head['output_scale'], self.y.std(0), atol=1e-12)
        p=self.q[:5]+.25
        jac=quadratic_reference_jacobian(self.q,self.y,p)
        np.testing.assert_allclose(jac,np.broadcast_to(self.coefficients.T,jac.shape),atol=1e-10)
        with self.assertRaises(ValueError):train_coordination(self.q,self.y[:,:2],probe_q=p,reference_jacobian=jac,epochs=1,width=12)

    def test_legacy_schema14_zero_coordinates_and_directions_remain_readable(self):
        # Tiny test-only legacy heads cover the old runtime format without
        # retaining separate historical trainer suites or training-log fixtures.
        c = deepcopy(self.config)
        c['schema_version'] = 14
        def legacy_head(mids, indices, axis, flag):
            first = np.zeros((2, 4))
            first[:, axis] = 1
            b = dict(motor_ids=mids, output_indices=indices, input_weights=[1.]*4,
                monotonic_inputs=[axis], hidden_activation='mixed_softplus',
                layers=[dict(weight=first.tolist(), bias=[0.,0.], activation='mixed_softplus'),
                    dict(weight=np.ones((len(mids),2)).tolist(), bias=[0.]*len(mids), activation='identity')],
                **{flag: True})
            if len(mids) == 1:b['monotonic_direction'] = 1
            else:b['output_directions'] = [1,-1]
            return b
        c['network']['branches'] = [c['network']['branches'][0],
            legacy_head([8,27], [3,5], 0, 'shared_thumb_side'),
            legacy_head([14], [4], 1, 'thumb_monotonic_opposition')]
        mapper = self.mapper(c)
        q = np.array([10.,-5.,30.,40.])
        np.testing.assert_array_equal(list(mapper.predict(np.zeros(4)).values()), np.zeros(6))
        np.testing.assert_allclose(list(mapper.predict(q).values()), motor_forward(c,q), atol=1e-10, rtol=0)
        np.testing.assert_allclose(list(mapper.predict(q+mapper.q_reference,relative=False).values()),
                                   list(mapper.predict(q).values()), atol=1e-10, rtol=0)
        base = motor_forward(c,q)
        more = q.copy();more[0] += 1
        delta = motor_forward(c,more)-base
        self.assertGreater(delta[3],0);self.assertLess(delta[5],0)
        more = q.copy();more[1] += 1
        self.assertGreater((motor_forward(c,more)-base)[4],0)

    def test_fold_reload_uses_declared_shared_head_and_flex_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);c=deepcopy(self.source)
            c['thumb_flex_ext_training']['reference_report']=str(root/'base')
            flex={k:deepcopy(c['network'][k])for k in('input_center','input_scale')}
            for k in('output_center','output_scale'):flex[k]=c['network'][k][:3]
            flex.update(architecture='role_weighted_branches',branches=[deepcopy(c['network']['branches'][0])])
            c['group_validation_reports']=dict(flex_ext=str(root/'flex'),coordination=str(root/'coordination'))
            for name,head in(('base',c['network']),('flex',flex),('coordination',self.head)):
                dest=root/name/'fold_models/fixture';dest.mkdir(parents=True)
                (dest/'0.json').write_text(json.dumps(head))
            loaded=load_thumb_fold(c,root/'unused','fixture','0');self.mapper(loaded)
            np.testing.assert_array_equal(forward(loaded['network'],self.q),forward(self.config['network'],self.q))


if __name__=='__main__':unittest.main()
