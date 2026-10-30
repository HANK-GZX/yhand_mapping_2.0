#!/usr/bin/env python3
"""Refit thumb flex/ext only, prioritizing q2/q3 and guaranteeing bend direction."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil
import numpy as np

from fit_left_hand_q_to_motor_mlp import SCRIPTS,sha,load_data,delta_angles,grouped_folds,metrics
from fit_thumb_spatial_mlp import workspace_keys,macro_error
from thumb_monotonic_training import train_thumb_flex_ext,replace_thumb_flex_ext
from mlp_q_to_motor import MODEL_DIR,MLPFingerMapper,forward,motor_forward


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model-dir',type=Path,default=MODEL_DIR)
    parser.add_argument('--reference-report',type=Path,default=SCRIPTS/'motor_test_logs/thumb_grouped_mlp/20261028_three_groups_warmstart')
    parser.add_argument('--report-dir',type=Path)
    parser.add_argument('--epochs',type=int,default=2200)
    parser.add_argument('--width',type=int,default=32)
    parser.add_argument('--secondary-input-weight',type=float,default=.25)
    args=parser.parse_args()
    if args.epochs<1 or not 4<=args.width<=256 or not 0<args.secondary_input_weight<=1:parser.error('训练参数范围不正确')
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    source=args.source_model_dir/'left_thumb_q_to_motor_mlp.json';old=MLPFingerMapper(source,finger='thumb').config
    ref=json.loads((args.reference_report/'summary.json').read_text())
    if sha(source)!=ref['new_sha256']:raise ValueError('来源与参考分组验证档案不匹配，请指定对应备份')
    a=np.load(args.reference_report/'validation_predictions.npz');absolute,y,meta,provenance=load_data(SCRIPTS/'motor_test_logs/data_collection','thumb');q=delta_angles(absolute,old['q_reference'])
    for key,value in [('q_delta',q),('motor_feedback',y),('point_ids',np.array([int(r['point_id'])for r in meta]))]:np.testing.assert_array_equal(a[key],value)
    assert provenance==old['source'];keep=a['eligible'];combo=a['combo']
    b=json.loads((SCRIPTS/'motor_test_logs/workspace_boundaries/left/boundary_models.json').read_text())['combos'];keys=workspace_keys(y,combo,b)
    out=args.report_dir or SCRIPTS/'motor_test_logs/thumb_monotonic_mlp'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir();(out/'previous_models').mkdir()
    for path in args.source_model_dir.glob('left_*_q_to_motor_mlp.json'):shutil.copyfile(path,out/'models'/path.name);shutil.copyfile(path,out/'previous_models'/path.name)
    def fit(mask,seed):return train_thumb_flex_ext(q[mask],y[mask,:3],epochs=args.epochs,width=args.width,seed=seed,secondary=args.secondary_input_weight)
    ev={};arrays={k:a[k]for k in('q_absolute','q_delta','q_reference','motor_ids','motor_feedback','point_ids','combo','eligible','pose_group','pose_fold','workspace_fold','time_group')}
    schemes=[('pose_group_5fold',a['pose_fold']),('workspace_block_3fold',a['workspace_fold']),('leave_one_combo_out',combo),('zero_interval_3fold',grouped_folds(a['time_group'],count=3,seed=20261028))]
    for scheme,labels in schemes:
        before=a[scheme];pred=before.copy();details=[];folder=out/'fold_models'/scheme;folder.mkdir(parents=True)
        for i,label in enumerate(np.unique(labels)):
            test=labels==label;train=(~test)&keep;head=fit(train,20261029+i)
            base=json.loads((args.reference_report/'fold_models'/scheme/f'{label}.json').read_text());np.testing.assert_allclose(forward(base,q[test]),before[test],atol=1e-9,rtol=0)
            fold=deepcopy(old);fold['network']=base;replacement=replace_thumb_flex_ext(fold,head)
            pred[test,:3]=forward(head,q[test]);np.testing.assert_array_equal(forward(replacement['network'],q[test])[:,3:],before[test,3:])
            (folder/f'{label}.json').write_text(json.dumps(head,indent=2)+'\n');mask=test&keep
            details.append(dict(label=str(label),train=int(train.sum()),test=int(mask.sum()),epochs=args.epochs))
            if mask.any():print(scheme,label,'flex/ext MAE',abs(pred[mask,:3]-y[mask,:3]).mean(0).round(3).tolist(),flush=True)
        value=dict(mlp=metrics(y[keep],pred[keep]),previous_mlp=metrics(y[keep],before[keep]),n_test=int(keep.sum()),folds=details,
            macro_voxel_mae_motor_deg=macro_error(y[keep],pred[keep],combo[keep],keys[keep]).tolist(),
            previous_macro_voxel_mae_motor_deg=macro_error(y[keep],before[keep],combo[keep],keys[keep]).tolist(),
            all_recorded_diagnostic_mlp=metrics(y,pred),all_recorded_diagnostic_previous_mlp=metrics(y,before),
            scope='Same original 7278-point policy domain; only M4/M6/M7 recomputed. Preserved outputs inherit their unchanged fold predictions. Not independent final hardware testing.')
        ev[scheme]=value;arrays[scheme]=pred;arrays['previous_'+scheme]=before
    head=fit(keep,20261029);c=replace_thumb_flex_ext(old,head);c['validation']=ev
    c['previous_three_group_validation']=deepcopy(old['validation'])
    c['thumb_flex_ext_training']=dict(seed=20261029,epochs=args.epochs,width=args.width,optimizer='AdamW',lr=.01,weight_decay=0,
        input_weights=[args.secondary_input_weight,args.secondary_input_weight,1.,1.],priority_penalty=.001,
        activation='mixed_softplus_tanh',positive_weight_parameterization='softplus(raw_weight)',
        monotonic_inputs=[2,3],output_directions=[-1,1,1],trainable_motor_ids=[4,6,7],
        shared_hidden_layers=2,source=provenance,reference_report=str(args.reference_report),torch_version=str(torch.__version__),
        no_pseudo_labels=True,no_added_samples=True,no_linear_baseline=True,no_runtime_output_clipping=True,
        selection='Fixed epoch candidate; group metrics used for review, not an independent final test')
    groups=deepcopy(old['training']['groups']);groups[0].update(architecture=[4,args.width,args.width,3],activation='mixed_softplus_tanh',monotonic_inputs=[2,3],input_weights=c['thumb_flex_ext_training']['input_weights'])
    c['training']=dict(architecture_type='thumb_three_groups_monotonic_flex_ext',groups=groups,details='thumb_flex_ext_training; other two groups copied unchanged')
    c['preserved_outputs']=dict(motor_ids=[8,14,27],source_model_sha256=sha(source),source_training=old['training'],note='Networks, normalization, reference and output policy copied unchanged')
    c['fit_limitation']='q2/q3 signed monotonicity is structural at fixed other angles, not a minimum slope, amplitude, physical tension or human/robot alignment guarantee. q0/q1 remain weak unconstrained inputs.'
    path=out/'models'/source.name;path.write_text(json.dumps(c,ensure_ascii=False,indent=2,allow_nan=False)+'\n');MLPFingerMapper(path,finger='thumb')
    probe=np.random.default_rng(29).uniform([-50,-70,-70,-70],[70,90,150,150],(5000,4));base=motor_forward(c,probe)
    np.testing.assert_array_equal(base[:,3:],motor_forward(old,probe)[:,3:]);np.testing.assert_array_equal(motor_forward(c,np.zeros(4)),np.zeros(6))
    changes=[]
    for axis in(2,3):
        more=probe.copy();more[:,axis]+=1;diff=motor_forward(c,more)-base
        assert diff[:,0].max()<=1e-9 and diff[:,1:3].min()>=-1e-9;changes.append(dict(axis=axis,M4_max=float(diff[:,0].max()),M6_M7_min=float(diff[:,1:3].min())))
    summary=dict(previous_sha256=sha(source),new_sha256=sha(path),training_points=int(keep.sum()),source=provenance,training=c['thumb_flex_ext_training'],validation=ev,
        previous_neutral_raw=old['raw_prediction_at_reference_motor_deg'],new_neutral_raw=c['raw_prediction_at_reference_motor_deg'],
        monotonic_checks=changes,preserved_output_max_difference=0.,hardware_started=False,active_model_changed=False)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');np.savez_compressed(out/'validation_predictions.npz',**arrays)
    shutil.copyfile(args.reference_report/'point_log_snapshot.json',out/'point_log_snapshot.json')
    print('Staged',path,flush=True);print(json.dumps({name:dict(old=v['previous_mlp']['mae_motor_deg'][:3],new=v['mlp']['mae_motor_deg'][:3])for name,v in ev.items()},indent=2),flush=True)


if __name__=='__main__':main()
