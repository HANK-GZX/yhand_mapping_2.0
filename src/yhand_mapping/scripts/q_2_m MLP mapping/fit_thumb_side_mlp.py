#!/usr/bin/env python3
"""Refit shared M8/M27 thumb side MLP, prioritizing q0 with signed monotonicity."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil
import numpy as np

from fit_left_hand_q_to_motor_mlp import SCRIPTS,sha,load_data,delta_angles,grouped_folds,metrics
from fit_thumb_spatial_mlp import workspace_keys,macro_error
from thumb_monotonic_training import replace_thumb_flex_ext
from thumb_side_training import train_thumb_side,replace_thumb_side
from thumb_fold_models import load_thumb_fold,group_validation_reports
from mlp_q_to_motor import MODEL_DIR,MLPFingerMapper,forward,motor_forward


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model-dir',type=Path,default=MODEL_DIR)
    parser.add_argument('--reference-report',type=Path,default=SCRIPTS/'motor_test_logs/thumb_side_mlp/20261029_q0_priority')
    parser.add_argument('--report-dir',type=Path)
    parser.add_argument('--epochs',type=int,default=2200)
    parser.add_argument('--width',type=int,default=32)
    parser.add_argument('--secondary-input-weight',type=float,default=.25)
    parser.add_argument('--m8-q1-sensitivity-penalty',type=float,default=.1,help='Training-only M8 output q1 derivative penalty; zero reproduces original side fit')
    args=parser.parse_args()
    if args.epochs<1 or not 4<=args.width<=256 or not 0<args.secondary_input_weight<=1 or not np.isfinite(args.m8_q1_sensitivity_penalty)or args.m8_q1_sensitivity_penalty<0:parser.error('训练参数范围不正确')
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    source=args.source_model_dir/'left_thumb_q_to_motor_mlp.json';old=MLPFingerMapper(source,finger='thumb').config
    ref=json.loads((args.reference_report/'summary.json').read_text())
    if sha(source)!=ref['new_sha256']:raise ValueError('来源与参考分组验证档案不匹配，请指定对应备份')
    flex_report=Path(group_validation_reports(old,args.reference_report)['flex_ext'])
    a=np.load(args.reference_report/'validation_predictions.npz');absolute,y,meta,provenance=load_data(SCRIPTS/'motor_test_logs/data_collection','thumb');q=delta_angles(absolute,old['q_reference'])
    for key,value in [('q_delta',q),('motor_feedback',y),('point_ids',np.array([int(r['point_id'])for r in meta]))]:np.testing.assert_array_equal(a[key],value)
    assert provenance==old['source'];keep=a['eligible'];combo=a['combo']
    b=json.loads((SCRIPTS/'motor_test_logs/workspace_boundaries/left/boundary_models.json').read_text())['combos'];keys=workspace_keys(y,combo,b)
    out=args.report_dir or SCRIPTS/'motor_test_logs/thumb_side_mlp'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir();(out/'previous_models').mkdir()
    for path in args.source_model_dir.glob('left_*_q_to_motor_mlp.json'):shutil.copyfile(path,out/'models'/path.name);shutil.copyfile(path,out/'previous_models'/path.name)
    def fit(mask,seed):return train_thumb_side(q[mask],y[mask][:,[3,5]],epochs=args.epochs,width=args.width,seed=seed,secondary=args.secondary_input_weight,m8_q1_sensitivity_penalty=args.m8_q1_sensitivity_penalty)
    ev={};arrays={k:a[k]for k in('q_absolute','q_delta','q_reference','motor_ids','motor_feedback','point_ids','combo','eligible','pose_group','pose_fold','workspace_fold','time_group')}
    schemes=[('pose_group_5fold',a['pose_fold']),('workspace_block_3fold',a['workspace_fold']),('leave_one_combo_out',combo),('zero_interval_3fold',grouped_folds(a['time_group'],count=3,seed=20261028))]
    for scheme,labels in schemes:
        before=a[scheme];pred=before.copy();details=[];folder=out/'fold_models'/scheme;folder.mkdir(parents=True)
        for i,label in enumerate(np.unique(labels)):
            test=labels==label;train=(~test)&keep;head=fit(train,20261029+i)
            fold=load_thumb_fold(old,args.reference_report,scheme,label)
            frozen=forward(fold['network'],q[test])
            np.testing.assert_allclose(frozen,before[test],atol=1e-9,rtol=0)
            replacement=replace_thumb_side(fold,head);side=forward(head,q[test]);pred[test,3]=side[:,0];pred[test,5]=side[:,1]
            np.testing.assert_array_equal(forward(replacement['network'],q[test])[:,[0,1,2,4]],frozen[:,[0,1,2,4]])
            (folder/f'{label}.json').write_text(json.dumps(head,indent=2)+'\n');mask=test&keep
            details.append(dict(label=str(label),train=int(train.sum()),test=int(mask.sum()),epochs=args.epochs))
            if mask.any():print(scheme,label,'M8/M27 MAE',abs(pred[mask][:,[3,5]]-y[mask][:,[3,5]]).mean(0).round(3).tolist(),flush=True)
        value=dict(mlp=metrics(y[keep],pred[keep]),previous_mlp=metrics(y[keep],before[keep]),n_test=int(keep.sum()),folds=details,
            macro_voxel_mae_motor_deg=macro_error(y[keep],pred[keep],combo[keep],keys[keep]).tolist(),
            previous_macro_voxel_mae_motor_deg=macro_error(y[keep],before[keep],combo[keep],keys[keep]).tolist(),
            all_recorded_diagnostic_mlp=metrics(y,pred),all_recorded_diagnostic_previous_mlp=metrics(y,before),
            scope='Same original 7278-point policy domain; only M8/M27 recomputed. Preserved outputs inherit their unchanged fold predictions. Not independent final hardware testing.')
        ev[scheme]=value;arrays[scheme]=pred;arrays['previous_'+scheme]=before
    head=fit(keep,20261029);c=replace_thumb_side(old,head);c['validation']=ev
    c['group_validation_reports']=group_validation_reports(old,args.reference_report)
    c['group_validation_reports']['side']=str(out.resolve())
    c['previous_side_validation']=deepcopy(old['validation'])
    c['previous_side_training']=deepcopy(old.get('thumb_side_training'))
    c['thumb_side_training']=dict(seed=20261029,epochs=args.epochs,width=args.width,optimizer='AdamW',lr=.01,weight_decay=0,
        input_weights=[1.,args.secondary_input_weight,args.secondary_input_weight,args.secondary_input_weight],priority_penalty=.001,
        activation='mixed_softplus_tanh',positive_weight_parameterization='softplus(raw_weight)',
        monotonic_inputs=[0],output_directions=[1,-1],trainable_motor_ids=[8,27],
        m8_q1_sensitivity_penalty=args.m8_q1_sensitivity_penalty,
        derivative_coordinates='standardized M8 output w.r.t. standardized q1 before shared input gain',
        regularized_motor_ids=[8]if args.m8_q1_sensitivity_penalty else [],
        shared_hidden_layers=2,source=provenance,reference_report=str(args.reference_report),flex_ext_validation_report=str(flex_report),torch_version=str(torch.__version__),
        no_pseudo_labels=True,no_added_samples=True,no_linear_baseline=True,no_runtime_output_clipping=True,
        selection='Fixed epoch candidate; group metrics used for review, not an independent final test')
    c['input_priority'].pop('training_only_output_sensitivity',None)
    if args.m8_q1_sensitivity_penalty:
        c['mapping_revision']='thumb_M8_q1_reduced_shared_side_20261029'
        c['side_motor_mapping']+=' M8-only q1 sensitivity regularized during training; shared input gains and M27 priority unchanged.'
        c['input_priority']['training_only_output_sensitivity']=dict(motor_ids=[8],q_axes=[1],coefficient=args.m8_q1_sensitivity_penalty,coordinates=c['thumb_side_training']['derivative_coordinates'])
        c['input_priority']['note']+=' M8-only q1 output sensitivity penalty applied during training; per_motor_weights remain shared runtime input gains.'
    groups=deepcopy(old['training']['groups']);groups[1].update(architecture=[4,args.width,args.width,2],activation='mixed_softplus_tanh',monotonic_inputs=[0],input_weights=c['thumb_side_training']['input_weights'])
    c['training']=dict(architecture_type='thumb_three_groups_monotonic_flex_ext_and_side',groups=groups,details='thumb_side_training; flex/ext and opposition copied unchanged')
    c['preserved_outputs']=dict(motor_ids=[4,6,7,14],source_model_sha256=sha(source),source_training=old['training'],note='Networks, normalization, reference and output policy copied unchanged')
    c['fit_limitation']='q0 signed monotonicity of M8(+)/M27(-) is structural at fixed other angles, not a minimum slope, amplitude, physical tension or human/robot alignment guarantee. q1/q2/q3 remain weak unconstrained side inputs; preserved flex/ext retain their q2/q3 monotonicity.'
    path=out/'models'/source.name;path.write_text(json.dumps(c,ensure_ascii=False,indent=2,allow_nan=False)+'\n');MLPFingerMapper(path,finger='thumb')
    probe=np.random.default_rng(29).uniform([-50,-70,-70,-70],[70,90,150,150],(5000,4));base=motor_forward(c,probe)
    np.testing.assert_array_equal(base[:,[0,1,2,4]],motor_forward(old,probe)[:,[0,1,2,4]]);np.testing.assert_array_equal(motor_forward(c,np.zeros(4)),np.zeros(6))
    changes=[]
    for axis in(0,):
        more=probe.copy();more[:,axis]+=1;diff=motor_forward(c,more)-base
        assert diff[:,3].min()>=-1e-9 and diff[:,5].max()<=1e-9;changes.append(dict(axis=axis,M8_min=float(diff[:,3].min()),M27_max=float(diff[:,5].max())))
    summary=dict(previous_sha256=sha(source),new_sha256=sha(path),training_points=int(keep.sum()),source=provenance,training=c['thumb_side_training'],validation=ev,
        previous_neutral_raw=old['raw_prediction_at_reference_motor_deg'],new_neutral_raw=c['raw_prediction_at_reference_motor_deg'],
        monotonic_checks=changes,preserved_output_max_difference=0.,hardware_started=False,active_model_changed=False)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');np.savez_compressed(out/'validation_predictions.npz',**arrays)
    shutil.copyfile(args.reference_report/'point_log_snapshot.json',out/'point_log_snapshot.json')
    print('Staged',path,flush=True);print(json.dumps({name:dict(old=np.array(v['previous_mlp']['mae_motor_deg'])[[3,5]].tolist(),new=np.array(v['mlp']['mae_motor_deg'])[[3,5]].tolist())for name,v in ev.items()},indent=2),flush=True)


if __name__=='__main__':main()
