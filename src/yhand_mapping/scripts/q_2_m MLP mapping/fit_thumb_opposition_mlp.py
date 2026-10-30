#!/usr/bin/env python3
"""Refit only M14: q1 primary, q0 secondary, q2/q3 minimally weighted."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil
import numpy as np
from fit_left_hand_q_to_motor_mlp import SCRIPTS,sha,load_data,delta_angles,grouped_folds,metrics
from fit_thumb_spatial_mlp import workspace_keys,macro_error
from thumb_fold_models import load_thumb_fold,group_validation_reports
from thumb_opposition_training import train_thumb_opposition,replace_thumb_opposition,INPUT_WEIGHTS,SENSITIVITY_PENALTIES
from mlp_q_to_motor import MODEL_DIR,MLPFingerMapper,forward,motor_forward

PRESERVED=[0,1,2,3,5]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model-dir',type=Path,default=MODEL_DIR)
    parser.add_argument('--reference-report',type=Path,default=SCRIPTS/'motor_test_logs/thumb_side_mlp/20261029_M8_q1_reduced')
    parser.add_argument('--report-dir',type=Path)
    parser.add_argument('--epochs',type=int,default=2200)
    parser.add_argument('--hidden',nargs=2,type=int,default=[64,32])
    parser.add_argument('--input-weights',nargs=4,type=float,default=INPUT_WEIGHTS)
    parser.add_argument('--sensitivity-penalties',nargs=4,type=float,default=SENSITIVITY_PENALTIES)
    parser.add_argument('--priority-penalty',type=float,default=.001)
    parser.add_argument('--lr',type=float,default=.005)
    args=parser.parse_args()
    if (args.epochs<1 or any(not 4<=w<=256 for w in args.hidden)
            or not all(np.isfinite(v)and 0<v<=1 for v in args.input_weights)
            or not all(np.isfinite(v)and v>=0 for v in args.sensitivity_penalties)
            or not np.isfinite(args.priority_penalty)or args.priority_penalty<0 or not np.isfinite(args.lr)or args.lr<=0):parser.error('对掌训练参数不正确')
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    source=args.source_model_dir/'left_thumb_q_to_motor_mlp.json';old=MLPFingerMapper(source,finger='thumb').config
    ref=json.loads((args.reference_report/'summary.json').read_text())
    if sha(source)!=ref['new_sha256']:raise ValueError('来源与参考验证档案不匹配，请指定对应备份')
    a=np.load(args.reference_report/'validation_predictions.npz');absolute,y,meta,provenance=load_data(SCRIPTS/'motor_test_logs/data_collection','thumb');q=delta_angles(absolute,old['q_reference'])
    for key,value in [('q_delta',q),('motor_feedback',y),('point_ids',np.array([int(r['point_id'])for r in meta]))]:np.testing.assert_array_equal(a[key],value)
    assert provenance==old['source'];keep=a['eligible'];combo=a['combo']
    boundary=json.loads((SCRIPTS/'motor_test_logs/workspace_boundaries/left/boundary_models.json').read_text())['combos'];keys=workspace_keys(y,combo,boundary)
    out=args.report_dir or SCRIPTS/'motor_test_logs/thumb_opposition_mlp'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir();(out/'previous_models').mkdir()
    for path in args.source_model_dir.glob('left_*_q_to_motor_mlp.json'):shutil.copyfile(path,out/'models'/path.name);shutil.copyfile(path,out/'previous_models'/path.name)
    def fit(mask,seed):return train_thumb_opposition(q[mask],y[mask][:,[4]],epochs=args.epochs,hidden=tuple(args.hidden),seed=seed,lr=args.lr,input_weights=args.input_weights,sensitivity_penalties=args.sensitivity_penalties,priority_penalty=args.priority_penalty)
    ev={};arrays={k:a[k]for k in('q_absolute','q_delta','q_reference','motor_ids','motor_feedback','point_ids','combo','eligible','pose_group','pose_fold','workspace_fold','time_group')}
    schemes=[('pose_group_5fold',a['pose_fold']),('workspace_block_3fold',a['workspace_fold']),('leave_one_combo_out',combo),('zero_interval_3fold',grouped_folds(a['time_group'],count=3,seed=20261028))]
    for scheme,labels in schemes:
        before=a[scheme];pred=before.copy();details=[];folder=out/'fold_models'/scheme;folder.mkdir(parents=True)
        for i,label in enumerate(np.unique(labels)):
            test=labels==label;train=(~test)&keep;head=fit(train,20261029+i)
            fold=load_thumb_fold(old,args.reference_report,scheme,label)
            frozen=forward(fold['network'],q[test])
            np.testing.assert_allclose(frozen,before[test],atol=1e-9,rtol=0)
            replacement=replace_thumb_opposition(fold,head);pred[test,4]=forward(head,q[test])[:,0]
            np.testing.assert_array_equal(forward(replacement['network'],q[test])[:,PRESERVED],frozen[:,PRESERVED])
            (folder/f'{label}.json').write_text(json.dumps(head,indent=2)+'\n');mask=test&keep
            details.append(dict(label=str(label),train=int(train.sum()),test=int(mask.sum()),epochs=args.epochs))
            if mask.any():print(scheme,label,'M14 MAE',round(float(abs(pred[mask,4]-y[mask,4]).mean()),3),flush=True)
        ev[scheme]=dict(mlp=metrics(y[keep],pred[keep]),previous_mlp=metrics(y[keep],before[keep]),n_test=int(keep.sum()),folds=details,
            macro_voxel_mae_motor_deg=macro_error(y[keep],pred[keep],combo[keep],keys[keep]).tolist(),previous_macro_voxel_mae_motor_deg=macro_error(y[keep],before[keep],combo[keep],keys[keep]).tolist(),
            all_recorded_diagnostic_mlp=metrics(y,pred),all_recorded_diagnostic_previous_mlp=metrics(y,before),
            scope='Same 7278-point policy domain; only M14 recomputed. Other five outputs inherit frozen group-fold predictions. Not independent hardware testing.')
        arrays[scheme]=pred;arrays['previous_'+scheme]=before
    head=fit(keep,20261029);c=replace_thumb_opposition(old,head);c['validation']=ev
    c['previous_opposition_validation']=deepcopy(old['validation'])
    c['group_validation_reports']=group_validation_reports(old,args.reference_report);c['group_validation_reports']['opposition']=str(out.resolve())
    c['thumb_opposition_training']=dict(seed=20261029,epochs=args.epochs,hidden=args.hidden,optimizer='AdamW',lr=args.lr,weight_decay=0,
        input_weights=list(args.input_weights),sensitivity_penalties=list(args.sensitivity_penalties),priority_penalty=args.priority_penalty,
        derivative_coordinates='standardized M14 output w.r.t. standardized angles before input gains',activation='tanh',trainable_motor_ids=[14],
        source=provenance,reference_report=str(args.reference_report),torch_version=str(torch.__version__),no_added_samples=True,no_pseudo_labels=True,no_linear_baseline=True,no_runtime_compensation=True,
        selection='Fixed candidate; validation used for review, not epoch selection or independent final test')
    groups=deepcopy(old['training']['groups']);groups[2].update(architecture=[4,*args.hidden,1],activation='tanh',input_weights=list(args.input_weights))
    c['training']=dict(architecture_type='thumb_three_groups_monotonic_flex_ext_and_side',groups=groups,details='thumb_opposition_training; flex/ext and side copied unchanged')
    c['preserved_outputs']=dict(motor_ids=[4,6,7,8,27],source_model_sha256=sha(source),source_training=old['training'],note='Networks, normalization, reference and output alignment copied unchanged')
    c['fit_limitation']='M14 input priorities and derivative penalties are soft preferences on training states, not global Jacobian ranks, directional, amplitude, physical tension or alignment guarantees. Flex/ext and side mappings are unchanged.'
    path=out/'models'/source.name;path.write_text(json.dumps(c,ensure_ascii=False,indent=2,allow_nan=False)+'\n');MLPFingerMapper(path,finger='thumb')
    probe=np.random.default_rng(29).uniform([-50,-70,-70,-70],[70,90,150,150],(5000,4));base=motor_forward(c,probe)
    np.testing.assert_array_equal(base[:,PRESERVED],motor_forward(old,probe)[:,PRESERVED]);np.testing.assert_array_equal(motor_forward(c,np.zeros(4)),np.zeros(6))
    def slopes(config):
        value=motor_forward(config,q[keep])[:,4];result=[]
        for axis in range(4):
            more=q[keep].copy();more[:,axis]+=.01;result.append(float(np.mean(abs((motor_forward(config,more)[:,4]-value)/.01))))
        return result
    summary=dict(previous_sha256=sha(source),new_sha256=sha(path),training_points=int(keep.sum()),source=provenance,training=c['thumb_opposition_training'],validation=ev,
        training_mean_abs_input_slopes=dict(old=slopes(old),new=slopes(c)),previous_neutral_raw=old['raw_prediction_at_reference_motor_deg'],new_neutral_raw=c['raw_prediction_at_reference_motor_deg'],
        preserved_output_max_difference=0.,hardware_started=False,active_model_changed=False)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');np.savez_compressed(out/'validation_predictions.npz',**arrays)
    shutil.copyfile(args.reference_report/'point_log_snapshot.json',out/'point_log_snapshot.json')
    print('Staged',path,flush=True);print(json.dumps({k:dict(old=v['previous_mlp']['mae_motor_deg'][4],new=v['mlp']['mae_motor_deg'][4])for k,v in ev.items()},indent=2),flush=True)


if __name__=='__main__':main()
