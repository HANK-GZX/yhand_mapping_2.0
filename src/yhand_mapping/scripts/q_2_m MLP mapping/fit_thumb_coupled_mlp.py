#!/usr/bin/env python3
"""Restore thumb side/opposition coupling using real feedback and direction priors."""
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
from thumb_coupled_training import train_coupled_heads,replace_coupled_heads,SIDE_GAINS,OPPOSITION_GAINS,DIRECTIONS
from mlp_q_to_motor import MODEL_DIR,MLPFingerMapper,forward,motor_forward


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-model-dir',type=Path,default=MODEL_DIR)
    p.add_argument('--reference-report',type=Path,default=SCRIPTS/'motor_test_logs/thumb_opposition_mlp/20261029_q1_primary')
    p.add_argument('--runtime-inputs',type=Path,default=SCRIPTS/'motor_test_logs/thumb_q_to_motor_test/analysis/mlp_linear_20261029_1316/comparison_arrays.npz')
    p.add_argument('--report-dir',type=Path);p.add_argument('--epochs',type=int,default=2200);p.add_argument('--width',type=int,default=32)
    p.add_argument('--direction-penalty',type=float,default=.2)
    args=p.parse_args()
    if args.epochs<1 or not 4<=args.width<=256 or not np.isfinite(args.direction_penalty)or args.direction_penalty<0:p.error('训练参数不正确')
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    source=args.source_model_dir/'left_thumb_q_to_motor_mlp.json';old=MLPFingerMapper(source,finger='thumb').config
    ref=json.loads((args.reference_report/'summary.json').read_text())
    if sha(source)!=ref['new_sha256']:raise ValueError('来源与参考档案不匹配，请指定对应备份')
    a=np.load(args.reference_report/'validation_predictions.npz');absolute,y,meta,provenance=load_data(SCRIPTS/'motor_test_logs/data_collection','thumb');q=delta_angles(absolute,old['q_reference']);keep=a['eligible'];combo=a['combo']
    for key,value in [('q_delta',q),('motor_feedback',y),('point_ids',np.array([int(r['point_id'])for r in meta]))]:np.testing.assert_array_equal(a[key],value)
    assert old['source']==provenance
    runtime_archive=np.load(args.runtime_inputs);runtime=np.concatenate([runtime_archive['mlp_q'],runtime_archive['linear_q']])
    out=args.report_dir or SCRIPTS/'motor_test_logs/thumb_coupled_mlp'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir();(out/'previous_models').mkdir()
    for path in args.source_model_dir.glob('left_*_q_to_motor_mlp.json'):shutil.copyfile(path,out/'models'/path.name);shutil.copyfile(path,out/'previous_models'/path.name)
    def fit(mask,seed):return train_coupled_heads(q[mask],y[mask][:,[3,5,4]],runtime_q=runtime,epochs=args.epochs,width=args.width,seed=seed,direction_penalty=args.direction_penalty)
    boundary=json.loads((SCRIPTS/'motor_test_logs/workspace_boundaries/left/boundary_models.json').read_text())['combos'];keys=workspace_keys(y,combo,boundary)
    arrays={k:a[k]for k in('q_absolute','q_delta','q_reference','motor_ids','motor_feedback','point_ids','combo','eligible','pose_group','pose_fold','workspace_fold','time_group')};ev={}
    schemes=[('pose_group_5fold',a['pose_fold']),('workspace_block_3fold',a['workspace_fold']),('leave_one_combo_out',combo),('zero_interval_3fold',grouped_folds(a['time_group'],count=3,seed=20261028))]
    for scheme,labels in schemes:
        before=a[scheme];pred=before.copy();details=[]
        for i,label in enumerate(np.unique(labels)):
            test=labels==label;train=(~test)&keep;side,opp=fit(train,20261029+i)
            fold=load_thumb_fold(old,args.reference_report,scheme,label);frozen=forward(fold['network'],q[test]);np.testing.assert_allclose(frozen,before[test],atol=1e-9,rtol=0)
            c=replace_coupled_heads(fold,side,opp);np.testing.assert_array_equal(forward(c['network'],q[test])[:,:3],frozen[:,:3])
            sv=forward(side,q[test]);ov=forward(opp,q[test]);pred[test,3]=sv[:,0];pred[test,5]=sv[:,1];pred[test,4]=ov[:,0]
            for name,head in(('side',side),('opposition',opp)):
                folder=out/name/'fold_models'/scheme;folder.mkdir(parents=True,exist_ok=True);(folder/f'{label}.json').write_text(json.dumps(head,indent=2)+'\n')
            mask=test&keep;details.append(dict(label=str(label),train=int(train.sum()),test=int(mask.sum()),epochs=args.epochs))
            if mask.any():print(scheme,label,'M8/M27/M14 MAE',np.mean(abs(pred[mask][:,[3,5,4]]-y[mask][:,[3,5,4]]),0).round(3).tolist(),flush=True)
        ev[scheme]=dict(mlp=metrics(y[keep],pred[keep]),previous_mlp=metrics(y[keep],before[keep]),folds=details,n_test=int(keep.sum()),macro_voxel_mae_motor_deg=macro_error(y[keep],pred[keep],combo[keep],keys[keep]).tolist(),previous_macro_voxel_mae_motor_deg=macro_error(y[keep],before[keep],combo[keep],keys[keep]).tolist(),all_recorded_diagnostic_mlp=metrics(y,pred),all_recorded_diagnostic_previous_mlp=metrics(y,before),scope='Only M8/M27/M14 refitted; M4/M6/M7 archived predictions retained. Runtime q-only probes are domain-informed direction regularization, not independent hardware validation.')
        arrays[scheme]=pred;arrays['previous_'+scheme]=before
    side,opp=fit(keep,20261029);c=replace_coupled_heads(old,side,opp);c['validation']=ev;c['previous_coupled_validation']=deepcopy(old['validation'])
    c['group_validation_reports']=group_validation_reports(old,args.reference_report);c['group_validation_reports'].update(side=str((out/'side').resolve()),opposition=str((out/'opposition').resolve()))
    common=dict(seed=20261029,epochs=args.epochs,width=args.width,optimizer='AdamW',lr=.01,weight_decay=0,activation='mixed_softplus',direction_penalty=args.direction_penalty,direction_priors=DIRECTIONS.tolist(),source=provenance,reference_report=str(args.reference_report),runtime_probe_archive=str(args.runtime_inputs),runtime_probe_archive_sha256=sha(args.runtime_inputs),runtime_probe_angles=len(runtime),runtime_motor_labels_used=False,labeled_training_points=int(keep.sum()),no_new_collection=True,no_position_pseudo_labels=True,no_linear_baseline=True,no_runtime_compensation=True,torch_version=str(torch.__version__),selection='Fixed candidate and epochs; validation reviewed, not an independent final test')
    c['thumb_coupled_training']=common
    c['previous_side_training']=deepcopy(old['thumb_side_training']);c['previous_opposition_training']=deepcopy(old['thumb_opposition_training'])
    c['thumb_side_training']=dict(common,trainable_motor_ids=[8,27],input_weights=list(SIDE_GAINS),flex_ext_validation_report=c['group_validation_reports']['flex_ext'],m8_q1_sensitivity_penalty=0.)
    c['thumb_opposition_training']=dict(common,trainable_motor_ids=[14],input_weights=list(OPPOSITION_GAINS),monotonic_inputs=[1],monotonic_direction=1)
    groups=deepcopy(old['training']['groups']);groups[1].update(architecture=[4,args.width,args.width,2],activation='mixed_softplus',input_weights=list(SIDE_GAINS));groups[2].update(architecture=[4,args.width,args.width,1],activation='mixed_softplus',input_weights=list(OPPOSITION_GAINS),monotonic_inputs=[1])
    c['training']=dict(architecture_type='thumb_three_groups_monotonic_coupled',groups=groups,details='thumb_coupled_training; flex/ext copied unchanged')
    c['preserved_outputs']=dict(motor_ids=[4,6,7],source_model_sha256=sha(source),source_training=old['training'],note='Flex/ext network, common normalization, reference and zero alignment unchanged')
    c['fit_limitation']='Only q0 side and q1 opposition monotonicity are structural. Other coupling signs are soft, state-dependent priors from dominant linear behavior, not universal direction or amplitude guarantees. No linear motor commands or feedback labels used in optimization.'
    path=out/'models'/source.name;path.write_text(json.dumps(c,ensure_ascii=False,indent=2,allow_nan=False)+'\n');MLPFingerMapper(path,finger='thumb')
    probe=np.random.default_rng(29).uniform([-50,-70,-70,-70],[70,90,150,150],(5000,4));base=motor_forward(c,probe)
    np.testing.assert_array_equal(base[:,:3],motor_forward(old,probe)[:,:3]);np.testing.assert_array_equal(motor_forward(c,np.zeros(4)),np.zeros(6))
    more=probe.copy();more[:,0]+=1;delta=motor_forward(c,more)-base;assert delta[:,3].min()>=-1e-9 and delta[:,5].max()<=1e-9
    more=probe.copy();more[:,1]-=1;delta=motor_forward(c,more)-base;assert delta[:,4].max()<=1e-9
    summary=dict(previous_sha256=sha(source),new_sha256=sha(path),training=common,validation=ev,training_points=int(keep.sum()),source=provenance,previous_neutral_raw=old['raw_prediction_at_reference_motor_deg'],new_neutral_raw=c['raw_prediction_at_reference_motor_deg'],preserved_output_max_difference=0.,structural_check_points=5000,hardware_started=False,active_model_changed=False)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');np.savez_compressed(out/'validation_predictions.npz',**arrays);shutil.copyfile(args.reference_report/'point_log_snapshot.json',out/'point_log_snapshot.json');print('Staged',path,flush=True)


if __name__=='__main__':main()
