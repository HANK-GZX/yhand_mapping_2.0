#!/usr/bin/env python3
"""Stage thumb shared-MLP retraining with 3-D workspace-aware model selection."""
import argparse
from copy import deepcopy
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from fit_left_hand_q_to_motor_mlp import SCRIPTS,load_data,delta_angles,pose_groups,grouped_folds,metrics,sha,train as legacy_train
from mlp_q_to_motor import MODEL_DIR,MLPFingerMapper,forward,motor_forward
from thumb_spatial_training import train_thumb,IDS


def workspace_keys(y,combo,boundaries,bins=5):
    """Fixed evaluation cells from saved envelopes, not training normalization."""
    keys=np.empty(len(y),dtype=object)
    for name in np.unique(combo):
        rows=np.flatnonzero(combo==name);b=boundaries[str(name)];axes=[b['axes'][k]for k in('x_motor','y_motor','z_motor')]
        xyz=y[rows][:,[IDS.index(mid)for mid in axes]]
        lo=np.array([min(b['x_grid']),np.min(b['y_min']),min(b['z_grid'])]);hi=np.array([max(b['x_grid']),np.max(b['y_max']),max(b['z_grid'])])
        cells=np.clip(np.floor((xyz-lo)/np.maximum(hi-lo,1e-8)*bins).astype(int),0,bins-1)
        keys[rows]=[f'{name}:{a},{c},{d}'for a,c,d in cells]
    return keys


def macro_error(y,pred,combo,keys):
    values=[]
    for name in np.unique(combo):
        cells=[]
        for key in np.unique(keys[combo==name]):
            rows=keys==key;cells.append(abs(y[rows]-pred[rows]).mean(0))
        values.append(np.mean(cells,axis=0))
    return np.mean(values,axis=0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model-dir',type=Path,default=MODEL_DIR)
    parser.add_argument('--data-root',type=Path,default=SCRIPTS/'motor_test_logs/data_collection')
    parser.add_argument('--report-dir',type=Path)
    parser.add_argument('--epochs',type=int,default=2500)
    args=parser.parse_args(argv)
    if args.epochs<1:parser.error('epochs必须为正')
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    from data_collection_config import THUMB_EXCLUDED_COMBO_IDS
    source_path=args.source_model_dir/'left_thumb_q_to_motor_mlp.json';old=MLPFingerMapper(source_path,finger='thumb').config
    absolute,y,meta,provenance=load_data(args.data_root.resolve(),'thumb');q=delta_angles(absolute,old['q_reference'])
    if any(old['source'][k]!=provenance[k]for k in old['source']):raise ValueError('原始拇指数据来源改变')
    combo=np.array([m['combo_id']for m in meta]);pid=np.array([int(m['point_id'])for m in meta]);eligible=~np.isin(combo,list(THUMB_EXCLUDED_COMBO_IDS))
    if eligible.sum()<100:raise ValueError('当前策略保留的训练点不足')
    boundary_path=SCRIPTS/'motor_test_logs/workspace_boundaries/left/boundary_models.json'
    boundaries=json.loads(boundary_path.read_text())['combos'];axes={name:[boundaries[name]['axes'][k]for k in('x_motor','y_motor','z_motor')]for name in np.unique(combo)}
    report=args.report_dir or SCRIPTS/'motor_test_logs/thumb_spatial_mlp'/datetime.now().strftime('%Y%m%d_%H%M%S')
    report.mkdir(parents=True,exist_ok=True);backup=report/'previous_models';models=report/'models';backup.mkdir();models.mkdir()
    for p in args.source_model_dir.glob('left_*_q_to_motor_mlp.json'):
        shutil.copyfile(p,backup/p.name);shutil.copyfile(p,models/p.name)
    groups=pose_groups(q);pose_fold=grouped_folds(groups,seed=20261026);keys=workspace_keys(y,combo,boundaries)
    baseline_path=SCRIPTS/'motor_test_logs/q_to_motor_delta_retrain/20261027_093149/mlp/report/thumb/validation_predictions.npz'
    archive=np.load(baseline_path)
    for key,value in [('q_delta',q),('motor_feedback',y),('point_ids',pid),('pose_fold',pose_fold)]:np.testing.assert_array_equal(archive[key],value)
    for scheme in('pose_group_5fold','leave_one_combo_out'):
        if not np.allclose(metrics(y,archive[f'{scheme}_mlp'])['mae_motor_deg'],old['validation'][scheme]['mlp']['mae_motor_deg'],atol=1e-8,rtol=0):
            raise ValueError('来源模型与固定原始验证档案不匹配；请按README用--source-model-dir指定替换前32×32模型备份，不能把原档案当作新模型的验证结果')
    for scheme in('pose_group_5fold','leave_one_combo_out'):
        if not np.isfinite(archive[f'{scheme}_mlp']).all():raise ValueError('原模型验证预测无效')
    all_predictions={};evaluations={};alternatives={};baseline_cache={}
    def fit(rows,seed,weighted):
        return train_thumb(q[rows],y[rows],combo[rows],axes,epochs=args.epochs,seed=seed,weighted=weighted,huber=weighted)[0]
    def evaluate(scheme,labels,weighted,tag,use_archive):
        pred=np.zeros_like(y);before=np.zeros_like(y);details=[]
        for i,label in enumerate(np.unique(labels)):
            test=labels==label;train=(~test)&eligible
            net=fit(train,20261028+i,weighted);pred[test]=forward(net,q[test])
            if use_archive:before[test]=archive[f'{scheme}_mlp'][test]
            elif scheme in baseline_cache:before[test]=baseline_cache[scheme][test]
            else:
                base,_=legacy_train(q[~test],y[~test],epochs=1193,width=32,seed=20261026+i)
                before[test]=forward(base,q[test])
            folder=report/'fold_models'/tag/scheme;folder.mkdir(parents=True,exist_ok=True)
            (folder/f'{label}.json').write_text(json.dumps(net,indent=2)+'\n')
            mask=test&eligible;details.append(dict(label=str(label),train=int(train.sum()),test=int(mask.sum()),epochs=args.epochs))
            if mask.any():print(f'{tag} {scheme} {label}: M6/M14 MAE {abs(pred[mask]-y[mask]).mean(0)[[1,4]].round(3).tolist()}',flush=True)
        baseline_cache[scheme]=before.copy()
        macro=macro_error(y[eligible],pred[eligible],combo[eligible],keys[eligible]);previous_macro=macro_error(y[eligible],before[eligible],combo[eligible],keys[eligible])
        result=dict(mlp=metrics(y[eligible],pred[eligible]),previous_mlp=metrics(y[eligible],before[eligible]),
            macro_voxel_mae_motor_deg=macro.tolist(),previous_macro_voxel_mae_motor_deg=previous_macro.tolist(),
            n_test=int(eligible.sum()),folds=details,
            all_recorded_diagnostic_mlp=metrics(y,pred),all_recorded_diagnostic_previous_mlp=metrics(y,before),
            scope='Same current-policy allowed population for both models; excluded historical combo is diagnostic only')
        return result,pred,before
    # The same fixed 7278-point test domain is used for both candidate variants.
    scale=np.maximum(y[eligible].std(0),1e-8)
    for tag,weighted in [('uniform_policy',False),('spatial_policy',True)]:
        result,pred,before=evaluate('pose_group_5fold',pose_fold,weighted,tag,True)
        mean=np.array(result['mlp']['mae_motor_deg']);macro=np.array(result['macro_voxel_mae_motor_deg'])
        score=float(.5*np.mean(mean/scale)+.5*np.mean(macro/scale))
        alternatives[tag]=dict(result=result,pose_selection_score=score,weighted=weighted,huber=weighted)
        all_predictions[f'{tag}_pose_group_5fold']=pred
        print(f'{tag}: selection_score={score:.6f}, MAE={mean.round(3).tolist()}, macro={macro.round(3).tolist()}',flush=True)
    # Include held-out 3-D blocks in selection: good local pose fit alone does
    # not establish generalization across a sparsely occupied workspace.
    workspace_folds=grouped_folds(workspace_keys(y,combo,boundaries,bins=3),count=3,seed=20261028)
    for tag in alternatives:
        result,pred,before=evaluate('workspace_block_3fold',workspace_folds,alternatives[tag]['weighted'],tag,False)
        workspace_score=float(.5*np.mean(np.array(result['mlp']['mae_motor_deg'])/scale)+.5*np.mean(np.array(result['macro_voxel_mae_motor_deg'])/scale))
        alternatives[tag].update(workspace_result=result,workspace_selection_score=workspace_score,
            selection_score=.5*(alternatives[tag]['pose_selection_score']+workspace_score))
        all_predictions[f'{tag}_workspace_block_3fold']=pred;all_predictions['previous_workspace_block_3fold']=before
    selected=min(alternatives,key=lambda name:alternatives[name]['selection_score']);weighted=alternatives[selected]['weighted']
    evaluations['pose_group_5fold']=alternatives[selected]['result'];all_predictions['pose_group_5fold']=all_predictions[f'{selected}_pose_group_5fold']
    all_predictions['previous_pose_group_5fold']=archive['pose_group_5fold_mlp']
    evaluations['workspace_block_3fold']=alternatives[selected]['workspace_result']
    all_predictions['workspace_block_3fold']=all_predictions[f'{selected}_workspace_block_3fold']
    print(f'Selected {selected}',flush=True)
    with(args.data_root/'thumb_data/left_hand/zero_calibration_events.csv').open()as stream:events=list(csv.DictReader(stream))
    event_ids=np.array(sorted(int(e['global_point_id'])for e in events));time_groups=np.searchsorted(event_ids,pid,side='left')
    schemes=[('leave_one_combo_out',combo,True),
             ('zero_interval_3fold',grouped_folds(time_groups,count=3,seed=20261028),False)]
    for scheme,labels,cached in schemes:
        result,pred,before=evaluate(scheme,labels,weighted,selected,cached);evaluations[scheme]=result
        all_predictions[scheme]=pred;all_predictions[f'previous_{scheme}']=before
    network,_,weights=train_thumb(q[eligible],y[eligible],combo[eligible],axes,epochs=args.epochs,weighted=weighted,huber=weighted)
    config=deepcopy(old);config.update(schema_version=2,mapping_revision='thumb_shared_mlp_3d_validated_20261028',network=network,
        training_points=int(eligible.sum()),raw_source_points=len(q),training_timeouts=int(sum(m['timeout']=='1'for m,k in zip(meta,eligible)if k)),
        training_scope='Original full result feedback; current collection-policy subset; no raw data deleted',source=provenance,
        previous_original_validation=deepcopy(old.get('previous_original_validation',old['validation'])),
        validation=evaluations,excluded_collection_combos=sorted(THUMB_EXCLUDED_COMBO_IDS),
        selected_point_ids_sha256=hashlib.sha256(json.dumps(pid[eligible].tolist()).encode()).hexdigest(),
        q_absolute_training_bounds_deg=np.column_stack([absolute[eligible].min(0),absolute[eligible].max(0)]).tolist(),
        q_delta_training_bounds_deg=np.column_stack([q[eligible].min(0),q[eligible].max(0)]).tolist(),
        motor_feedback_training_bounds_deg=np.column_stack([y[eligible].min(0),y[eligible].max(0)]).tolist(),
        reference_outside_training_q_indices=[i for i in range(4)if not q[eligible,i].min()<=0<=q[eligible,i].max()],
        raw_prediction_at_reference_motor_deg=forward(network,np.zeros(4)).tolist(),
        training=dict(architecture=[4,64,32,6],architecture_type='dense_shared',activation='tanh',seed=20261028,epochs=args.epochs,
            optimizer='AdamW',lr=.005,weight_decay=1e-4,loss='standardized SmoothL1 beta0.5'if weighted else'standardized MSE',
            selected_variant=selected,spatial_weighting=weights,boundary_model_sha256=sha(boundary_path),
            selection='Fixed epochs after pilot. Equal-weight pose-fold and 3-D-block standardized mean/macro-voxel MAE select candidate; not independent final testing',
            input_definition=old['training_input_definition'],target_definition=old['training_target_definition'],
            no_added_samples=True,no_pseudo_labels=True,no_linear_baseline=True,torch_version=str(torch.__version__)),
        fit_limitation='Held-out metrics use a matched current-policy domain; all-recorded metrics are separately diagnostic. Reference q1/q3 remains outside data support; zero subtraction is not physical zero calibration. No hardware validation.')
    out=models/source_path.name;out.write_text(json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    candidate=MLPFingerMapper(out,finger='thumb');np.testing.assert_array_equal(motor_forward(config,np.zeros(4)),np.zeros(6))
    np.testing.assert_allclose(motor_forward(config,q[:8]),[list(candidate.predict(row).values())for row in q[:8]],atol=1e-12,rtol=0)
    summary=dict(selected_variant=selected,training_points=int(eligible.sum()),raw_source_points=len(q),source=provenance,
        previous_sha256=sha(backup/source_path.name),new_sha256=sha(out),selection=alternatives,validation=evaluations,
        previous_neutral_raw=old['raw_prediction_at_reference_motor_deg'],new_neutral_raw=config['raw_prediction_at_reference_motor_deg'],
        q_reference_unchanged=config['q_reference']==old['q_reference'],controller_changed=False,hardware_started=False)
    (report/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (report/'point_log_snapshot.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(report/'validation_predictions.npz',q_absolute=absolute,q_delta=q,q_reference=np.array(old['q_reference']),
        input_coordinates=np.array('delta_glove_angles'),motor_feedback=y,motor_ids=np.array(IDS),combo=combo,point_ids=pid,
        eligible=eligible,pose_fold=pose_fold,pose_group=groups,workspace_fold=workspace_folds,time_group=time_groups,**all_predictions)
    print(json.dumps(dict(selected=selected,validation={s:dict(old=v['previous_mlp']['mae_motor_deg'],new=v['mlp']['mae_motor_deg'],n=v['n_test'])for s,v in evaluations.items()}),indent=2),flush=True)
    print(f'Staged {out}; live config unchanged',flush=True)


if __name__=='__main__':main()
