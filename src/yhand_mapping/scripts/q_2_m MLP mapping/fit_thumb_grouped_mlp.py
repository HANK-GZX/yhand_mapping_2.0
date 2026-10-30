#!/usr/bin/env python3
"""Stage three thumb MLPs on the same robot data/splits as the current dense MLP."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from fit_left_hand_q_to_motor_mlp import SCRIPTS,sha,load_data,delta_angles,grouped_folds,metrics
from fit_thumb_spatial_mlp import workspace_keys,macro_error
from thumb_spatial_training import train_thumb,IDS,THUMB_GROUPS
from mlp_q_to_motor import MODEL_DIR,MLPFingerMapper,forward,motor_forward


def grouped_config(source,network):
    c=deepcopy(source);c.update(schema_version=3,network=network,
        mapping_revision='thumb_three_group_mlp_20261028',
        input_priority=dict(applied_after_standardization=True,per_motor_weights={str(mid):[1.,1.,1.,1.]for mid in IDS},
            note='All four delta angles retained for each thumb group; no four-finger axis priorities'),
        side_motor_mapping='M8/M27 share a two-output MLP; M4/M6/M7 share a three-output MLP; M14 has its own MLP; all consume four delta angles')
    return c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model-dir',type=Path,default=MODEL_DIR)
    parser.add_argument('--reference-report',type=Path,default=SCRIPTS/'motor_test_logs/thumb_spatial_mlp/20261028_retrain')
    parser.add_argument('--report-dir',type=Path)
    parser.add_argument('--epochs',type=int,default=2500)
    parser.add_argument('--hidden',type=int,nargs=2,default=[32,32])
    parser.add_argument('--lr',type=float,default=.005)
    parser.add_argument('--initialize-from-shared',action='store_true')
    args=parser.parse_args()
    if args.epochs<1:parser.error('epochs必须为正')
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    source=args.source_model_dir/'left_thumb_q_to_motor_mlp.json';old=MLPFingerMapper(source,finger='thumb').config
    reference=json.loads((args.reference_report/'summary.json').read_text())
    if sha(source)!=reference['new_sha256']:raise ValueError('来源模型与固定分组验证档案不匹配，请指定该档案对应的共享网络备份')
    archive=np.load(args.reference_report/'validation_predictions.npz')
    absolute,y,meta,provenance=load_data(SCRIPTS/'motor_test_logs/data_collection','thumb');q=delta_angles(absolute,old['q_reference'])
    for key,value in [('q_delta',q),('motor_feedback',y),('point_ids',np.array([int(m['point_id'])for m in meta]))]:np.testing.assert_array_equal(archive[key],value)
    if old['source']!=provenance:raise ValueError('来源数据改变')
    keep=archive['eligible'];combo=archive['combo'];qref=np.array(old['q_reference'])
    boundary=SCRIPTS/'motor_test_logs/workspace_boundaries/left/boundary_models.json'
    boundaries=json.loads(boundary.read_text())['combos'];keys=workspace_keys(y,combo,boundaries)
    axes={name:[boundaries[name]['axes'][k]for k in('x_motor','y_motor','z_motor')]for name in np.unique(combo)}
    report=args.report_dir or SCRIPTS/'motor_test_logs/thumb_grouped_mlp'/datetime.now().strftime('%Y%m%d_%H%M%S')
    report.mkdir(parents=True,exist_ok=False);backup=report/'previous_models';models=report/'models';backup.mkdir();models.mkdir()
    for p in args.source_model_dir.glob('left_*_q_to_motor_mlp.json'):shutil.copyfile(p,backup/p.name);shutil.copyfile(p,models/p.name)
    def fit(mask,seed,initial=None):
        return train_thumb(q[mask],y[mask],combo[mask],axes,epochs=args.epochs,seed=seed,
            hidden=tuple(args.hidden),lr=args.lr,weighted=False,huber=False,grouped=True,initial_network=initial)[0]
    schemes=[('pose_group_5fold',archive['pose_fold']),('workspace_block_3fold',archive['workspace_fold']),
        ('leave_one_combo_out',combo),('zero_interval_3fold',grouped_folds(archive['time_group'],count=3,seed=20261028))]
    evaluation={};arrays={key:archive[key]for key in('q_absolute','q_delta','q_reference','motor_ids','motor_feedback','point_ids','combo','eligible','pose_fold','pose_group','workspace_fold','time_group')}
    for scheme,labels in schemes:
        pred=np.zeros_like(y);folds=[];before=archive[scheme]
        np.testing.assert_allclose(metrics(y[keep],before[keep])['mae_motor_deg'],old['validation'][scheme]['mlp']['mae_motor_deg'],atol=1e-9,rtol=0)
        folder=report/'fold_models'/scheme;folder.mkdir(parents=True)
        for i,label in enumerate(np.unique(labels)):
            test=labels==label;train=(~test)&keep
            initial=None
            if args.initialize_from_shared:
                initial=json.loads((args.reference_report/'fold_models/uniform_policy'/scheme/f'{label}.json').read_text())
                np.testing.assert_allclose(forward(initial,q[test]),before[test],atol=1e-9,rtol=0)
            net=fit(train,20261028+i,initial);pred[test]=forward(net,q[test])
            (folder/f'{label}.json').write_text(json.dumps(net,indent=2)+'\n')
            mask=test&keep;folds.append(dict(label=str(label),train=int(train.sum()),test=int(mask.sum())))
            if mask.any():print(scheme,label,'MAE',abs(pred[mask]-y[mask]).mean(0).round(3).tolist(),flush=True)
        value=dict(mlp=metrics(y[keep],pred[keep]),previous_mlp=metrics(y[keep],before[keep]),n_test=int(keep.sum()),folds=folds,
            macro_voxel_mae_motor_deg=macro_error(y[keep],pred[keep],combo[keep],keys[keep]).tolist(),
            previous_macro_voxel_mae_motor_deg=macro_error(y[keep],before[keep],combo[keep],keys[keep]).tolist(),
            all_recorded_diagnostic_mlp=metrics(y,pred),all_recorded_diagnostic_previous_mlp=metrics(y,before),
            scope='Same 7278-point policy population, unchanged raw labels/splits/hyperparameters; no new independent hardware test')
        evaluation[scheme]=value;arrays[scheme]=pred;arrays['previous_'+scheme]=before
    network=fit(keep,20261028,old['network']if args.initialize_from_shared else None);c=grouped_config(old,network);c['validation']=evaluation
    c['previous_shared_validation']=deepcopy(old['validation'])
    c['raw_prediction_at_reference_motor_deg']=forward(network,np.zeros(4)).tolist()
    c['training']=dict(architecture_type='thumb_three_independent_group_trunks',groups=[dict(name=name,motor_ids=[IDS[i]for i in indices],architecture=[4,*args.hidden,len(indices)])for name,indices in THUMB_GROUPS],
        activation='tanh',epochs=args.epochs,seed=20261028,optimizer='AdamW',lr=args.lr,weight_decay=1e-4,
        loss='standardized MSE',selected_variant='three_group_uniform_policy',spatial_weighting=dict(type='uniform'),
        input_definition=old['training_input_definition'],target_definition=old['training_target_definition'],
        no_added_samples=True,no_pseudo_labels=True,no_linear_baseline=True,no_monotonic_prior=True,
        reference_report=str(args.reference_report),boundary_model_sha256=sha(boundary),torch_version=str(torch.__version__),
        initialization='fold-specific shared-network parameters; no held-out data in initialization'if args.initialize_from_shared else'random',
        evaluation_note='User-requested group split; unchanged original labels and outer splits. Warm-start uses only the corresponding training-fold model. Existing validation reused, not independent final testing.')
    c['fit_limitation']='Three groups reduce feature competition across muscle roles; do not guarantee tendon tension, extrapolation or human/robot coordinate alignment. Reference q1/q3 remains outside training support; no hardware verification.'
    out=models/source.name;out.write_text(json.dumps(c,ensure_ascii=False,indent=2,allow_nan=False)+'\n');mapper=MLPFingerMapper(out,finger='thumb')
    np.testing.assert_array_equal(motor_forward(c,np.zeros(4)),np.zeros(6))
    np.testing.assert_allclose(motor_forward(c,q[:8]),[list(mapper.predict(row).values())for row in q[:8]],atol=1e-12,rtol=0)
    parameters=sum(np.asarray(layer['weight']).size+len(layer['bias'])for branch in network['branches']for layer in branch['layers'])
    summary=dict(previous_sha256=sha(source),new_sha256=sha(out),parameter_count=parameters,training_points=int(keep.sum()),source=provenance,
        validation=evaluation,previous_neutral_raw=old['raw_prediction_at_reference_motor_deg'],new_neutral_raw=c['raw_prediction_at_reference_motor_deg'],
        labels_unchanged=True,q_reference_unchanged=c['q_reference']==old['q_reference'],hardware_started=False,active_model_changed=False)
    (report/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');np.savez_compressed(report/'validation_predictions.npz',**arrays)
    shutil.copyfile(args.reference_report/'point_log_snapshot.json',report/'point_log_snapshot.json')
    print('Staged three-group thumb:',out,flush=True)
    print(json.dumps({name:dict(old=v['previous_mlp']['mae_motor_deg'],new=v['mlp']['mae_motor_deg'])for name,v in evaluation.items()},indent=2),flush=True)


if __name__=='__main__':main()
