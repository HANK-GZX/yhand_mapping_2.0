#!/usr/bin/env python3
"""Restore part of ring M10 payout while exactly preserving the shared trunk/M17."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np

from fit_left_hand_q_to_motor_mlp import SCRIPTS,load_data,delta_angles,pose_groups,grouped_folds,metrics,sha
from joint_antagonistic_training import train_pair
from ring_extensor_readout_training import pair_network,tune_readout,replace_readout
from mlp_q_to_motor import MODEL_DIR,MLPFingerMapper,forward,motor_forward


def evaluate(q,y,combo,source,args,report):
    base=source['joint_pair_training'];seed=base['seed'];width=base['width']
    groups=pose_groups(q);folds=grouped_folds(groups,seed=seed)
    all_predictions={};all_base={};evaluation={};epochs_selected=[]
    for scheme,labels in [('pose_group_5fold',folds),('leave_one_combo_out',combo)]:
        predicted=np.zeros_like(y);before=np.zeros_like(y);details=[]
        recorded=source['joint_pair_validation'][scheme]['folds']
        for i,label in enumerate(np.unique(labels)):
            entry=recorded[i]
            if str(label)!=entry['label']:raise ValueError('来源验证折与重建折不一致')
            base_epochs=entry.get('inner_selected_epochs',entry.get('epochs'));test=labels==label
            if base_epochs is None:raise ValueError('来源验证折缺少训练轮数')
            rows=np.flatnonzero(~test);inner=grouped_folds(groups[rows],seed=seed+i+1)
            fit_rows,val_rows=rows[inner!=0],rows[inner==0]
            kwargs=dict(epochs=base_epochs,width=width,seed=seed+i,pair_weight=base['pair_error_weight'],
                coordination_penalty=base['output_feature_coordination_penalty'],q3_gain=base['input_weights'][3],
                q3_penalty=base['q3_first_layer_penalty'])
            if base.get('convex_bend_inputs')is not None:
                kwargs.update(convex_bend_inputs=tuple(base['convex_bend_inputs']),flexor_convex_readout=base.get('flexor_convex_readout',False))
            # Never use the deployed full-data trunk for held-out validation.
            inner_head=None;chosen=args.epochs
            if args.epoch_selection=='inner':
                inner_head,_=train_pair(q[fit_rows],y[fit_rows],**kwargs)
                _,chosen=tune_readout(inner_head,q[fit_rows],y[fit_rows],epochs=args.epochs,
                    coordination_penalty=args.coordination_penalty,boundary_weight=args.boundary_weight,
                    validation=(q[val_rows],y[val_rows]))
            if args.reuse_fold_models:
                base_head=json.loads((args.reuse_fold_models/'fold_models'/scheme/str(label)/'base_outer.json').read_text())
            else:base_head,_=train_pair(q[~test],y[~test],**kwargs)
            tuned,_=tune_readout(base_head,q[~test],y[~test],epochs=chosen,
                coordination_penalty=args.coordination_penalty,boundary_weight=args.boundary_weight)
            predicted[test]=forward(tuned,q[test]);before[test]=forward(base_head,q[test])
            np.testing.assert_array_equal(predicted[test,1],before[test,1])
            folder=report/'fold_models'/scheme/str(label);folder.mkdir(parents=True)
            for name,head in [('base_inner',inner_head),('base_outer',base_head),('tuned_outer',tuned)]:
                if head is not None:(folder/f'{name}.json').write_text(json.dumps(head,indent=2)+'\n')
            details.append(dict(label=str(label),train=int((~test).sum()),test=int(test.sum()),
                rebuilt_base_epochs=base_epochs,selected_readout_epochs=chosen))
            if scheme=='pose_group_5fold':epochs_selected.append(chosen)
            print(f'{scheme} {label}: readout_epochs={chosen}, '
                f'M10 MAE {abs(before[test,0]-y[test,0]).mean():.3f} -> {abs(predicted[test,0]-y[test,0]).mean():.3f}; M17 exactly preserved',flush=True)
        evaluation[scheme]=dict(metrics(y,predicted),folds=details)
        all_predictions[scheme]=predicted;all_base[scheme]=before
    selected=args.epochs if args.epoch_selection=='fixed'else int(np.median(epochs_selected))
    return evaluation,all_predictions,all_base,selected,groups,folds


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model-dir',type=Path,default=MODEL_DIR)
    parser.add_argument('--data-root',type=Path,default=SCRIPTS/'motor_test_logs/data_collection')
    parser.add_argument('--report-dir',type=Path)
    parser.add_argument('--epochs',type=int,default=1500)
    parser.add_argument('--epoch-selection',choices=('fixed','inner'),default='fixed')
    parser.add_argument('--reuse-fold-models',type=Path,help='Optional verified prior evaluation artifact, never a deployed full-data trunk')
    parser.add_argument('--coordination-penalty',type=float,default=0.)
    parser.add_argument('--boundary-weight',type=float,default=4.)
    args=parser.parse_args(argv)
    if args.epochs<1 or args.coordination_penalty<0 or args.boundary_weight<1:parser.error('训练参数范围不正确')
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    source=args.source_model_dir/'left_ring_q_to_motor_mlp.json'
    old=MLPFingerMapper(source,finger='ring').config
    if old['schema_version']!=11:raise ValueError('需要已有版本11共享伸屈网络')
    if 'extensor_readout_training'in old:
        raise ValueError('来源已微调；请用--source-model-dir指定首次微调前的previous_models，避免累积修改及验证基线不一致')
    absolute,labels,meta,provenance=load_data(args.data_root.resolve(),'ring')
    if len(absolute)!=old['training_points']or provenance!=old['source']:raise ValueError('原始数据来源/数量改变')
    q=delta_angles(absolute,old['q_reference']);y=labels[:,:2]
    if args.reuse_fold_models:
        reused=json.loads((args.reuse_fold_models/'summary.json').read_text())
        if reused['previous_sha256']!=sha(source)or reused['source']!=provenance:
            raise ValueError('缓存训练折的原始模型/数据来源不一致')
        with np.load(args.reuse_fold_models/'validation_predictions.npz')as cached:
            np.testing.assert_array_equal(cached['q_delta'],q)
            np.testing.assert_array_equal(cached['motor_feedback'],labels)
            np.testing.assert_array_equal(cached['point_ids'],[int(r['point_id'])for r in meta])
    report=args.report_dir or SCRIPTS/'motor_test_logs/ring_extensor_readout'/datetime.now().strftime('%Y%m%d_%H%M%S')
    report.mkdir(parents=True,exist_ok=False);backup=report/'previous_models';models=report/'models';backup.mkdir();models.mkdir()
    for path in args.source_model_dir.glob('left_*_q_to_motor_mlp.json'):
        shutil.copyfile(path,backup/path.name);shutil.copyfile(path,models/path.name)
    print(f'Report {report}: {len(q)} original paired rows; exactly 33 M10 parameters trainable',flush=True)
    ev,pred,before,epochs,groups,folds=evaluate(q,y,np.array([r['combo_id']for r in meta]),old,args,report)
    head,_=tune_readout(pair_network(old),q,y,epochs=epochs,
        coordination_penalty=args.coordination_penalty,boundary_weight=args.boundary_weight)
    config=replace_readout(old,head)
    training=dict(trainable_motor_ids=[10],trainable_parameters=33,epochs=epochs,maximum_selection_epochs=args.epochs,
        optimizer='Adam',lr=.01,precision='float64 output readout only',coordination_penalty=args.coordination_penalty,
        source_pair_coordination_penalty=old['joint_pair_training']['output_feature_coordination_penalty'],
        boundary_weight=args.boundary_weight,boundary_definition='delta q1 OR q3 >= training-fold 80th percentile; original rows only',
        selection=('Fixed readout epochs; candidate-comparison validation, not independent final test'
            if args.epoch_selection=='fixed'else 'Weighted inner-validation M10 error; median readout epochs from five outer pose folds'),
        epoch_selection=args.epoch_selection,cv_base_cache=str(args.reuse_fold_models)if args.reuse_fold_models else None,
        normalization='Source shared network normalization frozen',frozen_motor_ids=[17,22,21],frozen_shared_hidden_layers=2,
        no_added_samples=True,no_pseudo_labels=True,no_linear_baseline=True,no_runtime_payout_offset=True,
        source=provenance,torch_version=str(torch.__version__))
    config.update(mapping_revision='ring_shared_m10_readout_refinement_20261028',
        extensor_readout_training=training,extensor_readout_validation=ev,
        training=dict(architecture_type='shared_antagonistic_pair_mlp',architecture=[4,32,32,2],updated_motor_ids=[10],
            note='Shared trunk/M17 frozen; extensor_readout_training describes current fine-tuning; joint_pair_training describes base fit'),
        preserved_outputs=dict(motor_ids=[17,22,21],source_model_sha256=sha(source),
            note='All shared hidden layers, M17 neuron, side networks, normalization and reference copied exactly'))
    for scheme,result in ev.items():
        for key,value in result.items():
            if key!='folds':config['validation'][scheme]['mlp'][key][0]=value[0]
    config['control_policy_validation_note']=('Readout validation rebuilds inner/outer base trunks without outer test data. '
        'M10 then fine-tuned on real training-fold feedback; M17 unchanged from each fold baseline. '
        'Old full-data trunk is used only for final fitting, never for held-out estimates. '
        'Validation is used for candidate selection, not an independent final test or hardware tension accuracy.')
    config['raw_prediction_at_reference_motor_deg']=forward(config['network'],np.zeros(4)).tolist()
    path=models/source.name;path.write_text(json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    candidate=MLPFingerMapper(path,finger='ring')
    probe=np.random.default_rng(28).uniform([-40,-50,-30,-50],[40,110,160,120],(5000,4))
    a,b=motor_forward(old,probe),motor_forward(config,probe)
    np.testing.assert_array_equal(a[:,1:],b[:,1:])
    np.testing.assert_array_equal(motor_forward(config,np.zeros(4)),np.zeros(4))
    for axis in(1,2,3):
        more=probe.copy();more[:,axis]+=.5
        assert (motor_forward(config,more)[:,0]-b[:,0]).min()>=-1e-10
    summary=dict(training_points=len(q),source=provenance,previous_sha256=sha(backup/source.name),new_sha256=sha(path),
        training=training,validation=ev,
        old_fold_mae={s:metrics(y,v)['mae_motor_deg']for s,v in before.items()},
        new_fold_mae={s:metrics(y,v)['mae_motor_deg']for s,v in pred.items()},
        checks=dict(m17_and_sides_max_difference=float(abs(b[:,1:]-a[:,1:]).max()),
            shared_trunk_preserved=True,normalization_preserved=True,zero_input_zero_output=True),hardware_started=False)
    (report/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (report/'head.json').write_text(json.dumps(head,indent=2)+'\n')
    (report/'point_log_snapshot.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(report/'validation_predictions.npz',q_absolute=absolute,q_delta=q,motor_feedback=labels,
        point_ids=[int(r['point_id'])for r in meta],pose_group=groups,pose_fold=folds,
        **pred,**{f'base_{s}':v for s,v in before.items()})
    print(json.dumps(dict(old=summary['old_fold_mae'],new=summary['new_fold_mae'],epochs=epochs,checks=summary['checks']),indent=2),flush=True)
    print(f'Staged: {path}',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
