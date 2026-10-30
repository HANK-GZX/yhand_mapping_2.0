#!/usr/bin/env python3
"""Fit one shared 4-W-W-2 ring M10/M17 MLP and preserve both side networks."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np

from fit_left_hand_q_to_motor_mlp import SCRIPTS,load_data,delta_angles,sha
from fit_monotonic_extensor import evaluate
from joint_antagonistic_training import train_pair
from mlp_q_to_motor import MODEL_DIR,MLPFingerMapper,forward,motor_forward


def assemble(previous,head):
    if previous['finger']!='ring' or previous['motor_order']!=[10,17,22,21]:
        raise ValueError('本入口仅用于无名指M10/M17联合网络')
    result=deepcopy(previous);network=result['network']
    for key in ('input_center','input_scale'):
        np.testing.assert_allclose(network[key],head[key],atol=1e-12,rtol=1e-12)
    pair=deepcopy(head['branches'][0])
    if pair['output_indices']!=[0,1]or not pair.get('shared_antagonistic_pair'):
        raise ValueError('必须提供单个共享双输出分支')
    sides=[deepcopy(b)for b in network['branches']if b['output_indices']in([2],[3])]
    if len(sides)!=2:raise ValueError('来源必须有两条独立侧摆网络')
    network['branches']=[pair,*sides]
    for key in ('output_center','output_scale'):network[key][:2]=head[key]
    for mid in (10,17):result['input_priority']['per_motor_weights'][str(mid)]=list(pair['input_weights'])
    result['schema_version']=11
    result['raw_prediction_at_reference_motor_deg']=forward(network,np.zeros(4)).tolist()
    # Remove current-structure claims that described two independent networks.
    historical={}
    for prefix in ('extensor','flexor'):
        for suffix in ('structure','training','validation'):
            key=f'{prefix}_{suffix}'
            if key in result:historical[key]=result.pop(key)
    if historical:result['previous_independent_pair_metadata']=historical
    # Replacing the full pair invalidates claims about a frozen-M17 readout fit.
    readout={key:result.pop(key)for key in('extensor_readout_training','extensor_readout_validation')if key in result}
    if readout:result['previous_extensor_readout_metadata']=readout
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model-dir',type=Path,default=MODEL_DIR)
    parser.add_argument('--data-root',type=Path,default=SCRIPTS/'motor_test_logs/data_collection')
    parser.add_argument('--report-dir',type=Path)
    parser.add_argument('--epochs',type=int,default=3500)
    parser.add_argument('--width',type=int,default=32)
    parser.add_argument('--seed',type=int,default=20261028)
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args(argv)
    if args.epochs<1 or args.width<4 or args.width>256 or args.threads<1:parser.error('epochs/threads必须为正，width为4..256')
    import torch
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    source=args.source_model_dir/'left_ring_q_to_motor_mlp.json'
    old=MLPFingerMapper(source,finger='ring').config
    if old['schema_version']not in(7,11):raise ValueError('需要版本7或11无名指来源模型')
    absolute,labels,meta,provenance=load_data(args.data_root.resolve(),'ring')
    if old['training_points']!=len(absolute)or old['source']!=provenance:
        raise ValueError('数据数量或来源改变，不允许静默更换训练数据')
    q=delta_angles(absolute,old['q_reference']);y=labels[:,:2]
    report=args.report_dir or SCRIPTS/'motor_test_logs/ring_joint_mlp'/datetime.now().strftime('%Y%m%d_%H%M%S')
    report.mkdir(parents=True,exist_ok=False)
    backup,models=report/'previous_models',report/'models';backup.mkdir();models.mkdir()
    for path in args.source_model_dir.glob('left_*_q_to_motor_mlp.json'):
        shutil.copyfile(path,backup/path.name);shutil.copyfile(path,models/path.name)
    print(f'{report}: {len(q)} unchanged paired rows, one 4-{args.width}-{args.width}-2 MLP',flush=True)
    def trainer(q,y,**kwargs):
        kwargs.pop('direction',None)
        return train_pair(q,y,**kwargs)
    evaluation,predictions,epochs,groups,folds=evaluate(q,y,np.array([r['combo_id']for r in meta]),
        epochs=args.epochs,width=args.width,seed=args.seed,trainer=trainer)
    head,_=train_pair(q,y,epochs=epochs,width=args.width,seed=args.seed)
    config=assemble(old,head)
    training=dict(epochs=epochs,maximum_selection_epochs=args.epochs,seed=args.seed,width=args.width,
        optimizer='AdamW',lr=.01,early_stop_patience=150,input_weights=[.25,1.,1.,1.],
        pair_error_weight=.2,paired_precision_definition='Training-fold standardized measured-label covariance plus 0.02 diagonal; inverse normalized to trace 2',
        output_feature_coordination_penalty=10.,q0_first_layer_penalty=.015,q3_first_layer_penalty=0.,
        selection='Median inner-selected epochs from five outer pose folds; early stop scores validation pair error plus output-feature coordination penalty',
        torch_version=str(torch.__version__),source=provenance,
        input_definition=old['training_input_definition'],target_definition=old['training_target_definition'],
        no_added_samples=True,no_pseudo_labels=True,no_linear_baseline=True,no_fixed_motor_ratio=True)
    config.update(mapping_revision='ring_shared_antagonistic_mlp_20261028',
        joint_antagonistic_structure=dict(motor_ids=[10,17],architecture=[4,args.width,args.width,2],
            shared_hidden_layers=2,independent_hidden_layers=0,activation='mixed_softplus',
            output='one dense layer with two learned output neurons; signed monotonic directions +1/-1',
            bending_monotonic_inputs=[1,2,3],
            guarantee='M10 payout and M17 winding are monotonic. Shared features and soft coordination are not tension guarantees.'),
        joint_pair_training=training,joint_pair_validation=evaluation,
        preserved_outputs=dict(motor_ids=[22,21],source_model_sha256=sha(source),source_training=old['training'],
            note='Side weights, normalization, zero alignment and 2 degree policy unchanged'),
        training=dict(architecture_type='shared_antagonistic_pair_mlp',updated_motor_ids=[10,17],
            architecture=[4,args.width,args.width,2],details='joint_pair_training; side training retained'))
    for scheme,result in evaluation.items():
        for metric,value in result.items():
            if metric!='folds':config['validation'][scheme]['mlp'][metric][:2]=value
    config['control_policy_validation_note']=('M10/M17 jointly evaluated on raw measured feedback; side validation inherited. '
        'Outer test data are excluded from scaling, covariance, gradients and epoch selection. '
        'Architecture screening preceded this evaluation, so this is not a fully nested architecture search or hardware accuracy.')
    path=models/source.name;path.write_text(json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    candidate=MLPFingerMapper(path,finger='ring')
    probe=np.random.default_rng(args.seed).uniform([-40,-50,-20,-50],[40,100,150,110],(5000,4))
    old_out,new_out=motor_forward(old,probe),motor_forward(config,probe)
    np.testing.assert_allclose(old_out[:,2:],new_out[:,2:],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(motor_forward(config,np.zeros(4)),np.zeros(4))
    changes=[]
    for axis in (1,2,3):
        more=probe.copy();more[:,axis]+=.5;diff=motor_forward(config,more)-new_out
        assert diff[:,0].min()>=-1e-10 and diff[:,1].max()<=1e-10
        changes.append([float(diff[:,0].min()),float(diff[:,1].max())])
    summary=dict(training_points=len(q),source=provenance,previous_sha256=sha(backup/source.name),new_sha256=sha(path),
        old_mae={s:old['validation'][s]['mlp']['mae_motor_deg'][:2]for s in evaluation},
        new_mae={s:v['mae_motor_deg']for s,v in evaluation.items()},
        paired_angle_error_mae={s:float(abs((v-y).sum(1)).mean())for s,v in predictions.items()},
        training=training,checks=dict(shared_pair_branches=1,side_max_difference=float(abs(new_out[:,2:]-old_out[:,2:]).max()),
            monotonic_half_degree=changes,zero_input_zero_output=True,same_zero_reference=config['q_reference']==old['q_reference'],
            same_side_policy=config['side_bending_policy']==old['side_bending_policy']),hardware_started=False)
    (report/'head.json').write_text(json.dumps(head,indent=2)+'\n')
    (report/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (report/'point_log_snapshot.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(report/'validation_predictions.npz',q_absolute=absolute,q_delta=q,motor_feedback=labels,
        q_reference=old['q_reference'],point_ids=[int(r['point_id'])for r in meta],pose_group=groups,pose_fold=folds,**predictions)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print(f'Staged: {path}',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
