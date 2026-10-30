#!/usr/bin/env python3
"""Stage a genuine shared MLP with sustained M17 contraction along q1."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np

from fit_left_hand_q_to_motor_mlp import SCRIPTS,load_data,delta_angles,pose_groups,grouped_folds,metrics,sha
from fit_ring_joint_mlp import assemble
from joint_antagonistic_training import train_pair
from mlp_q_to_motor import MODEL_DIR,MLPFingerMapper,forward,motor_forward


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model-dir',type=Path,default=MODEL_DIR)
    parser.add_argument('--data-root',type=Path,default=SCRIPTS/'motor_test_logs/data_collection')
    parser.add_argument('--report-dir',type=Path)
    parser.add_argument('--epochs',type=int,default=3000)
    parser.add_argument('--width',type=int,default=32)
    parser.add_argument('--coordination-penalty',type=float,default=10.)
    args=parser.parse_args(argv)
    if args.epochs<1 or not 4<=args.width<=256 or args.coordination_penalty<0:parser.error('训练参数范围不正确')
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    source=args.source_model_dir/'left_ring_q_to_motor_mlp.json';old=MLPFingerMapper(source,finger='ring').config
    if old['schema_version']!=11:raise ValueError('需要共享伸屈版本11来源')
    absolute,y,meta,provenance=load_data(args.data_root.resolve(),'ring')
    if len(y)!=old['training_points']or provenance!=old['source']:raise ValueError('原始实际反馈来源改变')
    q=delta_angles(absolute,old['q_reference']);paired=y[:,:2];seed=20261028
    report=args.report_dir or SCRIPTS/'motor_test_logs/ring_flexor_shared'/datetime.now().strftime('%Y%m%d_%H%M%S')
    report.mkdir(parents=True,exist_ok=False);(report/'previous_models').mkdir();(report/'models').mkdir()
    for path in args.source_model_dir.glob('left_*_q_to_motor_mlp.json'):
        shutil.copyfile(path,report/'previous_models'/path.name);shutil.copyfile(path,report/'models'/path.name)
    groups=pose_groups(q);folds=grouped_folds(groups,seed=seed);combos=np.array([r['combo_id']for r in meta]);ev={};predictions={}
    def fit(rows,fit_seed):
        return train_pair(q[rows],paired[rows],epochs=args.epochs,width=args.width,seed=fit_seed,
            coordination_penalty=args.coordination_penalty,convex_bend_inputs=(1,),flexor_convex_readout=True)[0]
    for scheme,labels in [('pose_group_5fold',folds),('leave_one_combo_out',combos)]:
        predicted=np.zeros_like(paired);details=[]
        for i,label in enumerate(np.unique(labels)):
            test=labels==label;head=fit(~test,seed+i);predicted[test]=forward(head,q[test])
            folder=report/'fold_models'/scheme;folder.mkdir(parents=True,exist_ok=True)
            (folder/f'{label}.json').write_text(json.dumps(head,indent=2)+'\n')
            details.append(dict(label=str(label),train=int((~test).sum()),test=int(test.sum()),epochs=args.epochs))
            print(f'{scheme} {label}: MAE {abs(predicted[test]-paired[test]).mean(0).round(4).tolist()}',flush=True)
        predictions[scheme]=predicted;ev[scheme]=dict(metrics(paired,predicted),folds=details)
    head=fit(np.ones(len(q),bool),seed);config=assemble(old,head)
    config['previous_joint_pair_training']=deepcopy(old['joint_pair_training'])
    config['previous_joint_pair_validation']=deepcopy(old['joint_pair_validation'])
    training=dict(epochs=args.epochs,seed=seed,width=args.width,optimizer='AdamW',lr=.01,weight_decay=0,
        input_weights=[.25,1.,1.,1.],pair_error_weight=.2,q0_first_layer_penalty=.015,q3_first_layer_penalty=0.,
        output_feature_coordination_penalty=args.coordination_penalty,convex_bend_inputs=[1],flexor_convex_readout=True,
        activation='mixed_softplus',trainable_motor_ids=[10,17],source=provenance,torch_version=str(torch.__version__),
        selection=f'Fixed {args.epochs}-epoch candidate after structural screening; grouped results used for model selection, not independent final test',
        no_added_samples=True,no_pseudo_labels=True,no_linear_baseline=True,no_runtime_contraction_offset=True,no_fixed_motor_ratio=True)
    config.update(mapping_revision='ring_shared_flexor_deep_contraction_20261028',joint_pair_training=training,joint_pair_validation=ev,
        joint_antagonistic_structure=dict(motor_ids=[10,17],architecture=[4,args.width,args.width,2],shared_hidden_layers=2,
            independent_hidden_layers=0,activation='mixed_softplus',bending_monotonic_inputs=[1,2,3],
            flexor_convex_bend_inputs=[1],
            feature_connections='First-layer concave units omit q1; M17 reads second-layer Softplus units; M10 can read all shared features',
            guarantee='M17 winding is nondecreasing and has a nondecreasing slope along q1 at fixed other inputs. No amplitude or tension guarantee.'),
        preserved_outputs=dict(motor_ids=[22,21],source_model_sha256=sha(source),note='All side networks, side normalization, 2deg policy, reference and controller limits copied'),
        training=dict(architecture_type='shared_antagonistic_pair_mlp',updated_motor_ids=[10,17],architecture=[4,args.width,args.width,2],details='joint_pair_training; side_bending_refinement_training retained'))
    for scheme,result in ev.items():
        for key,value in result.items():
            if key!='folds':config['validation'][scheme]['mlp'][key][:2]=value
    config['control_policy_validation_note']='New M10/M17 held-out raw-feedback metrics fit each outer training fold only. Architecture selection is not independent final testing. Side metrics inherited; no hardware tension claim.'
    path=report/'models'/source.name;path.write_text(json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    MLPFingerMapper(path,finger='ring')
    probe=np.random.default_rng(28).uniform([-40,-50,-30,-50],[40,110,160,120],(5000,4))
    a,b=motor_forward(old,probe),motor_forward(config,probe);np.testing.assert_array_equal(a[:,2:],b[:,2:])
    for axis in(1,2,3):
        more=probe.copy();more[:,axis]+=1.;delta=motor_forward(config,more)-b
        assert delta[:,0].min()>=-1e-10 and delta[:,1].max()<=1e-10
    more=probe.copy();less=probe.copy();more[:,1]+=1;less[:,1]-=1
    curvature=motor_forward(config,more)[:,1]+motor_forward(config,less)[:,1]-2*b[:,1]
    assert curvature.max()<=1e-9
    np.testing.assert_array_equal(motor_forward(config,np.zeros(4)),np.zeros(4))
    summary=dict(training_points=len(q),source=provenance,previous_sha256=sha(report/'previous_models'/source.name),new_sha256=sha(path),
        training=training,validation=ev,old_mae={s:old['validation'][s]['mlp']['mae_motor_deg'][:2]for s in ev},
        new_mae={s:values['mae_motor_deg']for s,values in ev.items()},checks=dict(sides_max_difference=float(abs(a[:,2:]-b[:,2:]).max()),
        q1_flexor_curvature_max=float(curvature.max()),shared_network=True,zero_input_zero=True),hardware_started=False)
    (report/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (report/'head.json').write_text(json.dumps(head,indent=2)+'\n');(report/'point_log_snapshot.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(report/'validation_predictions.npz',q_absolute=absolute,q_delta=q,motor_feedback=y,pose_group=groups,pose_fold=folds,
        point_ids=[int(r['point_id'])for r in meta],**predictions)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True);print(f'Staged {path}',flush=True)


if __name__=='__main__':main()
