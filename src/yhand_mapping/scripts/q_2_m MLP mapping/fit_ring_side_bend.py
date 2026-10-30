#!/usr/bin/env python3
"""Stage M22 bending-only refinement; preserve ring shared pair, M21 and q0."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import numpy as np

from fit_left_hand_q_to_motor_mlp import SCRIPTS,load_data,delta_angles,pose_groups,grouped_folds,metrics,sha
from mlp_q_to_motor import MODEL_DIR,MLPFingerMapper,forward,motor_forward
from monotonic_side_training import train_side
from ring_side_bend_training import side_network,fit_bending,replace_bending


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model-dir',type=Path,default=MODEL_DIR)
    parser.add_argument('--data-root',type=Path,default=SCRIPTS/'motor_test_logs/data_collection')
    parser.add_argument('--report-dir',type=Path)
    parser.add_argument('--epochs',type=int,default=1500)
    parser.add_argument('--width',type=int,default=8)
    parser.add_argument('--q3-penalty',type=float,default=.01)
    args=parser.parse_args(argv)
    if args.epochs<1 or not 2<=args.width<=256 or args.q3_penalty<0:parser.error('训练参数范围不正确')
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    path=args.source_model_dir/'left_ring_q_to_motor_mlp.json';old=MLPFingerMapper(path,finger='ring').config
    if old['schema_version']!=11 or 'side_bending_refinement_training'in old:raise ValueError('需要尚未做本轮侧摆微调的版本11模型')
    absolute,y,meta,provenance=load_data(args.data_root.resolve(),'ring');q=delta_angles(absolute,old['q_reference'])
    if len(q)!=old['training_points']or provenance!=old['source']:raise ValueError('原始训练数据改变')
    report=args.report_dir or SCRIPTS/'motor_test_logs/ring_side_bend'/datetime.now().strftime('%Y%m%d_%H%M%S')
    report.mkdir(parents=True,exist_ok=False);(report/'previous_models').mkdir();(report/'models').mkdir()
    for source in args.source_model_dir.glob('left_*_q_to_motor_mlp.json'):
        shutil.copyfile(source,report/'previous_models'/source.name);shutil.copyfile(source,report/'models'/source.name)
    groups=pose_groups(q);base_seed=old['side_training']['22']['seed'];folds=grouped_folds(groups,seed=base_seed)
    labels=np.array([r['combo_id']for r in meta]);evaluation={};predictions={};baseline={}
    for scheme,outer in [('pose_group_5fold',folds),('leave_one_combo_out',labels)]:
        predicted=np.zeros((len(q),1));before=np.zeros_like(predicted);details=[]
        records=old['side_validation']['22'][scheme]['folds']
        for i,label in enumerate(np.unique(outer)):
            record=records[i]
            if record['label']!=str(label):raise ValueError('来源验证分折不匹配')
            test=outer==label
            # Rebuild q0 ONLY on outer training rows; a deployed full-data q0
            # network must never be used to estimate held-out accuracy.
            base,_=train_side(q[~test],y[~test,2:3],epochs=record['inner_selected_epochs'],seed=base_seed+i,direction=-1)
            head=fit_bending(base,q[~test],y[~test,2:3],epochs=args.epochs,width=args.width,seed=20261028+i,q3_penalty=args.q3_penalty)
            predicted[test]=forward(head,q[test]);before[test]=forward(base,q[test])
            folder=report/'fold_models'/scheme/str(label);folder.mkdir(parents=True)
            for name,n in [('base',base),('refined',head)]:
                (folder/f'{name}.json').write_text(json.dumps(n,indent=2)+'\n')
            details.append(dict(label=str(label),train=int((~test).sum()),test=int(test.sum()),base_epochs=record['inner_selected_epochs'],bend_epochs=args.epochs))
            print(f'{scheme} {label}: M22 MAE {abs(before[test]-y[test,2:3]).mean():.3f} -> {abs(predicted[test]-y[test,2:3]).mean():.3f}',flush=True)
        evaluation[scheme]=dict(metrics(y[:,2:3],predicted),folds=details);predictions[scheme]=predicted;baseline[scheme]=before
    head=fit_bending(side_network(old),q,y[:,2:3],epochs=args.epochs,width=args.width,q3_penalty=args.q3_penalty)
    config=replace_bending(old,head)
    training=dict(motor_id=22,epochs=args.epochs,width=args.width,seed=20261028,optimizer='Adam',lr=.01,
        q3_first_layer_penalty=args.q3_penalty,architecture=[4,args.width,args.width,1],active_inputs=[1,2,3],
        monotonic_inputs=[1],activation='mixed_softplus_tanh',frozen_q0_network=True,frozen_normalization=True,
        preserved_motor_ids=[10,17,21],no_added_samples=True,no_pseudo_labels=True,no_linear_baseline=True,
        validation_note='Candidate-selection validation; each base q0 rebuilt on outer training rows. Not an independent final test.',source=provenance)
    config.update(mapping_revision='ring_m22_bending_mlp_20261028',side_bending_refinement_training=training,
        side_bending_refinement_validation=evaluation,
        preserved_outputs=dict(motor_ids=[10,17,21],source_model_sha256=sha(path),note='M22 q0 network, all normalization, zero reference and 2deg policy also frozen'),
        training=dict(architecture_type='shared_pair_and_separable_sides',updated_motor_ids=[22],details='Only M22 bending MLP updated; see side_bending_refinement_training'))
    config['side_structure']['bending_hidden_activations']['22']='mixed_softplus_tanh'
    config['side_structure']['m22_bending_monotonic_inputs']=[1]
    config['side_structure']['raw_prediction']='signed_monotonic_h(q0) + learned_c(q1,q2,q3); M22 c is nondecreasing in q1'
    config['side_training_scope']='Base q0/side fit; current M22 bending-only stage is in side_bending_refinement_training'
    config['control_policy_validation_note']='M22 raw-feedback candidate-selection validation; each frozen q0 rebuilt on outer training rows. Other muscle metrics inherited. No robot angle or tension accuracy claim.'
    config['raw_prediction_at_reference_motor_deg']=forward(config['network'],np.zeros(4)).tolist()
    for scheme,result in evaluation.items():
        for key,value in result.items():
            if key!='folds':config['validation'][scheme]['mlp'][key][2]=value[0]
    candidate=report/'models'/path.name;candidate.write_text(json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    MLPFingerMapper(candidate,finger='ring')
    probe=np.random.default_rng(28).uniform([-40,-50,-30,-50],[40,110,160,120],(5000,4))
    a,b=motor_forward(old,probe),motor_forward(config,probe);np.testing.assert_array_equal(a[:,[0,1,3]],b[:,[0,1,3]])
    anchor=np.zeros_like(probe);anchor[:,0]=probe[:,0]
    np.testing.assert_allclose(motor_forward(old,anchor),motor_forward(config,anchor),atol=1e-12,rtol=0)
    more=probe.copy();more[:,1]+=.5;assert (motor_forward(config,more)[:,2]-b[:,2]).min()>=-1e-10
    np.testing.assert_array_equal(motor_forward(config,np.zeros(4)),np.zeros(4))
    summary=dict(training=training,training_points=len(q),source=provenance,previous_sha256=sha(report/'previous_models'/path.name),new_sha256=sha(candidate),
        old_mae={s:metrics(y[:,2:3],v)['mae_motor_deg'][0]for s,v in baseline.items()},new_mae={s:v['mae_motor_deg'][0]for s,v in evaluation.items()},
        checks=dict(preserved_m10_m17_m21_max_difference=float(abs(a[:,[0,1,3]]-b[:,[0,1,3]]).max()),q0_network_preserved=True,zero_target=True),
        limits_changed=False,hardware_started=False)
    (report/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');(report/'head.json').write_text(json.dumps(head,indent=2)+'\n')
    np.savez_compressed(report/'validation_predictions.npz',q_delta=q,motor_feedback=y,pose_group=groups,pose_fold=folds,**predictions,**{f'base_{s}':v for s,v in baseline.items()})
    (report/'point_log_snapshot.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True);print(f'Staged {candidate}',flush=True)


if __name__=='__main__':main()
