#!/usr/bin/env python3
"""Train side MLPs with finger-specific q0 directions and separate bend effects."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np

from fit_left_hand_q_to_motor_mlp import SCRIPTS,SEED,load_data,delta_angles,sha
from fit_monotonic_extensor import evaluate
from mlp_q_to_motor import MODEL_DIR,MLPFingerMapper,forward,SIDE_MOTORS,SIDE_DIRECTIONS
from monotonic_side_training import train_side


def assemble_side_model(previous, heads, results):
    side_ids=SIDE_MOTORS[previous['finger']]
    directions=SIDE_DIRECTIONS[previous['finger']]
    side_start=len(previous['motor_order'])-2
    if set(heads)!=set(side_ids):raise ValueError('必须同时提供该手指两条侧摆头')
    config=deepcopy(previous);network=config['network']
    history={key:config.pop(key)for key in('side_bending_refinement_training','side_bending_refinement_validation')if key in config}
    if history:config['previous_side_bending_refinement_metadata']=history
    preserved=[deepcopy(b)for b in network['branches'] if all(i<side_start for i in b['output_indices'])]
    if sorted(i for b in preserved for i in b['output_indices'])!=list(range(side_start)):
        raise ValueError('来源必须完整覆盖伸屈输出')
    for index,mid in enumerate(side_ids,start=side_start):
        head=heads[mid]
        for key in ('input_center','input_scale'):
            np.testing.assert_allclose(network[key],head[key],atol=1e-12,rtol=1e-12)
        branch=deepcopy(head['branches'][0]);branch['output_indices']=[index]
        preserved.append(branch)
        for key in ('output_center','output_scale'):network[key][index]=head[key][0]
    network['branches']=preserved
    has_shallow=any(b['output_indices']==[1] and b.get('monotonic_inputs')==[1,2] for b in preserved)
    config['schema_version']=(10 if has_shallow else 9) if previous['finger']=='middle' else (11 if any(b.get('shared_antagonistic_pair')for b in preserved)else 7)
    config['side_structure']=dict(type='separable_monotonic_q0_plus_bending',
        motor_directions=dict(zip(map(str,side_ids),directions)),anchor_input_indices=[0],bending_input_indices=[1,2,3],
        input_weights=[1,.25,.25,.25],
        raw_prediction='signed_monotonic_h(q0) + unconstrained_c(q1,q2,q3)',
        aligned_target='h(q0)-h(0) + existing_bounded_winding_policy(c(bend)-c(0))',
        guarantee='Signed q0 monotonicity follows motor_directions, including after the side policy. '
                  'Bending correction is independent of q0. No amplitude or hardware tracking guarantee.')
    config['side_structure']['bending_hidden_activations']={str(mid):heads[mid]['branches'][0]['bending_layers'][0]['activation']for mid in side_ids}
    config['side_training']={str(mid):results[mid]['training']for mid in side_ids}
    config['side_validation']={str(mid):results[mid]['validation']for mid in side_ids}
    for index,mid in enumerate(side_ids,start=side_start):
        for scheme,values in results[mid]['validation'].items():
            for metric,value in values.items():
                if metric!='folds':config['validation'][scheme]['mlp'][metric][index]=value[0]
    config['raw_prediction_at_reference_motor_deg']=forward(network,np.zeros(4)).tolist()
    config['training']=dict(architecture_type='role_weighted_branches_with_separable_monotonic_sides',
        updated_motor_ids=list(side_ids),monotonic_motors=[previous['motor_order'][i]for b in preserved if 'monotonic_inputs'in b for i in b['output_indices']],
        side_head_architecture='q0 and bending:4→W→W→1 with disjoint input masks; per-motor activations in side_structure',
        validation_pose_bin_deg=2.,note='Per-side training is in side_training; extensor/flexor retain earlier training fields.')
    config['control_policy_validation_note']=('Side held-out metrics evaluate raw measured feedback predictions, '
        'before neutral alignment and the side winding cap, not hardware accuracy. Extensor/flexor metrics are inherited.')
    return config


def fit_one(mid,args,folder,previous):
    folder.mkdir(parents=True,exist_ok=True)
    finger=getattr(args,'finger','little');side_ids=SIDE_MOTORS[finger]
    absolute,labels,meta,source=load_data(args.data_root.resolve(),finger)
    if len(absolute)!=previous['training_points'] or source!=previous['source']:
        raise ValueError('原采集数据来源或数量改变')
    q=delta_angles(absolute,previous['q_reference']);index=previous['motor_order'].index(mid)
    direction=SIDE_DIRECTIONS[finger][side_ids.index(mid)];y=labels[:,index:index+1]
    combo=np.array([r['combo_id']for r in meta])
    print(f'M{mid}: {len(q)} original samples, q0 monotonic direction={direction}',flush=True)
    evaluation,predictions,epochs,groups,folds=evaluate(q,y,combo,epochs=args.epochs,
        width=args.width,seed=args.seed,direction=direction,trainer=train_side)
    head,_=train_side(q,y,epochs=epochs,width=args.width,seed=args.seed,direction=direction)
    training=dict(seed=args.seed,epochs=epochs,maximum_selection_epochs=args.epochs,width=args.width,
        optimizer='AdamW',lr=.01,weight_decay=0,early_stop_patience=120,input_priority_penalty=.001,
        selection='Median inner-selected epochs from five outer pose folds; final fit on original rows',
        torch_version=args.torch_version,no_added_samples=True,no_pseudo_labels=True,no_linear_baseline=True,
        input_definition=previous['training_input_definition'],target_definition=previous['training_target_definition'])
    result=dict(motor_id=mid,training_points=len(q),direction=direction,training=training,validation=evaluation,
        old_mae={s:previous['validation'][s]['mlp']['mae_motor_deg'][index]for s in evaluation},
        new_mae={s:evaluation[s]['mae_motor_deg'][0]for s in evaluation})
    (folder/'head.json').write_text(json.dumps(head,ensure_ascii=False,indent=2)+'\n')
    (folder/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (folder/'point_log_snapshot.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(folder/'validation_predictions.npz',q_absolute=absolute,q_delta=q,
        q_reference=previous['q_reference'],motor_feedback=labels,point_ids=[int(r['point_id'])for r in meta],
        combo=combo,pose_group=groups,pose_fold=folds,**predictions)
    print(f'M{mid}: old MAE {result["old_mae"]}; new MAE {result["new_mae"]}',flush=True)
    return head,result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finger',choices=['little','ring','middle'],default='little')
    parser.add_argument('--motor',choices=['18','20','22','21','9','24','both'],default='both')
    parser.add_argument('--source-model-dir',type=Path,default=MODEL_DIR)
    parser.add_argument('--data-root',type=Path,default=SCRIPTS/'motor_test_logs/data_collection')
    parser.add_argument('--report-dir',type=Path)
    parser.add_argument('--epochs',type=int,default=2400)
    parser.add_argument('--width',type=int,default=32)
    parser.add_argument('--seed',type=int,default=SEED)
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args(argv)
    if args.epochs<1 or not 2<=args.width<=256 or args.threads<1:parser.error('训练参数错误')
    import torch
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    args.torch_version=str(torch.__version__)
    side_ids=SIDE_MOTORS[args.finger]
    if args.motor!='both' and int(args.motor) not in side_ids:parser.error('电机不属于所选手指的侧摆')
    source=args.source_model_dir/f'left_{args.finger}_q_to_motor_mlp.json'
    previous=MLPFingerMapper(source,finger=args.finger).config
    if previous['schema_version'] not in (5,6,7,8,9,10,11):raise ValueError('需要单调伸肌来源模型')
    if args.motor=='both' and previous['schema_version']==5:parser.error('完整组装需要先训练单调屈肌；也可先单独导出侧摆头')
    report=args.report_dir or SCRIPTS/'motor_test_logs/q_to_motor_monotonic_side'/datetime.now().strftime('%Y%m%d_%H%M%S')
    report.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source,report/'previous_model.json')
    heads,results={},{}
    for mid in (side_ids if args.motor=='both' else(int(args.motor),)):
        folder=report/f'M{mid}' if args.motor=='both' else report
        heads[mid],results[mid]=fit_one(mid,args,folder,previous)
    if args.motor=='both':
        config=assemble_side_model(previous,heads,results)
        config['preserved_outputs']=dict(motor_ids=previous['motor_order'][:-2],source_model_sha256=sha(source),source_training=previous['training'])
        models=report/'models';models.mkdir()
        for path in args.source_model_dir.glob('left_*_q_to_motor_mlp.json'):shutil.copyfile(path,models/path.name)
        path=models/source.name;path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
        MLPFingerMapper(path,finger=args.finger)
    print(f'Report: {report}; no live model was replaced',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
