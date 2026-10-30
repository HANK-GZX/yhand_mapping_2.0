"""Independent M14 MLP: q1 primary, q0 secondary, q2/q3 very weak."""
from copy import deepcopy
import numpy as np
from mlp_q_to_motor import forward,validate_network
from role_weighted_training import export_network

INPUT_WEIGHTS=(.5,1.,.01,.01)
SENSITIVITY_PENALTIES=(.02,0.,.5,.5)


def train_thumb_opposition(q,y,*,epochs=2200,hidden=(64,32),seed=20261029,lr=.005,
        input_weights=INPUT_WEIGHTS,sensitivity_penalties=SENSITIVITY_PENALTIES,priority_penalty=.001):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    q=np.asarray(q,dtype=float);y=np.asarray(y,dtype=float)
    gains=np.asarray(input_weights,dtype=float);cost=np.asarray(sensitivity_penalties,dtype=float)
    if q.ndim!=2 or q.shape[1]!=4 or len(q)<2 or y.shape!=(len(q),1)or not np.isfinite(q).all()or not np.isfinite(y).all():
        raise ValueError('需要配对N×4角度与N×1原始M14反馈')
    if (epochs<1 or len(hidden)!=2 or any(type(w)is not int or not 4<=w<=256 for w in hidden)
            or gains.shape!=(4,)or not np.isfinite(gains).all()or not np.all((gains>0)&(gains<=1))
            or cost.shape!=(4,)or not np.isfinite(cost).all()or np.any(cost<0)
            or not np.isfinite(lr)or lr<=0 or not np.isfinite(priority_penalty)or priority_penalty<0):
        raise ValueError('对掌网络训练参数不正确')
    center,scale=q.mean(0),np.maximum(q.std(0),1e-8);yc,ys=y.mean(0),np.maximum(y.std(0),1e-8)
    tensor=lambda v:torch.as_tensor(v,dtype=torch.float32)
    x=tensor((q-center)/scale*gains).requires_grad_(bool(cost.any()));target=tensor((y-yc)/ys)
    net=nn.Sequential(nn.Linear(4,hidden[0]),nn.Tanh(),nn.Linear(hidden[0],hidden[1]),nn.Tanh(),nn.Linear(hidden[1],1))
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=0)
    gain_tensor=tensor(gains);cost_tensor=tensor(cost);weak_cost=tensor(1/gains**2-1)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True);x.grad=None;prediction=net(x)
        loss=(prediction-target).square().mean()+priority_penalty*(net[0].weight.square()*weak_cost).mean()
        if cost.any():
            derivative=torch.autograd.grad(prediction.sum(),x,create_graph=True)[0]*gain_tensor
            loss=loss+(derivative.square()*cost_tensor).sum(-1).mean()
        if not torch.isfinite(loss):raise ValueError('对掌训练损失不是有限值')
        loss.backward();optimizer.step()
    dense=export_network(net,center,scale,yc,ys);layers=dense.pop('layers')
    network=dict(dense,architecture='role_weighted_branches',branches=[dict(name='opposition',motor_ids=[14],
        output_indices=[0],input_weights=gains.tolist(),layers=layers)])
    validate_network(network,1)
    with torch.no_grad():expected=net(x).numpy()*ys+yc
    np.testing.assert_allclose(forward(network,q),expected,atol=3e-4,rtol=1e-5)
    return network


def replace_thumb_opposition(source,head):
    if source.get('finger')!='thumb'or source.get('motor_order')!=[4,6,7,8,14,27]:raise ValueError('需要拇指三组网络')
    validate_network(head,1)
    result=deepcopy(source);n=result['network']
    for key in('input_center','input_scale'):np.testing.assert_allclose(n[key],head[key],atol=1e-12,rtol=0)
    for key in('output_center','output_scale'):np.testing.assert_allclose([n[key][4]],head[key],atol=1e-12,rtol=0)
    matches=[i for i,b in enumerate(n['branches'])if b['output_indices']==[4]]
    if len(matches)!=1 or head['branches'][0]['output_indices']!=[0]or head['branches'][0].get('motor_ids')!=[14]:raise ValueError('需要M14单输出对掌网络')
    replacement=deepcopy(head['branches'][0]);replacement['output_indices']=[4];n['branches'][matches[0]]=replacement
    if replacement.get('thumb_monotonic_opposition'):result['schema_version']=14
    elif result.get('schema_version')==14:result['schema_version']=13
    result['input_priority']['per_motor_weights']['14']=list(replacement['input_weights'])
    result['input_priority']['note']='Thumb flex/ext q2/q3 priorities and side q0 priorities (including M8-only q1 regularization) preserved. M14 prioritizes q1, then q0, with minimal q2/q3 coupling.'
    result['mapping_revision']='thumb_M14_q1_primary_q0_secondary_20261029'
    result['opposition_motor_mapping']='M14 independent pure MLP: q1 primary, q0 secondary, very weak q2/q3; no fixed output compensation.'
    result['side_motor_mapping']=result['side_motor_mapping'].replace(' Flex/ext and M14 unchanged.','')
    result['raw_prediction_at_reference_motor_deg']=forward(n,np.zeros(4)).tolist()
    return result
