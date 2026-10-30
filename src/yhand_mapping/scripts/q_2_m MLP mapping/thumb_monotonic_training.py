"""One genuine shared M4/M6/M7 MLP, signed-monotone in thumb q2/q3."""
from copy import deepcopy
import numpy as np

from monotonic_extensor_training import build_extensor
from mlp_q_to_motor import forward,validate_network


def train_thumb_flex_ext(q,y,*,epochs=2200,width=32,seed=20261029,secondary=.25,
                         priority_penalty=.001,lr=.01):
    import torch
    torch.manual_seed(seed)
    q=np.asarray(q,dtype=float);y=np.asarray(y,dtype=float)
    if q.ndim!=2 or q.shape[1]!=4 or y.shape!=(len(q),3)or len(q)<2 or not np.isfinite(q).all()or not np.isfinite(y).all():
        raise ValueError('需要配对的N×4拇指角度与N×3原始M4/M6/M7反馈')
    if epochs<1 or not 4<=width<=256 or not 0<secondary<=1 or priority_penalty<0 or not np.isfinite(lr)or lr<=0:
        raise ValueError('拇指单调训练参数范围不正确')
    center,scale=q.mean(0),np.maximum(q.std(0),1e-8);yc,ys=y.mean(0),np.maximum(y.std(0),1e-8)
    gains=np.array([secondary,secondary,1.,1.]);tensor=lambda v:torch.as_tensor(v,dtype=torch.float32)
    x,target=tensor((q-center)/scale*gains),tensor((y-yc)/ys)
    net=build_extensor(width,monotonic_inputs=(2,3),hidden_activation='mixed_softplus_tanh',output_directions=(-1,1,1))
    with torch.no_grad():
        for axis in(0,1):net.layers[0].raw_weight[:,axis].normal_(0,.05)
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=0)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        weak=(net.layers[0].weight().square()*tensor(1/gains**2-1)).mean()
        loss=(net(x)-target).square().mean()+priority_penalty*weak
        if not torch.isfinite(loss):raise ValueError('拇指训练损失不是有限值')
        loss.backward();optimizer.step()
    layers=[dict(weight=layer.weight().detach().tolist(),bias=layer.bias.detach().tolist(),
        activation='identity'if i==2 else'mixed_softplus_tanh')for i,layer in enumerate(net.layers)]
    network=dict(architecture='role_weighted_branches',input_center=center.tolist(),input_scale=scale.tolist(),
        output_center=yc.tolist(),output_scale=ys.tolist(),branches=[dict(name='flex_ext',motor_ids=[4,6,7],
            output_indices=[0,1,2],input_weights=gains.tolist(),monotonic_inputs=[2,3],
            shared_thumb_flex_ext=True,output_directions=[-1,1,1],hidden_activation='mixed_softplus_tanh',layers=layers)])
    validate_network(network,3)
    with torch.no_grad():expected=net(x).numpy()*ys+yc
    np.testing.assert_allclose(forward(network,q),expected,atol=3e-4,rtol=1e-5)
    return network


def replace_thumb_flex_ext(source,head):
    if source.get('finger')!='thumb'or source['motor_order']!=[4,6,7,8,14,27]:raise ValueError('仅支持拇指三组网络')
    result=deepcopy(source);n=result['network']
    for key in('input_center','input_scale'):np.testing.assert_allclose(n[key],head[key],atol=1e-12,rtol=0)
    for key in('output_center','output_scale'):np.testing.assert_allclose(n[key][:3],head[key],atol=1e-12,rtol=0)
    matches=[i for i,b in enumerate(n['branches'])if b['output_indices']==[0,1,2]]
    if len(matches)!=1:raise ValueError('来源需要一个共享M4/M6/M7网络')
    n['branches'][matches[0]]=deepcopy(head['branches'][0])
    for mid in(4,6,7):result['input_priority']['per_motor_weights'][str(mid)]=list(head['branches'][0]['input_weights'])
    result['input_priority']['note']='Thumb M4/M6/M7 prioritize q2/q3; q0/q1 remain weak unconstrained inputs. Other groups retain their original input gains.'
    result.update(schema_version=15 if any(b.get('shared_thumb_coordination')for b in n['branches'])else 14 if any(b.get('thumb_monotonic_opposition')for b in n['branches'])else 13 if any(b.get('shared_thumb_side')for b in n['branches'])else 12,mapping_revision='thumb_q23_priority_monotonic_flex_ext_20261029',
        raw_prediction_at_reference_motor_deg=forward(n,np.zeros(4)).tolist(),
        thumb_joint_semantics=dict(q0='angle relative to palm plane',q1='angle relative to the plane perpendicular to palm',
            q2='proximal thumb flexion',q3='distal thumb flexion',note='User-defined meanings; existing delta reference retained'),
        side_motor_mapping='M8/M27 unchanged shared MLP; M4/M6/M7 shared MLP prioritizes q2/q3 with [-1,+1,+1] monotonic directions; M14 unchanged independent MLP')
    return result
