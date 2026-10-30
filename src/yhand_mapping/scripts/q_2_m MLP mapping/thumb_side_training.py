"""Shared M8/M27 neural side mapping, q0-primary and signed monotonic."""
from copy import deepcopy
import numpy as np
from monotonic_extensor_training import build_extensor
from mlp_q_to_motor import forward,validate_network


def side_forward_with_q1_derivative(net,x,q1_gain):
    """Training-only MLP JVP; derivative is w.r.t. standardized, ungained q1.

    Keeping this output-specific avoids changing M27's shared input gain.
    Autograd differentiates this expression into the same learned MLP weights.
    """
    import torch
    from torch.nn import functional as F
    value=x;derivative=torch.zeros_like(x);derivative[:,1]=q1_gain
    for i,layer in enumerate(net.layers):
        weight=layer.weight();value=F.linear(value,weight,layer.bias);derivative=F.linear(derivative,weight)
        if i<len(net.layers)-1:
            split=value.shape[-1]//2;bounded=torch.tanh(value[...,split:])
            derivative=torch.cat((torch.sigmoid(value[...,:split])*derivative[...,:split],
                (1-bounded.square())*derivative[...,split:]),dim=-1)
            value=torch.cat((F.softplus(value[...,:split]),bounded),dim=-1)
    return value*net.directions,derivative*net.directions


def train_thumb_side(q,y,*,epochs=2200,width=32,seed=20261029,secondary=.25,priority_penalty=.001,lr=.01,m8_q1_sensitivity_penalty=0.):
    import torch
    torch.manual_seed(seed)
    q=np.asarray(q,dtype=float);y=np.asarray(y,dtype=float)
    if q.ndim!=2 or q.shape[1]!=4 or y.shape!=(len(q),2)or len(q)<2 or not np.isfinite(q).all()or not np.isfinite(y).all():
        raise ValueError('需要配对N×4拇指角度和N×2原始M8/M27反馈')
    if epochs<1 or not 4<=width<=256 or not np.isfinite(secondary)or not 0<secondary<=1 or not np.isfinite(priority_penalty)or priority_penalty<0 or not np.isfinite(lr)or lr<=0 or not np.isfinite(m8_q1_sensitivity_penalty)or m8_q1_sensitivity_penalty<0:
        raise ValueError('拇指侧摆训练参数范围不正确')
    center,scale=q.mean(0),np.maximum(q.std(0),1e-8);yc,ys=y.mean(0),np.maximum(y.std(0),1e-8)
    gains=np.array([1.,secondary,secondary,secondary]);tensor=lambda v:torch.as_tensor(v,dtype=torch.float32)
    x,target=tensor((q-center)/scale*gains),tensor((y-yc)/ys)
    net=build_extensor(width,monotonic_inputs=(0,),hidden_activation='mixed_softplus_tanh',output_directions=(1,-1))
    with torch.no_grad():
        for axis in(1,2,3):net.layers[0].raw_weight[:,axis].normal_(0,.05)
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=0)
    weak_cost=tensor(1/gains**2-1)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True);weak=(net.layers[0].weight().square()*weak_cost).mean()
        if m8_q1_sensitivity_penalty:
            prediction,dq1=side_forward_with_q1_derivative(net,x,gains[1])
            loss=(prediction-target).square().mean()+priority_penalty*weak+m8_q1_sensitivity_penalty*dq1[:,0].square().mean()
        else:loss=(net(x)-target).square().mean()+priority_penalty*weak
        if not torch.isfinite(loss):raise ValueError('侧摆训练损失不是有限值')
        loss.backward();optimizer.step()
    layers=[dict(weight=layer.weight().detach().tolist(),bias=layer.bias.detach().tolist(),
        activation='identity'if i==2 else'mixed_softplus_tanh')for i,layer in enumerate(net.layers)]
    network=dict(architecture='role_weighted_branches',input_center=center.tolist(),input_scale=scale.tolist(),
        output_center=yc.tolist(),output_scale=ys.tolist(),branches=[dict(name='side',motor_ids=[8,27],
            output_indices=[0,1],input_weights=gains.tolist(),monotonic_inputs=[0],shared_thumb_side=True,
            output_directions=[1,-1],hidden_activation='mixed_softplus_tanh',layers=layers)])
    validate_network(network,2)
    with torch.no_grad():expected=net(x).numpy()*ys+yc
    np.testing.assert_allclose(forward(network,q),expected,atol=3e-4,rtol=1e-5)
    return network


def replace_thumb_side(source,head):
    if source.get('finger')!='thumb'or source['motor_order']!=[4,6,7,8,14,27]:raise ValueError('需要拇指三组网络')
    result=deepcopy(source);n=result['network']
    for key in('input_center','input_scale'):np.testing.assert_allclose(n[key],head[key],atol=1e-12,rtol=0)
    for key in('output_center','output_scale'):np.testing.assert_allclose(np.array(n[key])[[3,5]],head[key],atol=1e-12,rtol=0)
    matches=[i for i,b in enumerate(n['branches'])if b['output_indices']==[3,5]]
    if len(matches)!=1:raise ValueError('需要M8/M27共享侧摆来源')
    replacement=deepcopy(head['branches'][0]);replacement['output_indices']=[3,5];n['branches'][matches[0]]=replacement
    for mid in(8,27):result['input_priority']['per_motor_weights'][str(mid)]=list(replacement['input_weights'])
    result['input_priority']['note']='Thumb flex/ext retain q2/q3 priorities; side M8/M27 prioritize q0 with weaker q1/q2/q3 coupling. M14 is unchanged.'
    result.update(schema_version=14 if any(b.get('thumb_monotonic_opposition')for b in n['branches'])else 13,mapping_revision='thumb_q0_primary_monotonic_shared_side_20261029',
        raw_prediction_at_reference_motor_deg=forward(n,np.zeros(4)).tolist(),
        side_motor_mapping='M8/M27 share q0-primary signed-monotonic MLP: increasing q0 pays out M8 and winds M27; q1/q2/q3 weak unconstrained. Flex/ext and M14 unchanged.')
    return result
