"""Fine-tune M10's 33 output parameters inside the existing shared ring MLP."""
from copy import deepcopy
import numpy as np


def pair_network(config):
    n=config['network']
    if config['schema_version']!=11 or config['finger']!='ring':
        raise ValueError('需要版本11无名指共享网络')
    pair=next(b for b in n['branches']if b.get('shared_antagonistic_pair'))
    return dict(architecture='role_weighted_branches',
        input_center=list(n['input_center']),input_scale=list(n['input_scale']),
        output_center=list(n['output_center'][:2]),output_scale=list(n['output_scale'][:2]),
        branches=[deepcopy(pair)])


def features(network,q):
    b=network['branches'][0]
    x=(np.asarray(q,dtype=float)-network['input_center'])/network['input_scale']*b['input_weights']
    for layer in b['layers'][:-1]:
        x=x@np.asarray(layer['weight']).T+layer['bias']
        if layer['activation']=='mixed_softplus':
            split=x.shape[-1]//2
            x=np.concatenate((np.logaddexp(0,x[...,:split]),-np.logaddexp(0,-x[...,split:])),axis=-1)
        else:raise ValueError('此入口只微调现有混合Softplus共享网络')
    return x


def tune_readout(network,q,y,*,epochs=1500,coordination_penalty=0.,validation=None,patience=180,
                 boundary_weight=4.):
    import torch
    from torch.nn import functional as F
    q=np.asarray(q,dtype=float);y=np.asarray(y,dtype=float)
    if q.ndim!=2 or q.shape[1]!=4 or len(q)<2 or not np.isfinite(q).all()or not np.isfinite(y).all():
        raise ValueError('训练需要有限值的原始四角度配对数据')
    if epochs<1 or coordination_penalty<0 or boundary_weight<1:raise ValueError('微调参数范围不正确')
    if y.shape!=(len(q),2):raise ValueError('标签必须为原始实际M10/M17配对反馈')
    tensor=lambda value:torch.as_tensor(value,dtype=torch.float64)
    x=tensor(features(network,q));target=tensor((y[:,0]-network['output_center'][0])/network['output_scale'][0])
    thresholds=np.quantile(q[:,[1,3]],.8,axis=0)
    boundary=(q[:,1]>=thresholds[0])|(q[:,3]>=thresholds[1])
    sample_weights=tensor(np.where(boundary,boundary_weight,1.));sample_weights/=sample_weights.mean()
    old=network['branches'][0]['layers'][-1]
    initial=np.asarray(old['weight'][0]);anchor=tensor(old['weight'][1])
    weight=torch.nn.Parameter(tensor(np.log(np.expm1(initial))))
    bias=torch.nn.Parameter(tensor(old['bias'][0]))
    optimizer=torch.optim.Adam([weight,bias],lr=.01)
    if validation is not None:
        vq,vy=validation;vx=tensor(features(network,vq))
        vy=tensor((np.asarray(vy)[:,0]-network['output_center'][0])/network['output_scale'][0])
        vq=np.asarray(vq)
        val_boundary=(vq[:,1]>=thresholds[0])|(vq[:,3]>=thresholds[1])
        val_weights=tensor(np.where(val_boundary,boundary_weight,1.));val_weights/=val_weights.mean()
    best,chosen,state=float('inf'),0,None
    for epoch in range(1,epochs+1):
        optimizer.zero_grad(set_to_none=True)
        w=F.softplus(weight)
        error=x@w+bias-target
        penalty=coordination_penalty*(w-anchor).square().mean()
        loss=(sample_weights*error.square()).mean()+penalty
        if not torch.isfinite(loss):raise ValueError('M10微调损失不是有限值')
        loss.backward();optimizer.step()
        if validation is not None:
            with torch.no_grad():
                w=F.softplus(weight)
                score=float((val_weights*(vx@w+bias-vy).square()).mean()+coordination_penalty*(w-anchor).square().mean())
            if score<best-1e-10:
                best,chosen,state=score,epoch,(weight.detach().clone(),bias.detach().clone())
            if epoch-chosen>=patience:break
    if validation is not None:
        with torch.no_grad():weight.copy_(state[0]);bias.copy_(state[1])
    else:chosen=epochs
    result=deepcopy(network)
    result['branches'][0]['layers'][-1]['weight'][0]=F.softplus(weight).detach().tolist()
    result['branches'][0]['layers'][-1]['bias'][0]=float(bias.detach())
    from mlp_q_to_motor import forward,validate_network
    validate_network(result,2)
    with torch.no_grad():expected=(x@F.softplus(weight)+bias).numpy()*network['output_scale'][0]+network['output_center'][0]
    np.testing.assert_allclose(forward(result,q)[:,0],expected,atol=1e-9,rtol=1e-12)
    np.testing.assert_array_equal(forward(result,q)[:,1],forward(network,q)[:,1])
    return result,chosen


def replace_readout(config,head):
    result=deepcopy(config);before=pair_network(config)
    # Permit exactly the M10 output weight row and bias, no other change.
    restored=deepcopy(head)
    for key in('weight','bias'):
        restored['branches'][0]['layers'][-1][key][0]=deepcopy(before['branches'][0]['layers'][-1][key][0])
    if restored!=before:raise ValueError('微调修改了共享隐藏层、M17或标准化参数')
    pair=next(b for b in result['network']['branches']if b.get('shared_antagonistic_pair'))
    for key in('weight','bias'):pair['layers'][-1][key][0]=deepcopy(head['branches'][0]['layers'][-1][key][0])
    return result
