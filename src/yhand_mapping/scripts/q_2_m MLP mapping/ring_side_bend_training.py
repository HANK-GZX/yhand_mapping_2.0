"""Fit only M22's bending MLP against real feedback with its q0 MLP frozen."""
from copy import deepcopy
import numpy as np

from mlp_q_to_motor import forward,validate_network
from monotonic_extensor_training import build_extensor


def side_network(config,index=2):
    n=deepcopy(config['network']);branch=next(b for b in n['branches']if b['output_indices']==[index])
    branch['output_indices']=[0];n['branches']=[branch]
    for key in('output_center','output_scale'):n[key]=[n[key][index]]
    return n


def fit_bending(network,q,y,*,epochs=1500,seed=20261028,width=8,q3_penalty=.01):
    import torch
    torch.manual_seed(seed)
    q=np.asarray(q,dtype=float);y=np.asarray(y,dtype=float).reshape(-1,1)
    if q.shape!=(len(y),4)or len(q)<2 or not np.isfinite(q).all()or not np.isfinite(y).all():
        raise ValueError('需要有限值的原始角度与实际反馈配对')
    if epochs<1 or not 2<=width<=256 or q3_penalty<0:raise ValueError('训练参数范围不正确')
    branch=network['branches'][0]
    anchor=deepcopy(network);anchor['branches'][0].pop('bending_layers')
    tensor=lambda v:torch.as_tensor(v,dtype=torch.float32)
    x=tensor((q-network['input_center'])/network['input_scale']*branch['input_weights'])
    # This is the existing learned q0 network, not a polynomial baseline or label.
    h=tensor((forward(anchor,q)-network['output_center'])/network['output_scale'])
    target=tensor((y-network['output_center'])/network['output_scale'])
    net=build_extensor(width,monotonic_inputs=(1,),active_inputs=(1,2,3),hidden_activation='mixed_softplus_tanh')
    optimizer=torch.optim.Adam(net.parameters(),lr=.01)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss=(h+net(x)-target).square().mean()+q3_penalty*net.layers[0].weight()[:,3].square().mean()
        if not torch.isfinite(loss):raise ValueError('侧摆弯曲训练损失不是有限值')
        loss.backward();optimizer.step()
    result=deepcopy(network);b=result['branches'][0]
    b.update(bending_monotonic_inputs=[1],bending_hidden_activation='mixed_softplus_tanh',
        bending_layers=[dict(weight=l.weight().detach().tolist(),bias=l.bias.detach().tolist(),
            activation='identity'if i==2 else'mixed_softplus_tanh')for i,l in enumerate(net.layers)])
    validate_network(result,1)
    with torch.no_grad():expected=(h+net(x)).numpy()*network['output_scale']+network['output_center']
    np.testing.assert_allclose(forward(result,q),expected,atol=3e-4,rtol=1e-5)
    return result


def replace_bending(config,head):
    old=side_network(config);check=deepcopy(head)
    for key in('bending_layers','bending_hidden_activation','bending_monotonic_inputs'):
        if key in old['branches'][0]:check['branches'][0][key]=deepcopy(old['branches'][0][key])
        else:check['branches'][0].pop(key,None)
    if check!=old:raise ValueError('只能修改M22弯曲网络，q0网络和标准化必须保持')
    validate_network(head,1)
    result=deepcopy(config);branch=next(b for b in result['network']['branches']if b['output_indices']==[2])
    for key in('bending_layers','bending_hidden_activation','bending_monotonic_inputs'):branch[key]=deepcopy(head['branches'][0][key])
    return result
