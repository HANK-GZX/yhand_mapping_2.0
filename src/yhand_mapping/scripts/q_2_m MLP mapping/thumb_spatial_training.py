"""Thumb shared MLP and train-fold-only weighting of 3-D recorded workspaces."""
from copy import deepcopy
import numpy as np

from mlp_q_to_motor import forward,validate_network
from role_weighted_training import export_network

IDS=(4,6,7,8,14,27)
THUMB_GROUPS=(('flex_ext',(0,1,2)),('side',(3,5)),('opposition',(4,)))


def split_dense_thumb(network):
    """Copy a six-output dense MLP into three independent, equivalent MLPs."""
    if 'branches'in network:raise ValueError('初始化必须来自六输出共享MLP')
    validate_network(network,6)
    result={k:deepcopy(network[k])for k in('input_center','input_scale','output_center','output_scale')}
    result.update(architecture='role_weighted_branches',branches=[])
    for name,indices in THUMB_GROUPS:
        layers=deepcopy(network['layers']);layers[-1]['weight']=[layers[-1]['weight'][i]for i in indices];layers[-1]['bias']=[layers[-1]['bias'][i]for i in indices]
        result['branches'].append(dict(name=name,motor_ids=[IDS[i]for i in indices],output_indices=list(indices),input_weights=[1.]*4,layers=layers))
    validate_network(result,6)
    return result


def build_thumb_grouped(hidden=(32,32)):
    """Three independent feature trunks; paired/triplet muscles share a trunk."""
    import torch
    from torch import nn

    class GroupedThumb(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_indices=[list(indices)for _,indices in THUMB_GROUPS]
            self.register_buffer('input_weights',torch.ones(3,4))
            self.branches=nn.ModuleList([nn.Sequential(nn.Linear(4,hidden[0]),nn.Tanh(),
                nn.Linear(hidden[0],hidden[1]),nn.Tanh(),nn.Linear(hidden[1],len(indices)))
                for _,indices in THUMB_GROUPS])

        def forward(self,x):
            result=x.new_zeros((*x.shape[:-1],6))
            for branch,indices in zip(self.branches,self.output_indices):result[...,indices]=branch(x)
            return result

    return GroupedThumb()


def spatial_weights(y,combo,axis_motors,*,bins=5,edge_weight=1.5,density_power=.5):
    """Weight observed cells only; do not synthesize missing workspace labels."""
    y=np.asarray(y,dtype=float);combo=np.asarray(combo)
    if y.shape!=(len(combo),6)or not len(y)or not np.isfinite(y).all():raise ValueError('需要原始六电机反馈与组合标签')
    if type(bins)is not int or bins<2 or not np.isfinite(edge_weight)or edge_weight<1 or not 0<=density_power<=1:raise ValueError('空间权重参数范围不正确')
    weight=np.ones(len(y));details={};names=np.unique(combo)
    for name in names:
        rows=np.flatnonzero(combo==name);axes=axis_motors[str(name)]
        xyz=y[rows][:,[IDS.index(mid)for mid in axes]]
        lo,hi=xyz.min(0),xyz.max(0);unit=(xyz-lo)/np.maximum(hi-lo,1e-8)
        cells=np.minimum((unit*bins).astype(int),bins-1)
        _,inverse,count=np.unique(cells,axis=0,return_inverse=True,return_counts=True)
        local=count[inverse].astype(float)**(-density_power)
        edge=np.any((unit<.1)|(unit>.9),axis=1)
        local*=np.where(edge,edge_weight,1.)
        local/=local.mean()
        weight[rows]=local*len(y)/(len(names)*len(rows))
        details[str(name)]=dict(rows=len(rows),axis_motor_ids=list(axes),min=lo.tolist(),max=hi.tolist(),
            occupied_voxels=len(count),largest_voxel=int(count.max()),edge_rows=int(edge.sum()))
    # Clip the preliminary weights, then normalize. Export actual final bounds.
    weight=np.clip(weight,.25,4.);weight/=weight.mean()
    return weight,dict(groups=details,min=float(weight.min()),max=float(weight.max()),mean=float(weight.mean()),
        preliminary_clip=[.25,4.],bins=bins,density_power=density_power,edge_weight=edge_weight)


def train_thumb(q,y,combo,axis_motors,*,epochs=2500,seed=20261028,hidden=(64,32),weighted=True,
                huber=True,validation=None,patience=200,lr=.005,edge_weight=1.5,grouped=False,initial_network=None):
    """Learn recorded feedback; all normalization/weights use training rows only."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.manual_seed(seed)
    q=np.asarray(q,dtype=float);y=np.asarray(y,dtype=float);combo=np.asarray(combo)
    if q.shape!=(len(y),4)or y.shape[1:]!=(6,)or len(combo)!=len(y)or len(y)<2:
        raise ValueError('拇指训练需要配对N×4角度、N×6实际电机反馈与组合标签')
    if not np.isfinite(q).all()or not np.isfinite(y).all():raise ValueError('训练数据必须有限')
    if epochs<1 or len(hidden)!=2 or any(type(w)is not int or not 4<=w<=256 for w in hidden):raise ValueError('网络参数范围不正确')
    if not np.isfinite(lr)or lr<=0:raise ValueError('学习率必须为正')
    center,scale=q.mean(0),np.maximum(q.std(0),1e-8);out_center,out_scale=y.mean(0),np.maximum(y.std(0),1e-8)
    tensor=lambda v:torch.as_tensor(v,dtype=torch.float32)
    x,target=tensor((q-center)/scale),tensor((y-out_center)/out_scale)
    weights,weight_info=spatial_weights(y,combo,axis_motors,edge_weight=edge_weight)if weighted else(np.ones(len(y)),{'type':'uniform'})
    w=tensor(weights[:,None])
    net=build_thumb_grouped(hidden)if grouped else nn.Sequential(nn.Linear(4,hidden[0]),nn.Tanh(),nn.Linear(hidden[0],hidden[1]),nn.Tanh(),nn.Linear(hidden[1],6))
    if initial_network is not None:
        if not grouped:raise ValueError('当前仅支持三组网络从共享权重初始化')
        initial=split_dense_thumb(initial_network)
        for key,expected in [('input_center',center),('input_scale',scale),('output_center',out_center),('output_scale',out_scale)]:
            np.testing.assert_allclose(initial[key],expected,atol=1e-10,rtol=0)
        with torch.no_grad():
            for branch,saved in zip(net.branches,initial['branches']):
                linear=[layer for layer in branch if isinstance(layer,nn.Linear)]
                for layer,state in zip(linear,saved['layers']):
                    layer.weight.copy_(tensor(state['weight']));layer.bias.copy_(tensor(state['bias']))
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
    if validation is not None:
        vq,vy=validation;vx=tensor((np.asarray(vq)-center)/scale);vy=tensor((np.asarray(vy)-out_center)/out_scale)
    best,chosen,state=float('inf'),0,None
    for epoch in range(1,epochs+1):
        optimizer.zero_grad(set_to_none=True);pred=net(x)
        error=F.smooth_l1_loss(pred,target,reduction='none',beta=.5)if huber else(pred-target).square()
        loss=(w*error).mean()
        if not torch.isfinite(loss):raise ValueError('拇指训练损失不是有限值')
        loss.backward();optimizer.step()
        if validation is not None:
            with torch.no_grad():score=float((net(vx)-vy).square().mean())
            if score<best-1e-6:
                best,chosen=score,epoch;state=deepcopy(net.state_dict())
            if epoch-chosen>=patience:break
    if validation is not None:net.load_state_dict(state)
    else:chosen=epochs
    network=export_network(net,center,scale,out_center,out_scale)
    if grouped:
        for branch,(name,indices)in zip(network['branches'],THUMB_GROUPS):
            branch.update(name=name,motor_ids=[IDS[i]for i in indices])
    validate_network(network,6)
    with torch.no_grad():expected=net(x).numpy()*out_scale+out_center
    np.testing.assert_allclose(forward(network,q),expected,atol=2e-4,rtol=1e-5)
    return network,chosen,weight_info
