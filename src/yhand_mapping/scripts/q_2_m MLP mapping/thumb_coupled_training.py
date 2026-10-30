"""Two true MLPs: restored thumb coupling, no polynomial or position pseudo-labels."""
from copy import deepcopy
import numpy as np
from monotonic_extensor_training import build_extensor
from mlp_q_to_motor import forward,validate_network
from thumb_side_training import replace_thumb_side
from thumb_opposition_training import replace_thumb_opposition

SIDE_GAINS=(1.,.5,1.,1.)
OPPOSITION_GAINS=(1.,1.,1.,.5)
# Direction priors inferred from linear's dominant couplings, not its positions.
DIRECTIONS=np.array([[1.,-1.,1.,-1.],[-1.,1.,-1.,1.],[-1.,1.,-1.,1.]])


def train_coupled_heads(q,y,*,runtime_q=None,epochs=2200,width=32,seed=20261029,direction_penalty=.2,lr=.01):
    import torch
    from torch import nn
    q=np.asarray(q,dtype=float);y=np.asarray(y,dtype=float)
    if q.ndim!=2 or q.shape[1]!=4 or len(q)<2 or y.shape!=(len(q),3)or not np.isfinite(q).all()or not np.isfinite(y).all():raise ValueError('需要N×4角度和N×3真实M8/M27/M14反馈')
    if epochs<1 or not 4<=width<=256 or not np.isfinite(direction_penalty)or direction_penalty<0 or not np.isfinite(lr)or lr<=0:raise ValueError('耦合MLP训练参数不正确')
    if runtime_q is not None:
        runtime_q=np.asarray(runtime_q,dtype=float)
        if runtime_q.ndim!=2 or runtime_q.shape[1]!=4 or not len(runtime_q)or not np.isfinite(runtime_q).all():raise ValueError('运行域角度必须为有限N×4输入，不含电机标签')
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    center,scale=q.mean(0),np.maximum(q.std(0),1e-8);yc,ys=y.mean(0),np.maximum(y.std(0),1e-8)
    tensor=lambda v:torch.as_tensor(v,dtype=torch.float32)
    class Coupled(nn.Module):
        def __init__(self):
            super().__init__();self.side=build_extensor(width,monotonic_inputs=(0,),hidden_activation='mixed_softplus',output_directions=(1,-1))
            self.opposition=build_extensor(width,monotonic_inputs=(1,),hidden_activation='mixed_softplus')
            self.register_buffer('side_gains',tensor(SIDE_GAINS));self.register_buffer('opp_gains',tensor(OPPOSITION_GAINS))
            with torch.no_grad():
                for net,axes in((self.side,(1,2,3)),(self.opposition,(0,2,3))):
                    for axis in axes:net.layers[0].raw_weight[:,axis].normal_(0,.05)
        def forward(self,x):return torch.cat((self.side(x*self.side_gains),self.opposition(x*self.opp_gains)),dim=-1)
    net=Coupled();x=tensor((q-center)/scale);target=tensor((y-yc)/ys)
    probes=q[rng.choice(len(q),min(256,len(q)),replace=False)]
    if runtime_q is not None:probes=np.concatenate((probes,runtime_q[rng.choice(len(runtime_q),min(256,len(runtime_q)),replace=False)]))
    p=tensor((probes-center)/scale).requires_grad_(direction_penalty>0);sign=tensor(DIRECTIONS)
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=0)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True);p.grad=None;loss=(net(x)-target).square().mean()
        for branch,gain in((net.side,SIDE_GAINS),(net.opposition,OPPOSITION_GAINS)):
            loss=loss+.0005*(branch.layers[0].weight().square()*tensor(1/np.array(gain)**2-1)).mean()
        if direction_penalty:
            pred=net(p);derivatives=torch.stack([torch.autograd.grad(pred[:,i].sum(),p,create_graph=True,retain_graph=True)[0]for i in range(3)],dim=1)
            loss=loss+direction_penalty*torch.relu(-derivatives*sign).square().mean()
        if not torch.isfinite(loss):raise ValueError('耦合MLP损失不是有限值')
        loss.backward();optimizer.step()
    def export(branch,indices,name,mids,gains):
        layers=[dict(weight=l.weight().detach().tolist(),bias=l.bias.detach().tolist(),activation='identity'if i==2 else'mixed_softplus')for i,l in enumerate(branch.layers)]
        b=dict(name=name,motor_ids=mids,output_indices=list(range(len(indices))),input_weights=list(gains),layers=layers,hidden_activation='mixed_softplus')
        if name=='side':b.update(shared_thumb_side=True,monotonic_inputs=[0],output_directions=[1,-1])
        else:b.update(thumb_monotonic_opposition=True,monotonic_inputs=[1],monotonic_direction=1)
        n=dict(architecture='role_weighted_branches',input_center=center.tolist(),input_scale=scale.tolist(),output_center=yc[indices].tolist(),output_scale=ys[indices].tolist(),branches=[b]);validate_network(n,len(indices));return n
    side=export(net.side,[0,1],'side',[8,27],SIDE_GAINS);opp=export(net.opposition,[2],'opposition',[14],OPPOSITION_GAINS)
    with torch.no_grad():expected=net(x).numpy()*ys+yc
    np.testing.assert_allclose(np.column_stack((forward(side,q),forward(opp,q))),expected,atol=4e-4,rtol=1e-5)
    return side,opp


def replace_coupled_heads(source,side,opposition):
    c=replace_thumb_opposition(replace_thumb_side(source,side),opposition)
    c['mapping_revision']='thumb_restored_coupling_linear_reference_20261029'
    c['input_priority'].pop('training_only_output_sensitivity',None)
    c['input_priority']['note']='Flex/ext preserved. Side restores q2/q3 coupling and q1 direction prior; M14 restores all-angle coupling with structural q1 monotonicity. Coupling sign penalties are training-only soft priors.'
    c['side_motor_mapping']='M8/M27 shared true MLP, structural q0 monotonic [+,-], restored q2/q3 coupling; soft direction priors follow dominant linear behavior.'
    c['opposition_motor_mapping']='M14 independent true MLP, structural q1 monotonic +; restored q0/q2/q3 coupling. Mixed Softplus paths support both signs without tanh-only saturation.'
    return c
