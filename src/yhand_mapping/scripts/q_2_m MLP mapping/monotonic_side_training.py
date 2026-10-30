"""Pure-MLP side head: signed monotonic h(q0) plus independent c(q1,q2,q3)."""
import numpy as np

from monotonic_extensor_training import build_extensor


def train_side(q, y, *, epochs=2400, seed=20261026, width=32,
               validation=None, patience=120, direction=1):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    q=np.asarray(q,dtype=float);y=np.asarray(y,dtype=float).reshape(-1,1)
    center,scale=q.mean(0),np.maximum(q.std(0),1e-8)
    out_center,out_scale=y.mean(0),np.maximum(y.std(0),1e-8)
    gains=np.array([1.,.25,.25,.25])
    tensor=lambda value:torch.as_tensor(value,dtype=torch.float32)
    x,target=tensor((q-center)/scale*gains),tensor((y-out_center)/out_scale)

    class Side(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor=build_extensor(width,direction=direction,monotonic_inputs=(0,),active_inputs=(0,))
            self.bending=nn.Sequential(nn.Linear(4,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,1))
            nn.init.zeros_(self.bending[-1].weight);nn.init.zeros_(self.bending[-1].bias)
            self.register_buffer('bend_mask',torch.tensor([0.,1.,1.,1.]))

        def forward(self,value):
            return self.anchor(value)+self.bending(value*self.bend_mask)

    net=Side()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.01,weight_decay=0)
    if validation is not None:
        vq,vy=validation
        vx=tensor((np.asarray(vq)-center)/scale*gains)
        vy=tensor((np.asarray(vy).reshape(-1,1)-out_center)/out_scale)
    best,chosen,state=float('inf'),0,None
    for epoch in range(1,epochs+1):
        optimizer.zero_grad(set_to_none=True)
        penalty=.001*15*(net.bending[0].weight*net.bend_mask).square().mean()
        loss=(net(x)-target).square().mean()+penalty
        if not torch.isfinite(loss):raise ValueError('侧摆训练损失不是有限值')
        loss.backward();optimizer.step()
        if validation is not None:
            with torch.no_grad():score=float((net(vx)-vy).square().mean())
            if score<best-1e-6:
                best,chosen=score,epoch
                state={k:v.detach().clone()for k,v in net.state_dict().items()}
            if epoch-chosen>=patience:break
    if validation is not None:net.load_state_dict(state)
    else:chosen=epochs
    anchor=[dict(weight=layer.weight().detach().tolist(),bias=layer.bias.detach().tolist(),
        activation='identity' if i==len(net.anchor.layers)-1 else 'mixed_softplus')for i,layer in enumerate(net.anchor.layers)]
    bending=[dict(weight=layer.weight.detach().tolist(),bias=layer.bias.detach().tolist(),activation='tanh')
        for layer in net.bending if isinstance(layer,nn.Linear)]
    bending[-1]['activation']='identity'
    for row in bending[0]['weight']:row[0]=0.
    network=dict(architecture='role_weighted_branches',input_center=center.tolist(),input_scale=scale.tolist(),
        output_center=out_center.tolist(),output_scale=out_scale.tolist(),branches=[dict(output_indices=[0],
        input_weights=gains.tolist(),monotonic_inputs=[0],monotonic_direction=direction,
        side_decomposition='q0_plus_bending',layers=anchor,bending_layers=bending)])
    from mlp_q_to_motor import forward,validate_network
    validate_network(network,1)
    with torch.no_grad():expected=net(x).numpy()*out_scale+out_center
    np.testing.assert_allclose(forward(network,q),expected,atol=3e-4,rtol=1e-5)
    return network,chosen
