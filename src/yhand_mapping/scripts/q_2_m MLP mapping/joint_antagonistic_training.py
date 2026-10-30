"""One shared 4-W-W-2 MLP for an antagonistic pair, with measured paired labels."""
import numpy as np
from monotonic_extensor_training import build_extensor


def train_pair(q,y,*,epochs=3000,width=32,seed=20261028,validation=None,
               patience=150,pair_weight=.2,q3_penalty=0.,q3_gain=1.,activation='mixed_softplus',
               coordination_penalty=10.,convex_bend_inputs=None,flexor_convex_readout=False):
    import torch
    torch.manual_seed(seed)
    q=np.asarray(q,dtype=float);y=np.asarray(y,dtype=float)
    if q.ndim!=2 or q.shape[1]!=4 or y.shape!=(len(q),2):
        raise ValueError('联合伸屈训练需要N×4输入和同一行N×2实际反馈标签')
    if not np.isfinite(q).all()or not np.isfinite(y).all():raise ValueError('输入/标签不是有限值')
    center,scale=q.mean(0),np.maximum(q.std(0),1e-8)
    out_center,out_scale=y.mean(0),np.maximum(y.std(0),1e-8)
    gains=np.array([.25,1.,1.,q3_gain])
    tensor=lambda value:torch.as_tensor(value,dtype=torch.float32)
    x=tensor((q-center)/scale*gains);target=tensor((y-out_center)/out_scale)
    # Joint statistics come from this training fold's actual paired feedback.
    # They weight correlated prediction errors, never set a fixed motor ratio.
    corr=np.cov(((y-out_center)/out_scale).T,bias=True)
    precision=np.linalg.inv(corr+.02*np.eye(2));precision/=np.trace(precision)/2
    pair_precision=tensor(precision)
    net=build_extensor(width,hidden_activation=activation,output_directions=(1,-1),
        convex_bend_inputs=convex_bend_inputs,flexor_convex_readout=flexor_convex_readout)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.01,weight_decay=0)
    def criterion(pred,labels):
        error=pred-labels
        return error.square().mean()+pair_weight*((error@pair_precision)*error).sum(1).mean()
    if validation is not None:
        vq,vy=validation;vx=tensor((np.asarray(vq)-center)/scale*gains)
        vy=tensor((np.asarray(vy)-out_center)/out_scale)
    best,chosen,state=float('inf'),0,None
    for epoch in range(1,epochs+1):
        optimizer.zero_grad(set_to_none=True)
        first=net.layers[0].weight()
        penalty=.015*first[:,0].square().mean()+q3_penalty*first[:,3].square().mean()
        last=net.layers[-1].weight()
        penalty=penalty+coordination_penalty*(last[0]-last[1]).square().mean()
        loss=criterion(net(x),target)+penalty
        if not torch.isfinite(loss):raise ValueError('联合伸屈损失不是有限值')
        loss.backward();optimizer.step()
        if validation is not None:
            with torch.no_grad():
                score=float(criterion(net(vx),vy)+coordination_penalty*(net.layers[-1].weight()[0]-net.layers[-1].weight()[1]).square().mean())
            if score<best-1e-6:
                best,chosen=score,epoch
                state={k:v.detach().clone()for k,v in net.state_dict().items()}
            if epoch-chosen>=patience:break
    if validation is not None:net.load_state_dict(state)
    else:chosen=epochs
    layers=[dict(weight=layer.weight().detach().tolist(),bias=layer.bias.detach().tolist(),
                 activation='identity'if i==len(net.layers)-1 else activation)
            for i,layer in enumerate(net.layers)]
    n=dict(architecture='role_weighted_branches',input_center=center.tolist(),input_scale=scale.tolist(),
        output_center=out_center.tolist(),output_scale=out_scale.tolist(),branches=[dict(output_indices=[0,1],
            input_weights=gains.tolist(),monotonic_inputs=[1,2,3],output_directions=[1,-1],
            shared_antagonistic_pair=True,hidden_activation=activation,layers=layers)])
    if convex_bend_inputs is not None:n['branches'][0]['convex_bend_inputs']=list(convex_bend_inputs)
    if flexor_convex_readout:n['branches'][0]['flexor_convex_readout']=True
    from mlp_q_to_motor import forward,validate_network
    validate_network(n,2)
    with torch.no_grad():expected=net(x).numpy()*out_scale+out_center
    np.testing.assert_allclose(forward(n,q),expected,atol=3e-4,rtol=1e-5)
    return n,chosen
