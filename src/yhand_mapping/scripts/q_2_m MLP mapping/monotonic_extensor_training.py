"""Pure neural signed heads, monotonic in selected joint inputs by construction.

Legacy mixed convex/concave Softplus units and bounded-negative Softplus/tanh
units both avoid restricting the entire network to a convex function.
No polynomial fit, baseline, pseudo-labels or runtime fallback.
Torch is loaded only when constructing/training the head.
"""
import numpy as np


def build_extensor(width=32, *, direction=1, monotonic_inputs=(1, 2, 3), active_inputs=None,
                   hidden_activation='mixed_softplus', output_directions=None,convex_bend_inputs=None,
                   flexor_convex_readout=False):
    if type(direction) is not int or direction not in (-1, 1):
        raise ValueError('单调方向必须为+1或-1')
    if hidden_activation not in ('mixed_softplus', 'mixed_softplus_75', 'mixed_softplus_tanh', 'softplus'):
        raise ValueError('未知单调隐藏层激活')
    if output_directions is not None and tuple(output_directions)not in((1,-1),(-1,1,1)):
        raise ValueError('共享肌肉输出方向需要双输出[+1,-1]或拇指屈伸[-1,+1,+1]')
    if output_directions is not None and len(output_directions)==3 and convex_bend_inputs is not None:
        raise ValueError('拇指三输出不套用四指弯曲曲率连接规则')
    if convex_bend_inputs is not None and (hidden_activation!='mixed_softplus'
            or not convex_bend_inputs or any(type(i)is not int or i not in(1,2)for i in convex_bend_inputs)):
        raise ValueError('弯曲曲率结构只支持混合Softplus的q1/q2路径')
    if flexor_convex_readout and (convex_bend_inputs is None or output_directions is None):
        raise ValueError('屈肌曲率读出需要共享双输出与曲率输入')
    import torch
    from torch import nn
    from torch.nn import functional as F

    class PositiveDense(nn.Module):
        def __init__(self, inputs, outputs, *, first=False,second=False):
            super().__init__()
            self.first = first
            initial = torch.empty(outputs, inputs).uniform_(.5, 1.5)/inputs
            self.raw_weight = nn.Parameter(torch.log(torch.expm1(initial)))
            self.bias = nn.Parameter(torch.linspace(-1.5, 1.5, outputs) if outputs > 1 else torch.zeros(1))
            if first:
                with torch.no_grad():
                    self.raw_weight[:, 0].normal_(0, .05)
            mask=torch.ones(outputs,inputs)
            if convex_bend_inputs is not None:
                if first:
                    for axis in convex_bend_inputs:mask[outputs//2:,axis]=0
                if second and not flexor_convex_readout:mask[outputs//2:,:inputs//2]=0
                if not first and not second and flexor_convex_readout:mask[1,inputs//2:]=0
            self.register_buffer('structural_mask',mask)

        def weight(self):
            if self.first:
                columns=[]
                for axis in range(4):
                    value=self.raw_weight[:, axis:axis+1]
                    if active_inputs is not None and axis not in active_inputs:
                        value=torch.zeros_like(value)
                    elif axis in monotonic_inputs:
                        value=F.softplus(value)
                    columns.append(value)
                return torch.cat(columns, dim=1)*self.structural_mask
            return F.softplus(self.raw_weight)*self.structural_mask

        def forward(self, value):
            return F.linear(value, self.weight(), self.bias)

    class Extensor(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([PositiveDense(4, width, first=True),
                                         PositiveDense(width, width,second=True),
                                         PositiveDense(width, 1 if output_directions is None else len(output_directions))])
            self.register_buffer('directions',torch.tensor(direction if output_directions is None else output_directions))

        def forward(self, value):
            for i, layer in enumerate(self.layers):
                value = layer(value)
                if i < len(self.layers)-1:
                    if hidden_activation == 'softplus':
                        value = F.softplus(value)
                    elif hidden_activation == 'mixed_softplus_tanh':
                        split = value.shape[-1]//2
                        value = torch.cat((F.softplus(value[..., :split]),
                                           torch.tanh(value[..., split:])), dim=-1)
                    else:
                        split = (3*value.shape[-1]//4 if hidden_activation == 'mixed_softplus_75'
                                 else value.shape[-1]//2)
                        value = torch.cat((F.softplus(value[..., :split]),
                                           -F.softplus(-value[..., split:])), dim=-1)
            return self.directions*value

    return Extensor()


def train_extensor(q, y, *, epochs=1600, seed=20261026, width=32,
                   validation=None, patience=120, direction=1):
    """Use original feedback labels and train-only normalization, as before."""
    import torch
    torch.manual_seed(seed)
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    center, scale = q.mean(0), np.maximum(q.std(0), 1e-8)
    out_center, out_scale = y.mean(0), np.maximum(y.std(0), 1e-8)
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32)
    gains = np.array([.25, 1., 1., 1.])
    x, target = tensor((q-center)/scale*gains), tensor((y-out_center)/out_scale)
    net = build_extensor(width, direction=direction)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.01, weight_decay=0)
    if validation is not None:
        vq, vy = validation
        vx = tensor((np.asarray(vq)-center)/scale*gains)
        vy = tensor((np.asarray(vy).reshape(-1, 1)-out_center)/out_scale)
    best, chosen, state = float('inf'), 0, None
    for epoch in range(1, epochs+1):
        optimizer.zero_grad(set_to_none=True)
        # Same weak q0 preference, evaluated on effective first-layer weights.
        penalty = .001*(net.layers[0].weight()[:, 0].square()*15).mean()
        loss = (net(x)-target).square().mean()+penalty
        if not torch.isfinite(loss):
            raise ValueError('单调肌肉训练损失不是有限值')
        loss.backward()
        optimizer.step()
        if validation is not None:
            with torch.no_grad():
                score = float((net(vx)-vy).square().mean())
            if score < best-1e-6:
                best, chosen = score, epoch
                state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            if epoch-chosen >= patience:
                break
    if validation is not None:
        net.load_state_dict(state)
    else:
        chosen = epochs
    layers = [{'weight': layer.weight().detach().tolist(), 'bias': layer.bias.detach().tolist(),
               'activation': 'identity' if i == len(net.layers)-1 else 'mixed_softplus'}
              for i, layer in enumerate(net.layers)]
    network = dict(architecture='role_weighted_branches', input_center=center.tolist(), input_scale=scale.tolist(),
                   output_center=out_center.tolist(), output_scale=out_scale.tolist(),
                   branches=[dict(output_indices=[0], input_weights=gains.tolist(),
                                  monotonic_inputs=[1, 2, 3], monotonic_direction=direction, layers=layers)])
    from mlp_q_to_motor import forward, validate_network
    validate_network(network, 1)
    with torch.no_grad():
        expected = net(x).numpy()*out_scale+out_center
    np.testing.assert_allclose(forward(network, q), expected, atol=3e-4, rtol=1e-5)
    return network, chosen
