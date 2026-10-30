"""One shared posture encoder, three genuine neural heads for M8/M14/M27."""
from copy import deepcopy
import numpy as np
from mlp_q_to_motor import forward, validate_network

GRADIENT_SCALES = np.array([[6., 5., 3., 2.], [6., 8., 4., 1.], [4., 2., 1.5, 2.5]])
GRADIENT_UPPER = np.array([[12., 10., 6., 4.], [10., 12., 7., 3.], [6., 4., 3., 4.]])


def quadratic_reference_jacobian(q, y, probes):
    """Training-fold-only reference slopes; never a deployed motor baseline."""
    center, scale = q.mean(0), np.maximum(q.std(0), 1e-8)
    z = (q-center)/scale
    features = np.column_stack([np.ones(len(q)), z, z*z]+
        [z[:, i]*z[:, j] for i in range(4) for j in range(i+1, 4)])
    coef = np.linalg.lstsq(features, y, rcond=None)[0]
    x = (probes-center)/scale
    jac = np.broadcast_to(coef[1:5].T, (len(x), 3, 4)).copy()
    jac += 2*x[:, None, :]*coef[5:9].T[None]
    column = 9
    for i in range(4):
        for j in range(i+1, 4):
            jac[:, :, i] += x[:, j, None]*coef[column]
            jac[:, :, j] += x[:, i, None]*coef[column]
            column += 1
    return jac/scale


def train_coordination(q, y, *, probe_q, reference_jacobian, epochs=1600,
                       width=36, head_widths=(12, 24, 12), seed=20261029,
                       lr=.008, gradient_penalty=.4):
    import torch
    from torch import nn
    from torch.nn import functional as F
    q, y, probes, ref = (np.asarray(v, dtype=float) for v in (q, y, probe_q, reference_jacobian))
    if (q.ndim != 2 or q.shape[1] != 4 or len(q) < 2 or y.shape != (len(q), 3)
            or probes.ndim != 2 or probes.shape[1] != 4 or not len(probes)
            or ref.shape != (len(probes), 3, 4) or not all(np.isfinite(v).all() for v in (q, y, probes, ref))):
        raise ValueError('需要N×4角度、N×3真实反馈和无位置标签的角度/参考导数探针')
    if (epochs < 1 or width < 12 or width > 252 or width % 6 or len(head_widths) != 3
            or any(type(h) is not int or h < 4 or h > 256 or h % 2 for h in head_widths)
            or not np.isfinite(lr) or lr <= 0 or not np.isfinite(gradient_penalty) or gradient_penalty < 0):
        raise ValueError('共享对掌侧摆网络参数不正确')
    torch.manual_seed(seed)
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32)
    center, scale = q.mean(0), np.maximum(q.std(0), 1e-8)
    yc, ys = y.mean(0), np.maximum(y.std(0), 1e-8)
    size, split = width//6, width//2
    groups = [np.r_[np.arange(i*size, (i+1)*size), np.arange(split+i*size, split+(i+1)*size)] for i in range(3)]
    common, side, opposition = groups
    second_mask = np.zeros((width, width))
    for rows, cols in ((common, common), (side, np.r_[common, side]), (opposition, np.r_[common, opposition])):
        second_mask[np.ix_(rows, cols)] = 1

    class Dense(nn.Module):
        def __init__(self, inputs, outputs, mask=None, first=False):
            super().__init__()
            w = torch.empty(outputs, inputs).uniform_(.5, 1.5)/inputs
            self.raw = nn.Parameter(torch.log(torch.expm1(w)))
            self.bias = nn.Parameter(torch.linspace(-1.5, 1.5, outputs) if outputs > 1 else torch.zeros(1))
            self.first = first
            self.register_buffer('mask', torch.ones(outputs, inputs) if mask is None else tensor(mask))
            if first:
                with torch.no_grad():
                    self.raw[:, 2:].normal_(0, .1)
                    self.raw[opposition, 0].normal_(0, .05)
                    self.raw[side, 1].normal_(0, .05)
        def weight(self):
            if not self.first:return F.softplus(self.raw)*self.mask
            q0 = F.softplus(self.raw[:, 0]);q0 = q0.index_copy(0, torch.as_tensor(opposition), self.raw[opposition, 0])
            q1 = -F.softplus(self.raw[:, 1]);q1 = q1.index_copy(0, torch.as_tensor(side), self.raw[side, 1])
            return torch.cat((q0[:, None], q1[:, None], self.raw[:, 2:]), dim=1)
        def forward(self, x):return F.linear(x, self.weight(), self.bias)
    def activate(x):
        k = x.shape[-1]//2
        return torch.cat((F.softplus(x[..., :k]), -F.softplus(-x[..., k:])), dim=-1)
    class Coordination(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.ModuleList((Dense(4, width, first=True), Dense(width, width, second_mask)))
            heads = []
            for i, h in enumerate(head_widths):
                allowed = np.r_[common, opposition if i == 1 else side]
                mask = np.zeros((h, width));mask[:, allowed] = 1
                heads.append(nn.ModuleList((Dense(width, h, mask), Dense(h, 1))))
            self.heads = nn.ModuleList(heads)
        def forward(self, x):
            for layer in self.shared:x = activate(layer(x))
            return torch.cat([head[1](activate(head[0](x)))*direction
                for head, direction in zip(self.heads, (1., -1., -1.))], dim=1)
    net = Coordination()
    x, target = tensor((q-center)/scale), tensor((y-yc)/ys)
    p = tensor((probes-center)/scale).requires_grad_(True)
    signs = np.sign(ref)
    lower = np.minimum(.8*abs(ref), GRADIENT_SCALES)
    upper = np.minimum(1.35*abs(ref)+.1, GRADIENT_UPPER)
    valid = abs(ref) >= .05
    for output, axis, direction in ((0, 0, 1), (1, 1, 1), (2, 0, -1)):
        valid[:, output, axis] &= ref[:, output, axis]*direction >= 0
    sign, lower, upper, valid = map(tensor, (signs, lower, upper, valid))
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True);p.grad = None
        fit = (net(x)-target).square().mean()
        prior = x.new_zeros(())
        ramp = min(1., max(0., (epoch-250)/250))
        if gradient_penalty and ramp:
            prediction = net(p)
            jac = torch.stack([torch.autograd.grad(prediction[:, j].sum(), p, create_graph=True,
                retain_graph=True)[0] for j in range(3)], dim=1)*tensor(ys)[None, :, None]/tensor(scale)
            signed = jac*sign
            cost = (torch.relu(lower-signed)/tensor(GRADIENT_SCALES)).square()
            cost = cost+(torch.relu(signed-upper)/tensor(GRADIENT_SCALES)).square()
            prior = (cost*valid).sum()/valid.sum().clamp_min(1)
        loss = fit+gradient_penalty*ramp*prior
        if not torch.isfinite(loss):raise ValueError('共享三头MLP损失不是有限值')
        loss.backward();optimizer.step()
        if epoch == 0 or (epoch+1) % 200 == 0 or epoch == epochs-1:
            history.append(dict(epoch=epoch+1, fit=float(fit.detach()), prior=float(prior.detach())))
    def export(layer, activation):
        return dict(weight=layer.weight().detach().tolist(), bias=layer.bias.detach().tolist(), activation=activation)
    branch = dict(name='side_opposition', motor_ids=[8, 14, 27], output_indices=[0, 1, 2],
        input_weights=[1., 1., 1., 1.], shared_thumb_coordination=True,
        shared_layers=[export(l, 'mixed_softplus') for l in net.shared],
        feature_groups=dict(common=common.tolist(), side=side.tolist(), opposition=opposition.tolist()),
        heads=[dict(motor_id=mid, direction=direction, layers=[export(head[0], 'mixed_softplus'), export(head[1], 'identity')])
               for mid, direction, head in zip((8, 14, 27), (1, -1, -1), net.heads)])
    head = dict(architecture='role_weighted_branches', input_center=center.tolist(), input_scale=scale.tolist(),
        output_center=yc.tolist(), output_scale=ys.tolist(), branches=[branch])
    validate_network(head, 3)
    with torch.no_grad():expected = net(x).numpy()*ys+yc
    np.testing.assert_allclose(forward(head, q), expected, atol=8e-4, rtol=1e-5)
    return head, dict(history=history, width=width, head_widths=list(head_widths),
        common_features=len(common), labeled_points=len(q), angle_probe_points=len(probes))


def replace_coordination(source, head):
    c = deepcopy(source);n = c['network']
    for key in ('input_center', 'input_scale'):np.testing.assert_allclose(n[key], head[key], atol=1e-12, rtol=0)
    for key in ('output_center', 'output_scale'):np.testing.assert_allclose(n[key][3:], head[key], atol=1e-12, rtol=0)
    flex = [b for b in n['branches'] if b['output_indices'] == [0, 1, 2]]
    if len(flex) != 1 or source.get('finger') != 'thumb':raise ValueError('需要拇指共享屈伸来源')
    branch = deepcopy(head['branches'][0]);branch['output_indices'] = [3, 4, 5]
    n['branches'] = [flex[0], branch]
    c.update(schema_version=15, mapping_revision='thumb_shared_posture_three_heads_20261029',
        raw_prediction_at_reference_motor_deg=forward(n, np.zeros(4)).tolist())
    for mid in (8, 14, 27):c['input_priority']['per_motor_weights'][str(mid)] = [1., 1., 1., 1.]
    c['input_priority']['note'] = 'M8/M14/M27 share posture features with independent neural heads; channel priorities enter training losses. Flex/ext copied unchanged.'
    c['side_motor_mapping'] = 'Shared posture MLP, independent M8/M27 heads; q0 [+,-] structurally preserved.'
    c['opposition_motor_mapping'] = 'M14 reads common and dedicated posture features; q1 positive monotonicity structurally preserved.'
    return c
