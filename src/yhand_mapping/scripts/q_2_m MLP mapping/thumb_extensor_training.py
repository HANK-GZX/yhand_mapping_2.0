"""Refine M6/M7 inside one shared MLP while preserving the M4 subgraph.

The expanded dense layers embed frozen source neurons in the same activation
halves. New neurons and both extensor readout rows are learned together from
real paired feedback. Inference uses the existing three-layer MLP evaluator.
"""
from copy import deepcopy
import numpy as np

from mlp_q_to_motor import forward, validate_network
from thumb_monotonic_training import replace_thumb_flex_ext

# Conservative, training-only physical slope priors motivated by linear replay.
# Rows M6/M7; columns q0/q1/q2/q3. M6 q2/q3 have no positive slope floor.
PRIOR_DIRECTIONS = np.array([[1., -1., 1., 1.], [1., -1., 1., 1.]])
PRIOR_FLOORS = np.array([[2.5, 2., 0., 0.], [.5, 2., 2.4, .9]])
PRIOR_SCALES = np.array([3., 5., 2.5, 1.5])
# Soft M7 ceilings prevent a slope floor from producing excessive payout.
M7_SLOPE_CEILINGS = np.array([1.5, 6., 3., 1.6])


def train_extensor_refinement(source, q, y, *, runtime_q, epochs=800,
                              extra_width=16, seed=20261029, lr=.008,
                              slope_penalty=.15):
    import torch
    from torch import nn
    from torch.nn import functional as F

    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    runtime_q = np.asarray(runtime_q, dtype=float)
    if (q.ndim != 2 or q.shape[1] != 4 or len(q) < 2
            or y.shape != (len(q), 3) or not np.isfinite(q).all()
            or not np.isfinite(y).all() or runtime_q.ndim != 2
            or runtime_q.shape[1] != 4 or not len(runtime_q)
            or not np.isfinite(runtime_q).all()):
        raise ValueError('需要N×4角度、真实N×3 M4/M6/M7反馈及无标签运行域角度')
    if (epochs < 1 or extra_width < 4 or extra_width % 2
            or not np.isfinite(lr) or lr <= 0
            or not np.isfinite(slope_penalty) or slope_penalty < 0 or extra_width % 4):
        raise ValueError('伸肌微调参数范围不正确')
    n = source['network']
    branches = [b for b in n['branches'] if b['output_indices'] == [0, 1, 2]]
    if len(branches) != 1:
        raise ValueError('需要一个共享M4/M6/M7来源网络')
    branch = branches[0]
    layers = branch['layers']
    old_width = len(layers[0]['bias'])
    width = old_width + extra_width
    if (not branch.get('shared_thumb_flex_ext')
            or branch.get('hidden_activation') != 'mixed_softplus_tanh'
            or branch.get('monotonic_inputs') != [2, 3]
            or branch.get('output_directions') != [-1, 1, 1]
            or old_width % 2 or len(layers[1]['bias']) != old_width
            or width > 256):
        raise ValueError('仅支持偶数等宽的单调拇指共享屈伸MLP')
    tensor = lambda v: torch.as_tensor(v, dtype=torch.float32)
    center, scale = np.array(n['input_center']), np.array(n['input_scale'])
    yc, ys = np.array(n['output_center'][:3]), np.array(n['output_scale'][:3])
    # A fold must use that fold's frozen normalization and source features.
    np.testing.assert_allclose(center, q.mean(0), atol=1e-12, rtol=0)
    np.testing.assert_allclose(scale, np.maximum(q.std(0), 1e-8), atol=1e-12, rtol=0)
    np.testing.assert_allclose(yc, y.mean(0), atol=1e-12, rtol=0)
    np.testing.assert_allclose(ys, np.maximum(y.std(0), 1e-8), atol=1e-12, rtol=0)
    old_idx = np.r_[np.arange(old_width//2),
                    np.arange(width//2, width//2 + old_width//2)]
    new_idx = np.setdiff1d(np.arange(width), old_idx)
    coupling_rows = np.r_[np.arange(extra_width//4),
                          np.arange(extra_width//2, 3*extra_width//4)]
    bend_rows = np.setdiff1d(np.arange(extra_width), coupling_rows)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    class Expanded(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('old_idx', torch.as_tensor(old_idx))
            self.register_buffer('new_idx', torch.as_tensor(new_idx))
            for i, layer in enumerate(layers):
                self.register_buffer(f'w{i}', tensor(layer['weight']))
                self.register_buffer(f'b{i}', tensor(layer['bias']))
            # New first-layer neurons can learn signed q0/q1, positive q2/q3.
            first = tensor(rng.uniform(.08, .3, (extra_width, 4)))
            first[:, 1] *= -1
            first[coupling_rows, 0] = 1.
            first[coupling_rows, 1] = -.5
            first[:, 2:] = torch.log(torch.expm1(first[:, 2:]))
            self.first = nn.Parameter(first)
            self.first_bias = nn.Parameter(torch.linspace(-1.5, 1.5, extra_width))
            self.second = nn.Parameter(torch.full((extra_width, width), -3.))
            self.second_bias = nn.Parameter(torch.linspace(-1.5, 1.5, extra_width))
            initial = torch.full((2, width), 1e-4)
            initial[:, self.old_idx] = self.w2[1:]
            self.readout = nn.Parameter(torch.log(torch.expm1(initial)))
            self.readout_bias = nn.Parameter(self.b2[1:].clone())
            first_mask = torch.ones(extra_width, 4)
            first_mask[coupling_rows, 2:] = 0
            second_mask = torch.ones(extra_width, width)
            second_mask[coupling_rows] = 0
            second_mask[np.ix_(coupling_rows, new_idx[coupling_rows])] = 1
            readout_mask = torch.ones(2, width)
            readout_mask[0, new_idx[bend_rows]] = 0
            self.register_buffer('first_mask', first_mask)
            self.register_buffer('second_mask', second_mask)
            self.register_buffer('readout_mask', readout_mask)

        def effective(self):
            first = torch.cat((self.first[:, :2], F.softplus(self.first[:, 2:])), dim=1)*self.first_mask
            w0 = self.w0.new_zeros(width, 4).index_copy(0, self.old_idx, self.w0)
            w0 = w0.index_copy(0, self.new_idx, first)
            b0 = self.b0.new_zeros(width).index_copy(0, self.old_idx, self.b0)
            b0 = b0.index_copy(0, self.new_idx, self.first_bias)
            old_rows = self.w1.new_zeros(old_width, width).index_copy(1, self.old_idx, self.w1)
            w1 = self.w1.new_zeros(width, width).index_copy(0, self.old_idx, old_rows)
            w1 = w1.index_copy(0, self.new_idx, F.softplus(self.second)*self.second_mask)
            b1 = self.b1.new_zeros(width).index_copy(0, self.old_idx, self.b1)
            b1 = b1.index_copy(0, self.new_idx, self.second_bias)
            m4 = self.w2.new_zeros(1, width).index_copy(1, self.old_idx, self.w2[:1])
            w2 = torch.cat((m4, F.softplus(self.readout)*self.readout_mask), dim=0)
            b2 = torch.cat((self.b2[:1], self.readout_bias))
            return ((w0, b0), (w1, b1), (w2, b2))

        def forward(self, x):
            for i, (w, b) in enumerate(self.effective()):
                x = F.linear(x, w, b)
                if i < 2:
                    x = torch.cat((F.softplus(x[:, :width//2]),
                                   torch.tanh(x[:, width//2:])), dim=1)
            return x * x.new_tensor([-1., 1., 1.])

    model = Expanded()
    gains = tensor(branch['input_weights'])
    x = tensor((q-center)/scale) * gains
    target = tensor((y-yc)/ys)
    # Only normal, moderate deep-bend runtime inputs; no runtime motor targets.
    mask = ((runtime_q[:, 0] >= -15) & (runtime_q[:, 0] <= 35)
            & (runtime_q[:, 1] >= -25) & (runtime_q[:, 1] <= 20)
            & (runtime_q[:, 2] >= 50) & (runtime_q[:, 3] >= 20))
    domain = runtime_q[mask]
    if not len(domain):
        raise ValueError('运行域中没有可用的正常深弯输入')
    sampled = domain[rng.choice(len(domain), min(192, len(domain)), replace=False)]
    anchors = np.array([[a, 0., b, c] for a in (0., 10., 20.)
                        for b in (60., 80., 100.) for c in (30., 50., 80.)])
    probe = tensor((np.concatenate((sampled, anchors))-center)/scale).requires_grad_(slope_penalty > 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    fit_history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        probe.grad = None
        fit_loss = (model(x)[:, 1:] - target[:, 1:]).square().mean()
        slope_loss = x.new_zeros(())
        if slope_penalty:
            prediction = model(probe*gains)
            grad = torch.stack([torch.autograd.grad(prediction[:, i].sum(), probe,
                create_graph=True, retain_graph=True)[0] for i in (1, 2)], dim=1)
            physical_grad = grad * tensor(ys[1:])[None, :, None] / tensor(scale)
            shortfall = torch.relu(tensor(PRIOR_FLOORS)-physical_grad*tensor(PRIOR_DIRECTIONS))
            slope_loss = (shortfall/tensor(PRIOR_SCALES)).square().mean()
            excess = torch.relu(physical_grad[:, 1]*tensor(PRIOR_DIRECTIONS[1])-tensor(M7_SLOPE_CEILINGS))
            slope_loss = slope_loss+(excess/tensor(PRIOR_SCALES)).square().mean()
        loss = fit_loss + slope_penalty*slope_loss
        if not torch.isfinite(loss):
            raise ValueError('伸肌微调损失不是有限值')
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch+1) % 100 == 0 or epoch == epochs-1:
            fit_history.append(dict(epoch=epoch+1, fit=float(fit_loss.detach()),
                                    slope=float(slope_loss.detach())))
    head = {key: deepcopy(n[key]) for key in ('input_center', 'input_scale')}
    head.update(architecture='role_weighted_branches', output_center=yc.tolist(), output_scale=ys.tolist())
    b = deepcopy(branch)
    b['layers'] = [dict(weight=w.detach().tolist(), bias=bias.detach().tolist(),
        activation='identity' if i == 2 else 'mixed_softplus_tanh')
        for i, (w, bias) in enumerate(model.effective())]
    b['frozen_m4_embedding'] = dict(source_width=old_width, embedded_indices=old_idx.tolist(),
        new_indices=new_idx.tolist(), coupling_indices=new_idx[coupling_rows].tolist(),
        bend_indices=new_idx[bend_rows].tolist(),
        note='Frozen source neurons/M4 readout. New q0/q1 features shared by M6/M7; additional bending features read by M7. Both extensor readouts trainable.')
    head['branches'] = [b]
    validate_network(head, 3)
    with torch.no_grad():
        expected = model(x).numpy()*ys+yc
    np.testing.assert_allclose(forward(head, q), expected, atol=8e-4, rtol=1e-5)
    return head, dict(history=fit_history, runtime_probe_points=len(sampled),
                      synthetic_angle_anchors=len(anchors), labeled_points=len(q))


def replace_extensors(source, head):
    result = replace_thumb_flex_ext(source, head)
    result['mapping_revision'] = 'thumb_shared_expanded_extensors_20261029'
    result['input_priority']['note'] = ('Runtime input gains preserved; M4 subgraph frozen. '
        'New shared features and M6/M7 readouts use real paired labels and training-only deep-bend slope priors.')
    return result
