"""M6 feature selection with signed q0/q1 neurons inside the shared thumb MLP."""
from copy import deepcopy
import numpy as np

from mlp_q_to_motor import forward, validate_network
from thumb_monotonic_training import replace_thumb_flex_ext

SLOPE_DIRECTIONS = np.array([1., -1., 1., 1.])
SLOPE_SCALES = np.array([5., 6., 3., 2.])
SLOPE_CEILINGS = np.array([6., 8., 4.5, 3.])


def shared_features(network, q):
    """Frozen hidden values and exact local derivatives in physical joint units."""
    branch = network['branches'][0]
    x = np.atleast_2d(np.asarray(q, dtype=float))
    gain = np.array(branch['input_weights'])/network['input_scale']
    x = (x-network['input_center'])*gain
    jac = np.broadcast_to(np.diag(gain), (len(x), 4, 4)).copy()
    for layer in branch['layers'][:-1]:
        w = np.array(layer['weight'])
        z = x@w.T+layer['bias']
        jac = np.einsum('oi,nij->noj', w, jac)
        k = z.shape[1]//2
        x = np.c_[np.logaddexp(0, z[:, :k]), np.tanh(z[:, k:])]
        derivative = np.c_[np.exp(-np.logaddexp(0, -z[:, :k])), 1-np.tanh(z[:, k:])**2]
        jac *= derivative[:, :, None]
    return x, jac


def train_short_extensor(source, q, y, *, runtime_q, epochs=1400,
                         seed=20261029, lr=.015, slope_penalty=2.,
                         freeze_source_readout=False):
    import torch
    from torch import nn
    from torch.nn import functional as F
    q, y, runtime_q = (np.asarray(v, dtype=float) for v in (q, y, runtime_q))
    if (q.ndim != 2 or q.shape[1] != 4 or len(q) < 2 or y.shape != (len(q), 3)
            or runtime_q.ndim != 2 or runtime_q.shape[1] != 4 or not len(runtime_q)
            or not all(np.isfinite(v).all() for v in (q, y, runtime_q))):
        raise ValueError('需要真实N×4角度/N×3反馈与有限的运行域角度探针')
    if epochs < 1 or not np.isfinite(lr) or lr <= 0 or not np.isfinite(slope_penalty) or slope_penalty < 0:
        raise ValueError('M6训练参数不正确')
    n = source['network']
    b = n['branches'][0]
    if (not b.get('shared_thumb_flex_ext') or b.get('hidden_activation') != 'mixed_softplus_tanh'
            or b.get('output_indices') != [0, 1, 2] or 'frozen_m4_embedding' not in b):
        raise ValueError('需要当前具有共享弯曲特征的M4/M6/M7网络')
    for key, expected in [('input_center', q.mean(0)), ('input_scale', np.maximum(q.std(0), 1e-8)),
                          ('output_center', y.mean(0)), ('output_scale', np.maximum(y.std(0), 1e-8))]:
        actual = n[key] if key.startswith('input') else n[key][:3]
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    tensor = lambda v: torch.as_tensor(v, dtype=torch.float32)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    h, _ = shared_features(n, q)
    yc, ys = n['output_center'][1], n['output_scale'][1]
    x, target = tensor(h), tensor((y[:, 1]-yc)/ys)
    routing = b['frozen_m4_embedding']
    mask = np.ones(h.shape[1])
    mask[routing['coupling_indices']] = 0
    initial = np.array(b['layers'][-1]['weight'])[1]
    old_width = h.shape[1]
    extra = 8
    width = old_width+extra
    if old_width % 2 or width > 256:
        raise ValueError('共享屈伸隐藏层必须偶数等宽且扩展后不超过256')
    old_indices = np.r_[np.arange(old_width//2), np.arange(width//2, width//2+old_width//2)]
    new_indices = np.setdiff1d(np.arange(width), old_indices)
    raw_weight = nn.Parameter(tensor(np.log(np.expm1(np.r_[np.maximum(initial, 1e-4), np.full(extra, .02)]))))
    bias = nn.Parameter(tensor(float(b['layers'][-1]['bias'][1])))
    active_mask = tensor(np.r_[mask, np.ones(extra)])
    first_raw = nn.Parameter(tensor(np.log(np.expm1(np.tile([.8, .8], (extra, 1))))))
    first_bias = nn.Parameter(torch.linspace(-1.5, 1.5, extra))
    second_raw = nn.Parameter(torch.full((extra, extra), -2.))
    second_bias = nn.Parameter(torch.linspace(-1.5, 1.5, extra))
    def activate(z):
        return torch.cat((F.softplus(z[:, :extra//2]), torch.tanh(z[:, extra//2:])), dim=1)
    def first_weights():
        return F.softplus(first_raw)*first_raw.new_tensor([1., -1.])
    def new_features(z):
        z = activate(F.linear(z, first_weights(), first_bias))
        return activate(F.linear(z, F.softplus(second_raw), second_bias))
    def effective_readout():
        if freeze_source_readout:
            return torch.cat((tensor(initial)*tensor(mask), F.softplus(raw_weight[old_width:])))
        return F.softplus(raw_weight)*active_mask
    normalized = tensor((q-n['input_center'])/n['input_scale'])
    gains = tensor(b['input_weights'])
    domain = runtime_q[(runtime_q[:, 0] >= -15) & (runtime_q[:, 0] <= 35)
        & (runtime_q[:, 1] >= -25) & (runtime_q[:, 1] <= 20)
        & (runtime_q[:, 2] >= 15) & (runtime_q[:, 2] <= 105)
        & (runtime_q[:, 3] >= 15) & (runtime_q[:, 3] <= 95)]
    if not len(domain):
        raise ValueError('没有可用的中等/深弯角度探针')
    sampled = domain[rng.choice(len(domain), min(256, len(domain)), replace=False)]
    anchors = np.array([[a, b, c, d] for a in (0., 10., 20., 30.)
        for b in (-20., -10., 0., 10.) for c in (20., 35., 50., 70., 90.) for d in (30., 50., 70.)])
    probes = np.concatenate((sampled, anchors))
    probe = tensor((probes-n['input_center'])/n['input_scale']).requires_grad_(True)
    _, jac = shared_features(n, probes)
    derivative = tensor(jac*ys)
    floors = np.tile([2.5, 2., 2., 0.], (len(probes), 1))
    medium_opposition = (probes[:, 2] <= 50) & (probes[:, 1] <= 0)
    floors[medium_opposition, 0] = 3.5
    floors[medium_opposition, 1] = 4.5
    floors[medium_opposition, 2] = 2.4
    floors = tensor(floors)
    optimizer = torch.optim.AdamW([raw_weight, bias, first_raw, first_bias, second_raw, second_bias], lr=lr, weight_decay=0)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        probe.grad = None
        weight = effective_readout()
        features = torch.cat((x, new_features((normalized*gains)[:, :2])), dim=1)
        fit = (features@weight+bias-target).square().mean()
        new_prediction = new_features((probe*gains)[:, :2])@weight[old_width:]
        new_grad = torch.autograd.grad(new_prediction.sum(), probe, create_graph=True)[0]*ys/tensor(n['input_scale'])
        grad = (torch.einsum('nha,h->na', derivative, weight[:old_width])+new_grad)*tensor(SLOPE_DIRECTIONS)
        short = torch.relu(floors-grad)/tensor(SLOPE_SCALES)
        excess = torch.relu(grad-tensor(SLOPE_CEILINGS))/tensor(SLOPE_SCALES)
        slope = short.square().mean()+excess.square().mean()
        loss = fit+slope_penalty*slope
        if not torch.isfinite(loss):
            raise ValueError('M6损失不是有限值')
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch+1) % 200 == 0 or epoch == epochs-1:
            history.append(dict(epoch=epoch+1, fit=float(fit.detach()), slope=float(slope.detach())))
    head = {key: deepcopy(n[key]) for key in ('input_center', 'input_scale')}
    head.update(architecture='role_weighted_branches', output_center=n['output_center'][:3],
                output_scale=n['output_scale'][:3], branches=[deepcopy(b)])
    with torch.no_grad():
        weights = effective_readout().numpy()
        offset = float(bias)
    for i, layer in enumerate(b['layers']):
        w, offset_source = np.array(layer['weight']), np.array(layer['bias'])
        if i == 0:
            expanded = np.zeros((width, 4));expanded[old_indices] = w
            expanded[new_indices, :2] = first_weights().detach().numpy()
            offsets = np.zeros(width);offsets[old_indices] = offset_source;offsets[new_indices] = first_bias.detach().numpy()
        elif i == 1:
            expanded = np.zeros((width, width));expanded[np.ix_(old_indices, old_indices)] = w
            expanded[np.ix_(new_indices, new_indices)] = F.softplus(second_raw).detach().numpy()
            offsets = np.zeros(width);offsets[old_indices] = offset_source;offsets[new_indices] = second_bias.detach().numpy()
        else:
            expanded = np.zeros((3, width));expanded[:, old_indices] = w
            expanded[1, old_indices] = weights[:old_width];expanded[1, new_indices] = weights[old_width:]
            offsets = offset_source.copy();offsets[1] = offset
        head['branches'][0]['layers'][i] = dict(weight=expanded.tolist(), bias=offsets.tolist(),
            activation='identity' if i == 2 else 'mixed_softplus_tanh')
    head['branches'][0]['M6_feature_routing'] = dict(
        disabled_coupling_indices=old_indices[routing['coupling_indices']].tolist(),
        unlocked_bend_indices=[] if freeze_source_readout else old_indices[routing['bend_indices']].tolist(),
        source_hidden_indices=old_indices.tolist(), signed_q01_indices=new_indices.tolist(),
        source_hidden_layers_frozen=True, source_readout_frozen=freeze_source_readout,
        added_hidden_neurons_per_layer=extra,
        preserved_output_indices=[0, 2], note='M6 uses shared bending features and new structurally signed q0+/q1- neurons; source subgraph and M4/M7 frozen')
    # Remap archived path indices after widening activation halves.
    embedding = head['branches'][0]['frozen_m4_embedding']
    for key in ('embedded_indices', 'new_indices', 'coupling_indices', 'bend_indices'):
        embedding[key] = old_indices[routing[key]].tolist()
    validate_network(head, 3)
    with torch.no_grad():
        # Re-evaluate after the final optimizer step.
        expected = torch.cat((x, new_features((normalized*gains)[:, :2])), dim=1).numpy()@weights+offset
    np.testing.assert_allclose(forward(head, q)[:, 1], expected*ys+yc, atol=8e-4, rtol=1e-5)
    return head, dict(history=history, runtime_probe_points=len(sampled), synthetic_angle_anchors=len(anchors),
        labeled_points=len(q), width=width, source_width=old_width, added_neurons_per_layer=extra,
        disabled_coupling_indices=old_indices[routing['coupling_indices']].tolist(),
        unlocked_bend_indices=[] if freeze_source_readout else old_indices[routing['bend_indices']].tolist(),
        source_readout_frozen=freeze_source_readout, slope_floors=[2.5, 2., 2., 0.],
        medium_opposition_floors=[3.5, 4.5, 2.4, 0.], slope_ceilings=SLOPE_CEILINGS.tolist())


def replace_short_extensor(source, head):
    c = replace_thumb_flex_ext(source, head)
    c['mapping_revision'] = 'thumb_M6_q1_medium_bend_features_20261029'
    c['input_priority']['note'] = ('M6 readout and new q0+/q1- neural features trained; '
        'direct positive-q1 coupling readout excluded and existing bend features available. '
        'Source hidden subgraph, M4/M7 and side/opposition preserved. '
        'Medium/deep-bend angle-only derivative priors are training-only.')
    return c
