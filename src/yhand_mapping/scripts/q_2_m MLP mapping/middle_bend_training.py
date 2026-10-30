"""Data-fitted neural heads with bend-direction constraints and weak q3 coupling.

No baseline, generated motor labels or post-prediction angle compensation.
"""
import numpy as np

from monotonic_extensor_training import build_extensor


def train_bend(q, y, *, epochs=2000, seed=20261027, width=32, validation=None,
               patience=150, direction=1, q3_penalty=.03, activation='mixed_softplus_75',
               deep_weight=2.0):
    import torch
    torch.manual_seed(seed)
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    center, scale = q.mean(0), np.maximum(q.std(0), 1e-8)
    out_center, out_scale = y.mean(0), np.maximum(y.std(0), 1e-8)
    gains = np.array([.25, 1., 1., .5])
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32)
    x = tensor((q-center)/scale*gains)
    target = tensor((y-out_center)/out_scale)
    # Upweight original joint-space boundary rows; no labels are synthesized.
    thresholds = np.quantile(q[:, 1:3], .8, axis=0)
    boundary = (q[:, 1] >= thresholds[0]) & (q[:, 2] >= thresholds[1])
    weights = tensor(np.where(boundary, deep_weight, 1.)[:, None])
    weights /= weights.mean()
    axes = (1, 2, 3) if direction == 1 else (1, 2)
    net = build_extensor(width, direction=direction, monotonic_inputs=axes,
                         hidden_activation=activation)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.01, weight_decay=0)
    if validation is not None:
        vq, vy = validation
        vx = tensor((np.asarray(vq)-center)/scale*gains)
        vy = tensor((np.asarray(vy).reshape(-1, 1)-out_center)/out_scale)
    best, chosen, state = float('inf'), 0, None
    for epoch in range(1, epochs+1):
        optimizer.zero_grad(set_to_none=True)
        first = net.layers[0].weight()
        penalty = .015*first[:, 0].square().mean()+q3_penalty*first[:, 3].square().mean()
        loss = (weights*(net(x)-target).square()).mean()+penalty
        if not torch.isfinite(loss):
            raise ValueError('中指伸屈网络损失不是有限值')
        loss.backward()
        optimizer.step()
        if validation is not None:
            with torch.no_grad():
                score = float((net(vx)-vy).square().mean())
            if score < best-1e-6:
                best, chosen = score, epoch
                state = {k:v.detach().clone() for k,v in net.state_dict().items()}
            if epoch-chosen >= patience:
                break
    if validation is not None:
        net.load_state_dict(state)
    else:
        chosen = epochs
    layers = [dict(weight=layer.weight().detach().tolist(), bias=layer.bias.detach().tolist(),
                   activation='identity' if i == len(net.layers)-1 else activation)
              for i,layer in enumerate(net.layers)]
    network = dict(architecture='role_weighted_branches', input_center=center.tolist(),
        input_scale=scale.tolist(), output_center=out_center.tolist(), output_scale=out_scale.tolist(),
        branches=[dict(output_indices=[0], input_weights=gains.tolist(), monotonic_inputs=list(axes),
                       monotonic_direction=direction, hidden_activation=activation, layers=layers)])
    from mlp_q_to_motor import forward, validate_network
    validate_network(network, 1)
    with torch.no_grad():
        expected = net(x).numpy()*out_scale+out_center
    np.testing.assert_allclose(forward(network, q), expected, atol=3e-4, rtol=1e-5)
    return network, chosen
