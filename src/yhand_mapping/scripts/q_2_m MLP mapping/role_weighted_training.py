"""Role-specific feature priorities. Torch is imported only during training."""
import numpy as np

SECONDARY_WEIGHT = .25
PRIORITY_PENALTY = .001


def role_input_weights(finger, motor_ids, secondary=SECONDARY_WEIGHT):
    if not np.isfinite(secondary) or not 0 < secondary <= 1:
        raise ValueError('次要输入权重必须在(0,1]内')
    if finger == 'thumb':
        # Thumb glove columns have different anatomical meanings; keep its policy.
        return np.ones((len(motor_ids), 4))
    side_motors = {'index': {1, 28}, 'middle': {9, 24}, 'ring': {22, 21}, 'little': {18, 20}}
    if finger not in side_motors:
        raise ValueError('未知手指')
    return np.array([[1, secondary, secondary, secondary] if mid in side_motors[finger]
                     else [secondary, 1, 1, 1] for mid in motor_ids], dtype=float)


def build_network(outputs, width, input_weights=None, *, zero_output=False):
    import torch
    from torch import nn
    gains = np.ones((outputs, 4)) if input_weights is None else np.asarray(input_weights, dtype=float)
    if gains.shape != (outputs, 4) or not np.isfinite(gains).all() or not np.all((gains > 0) & (gains <= 1)):
        raise ValueError('每路输出需要4个(0,1]内的输入权重')

    def dense(count):
        net = nn.Sequential(nn.Linear(4, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, count))
        if zero_output:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        return net

    if np.all(gains == 1):
        return dense(outputs)

    class WeightedBranches(nn.Module):
        def __init__(self):
            super().__init__()
            unique, inverse = np.unique(gains, axis=0, return_inverse=True)
            self.output_indices = [np.flatnonzero(inverse == i).tolist() for i in range(len(unique))]
            self.branches = nn.ModuleList([dense(len(indices)) for indices in self.output_indices])
            self.register_buffer('input_weights', torch.tensor(unique, dtype=torch.float32))
            self.outputs = outputs

        def forward(self, x):
            result = x.new_zeros((*x.shape[:-1], self.outputs))
            for i, (branch, indices) in enumerate(zip(self.branches, self.output_indices)):
                result[..., indices] = branch(x*self.input_weights[i])
            return result

    return WeightedBranches()


def input_priority_penalty(net):
    """Extra first-layer L2 cost makes compensating weak gains more expensive."""
    if not hasattr(net, 'branches'):
        return next(net.parameters()).new_zeros(())
    penalties = [(branch[0].weight.square()*(net.input_weights[i].reciprocal().square()-1)).mean()
                 for i, branch in enumerate(net.branches)]
    return sum(penalties)/len(penalties)


def export_network(net, input_center, input_scale, output_center, output_scale):
    from torch import nn

    def layers(sequence):
        result = [{'weight': layer.weight.detach().tolist(), 'bias': layer.bias.detach().tolist(),
                   'activation': 'tanh'} for layer in sequence if isinstance(layer, nn.Linear)]
        result[-1]['activation'] = 'identity'
        return result

    result = {'input_center': input_center.tolist(), 'input_scale': input_scale.tolist(),
              'output_center': output_center.tolist(), 'output_scale': output_scale.tolist()}
    if hasattr(net, 'branches'):
        result.update(architecture='role_weighted_branches', branches=[
            {'output_indices': indices, 'input_weights': net.input_weights[i].detach().tolist(),
             'layers': layers(branch)}
            for i, (branch, indices) in enumerate(zip(net.branches, net.output_indices))])
    else:
        result['layers'] = layers(net)
    return result
