import torch

"""
Paper: https://doi.org/10.3390/electronics11223799

Formula: f(z) = ReLU(z - b*a) + c*a
    
Variables:
    - x: Input tensor.
    - a: Channel-wise / Neuron-wise mean of the input.
    - b, c: Fixed hyperparameters (Recommended: b=c=0.17).
"""


def aoaf(x: torch.Tensor, *, b: float = 0.17, c: float = 0.17, inplace: bool = False):
    reduce_dims = [d for d in range(x.ndim) if d != 1]
    a = x.mean(dim=reduce_dims, keepdim=True)

    if inplace:
        x.sub_(b * a)
        x.relu_()
        x.add_(c * a)
        return x

    return torch.relu(x - b * a) + c * a
