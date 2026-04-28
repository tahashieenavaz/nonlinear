import torch


def aoaf(x: torch.Tensor, *, b: float = 0.17, c: float = 0.17, inplace: bool = False):
    reduce_dims = [d for d in range(x.ndim) if d != 1]
    a = x.mean(dim=reduce_dims, keepdim=True)

    if inplace:
        x.sub_(b * a)
        x.relu_()
        x.add_(c * a)
        return x

    return torch.relu(x - b * a) + c * a
