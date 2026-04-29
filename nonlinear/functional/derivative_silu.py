import torch


def derivative_silu(x: torch.Tensor, *, inplace: bool = False):
    if inplace:
        a = torch.sigmoid(x)
        x.mul_(1 - a)
        x.add_(1)
        x.mul_(a)
        return x
    else:
        a = torch.sigmoid(x)
        return a * (1 + x * (1 - a))
