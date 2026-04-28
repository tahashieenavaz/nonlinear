import torch


def asilu(x: torch.Tensor, *, inplace: bool = False):
    alpha = 1 / (1 + torch.exp(-x))

    if inplace:
        x.mul_(alpha)
        x.atan_()
        return x

    return torch.arctan(x * alpha)
