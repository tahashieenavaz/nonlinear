import torch


def selfarctan(x: torch.Tensor, *, inplace: bool = False):
    if inplace:
        return x.mul_(torch.arctan(x))
    return x * torch.arctan(x)
