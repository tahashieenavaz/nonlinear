import torch
import math


def calu(x: torch.Tensor, *, b: float = 0.5, inplace: bool = False):
    alpha = torch.arctan(x) / math.pi
    if inplace:
        x.mul_(alpha + b)
        return x
    return x * (alpha + b)
