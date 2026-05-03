import torch


def double_silu(x: torch.Tensor) -> torch.Tensor:
    a = 1 + torch.exp(-x)
    b = -x * 1 / a
    c = 1 + torch.exp(b)
    return x * 1 / c
