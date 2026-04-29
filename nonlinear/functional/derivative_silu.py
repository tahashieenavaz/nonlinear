import torch


def derivative_silu(x: torch.Tensor):
    a = torch.sigmoid(x)
    b = 1 - torch.sigmoid(x)
    c = 1 + x * b
    return a * c
