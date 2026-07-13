import torch


def bsilu(x: torch.Tensor, *, alpha: float = 1.67):
    return (x + alpha) * torch.sigmoid(x) - alpha / 2
