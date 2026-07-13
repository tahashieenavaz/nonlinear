import torch


def expexpish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.exp(-torch.exp(-x))
