import torch


def asilu(x: torch.Tensor) -> torch.Tensor:
    alpha = 1 / (1 + torch.exp(-x))
    return torch.arctan(x * alpha)
