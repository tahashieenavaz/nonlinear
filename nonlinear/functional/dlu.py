import torch


def dlu(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, x, x / (1 - x))
