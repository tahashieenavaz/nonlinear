import torch


def ada(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, x, x * torch.exp(x))
