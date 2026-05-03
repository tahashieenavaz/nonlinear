import torch


def selfarctan(x: torch.Tensor) -> torch.Tensor:
    return x * torch.arctan(x)
