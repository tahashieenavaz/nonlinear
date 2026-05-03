import torch


def self_arctan(x: torch.Tensor) -> torch.Tensor:
    return x * torch.arctan(x)
