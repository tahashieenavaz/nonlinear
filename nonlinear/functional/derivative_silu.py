import torch


def derivative_silu(x: torch.Tensor) -> torch.Tensor:
    a = torch.sigmoid(x)
    return a * (1 + x * (1 - a))
