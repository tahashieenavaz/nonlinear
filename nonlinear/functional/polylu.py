import torch


def polylu(x: torch.Tensor) -> torch.Tensor:
    denominator = 1 - x
    x_negative = 1 / denominator - 1
    return torch.where(x >= 0, x, x_negative)
