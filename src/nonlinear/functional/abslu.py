import torch


def abslu(x: torch.Tensor, alpha: float = 0.18) -> torch.Tensor:
    return torch.where(x >= 0, x, alpha * x.abs())
