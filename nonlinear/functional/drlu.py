import torch


def drlu(x: torch.Tensor, *, alpha: float = 0.08) -> torch.Tensor:
    return torch.where(x - alpha >= 0, x - alpha, 0)
