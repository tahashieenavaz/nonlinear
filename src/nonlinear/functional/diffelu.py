import torch


def diffelu(x: torch.Tensor, *, a: float = 0.3, b: float = 0.1) -> torch.Tensor:
    x_safe = x.clamp_max(0.0)
    negative_branch = a * (x_safe * torch.exp(x_safe) - b * torch.exp(b * x_safe))
    return torch.where(x < 0, negative_branch, x)
