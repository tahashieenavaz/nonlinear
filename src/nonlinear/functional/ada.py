import torch


def ada(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x < 0, x * torch.exp(x.clamp_max(0.0)), x)
