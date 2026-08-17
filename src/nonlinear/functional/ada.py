import torch


def ada(x: torch.Tensor) -> torch.Tensor:
    return x * torch.exp(x.clamp_max(0.0))
