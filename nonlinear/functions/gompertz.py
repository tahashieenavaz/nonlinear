import torch

def gompertz(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-torch.exp(-x))