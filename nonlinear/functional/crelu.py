import torch


def crelu(x: torch.Tensor) -> torch.Tensor:
    alpha = torch.relu(x)
    beta = torch.relu(-x)
    return torch.cat([alpha, beta], dim=1)
