import torch


def ada(x: torch.Tensor, *, inplace: bool = False) -> torch.Tensor:
    if inplace:
        negative_mask = x < 0
        x[negative_mask] *= torch.exp(x[negative_mask])
        return x

    return torch.where(x >= 0, x, x * torch.exp(x))
