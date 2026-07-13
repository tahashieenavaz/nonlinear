import torch


def ada(x: torch.Tensor, *, inplace: bool = False) -> torch.Tensor:
    if inplace:
        mask = x < 0
        x_neg = x[mask]
        x[mask] = x_neg * torch.exp(x_neg)
        return x

    return torch.where(x < 0, x * torch.exp(x.clamp_max(0.0)), x)
