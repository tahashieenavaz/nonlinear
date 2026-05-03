import torch


def ada(x: torch.Tensor, *, inplace: bool = False) -> torch.Tensor:
    if inplace:
        negative_mask = x < 0
        if negative_mask.any():
            x_negative = x[negative_mask]
            x[negative_mask] = x_negative * torch.exp(x_negative)
        return x

    exp_x = torch.exp(x)
    return torch.where(x >= 0, x, x * exp_x)
