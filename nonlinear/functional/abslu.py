import torch


def abslu(x: torch.Tensor, alpha: float = 0.5, inplace: bool = False) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(x, negative_slope=-alpha, inplace=inplace)
