import torch
import math


def calu(x: torch.Tensor, *, b: float = 0.5) -> torch.Tensor:
    alpha = torch.arctan(x) / math.pi
    return x * (alpha + b)
