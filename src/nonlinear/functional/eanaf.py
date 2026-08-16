import torch
from typing import Callable


def _g(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x / 2)


def _h(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x)


def eanaf(x: torch.Tensor, g: Callable = _g, h: Callable = _h) -> torch.Tensor:
    return x * g(h(x))
