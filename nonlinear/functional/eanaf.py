import torch
from typing import Callable


def __g(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x / 2)


def __h(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x)


def eanaf(x: torch.Tensor, g: Callable = __g, h: Callable = __h) -> torch.Tensor:
    return x * g(h(x))
