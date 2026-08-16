import torch
from typing import Callable
from nonlinear import ActivationFunction
from ..functional import eanaf
from ..functional.eanaf import __g
from ..functional.eanaf import __h


class EANAF(ActivationFunction):
    def __init__(self, *, g: Callable = __g, h: Callable = __h):
        super().__init__()
        self.g = g
        self.h = h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return eanaf(x, g=self.g, h=self.h)
