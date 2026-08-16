import torch
from typing import Callable
from nonlinear import ActivationFunction
from nonlinear.functional import eanaf
from nonlinear.functional.eanaf import __g
from nonlinear.functional.eanaf import __h


class EANAF(ActivationFunction):
    def __init__(self, *, g: Callable = __g, h: Callable = __h):
        super().__init__()
        self.g = g
        self.h = h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return eanaf(x, g=self.g, h=self.h)
