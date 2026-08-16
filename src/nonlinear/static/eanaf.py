import torch
from typing import Callable
from nonlinear import ActivationFunction
from nonlinear.functional import eanaf
from nonlinear.functional.eanaf import _g
from nonlinear.functional.eanaf import _h


class EANAF(ActivationFunction):
    def __init__(self, *, g: Callable = _g, h: Callable = _h):
        super().__init__()
        self.g = g
        self.h = h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return eanaf(x, g=self.g, h=self.h)
