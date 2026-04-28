import torch
from ..functional import aoaf
from ..ActivationFunction import ActivationFunction


class AOAF(ActivationFunction):
    def __init__(self, *, b: float = 0.17, c: float = 0.17, inplace: bool = False):
        super().__init__()
        self.b = b
        self.c = c
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return aoaf(x, b=self.b, c=self.c, inplace=self.inplace)
