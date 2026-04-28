import torch
from ..ActivationFunction import ActivationFunction
from ..functional import ada


class ADA(ActivationFunction):
    def __init__(self, *, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ada(x, inplace=self.inplace)
