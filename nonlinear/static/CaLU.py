import torch
from ..ActivationFunction import ActivationFunction
from ..functional import calu


class CaLU(ActivationFunction):
    def __init__(self, *, b: float = 0.5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return calu(x, b=self.b, inplace=self.inplace)
