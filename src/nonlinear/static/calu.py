import torch
from nonlinear import ActivationFunction
from nonlinear.functional import calu


class CaLU(ActivationFunction):
    def __init__(self, *, b: float = 0.5):
        super().__init__()
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return calu(x, b=self.b)
