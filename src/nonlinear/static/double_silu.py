import torch
from nonlinear import ActivationFunction
from nonlinear.functional import double_silu


class DoubleSiLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return double_silu(x)
