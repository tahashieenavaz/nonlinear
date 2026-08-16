import torch
from nonlinear import ActivationFunction
from nonlinear.functional import crelu


class CReLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return crelu(x)
