import torch
from ..activation_function import ActivationFunction
from ..functional import polylu


class PolyLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return polylu(x)
