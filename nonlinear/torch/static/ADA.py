import torch
from ..ActivationFunction import ActivationFunction
from ..functional import ada


class ADA(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ada(x)
