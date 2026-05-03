import torch
from ..ActivationFunction import ActivationFunction
from ..functional import double_silu


class DoubleSiLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return double_silu(x)
