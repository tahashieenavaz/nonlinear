import torch
from ..ActivationFunction import ActivationFunction
from ..functional import derivative_silu


class DerivativeSiLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return derivative_silu(x)
