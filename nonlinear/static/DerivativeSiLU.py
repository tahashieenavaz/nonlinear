import torch
from ..ActivationFunction import ActivationFunction
from ..functional import derivative_silu


class DerivativeSiLU(ActivationFunction):
    def __init__(self, *, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return derivative_silu(x, inplace=self.inplace)
