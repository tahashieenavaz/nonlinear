import torch
from ..activation_function import ActivationFunction
from ..functional import bsilu


class BSiLU(ActivationFunction):
    def __init__(self, *, alpha: float = 1.67):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return bsilu(x=x, alpha=self.alpha)
