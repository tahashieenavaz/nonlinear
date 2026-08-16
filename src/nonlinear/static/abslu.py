import torch
from ..functional import abslu
from nonlinear import ActivationFunction


class AbsLU(ActivationFunction):
    def __init__(self, *, alpha: float = 0.18):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return abslu(x=x, alpha=self.alpha)
