import torch
from ..functional import abslu
from ..activation_function import ActivationFunction


class AbsLU(ActivationFunction):
    def __init__(self, *, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return abslu(x, alpha=self.alpha)
