import torch
from ..functional import abslu
from ..ActivationFunction import ActivationFunction


class AbsLU(ActivationFunction):
    def __init__(self, *, alpha: float = 0.5, inplace: bool = False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return abslu(x, inplace=self.inplace, alpha=self.alpha)
