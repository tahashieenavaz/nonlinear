import torch
from ..ActivationFunction import ActivationFunction
from ..functional import drlu


class DRLU(ActivationFunction):
    def __init__(self, *, alpha: float = 0.08):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drlu(x, alpha=self.alpha)
