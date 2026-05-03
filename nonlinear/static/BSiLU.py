import torch
from ..ActivationFunction import ActivationFunction


class BSiLU(ActivationFunction):
    def __init__(self, *, alpha: float = 1.67):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.alpha) * torch.sigmoid(x) - self.alpha / 2
