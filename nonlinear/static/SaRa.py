import torch
from ..ActivationFunction import ActivationFunction


class SaRa(ActivationFunction):
    def __init__(self, *, alpha: float = 0.5, beta: float = 0.7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = 1 + self.alpha * torch.exp(-self.beta * x)
        return torch.where(x >= 0, x, x / delta)
