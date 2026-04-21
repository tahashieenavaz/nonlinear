import torch
from ..ActivationFunction import ActivationFunction


class DiffELU(ActivationFunction):
    def __init__(self, *, a: float = 0.3, b: float = 0.1):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = x * x.exp() - self.b * torch.exp(self.b * x)
        return torch.where(x >= 0, x, self.a * delta)
