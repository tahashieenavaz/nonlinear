import torch
from ..ActivationFunction import ActivationFunction


class NeLU(ActivationFunction):
    def __init__(self, *, alpha: float = 0.15):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, -self.alpha * torch.reciprocal(1 + x.square()))
