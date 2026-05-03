import torch
from ..ActivationFunction import ActivationFunction


class LogLog(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.exp(-x))
