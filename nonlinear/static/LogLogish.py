import torch
from ..ActivationFunction import ActivationFunction


class LogLogish(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = -torch.exp(x)
        return x * (1 - torch.exp(a))
