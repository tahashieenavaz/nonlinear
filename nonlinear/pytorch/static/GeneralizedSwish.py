import torch
from ..ActivationFunction import ActivationFunction


class GeneralizedSwish(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(torch.exp(-x))
