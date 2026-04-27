import torch
from ..ActivationFunction import ActivationFunction


class SelfArctan(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.arctan(x)
