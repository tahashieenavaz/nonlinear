import torch
from ..ActivationFunction import ActivationFunction


class ExpExpish(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.exp(-torch.exp(-x))
