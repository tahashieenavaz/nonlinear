import torch
from ..ActivationFunction import ActivationFunction


class MaxSig(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, torch.sigmoid(x))
