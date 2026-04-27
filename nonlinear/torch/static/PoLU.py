import torch
from enum import Enum
from ..ActivationFunction import ActivationFunction


class PoLUAlphaValues(float, Enum):
    ONE = 1
    ONE_HALF = 1.5
    TWO = 2


class PoLU(ActivationFunction):
    def __init__(self, *, alpha: PoLUAlphaValues = 1.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = 1 - x
        return torch.where(x >= 0, x, delta.pow(-self.alpha) - 1)
