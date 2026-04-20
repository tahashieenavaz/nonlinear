import torch
import math
from ..ActivationFunction import ActivationFunction


class CaLU(ActivationFunction):
    def __init__(self, *, b: float = 0.5):
        super().__init__()
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.arctan(x) / math.pi
        return x * (a + self.b)
