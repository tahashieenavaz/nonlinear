import torch
from ..ActivationFunction import ActivationFunction


class RReLU(ActivationFunction):
    def __init__(self, *, lower: float = 3.0, upper: float = 8.0):
        super().__init__()
        self.lower = lower
        self.upper = upper

    def forward(self, x):
        if self.training:
            a = torch.empty_like(x).uniform_(self.lower, self.upper)
        else:
            a = (self.lower + self.upper) / 2.0
        return torch.where(x >= 0, x, x / a)
