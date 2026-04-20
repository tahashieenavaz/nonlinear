import torch
from ..ActivationFunction import ActivationFunction


class GCU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x * torch.cos(x)
