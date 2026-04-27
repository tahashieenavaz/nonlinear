import torch
from ..ActivationFunction import ActivationFunction


class Elliot(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 + torch.div(0.5 * x, 1 + torch.abs(x))
