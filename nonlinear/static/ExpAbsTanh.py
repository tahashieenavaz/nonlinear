import torch
from ..ActivationFunction import ActivationFunction


class ExpAbsTanh(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = torch.tensor(torch.e)
        return x * torch.tanh(e.pow(x + 1) / (1 + torch.abs(e.pow(-x))))
