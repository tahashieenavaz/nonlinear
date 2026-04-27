import torch
from ..ActivationFunction import ActivationFunction


class CReLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.relu(x)
        b = torch.relu(-x)
        return torch.cat((a, b), dim=1)
