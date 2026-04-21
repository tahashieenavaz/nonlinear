import torch
from ..ActivationFunction import ActivationFunction


class LaLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi_laplace = torch.where(x >= 0, 1 - 0.5 * torch.exp(-x), 0.5 * torch.exp(x))
        return x * phi_laplace
