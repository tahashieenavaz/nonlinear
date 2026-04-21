import torch
from ..ActivationFunction import ActivationFunction


class ERF(ActivationFunction):
    def __init__(self, *, alpha: float = 1.0):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.erf(self.alpha * x)
