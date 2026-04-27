import torch
from ..ActivationFunction import ActivationFunction


class LearnableLiReLU(ActivationFunction):
    def __init__(self, *, a: float = 0.01):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(float(a)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * x + torch.relu(x)
