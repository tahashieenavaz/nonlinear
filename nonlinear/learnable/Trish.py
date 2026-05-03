import torch
from ..LearnableActivationFunction import LearnableActivationFunction


class Trish(LearnableActivationFunction):
    def __init__(self, *, alpha: float = 0.1, beta: float = 0.5):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        self.beta = torch.nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(torch.log(1 + torch.tanh(self.beta * x)))
