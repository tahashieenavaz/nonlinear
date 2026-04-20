import torch
from ..LearnableActivationFunction import LearnableActivationFunction


class StarReLU(LearnableActivationFunction):
    def __init__(self, channels: int, *, a: float = 0.8944, b: float = -0.4472):
        super().__init__()
        self.a = torch.nn.Parameter(torch.full((channels,), a))
        self.b = torch.nn.Parameter(torch.full((channels,), b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = self.a.view(shape)
        b = self.b.view(shape)
        return a * torch.relu(x).pow(2) + b