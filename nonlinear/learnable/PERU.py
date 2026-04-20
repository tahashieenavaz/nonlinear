import torch
from ..LearnableActivationFunction import LearnableActivationFunction

class PREU(LearnableActivationFunction):
    def __init__(self, channels: int, *, a: float = 1.0, b: float = 1.0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.full((channels,), a))
        self.b = torch.nn.Parameter(torch.full((channels,), b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = self.a.view(shape)
        b = self.b.view(shape)
        return torch.where(x >= 0, a * x, a * x * torch.exp(b * x))