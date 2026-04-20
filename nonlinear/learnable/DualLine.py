import torch
from ..LearnableActivationFunction import LearnableActivationFunction


class DualLine(LearnableActivationFunction):
    def __init__(
        self, channels: int, *, a: float = 1.0, b: float = 0.01, m: float = -0.22
    ):
        super().__init__()
        self.a = torch.nn.Parameter(torch.full((channels,), a))
        self.b = torch.nn.Parameter(torch.full((channels,), b))
        self.m = torch.nn.Parameter(torch.full((channels,), m))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = self.a.view(shape)
        b = self.b.view(shape)
        m = self.m.view(shape)
        return torch.where(x >= 0, a * x + m, b * x + m)
