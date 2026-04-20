import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class PiLU(ChannelBasedActivationFunction):
    def __init__(
        self, channels: int, *, a: float = 1.0, b: float = 0.01, c: float = 1.0
    ):
        super().__init__()
        self.a = torch.nn.Parameter(torch.full((channels,), a))
        self.b = torch.nn.Parameter(torch.full((channels,), b))
        self.c = torch.nn.Parameter(torch.full((channels,), c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = self.a.view(shape)
        b = self.b.view(shape)
        c = self.c.view(shape)
        return torch.where(x >= c, a * x + c * (1 - a), b * x + c * (1 - b))
