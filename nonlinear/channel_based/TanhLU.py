import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class TanhLU(ChannelBasedActivationFunction):
    def __init__(
        self, channels: int, *, a: float = 1.0, b: float = 1.0, c: float = 0.0
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
        return a * torch.tanh(c * x) + b * x
