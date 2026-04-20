import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class TaLU(ChannelBasedActivationFunction):
    def __init__(self, channels: int, *, a: float = -2.0, b: float = 4.0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.full((channels,), a))
        self.b = torch.nn.Parameter(torch.full((channels,), b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = torch.abs(self.a.view(shape))
        b = torch.abs(self.b.view(shape))
        tanh_x = torch.tanh(x)
        tanh_a = torch.tanh(a)
        return torch.where(x >= b, x, torch.where(x > a, tanh_x, tanh_a))
