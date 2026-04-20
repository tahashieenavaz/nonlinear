import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class PTELU(ChannelBasedActivationFunction):
    def __init__(self, channels: int, *, a: float = 1.0, b: float = 1.0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.full((channels,), a))
        self.b = torch.nn.Parameter(torch.full((channels,), b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = torch.abs(self.a.view(shape))
        b = torch.abs(self.b.view(shape))
        return torch.where(x >= 0, x, a * torch.tanh(b * x))
