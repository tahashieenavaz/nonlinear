import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class MPELU(ChannelBasedActivationFunction):
    def __init__(self, channels: int, *, a: float = 1.0, b: float = 1.0):
        super().__init__()
        self.a = self.get_parameter(channels=channels, initial_value=a)
        self.b = self.get_parameter(channels=channels, initial_value=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = self.a.view(shape)
        b = self.a.view(shape)
        return torch.where(x >= 0, x, a * (torch.exp(b * x) - 1))
