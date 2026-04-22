import torch
from math import log
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class PFELU(ChannelBasedActivationFunction):
    def __init__(self, channels: int, *, a: float = 1.0, b: float = 0.0):
        super().__init__()
        self.a = self.get_parameter(channels=channels, initial_value=a)
        self.b = self.get_parameter(channels=channels, initial_value=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = self.a.view(shape)
        b = self.a.view(shape)
        return torch.where(
            x >= 0, b + x, b + a * (torch.tensor(2.0).pow(x / log(2)) - 1)
        )
