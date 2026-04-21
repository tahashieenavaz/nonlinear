import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class DualLine(ChannelBasedActivationFunction):
    def __init__(
        self, channels: int, *, a: float = 1.0, b: float = 0.01, m: float = -0.22
    ):
        super().__init__()
        self.a = self.get_parameter(channels=channels, initial_value=a)
        self.b = self.get_parameter(channels=channels, initial_value=b)
        self.m = self.get_parameter(channels=channels, initial_value=m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = self.a.view(shape)
        b = self.b.view(shape)
        m = self.m.view(shape)
        return torch.where(x >= 0, a * x + m, b * x + m)
