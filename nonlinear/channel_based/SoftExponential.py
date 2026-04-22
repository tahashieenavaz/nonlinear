import torch
from math import log
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class SoftExponential(ChannelBasedActivationFunction):
    def __init__(self, channels: int, *, a: float = 1.0):
        super().__init__()
        self.a = self.get_parameter(channels=channels, initial_value=a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = self.a.view(shape)
        alef = (torch.exp(x) - 1) / a + a
        be = -log(1 - a * (x + a)) / a
        return torch.where(a > 0, alef, torch.where(a < 0, be, 0))
