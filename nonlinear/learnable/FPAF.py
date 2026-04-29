import torch
from typing import Callable
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


def identity(x: torch.Tensor):
    return x


class FPAF(ChannelBasedActivationFunction):
    def __init__(
        self,
        channels: int,
        *,
        mu: Callable[[torch.Tensor], torch.Tensor] = torch.sin,
        nu: Callable[[torch.Tensor], torch.Tensor] = torch.exp,
        a: float = 1.0,
        b: float = 0.05,
    ):
        super().__init__()
        self.mu = mu
        self.nu = nu
        self.a = self.get_parameter(channels=channels, initial_value=a)
        self.b = self.get_parameter(channels=channels, initial_value=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = self.a.view(shape)
        b = self.b.view(shape)
        return torch.where(x >= 0, a * self.mu(x), b * self.nu(x))
