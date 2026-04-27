import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class BLU(ChannelBasedActivationFunction):
    def __init__(
        self, channels: int, *, a: float = 0.0, a_min: float = -1.0, a_max: float = 1.0
    ):
        super().__init__()
        self.a_min = a_min
        self.a_max = a_max
        self.a = self.get_parameter(channels=channels, initial_value=a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        self.a.data.clamp_(self.a_min, self.a_max)
        a = self.a.view(shape)
        return a * (torch.sqrt(x * x + 1.0) - 1.0) + x
