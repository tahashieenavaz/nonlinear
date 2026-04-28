import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class TReLU(ChannelBasedActivationFunction):
    def __init__(self, channels: int, *, b: float = 0.01):
        super().__init__()
        self.b = self.get_parameter(channels=channels, initial_value=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        b = self.b.view(shape)
        return torch.where(x >= 0, x, torch.tanh(b * x))
