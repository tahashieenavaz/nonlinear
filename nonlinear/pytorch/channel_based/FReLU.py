import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class FReLU(ChannelBasedActivationFunction):
    def __init__(self, channels: int, *, b: float = 0.0):
        super().__init__()
        self.b = self.get_parameter(channels=channels, initial_value=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        b = self.b.view(shape)
        return torch.relu(x) + b
