import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class StarReLU(ChannelBasedActivationFunction):
    def __init__(self, channels: int, *, a: float = 0.8944, b: float = -0.4472):
        super().__init__()
        self.a = self.get_parameter(channels=channels, initial_value=a)
        self.b = self.get_parameter(channels=channels, initial_value=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = self.a.view(shape)
        b = self.b.view(shape)
        return a * torch.relu(x).pow(2) + b
