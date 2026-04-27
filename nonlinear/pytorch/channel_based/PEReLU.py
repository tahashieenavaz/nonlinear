import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction
from torch.nn.functional import elu


class PEReLU(ChannelBasedActivationFunction):
    def __init__(self, channels: int, *, a: float = 0.4, b: float = 0.3):
        super().__init__()
        self.a = self.get_parameter(channels=channels, initial_value=a)
        self.b = self.get_parameter(channels=channels, initial_value=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = self.a.view(shape)
        b = self.b.view(shape)
        return a * torch.relu(x) + b * elu(x) + (1 - a - b) * (-elu(-x))
