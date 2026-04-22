import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class FELU(ChannelBasedActivationFunction):
    def __init__(self, channels: int, *, a: float = 1.0):
        super().__init__()
        self.a = self.get_parameter(channels=channels, initial_value=a)

    def forward(self, x: torch.Tensor):
        shape = self.get_shape(x)
        a = self.a.view(shape)
        return torch.where(x >= 0, x, a * (torch.tensor(2.0).pow(x / torch.log(2)) - 1))
