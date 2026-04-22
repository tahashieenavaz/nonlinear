import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class DELU(ChannelBasedActivationFunction):
    def __init__(self, channels: int, *, a: float = 1.0):
        super().__init__()
        self.a = self.get_parameter(channels=channels, initial_value=a)

    def forward(self, x: torch.Tensor):
        shape = self.get_shape(x)
        a = self.a.view(shape)
        return torch.where(
            x < 0, x * torch.sigmoid(x), (a + 0.5) * x + torch.abs(torch.exp(-x) - 1.0)
        )
