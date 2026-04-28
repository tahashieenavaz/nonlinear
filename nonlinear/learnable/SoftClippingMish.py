import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class SoftClippingMish(ChannelBasedActivationFunction):
    def __init__(self, channels: int, *, a: float = 0.25):
        super().__init__()
        self.a = self.get_parameter(channels=channels, initial_value=a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = self.a.view(shape)
        return torch.relu(x * torch.tanh(torch.nn.functional.softplus(a * x)))
