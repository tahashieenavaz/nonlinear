import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class LeLeLU(ChannelBasedActivationFunction):
    def __init__(self, channels: int, *, a: float = 1.0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.full((channels,), a))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = self.a.view(shape)
        return torch.where(x >= 0, a * x, 0.01 * a * x)
