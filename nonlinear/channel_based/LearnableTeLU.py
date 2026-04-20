import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class LearnableTeLU(ChannelBasedActivationFunction):
    def __init__(self, channels: int, *, a: float = 1.0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.full((channels,), a))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = self.a.view(shape)
        return x * torch.tanh(torch.nn.functional.elu(a * x))
