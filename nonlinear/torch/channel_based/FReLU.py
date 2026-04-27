import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class FReLU(ChannelBasedActivationFunction):
    def __init__(self, channels: int, *, b: float = 0.0, inplace: bool = False):
        super().__init__()
        self.b = self.get_parameter(channels=channels, initial_value=b)
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        b = self.b.view(shape)

        if self.inplace:
            return torch.relu_(x).add_(b)

        return torch.relu(x) + b
