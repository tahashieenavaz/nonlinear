import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class ShiLU(ChannelBasedActivationFunction):
    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.initialize(x, ["a", "b"])
        a = self.a.view(self.parameter_shape(x))
        b = self.b.view(self.parameter_shape(x))
        return torch.relu(x) * a + b
