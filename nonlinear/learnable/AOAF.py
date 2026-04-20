import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class AOAF(ChannelBasedActivationFunction):
    def __init__(self, b: float = 0.17, c: float = 0.17):
        super().__init__()
        self.a = None
        self.b = b
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.initialize(x, "a")
        a = self.a.view(self.parameter_shape(x))
        return torch.relu(x - self.b * a) + self.c * a
