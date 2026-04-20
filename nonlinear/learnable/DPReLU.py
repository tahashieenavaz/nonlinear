import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class DPReLU(ChannelBasedActivationFunction):
    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None

    def forward(self, x: torch.Tensor):
        self.initialize(x, ["a", "b"], [1, 0.01])
        a = self.a.view(self.parameter_shape(x))
        b = self.b.view(self.parameter_shape(x))
        return torch.where(x >= 0, a * x, b * x)
