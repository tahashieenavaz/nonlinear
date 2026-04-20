import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class DualLine(ChannelBasedActivationFunction):
    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None
        self.m = None

    def forward(self, x: torch.Tensor):
        self.initialize(x, ["a", "b", "m"], [1, 0.01, -0.22])
        a = self.a.view(self.parameter_shape(x))
        b = self.b.view(self.parameter_shape(x))
        m = self.m.view(self.parameter_shape(x))
        return torch.where(x >= 0, a * x + m, b * x + m)
