import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class RMAF(ChannelBasedActivationFunction):
    def __init__(
        self, channels: int, *, b: float = 1.0, c: float = 1.0, a: float = 1.0
    ):
        super().__init__()
        self.a = self.get_parameter(channels=channels, initial_value=a)
        self.b = b
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = self.a.view(shape)
        inner_term = 0.25 * (1.0 + torch.exp(-x))
        denominator = (inner_term + 0.75) ** self.c
        bracket_term = self.b / denominator
        return bracket_term * a * x
