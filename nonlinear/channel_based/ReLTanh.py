import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class ReLTanh(ChannelBasedActivationFunction):
    def __init__(self, channels: int, *, a: float = 1.0, b: float = 0.01):
        super().__init__()
        self.a = self.get_parameter(channels=channels, initial_value=a)
        self.b = self.get_parameter(channels=channels, initial_value=b)

    def tanh_derivative(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.tanh(x)
        return 1 - t * t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = self.a.view(shape)
        b = self.b.view(shape)

        tanh_x = torch.tanh(x)
        tanh_a = torch.tanh(a)
        tanh_b = torch.tanh(b)

        dtanh_a = 1 - tanh_a * tanh_a
        dtanh_b = 1 - tanh_b * tanh_b

        return torch.where(
            x <= a,
            tanh_a + dtanh_a * (x - a),
            torch.where(
                x >= b,
                tanh_b + dtanh_b * (x - b),
                tanh_x,
            ),
        )
