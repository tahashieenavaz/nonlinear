import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class EPReLU(ChannelBasedActivationFunction):
    def __init__(self, channels: int, *, alpha: float = 0.2, a: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.a = self.get_parameter(channels=channels, initial_value=a)
        self.register_buffer("k", torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a = self.a.view(shape)
        if self.training:
            k = self.k.view(shape)
        else:
            k = torch.ones_like(self.k).view(shape)
        return torch.where(x >= 0, k * x, x / a)

    def step_epoch(self, epoch: int):
        is_odd = epoch % 2 != 0
        if is_odd:
            self.k.fill_(1.0)
            self.a.requires_grad = True
        else:
            self.k.uniform_(1.0 - self.alpha, 1.0 + self.alpha)
            self.a.requires_grad = False
