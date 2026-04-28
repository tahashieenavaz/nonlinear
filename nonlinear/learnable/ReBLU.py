import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction
from .BLU import BLU


class ReBLU(ChannelBasedActivationFunction):
    def __init__(
        self, channels: int, *, a: float = 0.0, a_min: float = -1.0, a_max: float = 1.0
    ):
        super().__init__()
        self.blu = BLU(channels=channels, a=a, a_min=a_min, a_max=a_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x < 0, 0, self.blu(x))
