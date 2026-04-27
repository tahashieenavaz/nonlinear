import torch
from .LearnableActivationFunction import LearnableActivationFunction


class ChannelBasedActivationFunction(LearnableActivationFunction):
    def get_shape(self, x: torch.Tensor) -> tuple[int, ...]:
        return (1, -1) + (1,) * (x.ndim - 2)

    def get_parameter(
        self, channels: int, initial_value: float | int
    ) -> torch.nn.Parameter:
        return torch.nn.Parameter(torch.full((channels,), float(initial_value)))
