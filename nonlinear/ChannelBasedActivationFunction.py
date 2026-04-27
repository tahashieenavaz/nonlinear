import torch
from .LearnableActivationFunction import LearnableActivationFunction


class ChannelBasedActivationFunction(LearnableActivationFunction):
    def __init__(self):
        super().__init__()
        self._shape_cache: dict[int, tuple[int, ...]] = {}

    def get_shape(self, x: torch.Tensor) -> tuple[int, ...]:
        ndim = x.ndim
        if ndim not in self._shape_cache:
            self._shape_cache[ndim] = (1, -1) + (1,) * (ndim - 2)
        return self._shape_cache[ndim]

    def get_parameter(
        self, channels: int, initial_value: float | int
    ) -> torch.nn.Parameter:
        return torch.nn.Parameter(torch.full((channels,), float(initial_value)))
