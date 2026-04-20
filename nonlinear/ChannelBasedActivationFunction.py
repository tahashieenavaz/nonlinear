import torch
from .LearnableActivationFunction import LearnableActivationFunction


class ChannelBasedActivationFunction(LearnableActivationFunction):
    def get_shape(self, x: torch.Tensor):
        return (1, -1) + (1,) * (x.ndim - 2)
