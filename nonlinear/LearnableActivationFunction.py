import torch
from .ActivationFunction import ActivationFunction

class LearnableActivationFunction(ActivationFunction):
    def get_shape(self, x: torch.Tensor):
        return (1, -1) + (1,) * (x.ndim - 2)