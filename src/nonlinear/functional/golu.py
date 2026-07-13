import torch
from nonlinear.helpers import gompertz


def golu(x: torch.Tensor):
    return x * gompertz(x)
