import torch
from nonlinear.functions import gompertz

def golu(x: torch.Tensor):
    return x * gompertz(x)