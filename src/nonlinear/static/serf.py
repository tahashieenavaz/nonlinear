import torch
from nonlinear import ActivationFunction


class Serf(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x * torch.erf(torch.log(1 + torch.exp(x)))
