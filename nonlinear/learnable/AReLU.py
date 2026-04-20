import torch
from ..LearnableActivationFunction import LearnableActivationFunction
from baloot import acceleration_device

class AReLU(LearnableActivationFunction):
    def __init__(self):
        super().__init__()
        self.__device = acceleration_device()
        self.a = torch.nn.Parameter(torch.tensor(0.9, requires_grad=True))
        self.b = torch.nn.Parameter(torch.tensor(2.0, requires_grad=True))
        self.a.to(self.__device)
        self.b.to(self.__device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        negative_slope = torch.clamp(self.a, 0.01, 0.99)
        positive_slope = 1 + torch.sigmoid(self.b)
        positive = positive_slope * torch.relu(x)
        negative = negative_slope * (-torch.relu(-x))
        return positive + negative
