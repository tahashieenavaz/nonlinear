import torch
from ..LearnableActivationFunction import LearnableActivationFunction


"""
AReLU: Attention-based Rectified Linear Unit (Chen et al., 2020)

Reference: https://arxiv.org/abs/2006.13858

This module implements a learnable activation function inspired by
element-wise attention mechanisms. AReLU generalizes ReLU by
introducing adaptive scaling factors for positive and negative
feature responses:

f(x) = beta · max(x, 0) − alpha · max(−x, 0)

where alpha and beta are trainable parameters.

This design improves the expressivity of standard ReLU while
maintaining efficiency and stability in deep neural networks.

Original Implementation: https://github.com/densechen/AReLU
"""


class AReLU(LearnableActivationFunction):
    def __init__(
        self,
        *,
        a: float = 0.9,
        b: float = 2.0,
        alpha: float = 0.0,
        beta: float = 0.99,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.a = torch.nn.Parameter(torch.tensor(a))
        self.b = torch.nn.Parameter(torch.tensor(b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        negative_slope = torch.clamp(self.a, min=self.alpha, max=self.beta)
        positive_slope = 1.0 + torch.sigmoid(self.b)
        return torch.where(x >= 0, positive_slope * x, negative_slope * x)
