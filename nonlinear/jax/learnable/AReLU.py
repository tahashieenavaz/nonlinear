import jax
from flax import linen as nn

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


class AReLU(nn.Module):
    a: float = 0.9
    b: float = 2.0
    alpha: float = 0.0
    beta: float = 0.99

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        a = self.param("a", lambda key: jax.numpy.array(self.a_init))
        b = self.param("b", lambda key: jax.numpy.array(self.b_init))
        negative_slope = jax.numpy.clip(a, self.alpha, self.beta)
        positive_slope = 1.0 + jax.nn.sigmoid(b)
        return jax.numpy.where(x >= 0, positive_slope * x, negative_slope * x)
