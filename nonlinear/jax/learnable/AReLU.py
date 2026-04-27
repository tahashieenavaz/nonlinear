import jax
from flax import linen as nn


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
