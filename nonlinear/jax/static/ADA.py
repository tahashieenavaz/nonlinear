import jax
from flax import linen as nn
from ..JAXActivationFunction import JAXActivationFunction


class ADA(JAXActivationFunction):

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.numpy.where(x >= 0, x, x * jax.numpy.exp(x))
