import jax
from ..JAXActivationFunction import JAXActivationFunction


class AbsLU(JAXActivationFunction):
    alpha: float = 0.5

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.numpy.where(x >= 0, jax.numpy.abs(x) * self.alpha)
