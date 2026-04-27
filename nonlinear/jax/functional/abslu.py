import jax


def abslu(x: jax.Array, alpha: float = 0.5) -> jax.Array:
    return jax.numpy.where(x >= 0, x, jax.numpy.abs(x) * alpha)
