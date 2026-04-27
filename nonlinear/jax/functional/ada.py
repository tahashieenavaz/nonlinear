import jax


def ada(x: jax.Array) -> jax.Array:
    return jax.numpy.where(x >= 0, x, x * jax.numpy.exp(x))
