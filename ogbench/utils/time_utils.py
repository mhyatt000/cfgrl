import jax
import jax.numpy as jnp


def spec(batch):
    def debug(x):
        return (type(x), getattr(x, "dtype", None), getattr(x, "shape", None))

    return jax.tree.map(debug, batch)


def log_standardize(x, min_val=1, max_val=1000):
    """
    Log-transform and normalize x ∈ [min_val, max_val] to [-1, 1] with mean 0.
    """
    log_x = jnp.log(x)
    log_min = jnp.log(min_val)
    log_max = jnp.log(max_val)

    # Shift and scale to [-1, 1]
    x_scaled = 2 * (log_x - log_min) / (log_max - log_min) - 1
    return x_scaled


def log_unstandardize(x_scaled, min_val=1, max_val=1000):
    """
    Inverse of log_standardize. Maps from [-1, 1] back to original range [min_val, max_val].
    """
    log_min = jnp.log(min_val)
    log_max = jnp.log(max_val)

    # Undo the scale and shift, then exponentiate
    log_x = 0.5 * (x_scaled + 1) * (log_max - log_min) + log_min
    x = jnp.exp(log_x)
    return x
