import flax.linen as nn
import jax
from jax._src import dtypes
import jax.numpy as jnp


def xavier_uniform_pytorchlike():
    def init(key, shape, dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        named_shape = core.as_named_shape(shape)
        if len(shape) == 2:  # Dense, [in, out]
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) == 4:  # Conv, [k, k, in, out]. Assumes patch-embed style conv.
            fan_in = shape[0] * shape[1] * shape[2]
            fan_out = shape[3]
        else:
            raise ValueError(f'Invalid shape {shape}')

        variance = 2 / (fan_in + fan_out)
        scale = jnp.sqrt(3 * variance)
        param = jax.random.uniform(key, shape, dtype, -1) * scale

        return param

    return init


class TrainConfig:
    def __init__(self, dtype):
        self.dtype = dtype

    def kern_init(self, name='default', zero=False):
        if zero or 'bias' in name:
            return nn.initializers.constant(0)
        return xavier_uniform_pytorchlike()

    def default_config(self):
        return {
            'kernel_init': self.kern_init(),
            'bias_init': self.kern_init('bias', zero=True),
            'dtype': self.dtype,
        }


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    hidden_size: int
    tc: TrainConfig
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        x = self.timestep_embedding(t)
        x = nn.Dense(
            self.hidden_size,
            kernel_init=nn.initializers.normal(0.02),
            bias_init=self.tc.kern_init('time_bias'),
            dtype=self.tc.dtype,
        )(x)
        x = nn.silu(x)
        x = nn.Dense(
            self.hidden_size, kernel_init=nn.initializers.normal(0.02), bias_init=self.tc.kern_init('time_bias')
        )(x)
        return x

    # t is between [0, 1].
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                            These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = jax.lax.convert_element_type(t, jnp.float32)
        # t = t * max_period
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        embedding = embedding.astype(self.tc.dtype)
        return embedding


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
