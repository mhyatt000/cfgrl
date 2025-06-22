import jax
import jaxtyping as jt  # noqa: F401
from jaxtyping import Array, Float, PyTree  # noqa: F401

Single = Float[Array, '1 *_']
Batched = Float[Array, 'bs ...']


def spec(tree: jt.PyTree):
    return jax.tree.map(lambda x: x.shape, tree)
