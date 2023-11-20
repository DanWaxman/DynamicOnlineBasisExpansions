import jax
from tensorflow_probability.substrates.jax import math as tfp_math
import objax
import copy
from jaxtyping import Float, Array


def softplus(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Alias for `jax.nn.softplus`.

    Args:
        x (Float[Array, ...]): x

    Returns:
        Float[Array, ...]: softplus(x)
    """
    return jax.nn.softplus(x)


def softplus_inv(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Alias for `tfp_math.softplus_inverse`.

    Args:
        x (Float[Array, ...]): x

    Returns:
        Float[Array, ...]: softplus^{-1}(x)
    """
    return tfp_math.softplus_inverse(x)


class ObjaxModuleWithDeepCopy(objax.Module):
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if isinstance(v, objax.BaseVar):
                # TODO: There are more correct ways to do this, but it works for now
                v = v.__class__(v.value)
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result
