from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp
from typing import List
from .models import DOGP, DOBE, DOAddHSGP
import numpy as np
from jaxtyping import Float, Array
import objax
import copy
import math
from .doebe import DOEBE

from typing import Tuple


class SDOEBE(DOEBE):
    def __init__(self, models: List[DOBE], Q: Float[Array, "M M"]):
        """Initialize an ensemble of (dynamic) incremental basis expansion (DIBE)
        models, contained in the list `models`.

        Args:
            models (List[DIBE]): The models to be ensembled
        """
        self.models = objax.ModuleList(models)
        self.w = objax.StateVar(jnp.ones(len(models)) / len(models))
        self.Q = objax.StateVar(Q)
        self.log_Q = objax.StateVar(jnp.log(Q))

    def fit(
        self, X: Float[Array, "N D"], y: Float[Array, "N 1"], return_ws=False
    ) -> Tuple[Float[Array, "N 1"], Float[Array, "N 1"]]:
        """Fits models according to (online) data `X` and `y`.

        Like in `pretrain`, it would be faster to vmap/pmap over the models
        if they are of the same class.

        Args:
            X (Float[Array, &quot;N D&quot;]): training X data
            y (Float[Array, &quot;N 1&quot;]): training y data

        Returns:
            Tuple[Float[Array, &quot;N 1&quot;], Float[Array, &quot;N 1&quot;]]: The mean and variance of the predictive distribution p(y_t | x_{1:t}, y_{1:t-1})
        """
        yhats = []
        cov_yhats = []
        ls = []

        # Model updates are the same as in the DOEBE case
        for j, model in enumerate(self.models):
            yhat, cov_yhat, l = model.predict_and_update(X, y)

            yhats.append(yhat)
            cov_yhats.append(cov_yhat)
            ls.append(l)

        yhat = jnp.vstack(yhats).T
        cov_yhat = jnp.vstack(cov_yhats).T
        ls = jnp.vstack(ls).T

        # Now, we calculate weights using a generalized version of the logsumexp trick
        def _step_weights(carry, i):
            log_w = jax.scipy.special.logsumexp(self.log_Q + carry[None], axis=1)

            ell = -jax.scipy.special.logsumexp(log_w - ls[i])

            log_w = log_w + (ell - ls[i])

            return log_w, log_w

        final_log_w, log_ws = lax.scan(
            _step_weights, jnp.log(self.w), jnp.arange(X.shape[0])
        )

        self.w = jnp.exp(final_log_w)

        ymean = jnp.sum(jnp.exp(log_ws) * yhat, axis=1)
        yvar = jnp.sum(
            (cov_yhat + (jnp.reshape(ymean, (-1, 1)) - yhat) ** 2) * jnp.exp(log_ws),
            axis=1,
        )

        if return_ws:
            return ymean, yvar, jnp.exp(log_ws)
        else:
            return ymean, yvar

    def fit_minibatched(self, X, y, n_batch=2000):
        ymeans = []
        yvars = []
        N = X.shape[0]

        for n in range(int(math.ceil(N / n_batch))):
            ymean, yvar = self.fit(
                X[n * n_batch : (n + 1) * n_batch], y[n * n_batch : (n + 1) * n_batch]
            )
            ymeans.append(ymean)
            yvars.append(yvar)

        ymean = jnp.concatenate(ymeans)
        yvar = jnp.concatenate(yvars)

        return ymean, yvar
