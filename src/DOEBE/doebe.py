from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp
from typing import List
from .models import DOBE
import numpy as np
from jaxtyping import Float, Array
import objax
import copy
import math

from typing import Tuple


class DOEBE(objax.Module):
    def __init__(self, models: List[DOBE]):
        """Creates a (dynamic) online ensemble of basis expansions

        Args:
            models (List[DOBE]): models to ensemble
        """
        self.models = objax.ModuleList(models)
        self.w = objax.StateVar(jnp.ones(len(models)) / len(models))

    def pretrain(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, "N 1"],
        lr: float = 1e-2,
        iters: int = 500,
        verbose: bool = False,
    ):
        """Pretrains models by maximizing the marginal likelihood, using Adam.

        This class will ensemble arbitrary DOBE models together. If all models are of the same type, it may be faster to refactor and vmap the pretraining.

        Args:
            X (Float[Array, &quot;N D&quot;]): input data
            y (Float[Array, &quot;N 1&quot;]): output data
            lr (float, optional): learning rate for Adam. Defaults to 1e-2.
            iters (int, optional): number of iterations for Adam. Defaults to 500.
            verbose (bool, optional): Defaults to False.
        """
        for model_idx in range(len(self.models)):
            opt = objax.optimizer.Adam(self.models[model_idx].vars())
            gv = objax.GradValues(
                self.models[model_idx].mnll, self.models[model_idx].vars()
            )

            @objax.Function.with_vars(
                self.models[model_idx].vars() + gv.vars() + opt.vars()
            )
            def train_op():
                df, f = gv(X, y)
                opt(lr, df)
                return f

            train_op = objax.Jit(train_op)

            for iter_idx in range(iters):
                f_value = train_op()
                if (iter_idx % 100 == 0 or iter_idx == iters - 1) and verbose:
                    print(iter_idx, f_value)

            self.models[model_idx].sigma_theta = self.models[
                model_idx
            ].var_theta * jnp.eye(self.models[model_idx].n_features)

    def pretrain_and_sample(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, "N"],
        lr: float = 1e-2,
        iters: int = 500,
        verbose: bool = False,
        n_samples: int = 10,
        sampling_type: str = "laplace",
    ):
        """Pretrains models by maximizing the marginal likelihood, using Adam.

        This class will ensemble arbitrary DOBE models together. If all models are of the same type, it may be faster to refactor and vmap the pretraining.

        Args:
            X (Float[Array, &quot;N D&quot;]): input data
            y (Float[Array, &quot;N 1&quot;]): output data
            lr (float, optional): learning rate for Adam. Defaults to 1e-2.
            iters (int, optional): number of iterations for Adam. Defaults to 500.
            verbose (bool, optional): Defaults to False.
            n_samples (int, optional): number of samples to generate for each model.
            sampling_type (str, optional): sample from Laplace approximation of the MLL (`"laplace"`) or a random pertubation of the trained hyperparameters (`"gaussian"`).
        """

        # First, pretrain the models
        self.pretrain(X, y, lr=lr, iters=iters, verbose=verbose)

        if type(sampling_type) is str:
            sampling_type = [sampling_type] * len(self.models)

        # repeat over all models in the ensemble
        for model_idx in range(len(self.models)):
            if sampling_type[model_idx] == "laplace":
                # calculate the negative Hessian of the MLL for Laplace approx (same as Hessian of NMLL)
                current_model = self.models[model_idx]
                hess = objax.Hessian(current_model.mnll, current_model.vars())(X, y)

                # Make dimensions work out nicely
                for i in range(len(hess)):
                    current_dim = jnp.atleast_2d(hess[i][i]).shape[0]
                    for j in range(len(hess)):
                        if current_dim == 1:
                            hess[i][j] = jnp.atleast_2d(hess[i][j])
                        else:
                            hess[i][j] = jnp.atleast_2d(hess[i][j]).T

                # Construct Hessians, and factor them with the SVD to generate random samples
                hess_matrix = jnp.block(hess)
                hess_inv = jnp.linalg.pinv(hess_matrix)
                u, s, _ = jnp.linalg.svd(hess_inv)
                inv_hess_factor = u @ jnp.diag(jnp.sqrt(s))

                # Make samples by making a deep copy, randomly sampling from the Laplace approx, and changing the copy
                for n_sample in range(n_samples - 1):
                    model_copy = copy.deepcopy(current_model)
                    random_sample = jnp.squeeze(
                        inv_hess_factor
                        @ objax.random.normal((inv_hess_factor.shape[0], 1))
                    )
                    d_offset = 0

                    for v_idx, v in enumerate(model_copy.vars().subset(objax.TrainVar)):
                        # TrainRefs like this are necessary for Objax internal reasons
                        ref = objax.TrainRef(v)
                        dim_v = jnp.atleast_1d(v.value).shape[0]
                        ref.value = v.value + jnp.squeeze(
                            random_sample[d_offset : d_offset + dim_v]
                        )
                        d_offset += dim_v

                    self.models.append(model_copy)
            elif sampling_type[model_idx] == "gaussian":
                # Comparitively a simpler way, just take a pertubation with WGN.
                current_model = self.models[model_idx]
                for n_sample in range(n_samples - 1):
                    model_copy = copy.deepcopy(current_model)
                    for v_idx, v in enumerate(model_copy.vars().subset(objax.TrainVar)):
                        ref = objax.TrainRef(v)
                        ref.value = v.value + objax.random.normal(v.value.shape) * 1e-3
                    self.models.append(model_copy)
            else:
                raise ValueError(
                    'Sampling type not recognized. It must be either "laplace" or "gaussian".'
                )

        # We added new models, so must redefine the weight vector
        self.w = objax.StateVar(jnp.ones(len(self.models)) / len(self.models))

        # We never updated the initial sigma_theta for each model
        for model_idx in range(len(self.models)):
            self.models[model_idx].sigma_theta = self.models[
                model_idx
            ].var_theta * jnp.eye(self.models[model_idx].n_features)

    def fit(
        self, X: Float[Array, "N D"], y: Float[Array, "N 1"], return_ws=False
    ) -> Tuple[Float[Array, "N 1"], Float[Array, "N 1"]]:
        """Fits models according to (online) data `X` and `y`.

        A trick used in Lu et al. is to only fit the models with nonzero
        weights. Since Jax does not work well with ragged arrays and the
        methods of this class are general, this trick is not done here. It
        may be faster to "batch" the data, where the trick is implemented.
        For example, if `w[0] == 0` after training on `X[:100]` and `y[:100]`,
        then the call `fit(X[100:], y[100:])` will only fit the models with
        nonzero weights.

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

        # The strategy here is to separate calculation of weights and predictive densities
        # This allows for simple implementation with lax scans
        for j, model in enumerate(self.models):
            # if model is inactive, no need to calculate anything
            if self.w[j] < 1e-16:
                yhats.append(jnp.zeros((1, X.shape[0])))
                cov_yhats.append(jnp.zeros((1, X.shape[0])))
                ls.append(jnp.zeros((1, X.shape[0])))
                continue

            yhat, cov_yhat, l = model.predict_and_update(X, y)

            yhats.append(yhat)
            cov_yhats.append(cov_yhat)
            ls.append(l)

        yhat = jnp.vstack(yhats).T
        cov_yhat = jnp.vstack(cov_yhats).T
        ls = jnp.vstack(ls).T

        # Next, use all predictive values to weight
        def _step_weights(carry, i):
            # Mark as inactive
            log_w = jnp.where(carry < jnp.log(1e-16), -jnp.inf, carry)

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
            if np.sum(self.w > 1 - 1e-16) > 0:
                ymean, yvar = self.fit(X[n * n_batch :], y[n * n_batch :])
                ymeans.append(ymean)
                yvars.append(yvar)
                break

            ymean, yvar = self.fit(
                X[n * n_batch : (n + 1) * n_batch], y[n * n_batch : (n + 1) * n_batch]
            )
            ymeans.append(ymean)
            yvars.append(yvar)

        ymean = jnp.concatenate(ymeans)
        yvar = jnp.concatenate(yvars)

        return ymean, yvar
