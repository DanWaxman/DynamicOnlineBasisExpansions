from __future__ import annotations

import jax.numpy as jnp
import jax

from abc import ABC, abstractmethod
from jaxtyping import Float, Array
from typing import Tuple
import objax
from .utils import softplus, softplus_inv, ObjaxModuleWithDeepCopy
import math


class DOBE(ABC, ObjaxModuleWithDeepCopy):
    def __init__(
        self,
        n_features: int,
        d_in: int,
        var_theta: Float[Array, "1"],
        var_rw: Float[Array, "1"],
        var_eps: Float[Array, "1"],
    ):
        """Initializes a Dynamic Online Basis Expansion.

        This is an abstract class, and is expected to be implemented with a `featurize` method

        Args:
            n_features (int): the feature dimension (F)
            d_in (int): the input dimension (D)
            var_theta (Float[Array, &quot;1&quot;]): the prior variance on theta
            var_rw (Float[Array, &quot;1&quot;]): the random walk variance
            var_eps (Float[Array, &quot;1&quot;]): the likelihood noise variance
        """
        self.n_features = n_features
        self.d_in = d_in
        # We transform with the inverse softplus operation to allow for unconstrained optimization
        self.transformed_var_theta = objax.TrainVar(softplus_inv(jnp.array(var_theta)))
        self.transformed_var_eps = objax.TrainVar(softplus_inv(jnp.asarray(var_eps)))
        self.var_rw = objax.StateVar(jnp.array(var_rw))

        self.mu_theta = objax.StateVar(jnp.zeros((self.n_features, 1)))
        self.sigma_theta = self.mu_theta * jnp.eye(self.n_features)

    @property
    def var_theta(self):
        return 1e-6 + jnp.minimum(softplus(self.transformed_var_theta.value), 1e2)

    @property
    def var_eps(self):
        return 1e-6 + jnp.minimum(softplus(self.transformed_var_eps.value), 1e2)

    @abstractmethod
    def featurize(self, X: Float[Array, "N D"]) -> Float[Array, "F N"]:
        """This function should "featurize" the input data, X, into a matrix of features, PHI."""
        pass

    def mnll(self, X: Float[Array, "N D"], y: Float[Array, "N"]) -> Float[Array, "1"]:
        """Calculates the marginal negative log likelihood (MNLL) using an optimized version of Eq (3.86) of [1]

        [1] Bishop, PRML

        Args:
            X (Float[Array, "N D"]): Input data
            y (Float[Array, "N"]) : Output data

        Returns:
            Float[Array, "1"]: The MNLL
        """
        jitter = 1e-6 * jnp.eye(self.n_features)
        y = y.squeeze()
        PHI = self.featurize(X).T
        N = X.shape[0]

        # See for example, Eq (3.86) in Bishop PRML
        # Note: One may be tempted to use the prior predictive directly,
        # i.e. y ~ N(0, PHI @ PHI^T + var_n * I)
        # This isn't possible in Jax implementations of the Gaussian logpdf
        # because this covariance is generally only positive semi-definite as sigma_n -> 0
        # and all Jax implementations I've found use the Cholesky decomposition
        # This should be more efficient anyways

        alpha = 1 / self.var_theta
        beta = 1 / self.var_eps

        A = alpha * jnp.eye(self.n_features) + beta * PHI.T @ PHI
        # Use Cholesky factor for numerical stability
        # Equivalent to m_N = beta A^-1 PHI^T y
        R, c = jax.scipy.linalg.cho_factor(A + jitter)
        m_N = beta * jax.scipy.linalg.cho_solve((R, c), PHI.T @ y)

        E_m_N = beta / 2 * jnp.sum((y - PHI @ m_N) ** 2) + alpha / 2 * m_N.T @ m_N

        return (
            -self.n_features / 2 * jnp.log(alpha)
            - N / 2 * jnp.log(beta)
            + E_m_N
            + 1 / 2 * jnp.sum(jnp.log(jnp.diag(R)))
            + N / 2 * jnp.log(2 * jnp.pi)
        )

    def predict(
        self, X: Float[Array, "N D"]
    ) -> Tuple[Float[Array, "N 1"], Float[Array, "N N"]]:
        """Predicts y_{t+1} given previous data and x_{t+1}

        Args:
            X (Float[Array, "N D"]): Input data

        Returns:
            Tuple[Float[Array, "N 1"], Float[Array, "N N"]]: output mean and covariance
        """
        sigma_theta = self.sigma_theta + self.var_rw * jnp.eye(self.n_features)

        phi_x = self.featurize(X)
        mu_yhat = phi_x.T @ self.theta_hat
        cov_yhat = phi_x.T @ sigma_hat @ phi_x + self.var_eps * jnp.eye(phi_x.shape[1])

        return mu_yhat, cov_yhat

    def predict_and_update(
        self, X: Float[Array, "N D"], y: Float[Array, "N 1"]
    ) -> Tuple[Float[Array, "N"], Float[Array, "N"], Float[Array, "N"]]:
        """Predicts and updates the DOBE using the recursive updates implemented in a lax scan.

        Args:
            X (Float[Array, "N D"]): input data
            y (Float[Array, "N D"]): output data

        Returns:
            Tuple[Float[Array, "N"], Float[Array, "N"], Float[Array, "N"]]: The output means, output variances, and Bayesian losses
        """
        y = y.squeeze()

        def _step_predict_and_update(carry, i):
            mu_theta, sigma_theta = carry

            # Predict step
            sigma_theta = sigma_theta + self.var_rw * jnp.eye(self.n_features)
            phi_x = self.featurize(X[i, None])
            mu_yhat = phi_x.T @ mu_theta
            cov_yhat = phi_x.T @ sigma_theta @ phi_x + self.var_eps * jnp.eye(
                phi_x.shape[1]
            )

            # Get loss
            l = -jax.scipy.stats.norm.logpdf(
                y[i], loc=mu_yhat, scale=jnp.sqrt(jnp.minimum(1e6, cov_yhat))
            )

            # Update step
            mu_theta = mu_theta + sigma_theta @ phi_x * (y[i] - mu_yhat) / cov_yhat
            sigma_theta = (
                sigma_theta - sigma_theta @ phi_x @ phi_x.T @ sigma_theta / cov_yhat
            )

            return (mu_theta, sigma_theta), (mu_yhat, cov_yhat, l)

        # lax scan over data to avoid slow Python for loops
        carry, res = jax.lax.scan(
            _step_predict_and_update,
            (self.mu_theta, self.sigma_theta),
            jnp.arange(X.shape[0]),
        )

        self.mu_theta = carry[0]
        self.sigma_theta = carry[1]

        mu_yhat = res[0].squeeze()
        var_yhat = res[1].squeeze()
        l = res[2].squeeze()

        return mu_yhat, var_yhat, l


class DOGP(DOBE):
    def __init__(
        self,
        n_rff: int,
        kernel_type: str,
        d_in: int,
        lengthscale: Float[Array, "D"],
        var_theta: float,
        var_rw: float,
        var_eps: float,
        train_lengthscale: bool = True,
        train_frequencies: bool = False,
    ):
        """Creates the DO-RFF model using the random Fourier feature (also known as the sparse spectrum) approximation to Gaussian processes [1].
        If lengthscales and frequencies are not trained, this is equivalent to the model of Li et al. [2].

        [1] Lazaro-Gredilla et al.
        [2] Li et al.

        Args:
            n_rff (int): number of random Fourier features to use
            kernel_type (str): type of kernel. Right now only `'se'` is available.
            d_in (int): the input dimension
            lengthscale (Float[Array, "D"]): lengthscale, which can be a vector of length d_in for ARD
            var_theta (Float[Array, &quot;1&quot;]): the prior variance on theta
            var_rw (Float[Array, &quot;1&quot;]): the random walk variance
            var_eps (Float[Array, &quot;1&quot;]): the likelihood noise variance
            train_lengthscale (bool, optional): whether to train the lengthscale with empirical Bayes. Defaults to True.
            train_frequencies (bool, optional): whether to train the frequencies with empirical Bayes. Defaults to False.
        """
        super().__init__(2 * n_rff, d_in, var_theta, var_rw, var_eps)

        freqs = objax.StateVar(self.sample_frequencies(kernel_type))
        if train_frequencies:
            self.freqs = objax.TrainVar(freqs.value)
        else:
            self.freqs = objax.StateVar(freqs.value)

        if train_lengthscale:
            self.transformed_lengthscale = objax.TrainVar(softplus_inv(lengthscale))
        else:
            self.transformed_lengthscale = objax.StateVar(softplus_inv(lengthscale))

    @property
    def lengthscale(self):
        return 1e-6 + jnp.minimum(softplus(self.transformed_lengthscale.value), 1e2)

    def sample_frequencies(self, kernel_type: str) -> Float[Array, "F D"]:
        """Samples frequencies for the RFF approximation from the kernel

        Args:
            kernel_type (str): type of kernel

        Raises:
            NotImplementedError: if unsupported kernel type is used

        Returns:
            Float[Array, "F D"]: random Fourier frequencies
        """
        if kernel_type in ["rbf", "se"]:
            return objax.random.normal((self.n_features // 2, self.d_in))
        else:
            raise NotImplementedError(f"Kernel type {kernel_type} not yet implemented")

    def featurize(self, X: Float[Array, "N H"]) -> Float[Array, "N_RFF N"]:
        freqs_times_x = self.freqs @ (X / jnp.atleast_2d(self.lengthscale)).T

        return jnp.vstack([jnp.cos(freqs_times_x), jnp.sin(freqs_times_x)]) / jnp.sqrt(
            self.n_features // 2
        )


class DORBF(DOBE):
    def __init__(
        self,
        x_h,
        lengthscale,
        var_theta,
        var_rw,
        var_eps,
        train_locs=True,
        train_lengthscale=True,
    ):
        super().__init__(x_h.shape[0], x_h.shape[1], var_theta, var_rw, var_eps)

        if train_locs:
            self.x_h = objax.TrainVar(x_h)
        else:
            self.x_h = objax.StateVar(x_h)

        if train_lengthscale:
            self.transformed_lengthscale = objax.TrainVar(softplus_inv(lengthscale))
        else:
            self.transformed_lengthscale = objax.StateVar(softplus_inv(lengthscale))

    @property
    def lengthscale(self):
        return softplus(self.transformed_lengthscale.value)

    def featurize(self, X: Float[Array, "N H"]) -> Float[Array, "N X_H"]:
        return jnp.exp(
            -jnp.sum(
                ((X - self.x_h.value[:, None, :]) / (math.sqrt(2) * self.lengthscale))
                ** 2,
                axis=2,
            )
        )


class DOAddPoly(DOBE):
    def __init__(self, order, d_in, var_theta, var_rw, var_eps):
        super().__init__((order + 1) * d_in, d_in, var_theta, var_rw, var_eps)
        self.order = order + 1

    def featurize(self, X: Float[Array, "N d_in"]) -> Float[Array, "Order N"]:
        return jnp.hstack(jax.vmap(jnp.vander, in_axes=(1, None))(X, self.order)).T


class DOLinear(DOBE):
    def __init__(self, d_in, var_theta, var_rw, var_eps, bias=True):
        super().__init__(d_in + int(bias), d_in, var_theta, var_rw, var_eps)
        self.bias = bias

    def featurize(self, X: Float[Array, "N D"]) -> Float[Array, "D N"]:
        if self.bias:
            return jnp.vstack([X.T, jnp.ones(X.shape[0])])
        else:
            return X.T


class DOAddHSGP(DOBE):
    def __init__(
        self, L, M, d_in, lengthscale, var_theta, var_rw, var_eps, kernel_type="rbf"
    ):
        super().__init__(M * d_in, d_in, var_theta, var_rw, var_eps)
        self.L = L
        self.M = M
        self.transformed_lengthscale = objax.TrainVar(
            softplus_inv(jnp.atleast_1d(jnp.asarray(lengthscale)))
        )
        self.kernel_type = kernel_type
        if kernel_type == "rbf":

            def get_diag(boundaries, lengthscale):
                return jnp.sqrt(jnp.sqrt(2 * jnp.pi) * lengthscale) * jnp.exp(
                    -0.25
                    * (lengthscale * jnp.pi / 2 / boundaries) ** 2
                    * jnp.arange(1, self.M + 1) ** 2
                )

        elif kernel_type == "matern32":

            def get_diag(boundaries, lengthscale):
                return (
                    2
                    * (math.sqrt(3) / lengthscale) ** 1.5
                    / (
                        (math.sqrt(3) / lengthscale) ** 2
                        + ((math.pi / 2 / boundaries) * jnp.arange(1, self.M + 1)) ** 2
                    )
                )

        else:
            raise NotImplementedError(f"Kernel type {kernel_type} not yet implemented")

        self.get_diag = jax.vmap(get_diag, in_axes=(0, 0))

    @property
    def lengthscale(self):
        return softplus(self.transformed_lengthscale.value)

    def featurize(self, X: Float[Array, "N D"]) -> Float[Array, "M N"]:
        diag = self.get_diag(self.L, self.lengthscale).reshape(self.d_in, 1, self.M)

        def get_phi(L, x):
            mat = jnp.tile(
                jnp.reshape(jnp.pi / (2 * L) * (x + L), (-1, 1)), self.M
            ) @ jnp.diag(jnp.linspace(1, self.M + 1, self.M))
            return jnp.sin(mat) / jnp.sqrt(L)

        return jnp.hstack(jax.vmap(get_phi, in_axes=(0, 1))(self.L, X) * diag).T
