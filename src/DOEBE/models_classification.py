from __future__ import annotations

import jax.numpy as jnp
import jax

from jaxtyping import Float, Array
from typing import Tuple
import objax
from .utils import softplus, softplus_inv, ObjaxModuleWithDeepCopy
import math

from jaxopt import LBFGS
from jax.scipy.special import expit
from .models import DOBE


class DOBE_Classification(DOBE, ObjaxModuleWithDeepCopy):
    """Creates a (dynamic) online basis expansion for classifcation.

    # WARNING: This code is not particularly well-optimized, and both efficiency and accuracy
        could likely be improved if classification is of particular interest (see, e.g.,
        Section 3.4 of the DOEBE paper). Implementations are inspired by GPC in sklearn and
        Bayesian logistic regression in bayes_logistic.
    """

    def mnll(
        self, X: Float[Array, "N D"], y: Float[Array, "N"], max_iters: int = 100
    ) -> Float[Array, "1"]:
        """Calculates the marginal negative log likelihood (MNLL) using algorithm 5.1 of [1].

        [1] Rasmussen & Williams (2006).
            Gaussian Processes for Machine Learning. MIT Press.

        Args:
            X (Float[Array, "N D"]): Input data
            y (Float[Array, "N"]) : Output data (0/1)

        Returns:
            Float[Array, "1"]: The MNLL
        """
        y = y.squeeze()
        PHI = self.featurize(X).T
        N = X.shape[0]
        K = self.var_theta * PHI @ PHI.T
        jitter = 1e-6 * jnp.eye(N)

        f = jnp.zeros_like(y)

        log_marginal_likelihood = -jnp.inf

        @objax.Function.with_vars(self.vars())
        def inner(f):
            pi = expit(f)
            W = pi * (1 - pi)

            W_sqrt = jnp.sqrt(W)
            W_sqrt_K = W_sqrt[:, jnp.newaxis] * K
            B = jnp.eye(W.shape[0]) + W_sqrt_K * W_sqrt
            L = jax.scipy.linalg.cholesky(B + jitter, lower=True)

            b = W * f + (y - pi)
            a = b - W_sqrt * jax.scipy.linalg.cho_solve((L, True), W_sqrt_K @ b)

            f = K @ a

            lml = (
                -0.5 * a.T @ f
                - jnp.log1p(jnp.exp(-(y * 2 - 1) * f)).sum()
                - jnp.log(jnp.diag(L)).sum()
            )

            return lml, f

        inner = objax.Jit(inner)

        for _ in range(max_iters):
            lml, f = inner(f)
            if (lml - log_marginal_likelihood) < 1e-10:
                break

            log_marginal_likelihood = lml

        return -log_marginal_likelihood

    def predict(self, X: Float[Array, "N D"]) -> Float[Array, "N 1"]:
        """Predicts y_{t+1} given previous data and x_{t+1}

        Args:
            X (Float[Array, "N D"]): Input data

        Returns:
            Tuple[Float[Array, "N 1"], Float[Array, "N N"]]: output mean and covariance
        """
        sigma_theta = self.sigma_theta + self.var_rw * jnp.eye(self.n_features)

        phi_x = self.featurize(X)
        mu_yhat = phi_x.T @ self.mu_theta
        cov_yhat = phi_x.T @ sigma_theta @ phi_x

        # Moderated output
        # See e.g. Murphy (2012) pg. 263.
        ksig = 1 / jnp.sqrt(1 + jnp.pi * jnp.diag(cov_yhat) / 8.0)
        return expit(ksig.reshape(-1, 1) * mu_yhat.reshape(-1, 1))

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

        def _f_log_posterior(theta, args):
            mu_theta, sigma_theta_inv, phi_x, y = args
            mu_yhat = phi_x.T @ theta

            py1 = expit(mu_yhat)
            eps = 1e-10

            neg_log_post = -(
                jnp.dot(y.T, jnp.log(py1 + eps))
                + jnp.dot((1.0 - y).T, jnp.log(1.0 - py1 + eps))
            ) + 0.5 * jnp.dot(
                (theta - mu_theta).T, jnp.dot(sigma_theta_inv, (theta - mu_theta))
            )

            return neg_log_post.sum()

        def _h_log_posterior(theta, mu_theta, sigma_theta_inv, phi_x, y):
            mu_yhat = phi_x.T @ theta

            py1 = expit(mu_yhat)

            S = py1 * (1.0 - py1)

            return phi_x @ phi_x.T * S[:, jnp.newaxis] + sigma_theta_inv

        def _step_predict_and_update(carry, i):
            mu_theta, sigma_theta, sigma_theta_inv = carry
            eps = 1e-10

            sigma_theta = sigma_theta + self.var_rw * jnp.eye(self.n_features)
            sigma_theta_inv = jax.scipy.linalg.inv(sigma_theta)

            # Predict step
            phi_x = self.featurize(X[i, None])
            opt = LBFGS(fun=_f_log_posterior, jit=True)
            res = opt.run(
                init_params=mu_theta, args=(mu_theta, sigma_theta_inv, phi_x, y[i])
            )

            # Get loss using probit approximation
            # See e.g. Murphy (2012) pg. 263.
            mu_yhat = phi_x.T @ mu_theta
            cov_yhat = phi_x.T @ sigma_theta @ phi_x
            ksig = 1 / jnp.sqrt(1 + jnp.pi * cov_yhat / 8)
            py1 = expit(ksig * mu_yhat)
            l = -(jnp.log(py1 + eps) * y[i] + jnp.log(1 - py1 + eps) * (1 - y[i]))

            # Update step
            sigma_theta_inv = _h_log_posterior(
                res.params, mu_theta, sigma_theta_inv, phi_x, y
            ).squeeze()
            sigma_theta = jnp.linalg.inv(sigma_theta_inv)
            mu_theta = res.params

            return (mu_theta, sigma_theta, sigma_theta_inv), (py1, l)

        # lax scan over data to avoid slow Python for loops
        jnp.linalg.inv(self.sigma_theta)
        carry, res = jax.lax.scan(
            _step_predict_and_update,
            (self.mu_theta, self.sigma_theta, jnp.linalg.inv(self.sigma_theta)),
            jnp.arange(X.shape[0]),
        )

        self.mu_theta = carry[0]
        self.sigma_theta = carry[1]

        mu_yhat = res[0].squeeze()
        var_yhat = res[0].squeeze()
        l = res[1].squeeze()

        return mu_yhat, var_yhat, l


class DOGP_Classification(DOBE_Classification):
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
        If lengthscales and frequencies are not trained, this is equivalent to the model of Lu et al. [2].

        [1] Lázaro-Gredilla, M., Quinonero-Candela, J., Rasmussen, C. E., & Figueiras-Vidal, A. R. (2010).
            Sparse spectrum Gaussian process regression. The Journal of Machine Learning Research, 11, 1865-1881
        [2] Lu, Q., Karanikolas, G. V., & Giannakis, G. B. (2022).
            Incremental ensemble Gaussian processes. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2), 1876-1893.


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


class DOAddHSGP_Classification(DOBE_Classification):
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


class DORBF_Classification(DOBE_Classification):
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
