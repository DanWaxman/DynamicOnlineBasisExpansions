from experiment_utils import *
import jax.numpy as jnp
import jax
from DOEBE.doebe import DOEBE
from DOEBE.sdoebe import SDOEBE
from DOEBE.models_classification import *
from tqdm import tqdm
import numpy as np
from scipy.cluster.vq import kmeans2
import seaborn as sns

import psutil

jax.config.update("jax_enable_x64", True)
sns.set_style("whitegrid")
sns.set_palette("colorblind")
from jax import config

config.update("jax_debug_nans", True)
import sys


def make_models(X, n_features=30):
    d = X.shape[1]
    M = 100 // d

    L = np.max([np.abs(np.min(X, axis=0)), np.abs(np.max(X, axis=0))], axis=0) * 1.5

    ls_guesses = [(jnp.max(X, axis=0) - jnp.min(X, axis=0)) / f for f in [0.1, 1, 10]]
    doehsgp = DOEBE(
        [
            DOAddHSGP_Classification(
                L, M, d, jnp.ones(d) * ls_guess, 1.0, 1e-3, 0.25, kernel_type="rbf"
            )
            for ls_guess in ls_guesses
        ]
    )
    oehsgp = DOEBE(
        [
            DOAddHSGP_Classification(
                L, M, d, jnp.ones(d) * ls_guess, 1.0, 0.0, 0.25, kernel_type="rbf"
            )
            for ls_guess in ls_guesses
        ]
    )

    doegp = DOEBE(
        [
            DOGP_Classification(
                n_features // 2,
                "rbf",
                d,
                ls_guess * jnp.ones(d),
                1.0,
                1e-3,
                0.25,
                train_lengthscale=True,
            )
            for ls_guess in ls_guesses
        ]
    )
    oegp = DOEBE(
        [
            DOGP_Classification(
                n_features // 2,
                "rbf",
                d,
                ls_guess * jnp.ones(d),
                1.0,
                0.0,
                0.25,
                train_lengthscale=True,
            )
            for ls_guess in ls_guesses
        ]
    )

    rbf_points = jnp.asarray(kmeans2(X[:1000], 100, minit="points")[0])
    mask = ~np.isnan(rbf_points).any(axis=1)
    rbf_points = rbf_points[mask]
    doerbf = DOEBE(
        [
            DORBF_Classification(
                rbf_points,
                ls_guess,
                1.0,
                1e-3,
                0.25,
                train_lengthscale=True,
                train_locs=True,
            )
            for ls_guess in ls_guesses
        ]
    )
    oerbf = DOEBE(
        [
            DORBF_Classification(
                rbf_points,
                ls_guess,
                1.0,
                0,
                0.25,
                train_lengthscale=True,
                train_locs=True,
            )
            for ls_guess in ls_guesses
        ]
    )
    delta = 0.01

    Q = (1 - delta) * np.eye(18 * 5)
    for i in range(15):
        Q[i, i + 3] = delta
        Q[i + 3, i] = delta
    for i in range(18, 18 * 5 - 12):
        Q[i, i + 12] = delta
        Q[i + 12, i] = delta

    sdoe = SDOEBE(
        [
            DOAddHSGP_Classification(
                L, M, d, jnp.ones(d) * ls_guess, 1.0, 1e-3, 0.25, kernel_type="rbf"
            )
            for ls_guess in ls_guesses
        ]
        + [
            DOAddHSGP_Classification(
                L, M, d, jnp.ones(d) * ls_guess, 1.0, 0.0, 0.25, kernel_type="rbf"
            )
            for ls_guess in ls_guesses
        ]
        + [
            DOGP_Classification(
                n_features // 2,
                "rbf",
                d,
                ls_guess * jnp.ones(d),
                1.0,
                1e-3,
                0.25,
                train_lengthscale=True,
            )
            for ls_guess in ls_guesses
        ]
        + [
            DOGP_Classification(
                n_features // 2,
                "rbf",
                d,
                ls_guess * jnp.ones(d),
                1.0,
                0.0,
                0.25,
                train_lengthscale=True,
            )
            for ls_guess in ls_guesses
        ]
        + [
            DORBF_Classification(
                rbf_points,
                ls_guess,
                1.0,
                1e-3,
                0.25,
                train_lengthscale=True,
                train_locs=True,
            )
            for ls_guess in ls_guesses
        ]
        + [
            DORBF_Classification(
                rbf_points,
                ls_guess,
                1.0,
                0,
                0.25,
                train_lengthscale=True,
                train_locs=True,
            )
            for ls_guess in ls_guesses
        ],
        Q=jnp.asarray(Q),
    )

    return [
        doehsgp,
        oehsgp,
        doegp,
        oegp,
        doerbf,
        oerbf,
        sdoe,
    ], [
        "DOE-HSGP",
        "OE-HSGP",
        "DOE-RFF",
        "OE-RFF",
        "DOE-RBF",
        "OE-RBF",
        "Our Model",
    ]


def pretrain(models, X, y, lrs, sample_type, n_samples=5):
    for model_idx, model in enumerate(tqdm(models)):
        model.pretrain_and_sample(
            X[:n_samples],
            y[:n_samples],
            lrs[model_idx],
            iters=150,
            sampling_type=sample_type[model_idx],
            verbose=True,
            n_samples=5,
            jit=False,
        )


def fit(models, X, y, n_pretrain=1000, n_batch=512):
    yhats = []
    yvars = []
    for model in tqdm(models):
        yhat, yvar = model.fit_minibatched(
            X[n_pretrain:], y[n_pretrain:], n_batch=n_batch
        )
        yhats.append(yhat)
        yvars.append(yvar)

    return yhats, yvars


if __name__ == "__main__":
    N_to_pretrain = 1000
    N_trials = 1
    N_models = 7
    SHUFFLE = True

    yhat_collection = []
    yvar_collection = []
    y_collection = []

    import objax
    import time

    t = int(time.time())
    objax.random.DEFAULT_GENERATOR.seed(t)

    model_names = ""
    print(f"Loading Data for banana")

    mat_obj = loadmat("data/banana.mat")["banana"]
    X = mat_obj["x"][0][0]
    X = (X - X[:N_to_pretrain].mean(0)) / X[:N_to_pretrain].std(0)
    y = (mat_obj["t"][0][0].squeeze() + 1) / 2

    if SHUFFLE:
        shuffle_idx = np.random.permutation(X.shape[0])

        X = X[shuffle_idx].squeeze()
        y = y[shuffle_idx].squeeze()

    from sklearn.linear_model import LogisticRegression
    from sklearn.gaussian_process import GaussianProcessClassifier

    clf = LogisticRegression(random_state=0).fit(X[:N_to_pretrain], y[:N_to_pretrain])
    print(clf.score(X, y))

    clf = GaussianProcessClassifier().fit(X[:N_to_pretrain], y[:N_to_pretrain])
    print(clf.score(X, y))

    X = jnp.array(X)
    y = jnp.array(y)

    yhat_trials = np.zeros((N_trials, N_models, y.shape[0] - N_to_pretrain))

    for trial in tqdm(range(N_trials)):
        print("Initializing Models")
        models, model_names = make_models(X)

        print("Pretraining Models")
        lrs = [2e-1] * 10
        sample_type = [
            "laplace",
            "laplace",
            "laplace",
            "laplace",
            "gaussian",
            "gaussian",
            ["laplace"] * 12 + ["gaussian"] * 6,
        ]

        pretrain(models, X, y, lrs, sample_type, n_samples=N_to_pretrain)

        print("Fitting Models")
        yhats, _ = fit(models, X, y, n_pretrain=N_to_pretrain)
        yhats = np.stack(yhats)
        yhat_trials[trial] = yhats

        for i in range(len(models)):
            print(y[-10:], yhats[i, -10:])
            print(
                i,
                np.sum(y[N_to_pretrain:] == np.round(yhats[i, :]))
                / (y.shape[0] - N_to_pretrain),
            )

    if SHUFFLE:
        np.save(f"yhat_trials_classification_shuffle_{t}", yhat_trials)
        np.save(f"y_shuffle_{t}", y)
    else:
        np.save(f"yhat_trials_classification_noshuffle_{t}", yhat_trials)
