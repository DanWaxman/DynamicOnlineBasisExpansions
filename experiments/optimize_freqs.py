from experiment_utils import *
import jax.numpy as jnp
import jax
from DOEBE.doebe import DOEBE
from DOEBE.models import *
from tqdm import tqdm
import numpy as np
from scipy.cluster.vq import kmeans2
import seaborn as sns

jax.config.update("jax_enable_x64", True)
sns.set_style("whitegrid")
sns.set_palette("colorblind")


def make_models(X, n_features=100):
    d = X.shape[1]
    M = 100 // d

    L = np.max([np.abs(np.min(X, axis=0)), np.abs(np.max(X, axis=0))], axis=0) * 1.5

    ls_guesses = [(jnp.max(X, axis=0) - jnp.min(X, axis=0)) / f for f in [0.1, 1, 10]]
    doegp = DOEBE(
        [
            DOGP(
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
    doegp_3 = DOEBE(
        [
            DOGP(
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
    doegp_2 = DOEBE(
        [
            DOGP(
                n_features // 2,
                "rbf",
                d,
                ls_guess * jnp.ones(d),
                1.0,
                1e-3,
                0.25,
                train_lengthscale=True,
                train_frequencies=True,
            )
            for ls_guess in ls_guesses
        ]
    )

    return [
        doegp,
        doegp_2,
    ], [
        "DOE-RFF",
        "DOE-OFF",
    ]


def pretrain(models, X, y, lrs, sample_type, n_samples=1000):
    for model_idx, model in enumerate(tqdm(models)):
        model.pretrain_and_sample(
            X[:n_samples],
            y[:n_samples],
            lrs[model_idx],
            sampling_type=sample_type[model_idx],
            verbose=False,
            n_samples=5,
        )


def fit(models, X, y, n_pretrain=1000, n_batch=2500):
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
    datasets = [
        "Friedman #1",
        "Friedman #2",
        "Elevators",
        "SARCOS",
        "Kuka #1",
        "CaData",
        "CPU Small",
    ]
    N_to_pretrain = 1000
    N_trials = 10
    N_models = 2

    yhat_collection = []
    yvar_collection = []
    y_collection = []

    model_names = ""
    for dataset in datasets:
        print(f"Loading Data for {dataset}")
        X, y = get_data(dataset)

        yhat_trials = np.zeros((N_trials, N_models, y.shape[0] - N_to_pretrain))
        yvar_trials = np.zeros((N_trials, N_models, y.shape[0] - N_to_pretrain))

        for trial in range(N_trials):
            print("Initializing Models")
            models, model_names = make_models(X)

            print("Pretraining Models")
            lrs = [1e-2] * 10
            sample_type = ["gaussian", "gaussian"]
            pretrain(models, X, y, lrs, sample_type, n_samples=N_to_pretrain)

            print("Fitting Models")
            yhats, yvars = fit(models, X, y, n_pretrain=N_to_pretrain)
            yhats = np.stack(yhats)
            yvars = np.stack(yvars)
            yhat_trials[trial] = yhats
            yvar_trials[trial] = yvars

            print("Plotting Results")

        yhat_collection.append(yhat_trials)
        yvar_collection.append(yvar_trials)
        y_collection.append(y[N_to_pretrain:])
        plot_results_avg(
            yhat_trials,
            yvar_trials,
            y[N_to_pretrain:],
            model_names,
            dataset,
            prefix="Exp4",
        )

    plot_summary_nmse(
        yhat_collection, y_collection, model_names, datasets, prefix="Exp4"
    )
    plot_summary_pll(
        yhat_collection,
        yvar_collection,
        y_collection,
        model_names,
        datasets,
        prefix="Exp4",
    )
