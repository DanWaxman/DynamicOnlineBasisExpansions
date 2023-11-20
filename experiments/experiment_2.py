import jax.numpy as jnp
import jax
from DOEBE.doebe import DOEBE
from DOEBE.sdoebe import SDOEBE
from DOEBE.models import *
from tqdm import tqdm
import numpy as np
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

jax.config.update("jax_enable_x64", True)
sns.set_style("whitegrid")
sns.set_palette("colorblind")
sns.set_context("paper")

from experiment_utils import *

X, y = get_data("kuka1")

d = X.shape[1]


def run_experiment(N_to_plot=196000):
    var_eps_vals = [0.0, 1e-3]
    L = np.max([np.abs(np.min(X, axis=0)), np.abs(np.max(X, axis=0))], axis=0) * 1.5
    M = 100 // d

    print("Pretraining DOEBE")
    ls_guess = jnp.ones(d)
    dogp_list = [
        DOAddHSGP(L, M, d, ls_guess, 1.0, var_eps, 0.25) for var_eps in var_eps_vals
    ]
    doebe = DOEBE(dogp_list)
    doebe.pretrain(X[:1000], y[:1000])

    print("Pretraining E-DOEBE")
    delta = 1e-2
    dogp_list = [
        DOAddHSGP(L, M, d, ls_guess, 1.0, var_eps, 0.25) for var_eps in var_eps_vals
    ]
    Q = jnp.array([[1 - delta, delta], [delta, 1 - delta]])
    sdoebe = SDOEBE(dogp_list, Q)
    sdoebe.pretrain(X[:1000], y[:1000])

    print("Fitting DOEBE")
    y_mean, y_var, ws = doebe.fit(
        X[1000 : 1000 + N_to_plot], y[1000 : 1000 + N_to_plot], return_ws=True
    )
    doebe_ws = ws[:, 1]

    print(
        scipy.stats.norm.logpdf(
            y[1000 : 1000 + N_to_plot], loc=y_mean, scale=jnp.sqrt(y_var)
        ).mean()
    )

    print("Fitting E-DOEBE")
    y_mean, y_var, ws = sdoebe.fit(
        X[1000 : 1000 + N_to_plot], y[1000 : 1000 + N_to_plot], return_ws=True
    )
    sdoebe_ws = ws[:, 1]

    print(
        scipy.stats.norm.logpdf(
            y[1000 : 1000 + N_to_plot], loc=y_mean, scale=jnp.sqrt(y_var)
        ).mean()
    )

    doebe_ws = np.asarray(doebe_ws)
    sdoebe_ws = np.asarray(sdoebe_ws)

    def moving_average(a, n=10):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    fig = plt.figure(figsize=(4, 2))
    plt.title("Weight of Dynamic Model on Kuka1")
    plt.xscale("log")
    plt.plot(moving_average(doebe_ws), label="DOEBE")
    plt.plot(moving_average(sdoebe_ws), label="E-DOEBE")
    plt.xlabel("Number of Data Points Seen")
    plt.ylabel("Weight of Dynamic Model")
    plt.legend()
    plt.savefig("plots/Exp2.pdf", bbox_inches="tight")


run_experiment()
