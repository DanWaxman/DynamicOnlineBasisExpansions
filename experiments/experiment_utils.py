from libsvmdata import fetch_libsvm
from scipy.io import loadmat
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from typing import Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import math


def get_data(dataset):
    if dataset == "friedman1":  # 10 dimensional
        from sklearn.datasets import make_friedman1

        X, y = make_friedman1(n_samples=40000, n_features=10, noise=0.1, random_state=0)
        y = (y - y.mean()) / y.std()
    elif dataset == "friedman2":  # 4 dimensional
        from sklearn.datasets import make_friedman2

        X, y = make_friedman2(n_samples=40000, noise=1, random_state=0)
        y = (y - y.mean()) / y.std()
    elif dataset == "elevators":  # 18 dimensional
        data = np.asarray(loadmat("data/elevators.mat")["data"])
        X = data[:, :-3]
        X = np.hstack([X, data[:, -2:-1]])
        y = data[:, -1]
    elif dataset == "SARCOS":  # 21 dimensional
        sarcos_data = loadmat("data/sarcos_inv.mat")["sarcos_inv"].astype(np.float64)

        X = sarcos_data[:, :21]

        y = sarcos_data[:, 21]
    elif dataset == "kuka1":  # 21 dimensional
        df = pd.read_csv("data/kuka1_offline.txt", sep=" ", header=None)
        X = df.iloc[:, :21].values
        y = df.iloc[:, 21].values

        df = pd.read_csv("data/kuka1_online.txt", sep=" ", header=None)
        X = np.concatenate([X, df.iloc[:, :21].values])
        y = np.concatenate([y, df.iloc[:, 21].values])
    elif dataset == "cadata":  # 8 dimensional
        X, y = fetch_libsvm("cadata")
    elif dataset == "cpusmall":  # 12 dimensional
        X, y = fetch_libsvm("cpusmall")
    else:
        raise NotImplementedError

    X = (X - X[:1000].mean(axis=0)) / (X[:1000].std(axis=0) + np.finfo(np.float64).eps)
    X = jnp.clip(X, -1e3, 1e3)
    y = (y - y[:1000].mean()) / y[:1000].std()
    y = jnp.clip(y, -1e3, 1e3)

    X = jnp.asarray(X)
    y = jnp.asarray(y)

    return X, y


def plot_results(yhats, yvars, y_true, model_names, dataset_name, prefix):
    plt.clf()
    plt.cla()
    fig = plt.gcf()
    ax = plt.gca()
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    fig.set_size_inches(4, 4)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 27, 33]

    m = 0
    for model_idx, model_name in enumerate(model_names):
        err = np.cumsum((y_true - yhats[model_idx]) ** 2) / np.arange(
            1, y_true.shape[0] + 1
        )
        m = max(m, np.max(err[1000:]))
        plt.plot(
            err, label=model_name, linestyle="--", dashes=(primes[model_idx] - 1, 1)
        )
    m = min(m, 1.0)

    plt.title(dataset_name)
    plt.xlim(0, len(y_true))
    plt.ylim([0, 1.1 * m])
    plt.ylabel("nMSE")
    plt.xlabel("Number of Data Points Seen")
    # plt.legend()

    plt.savefig(f"plots/{prefix}_NMSE_" + dataset_name + ".pdf", bbox_inches="tight")

    plt.clf()
    plt.cla()
    fig = plt.gcf()
    fig.set_size_inches(3.5, 4)

    pred_log_likelihoods = []
    for model_idx, model_name in enumerate(model_names):
        pred_log_likelihood = (
            jax.scipy.stats.norm.logpdf(
                y_true, yhats[model_idx], jnp.sqrt(yvars[model_idx])
            )
            .mean()
            .item()
        )
        pred_log_likelihoods.append(pred_log_likelihood)
        print(model_name, pred_log_likelihood)

    sns.barplot(x=model_names, y=pred_log_likelihoods)
    plt.ylim([-2, 2])
    plt.xticks(rotation=70)
    plt.title(dataset_name)
    plt.ylabel("Predictive Log Likelihood")
    plt.xlabel("Model")
    plt.savefig(f"plots/{prefix}_PLL_" + dataset_name + ".pdf", bbox_inches="tight")


def plot_results_avg(
    yhats, yvars, y_true, model_names, dataset_name, prefix, freq=100, color_idxs=None
):
    plt.clf()
    plt.cla()
    fig = plt.gcf()
    ax = plt.gca()
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    fig.set_size_inches(4, 4)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 27, 33]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"] + [
        [0.75, 0.75, 0.75]
    ]

    if color_idxs is None:
        color_idxs = list(range(len(model_names)))

    m = 0
    for model_idx, model_name in enumerate(model_names):
        err = np.cumsum(
            (y_true.reshape(1, -1) - yhats[:, model_idx]) ** 2, axis=1
        ) / np.reshape(np.arange(1, y_true.shape[0] + 1), (1, -1))
        err_mean = np.mean(err, axis=0)[::freq]
        err_std = np.std(err, axis=0)[::freq]
        m = max(m, np.max(err_mean[1000 // freq :]))
        plt.plot(
            np.arange(len(err_mean)) * freq,
            err_mean,
            label=model_name,
            linestyle="--",
            dashes=(primes[model_idx] - 1, 1),
            color=color_cycle[color_idxs[model_idx]],
        )
        plt.fill_between(
            np.arange(len(err_mean)) * freq,
            err_mean - err_std,
            err_mean + err_std,
            alpha=0.5,
            color=color_cycle[color_idxs[model_idx]],
        )
    m = min(m, 1.0)

    plt.title(dataset_name)
    plt.xlim(0, len(y_true))
    plt.ylim([0, 1.1 * m])
    plt.ylabel("nMSE")
    plt.xlabel("Number of Data Points Seen")

    plt.savefig(
        f"plots/{prefix}_NMSE_Avg_" + dataset_name + ".pdf", bbox_inches="tight"
    )


def plot_summary_nmse(
    yhat_collection,
    y_collection,
    model_names,
    datasets,
    bar_width=0.25,
    prefix="",
    color_idxs=None,
):
    fig = plt.subplots(figsize=(6.2, 1.5))

    bar_positions = [
        bar_width * (np.arange(len(model_names)) + n * len(model_names) + n)
        for n in range(len(datasets))
    ]

    if color_idxs is None:
        color_idxs = list(range(len(model_names)))

    colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"]) + [
        [0.75, 0.75, 0.75]
    ]
    colors = [colors[color_idx] for color_idx in color_idxs]

    plt.yscale("log")
    for d_idx, dataset in enumerate(datasets):
        yhat = yhat_collection[d_idx]
        y = y_collection[d_idx]

        nmse = np.mean((y[None] - yhat) ** 2, axis=2) / y.var()
        np.savetxt(f"results/{prefix}_{dataset}_nmse.out", nmse)
        nmse = np.nan_to_num(nmse, nan=np.inf)
        nmse_mean = np.mean(nmse, axis=0)
        nmse_std = np.std(nmse, axis=0)

        plt.bar(
            bar_positions[d_idx],
            nmse_mean,
            yerr=nmse_std,
            color=colors,
            width=bar_width,
        )
        min_nmse_idx = np.argmin(nmse_mean)
        plt.bar(
            bar_positions[d_idx][min_nmse_idx],
            nmse_mean[min_nmse_idx],
            color=colors[min_nmse_idx],
            width=bar_width,
            linewidth=2,
            edgecolor="black",
        )

    plt.ylim([0.0, 1.0])

    plt.ylabel("nMSE")
    plt.xticks(
        [np.mean(bar_position) for bar_position in bar_positions],
        [""] * len(bar_positions),
    )
    plt.savefig(f"plots/{prefix}_nMSE_Summary.pdf", bbox_inches="tight")


def plot_summary_pll(
    yhat_collection,
    yvar_collection,
    y_collection,
    model_names,
    datasets,
    bar_width=0.25,
    prefix="",
    color_idxs=None,
):
    fig = plt.subplots(figsize=(6.2, 1.5))

    bar_positions = [
        bar_width * (np.arange(len(model_names)) + n * len(model_names) + n)
        for n in range(len(datasets))
    ]

    if color_idxs is None:
        color_idxs = list(range(len(model_names)))

    colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"]) + [
        [0.75, 0.75, 0.75]
    ]
    colors = [colors[color_idx] for color_idx in color_idxs]

    for d_idx, dataset in enumerate(datasets):
        yhat = yhat_collection[d_idx]
        yvar = yvar_collection[d_idx]
        y = y_collection[d_idx]

        pll = jax.scipy.stats.norm.logpdf(y, yhat, jnp.sqrt(yvar)).mean(axis=2)
        pll = np.nan_to_num(pll, nan=-np.inf)
        np.savetxt(f"results/{prefix}_{dataset}_pll.out", pll)
        pll_mean = np.mean(pll, axis=0)
        pll_std = np.std(pll, axis=0)
        print(pll_mean, pll_std)

        plt.bar(
            bar_positions[d_idx],
            pll_mean,
            yerr=pll_std,
            color=colors,
            width=bar_width,
        )
        max_pll_idx = np.argmax(pll_mean)
        plt.bar(
            bar_positions[d_idx][max_pll_idx],
            pll_mean[max_pll_idx],
            color=colors[max_pll_idx],
            width=bar_width,
            linewidth=2,
            edgecolor="black",
        )
        plt.ylim([-2.5, 2.5])

    plt.ylabel("PLL")
    plt.xticks(
        [np.mean(bar_position) for bar_position in bar_positions], datasets, rotation=70
    )
    plt.savefig(f"plots/{prefix}_PLL_Summary.pdf", bbox_inches="tight")


if __name__ == "__main__":
    datasets = [
        "friedman1",
        "friedman2",
        "elevators",
        "SARCOS",
        "kuka1",
        "cadata",
        "cpusmall",
    ]
    for dataset in datasets:
        X, y = get_data(dataset)
        print(dataset, X.shape, y.shape)

    # sns.set_style('whitegrid')
    sns.set_palette("colorblind")

    model_names = [
        "DOE-HSGP",
        "OE-HSGP",
        "DOE-RFF",
        "OE-RFF",
        "DOE-RBF",
        "OE-RBF",
        "S-DOEBE",
    ]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"] + [
        [0.75, 0.75, 0.75]
    ]
    idxs = [0, 1, 2, 3, 8, 9, 10]
    colors = [color_cycle[i] for i in idxs]

    def f(m, c):
        return plt.plot([], [], marker=m, color=c, ls="none")[0]

    handles = [f("s", colors[i]) for i in range(len(model_names))]
    labels = model_names
    legend = plt.legend(
        handles,
        labels,
        ncol=3,  # int(math.ceil(len(model_names) / 2)),
        loc=3,
        framealpha=1,
        frameon=True,
        fontsize=10,
    )

    def export_legend(
        legend, filename="plots/Appendix_Exp3_Legend.pdf", expand=[-5, -0, 5, 0]
    ):
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, bbox_inches=bbox)

    export_legend(legend)

    yhat_collection = [np.random.randn(3, 100), np.random.randn(3, 100)]
    yvar_collection = [np.random.randn(3, 100) ** 2, np.random.randn(3, 100) ** 2]
    y_collection = [np.random.randn(100), np.random.randn(100)]
    model_names = ["model1", "model2", "model3"]
    datasets = ["dataset1", "dataset2"]
    plot_summary_nmse(yhat_collection, y_collection, model_names, datasets)
    plot_summary_pll(
        yhat_collection, yvar_collection, y_collection, model_names, datasets
    )
