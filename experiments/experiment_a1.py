import jax.numpy as jnp
import jax
from DOEBE.doebe import DOEBE
from DOEBE.models import *
from tqdm import tqdm
import numpy as np
from scipy.cluster.vq import kmeans2
import seaborn as sns

# get_data, plot_results, plot_summary_nmse, plot_summary_pll
from experiment_utils import *

jax.config.update("jax_enable_x64", True)
sns.set_style('whitegrid')
sns.set_palette('colorblind')


def make_models(X, n_features=100):
    d = X.shape[1]
    M = 100 // d

    L = np.max([np.abs(np.min(X, axis=0)), np.abs(
        np.max(X, axis=0))], axis=0)*1.5

    ls_guesses = [(jnp.max(X, axis=0) - jnp.min(X, axis=0)) /
                  f for f in [0.1, 1, 10]]
    dehsgp_1 = DOEBE([DOAddHSGP(L, M, d, jnp.ones(d) * ls_guess, 1.0,
                                1e-3, 0.25, kernel_type='rbf') for ls_guess in ls_guesses])
    dehsgp_2 = DOEBE([DOAddHSGP(L, M, d, jnp.ones(d) * ls_guess, 1.0,
                                1e-3, 0.25, kernel_type='rbf') for ls_guess in ls_guesses])

    diegp_1 = DOEBE([DOGP(n_features//2, 'rbf', d, ls_guess*jnp.ones(d), 1.0,
                          1e-3, 0.25, train_lengthscale=True) for ls_guess in ls_guesses])
    diegp_2 = DOEBE([DOGP(n_features//2, 'rbf', d, ls_guess*jnp.ones(d), 1.0,
                          1e-3, 0.25, train_lengthscale=True) for ls_guess in ls_guesses])

    for model_idx, model in enumerate(diegp_1.models):
        diegp_2.models[model_idx].freqs = model.freqs

    return [dehsgp_1, dehsgp_2, diegp_1, diegp_2], ['DOE-HSGP-MLE', 'DOE-HSGP-Sample', 'DOE-RFF-MLE', 'DOE-RFF-Sample']


def pretrain(models, X, y, lrs, sample_type, n_samples=1000):
    for model_idx, model in enumerate(tqdm(models)):
        if sample_type[model_idx] == 'none':
            model.pretrain(X[:n_samples], y[:n_samples],
                           lrs[model_idx], verbose=False)
        else:
            model.pretrain_and_sample(X[:n_samples], y[:n_samples], lrs[model_idx],
                                      sampling_type=sample_type[model_idx], verbose=False, n_samples=10)


def fit(models, X, y, n_pretrain=1000, n_batch=2500):
    yhats = []
    yvars = []
    for model in tqdm(models):
        yhat, yvar = model.fit_minibatched(
            X[n_pretrain:], y[n_pretrain:], n_batch=n_batch)
        yhats.append(yhat)
        yvars.append(yvar)

    return yhats, yvars


if __name__ == '__main__':
    ANALYSIS = True
    if ANALYSIS:
        from scipy.stats import wilcoxon
        np.set_printoptions(
            formatter={'float': lambda x: "{0:0.3f}".format(x)})

        def print_means_stds(name, arr):
            print(f'Results for {name}')
            print(np.mean(arr, axis=0), np.std(arr, axis=0))

        def print_significance(name, arr, test):
            print(f'Significance Results for {name}')
            print('> HSGP')
            print('>', wilcoxon(arr[:, 0], arr[:, 1], alternative=test))
            print('> RFF')
            print('>', wilcoxon(arr[:, 2], arr[:, 3], alternative=test))

        nmse_elevators = np.loadtxt('ExpA1_elevators_nmse.out')
        nmse_SARCOS = np.loadtxt('ExpA1_SARCOS_nmse.out')
        nmse_cpusmall = np.loadtxt('ExpA1_cpusmall_nmse.out')

        pll_elevators = np.loadtxt('ExpA1_elevators_pll.out')
        pll_SARCOS = np.loadtxt('ExpA1_SARCOS_pll.out')
        pll_cpusmall = np.loadtxt('ExpA1_cpusmall_pll.out')

        print_means_stds('Elevators NMSE', nmse_elevators)
        print_significance('Elevators NMSE', nmse_elevators, 'greater')
        print_means_stds('SARCOS NMSE', nmse_SARCOS)
        print_significance('SARCOS NMSE', nmse_SARCOS, 'greater')
        print_significance('SARCOS NMSE', nmse_SARCOS, 'less')
        print_means_stds('cpusmall NMSE', nmse_cpusmall)
        print_significance('cpusmall NMSE', nmse_cpusmall, 'greater')

        print_means_stds('Elevators pll', pll_elevators)
        print_significance('Elevators pll', pll_elevators, 'less')
        print_means_stds('SARCOS pll', pll_SARCOS)
        print_significance('SARCOS pll', pll_SARCOS, 'less')
        print_means_stds('cpusmall pll', pll_cpusmall)
        print_significance('cpusmall pll', pll_cpusmall, 'less')
    else:
        datasets = ['elevators',
                    'SARCOS',
                    'cpusmall']
        N_to_pretrain = 1000
        N_trials = 100
        N_models = 4

        yhat_collection = []
        yvar_collection = []
        y_collection = []

        model_names = ''
        for dataset in datasets:
            print(f'Loading Data for {dataset}')
            X, y = get_data(dataset)

            yhat_trials = np.zeros(
                (N_trials, N_models, y.shape[0] - N_to_pretrain))
            yvar_trials = np.zeros(
                (N_trials, N_models, y.shape[0] - N_to_pretrain))

            for trial in range(N_trials):
                print('Initializing Models')
                models, model_names = make_models(X)

                print('Pretraining Models')
                lrs = [1e-2] * 10
                sample_type = ['none', 'laplace', 'none', 'laplace',]
                pretrain(models, X, y, lrs, sample_type,
                         n_samples=N_to_pretrain)

                print('Fitting Models')
                yhats, yvars = fit(models, X, y, n_pretrain=N_to_pretrain)
                yhats = np.stack(yhats)
                yvars = np.stack(yvars)
                yhat_trials[trial] = yhats
                yvar_trials[trial] = yvars

                print('Plotting Results')

            yhat_collection.append(yhat_trials)
            yvar_collection.append(yvar_trials)
            y_collection.append(y[N_to_pretrain:])
            plot_results_avg(yhat_trials, yvar_trials,
                             y[N_to_pretrain:], model_names, dataset, prefix='ExpA1')
            # plot_results(yhats, yvars, y[N_to_pretrain:], model_names, dataset, prefix='Exp1')

        plot_summary_nmse(yhat_collection, y_collection,
                          model_names, datasets, prefix='ExpA1')
        plot_summary_pll(yhat_collection, yvar_collection,
                         y_collection, model_names, datasets, prefix='ExpA1')
