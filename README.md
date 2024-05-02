# Dynamic Online Ensembles of Basis Expansions
This repository contains the code for [Dynamic Online Ensembles of Basis Expansions](https://openreview.net/forum?id=aVOzWH1Nc5) (DOEBE), published in Transactions on Machine Learning Research (TMLR). The paper proposes a generalization of online Gaussian process methods using the random feature approximation to general basis expansions, including Hilbert space Gaussian processes. 

# Citation
The DOEBE paper is available open-access on [OpenReview](https://openreview.net/forum?id=aVOzWH1Nc5). If you use any code or results from this project, please consider citing the orignal paper:

```
@article{
waxman2024doebe,
title={Dynamic Ensembles of Basis Expansions},
author={Daniel Waxman and Petar M. Djuri\'c},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=aVOzWH1Nc5},
}
```

# Installation Instructions 

To install DOEBE, you can download the git repository and install the package using `pip`:

```
git clone https://github.com/DanWaxman/DynamicOnlineBasisExpansions
cd DynamicOnlineBasisExpansions/src
pip install -e .
```

# Example

Examples used for experiments in the paper can be found in `experiments/`. In general, creating a DOEBE starts with a list of `DOEBE.models.DOBE` instances. After instantiating a list of `DOBE` models, hyperparameters can be tuned with empirical Bayes via `doebe.pretrain(X, y)`. After that, the DOEBE model can be fit to data with `doebe.fit(X, y)`. See a minimal example below, which is a fragment of `experiments/experiment_2.py`.

```
import jax
import jax.numpy as jnp
import numpy as np
from DOEBE.doebe import DOEBE
from DOEBE.models import *

jax.config.update(
    "jax_enable_x64", True
)  # Use double precision, especially for empirical Bayes

from experiment_utils import *

# Get data
X, y = get_data("Kuka #1")

# Prepare GP lengthscale initialization as mentioned in the paper
d = X.shape[1]
L = np.max([np.abs(np.min(X, axis=0)), np.abs(np.max(X, axis=0))], axis=0) * 1.5
M = 100 // d

# Choose one model to be static, and one to be dynamic
var_eps_vals = [0.0, 1e-3]

# Create DOEBE Model of HSGPs
ls_guess = jnp.ones(d)
dogp_list = [
    DOAddHSGP(L, M, d, ls_guess, 1.0, var_eps, 0.25) for var_eps in var_eps_vals
]
doebe = DOEBE(dogp_list)

print("Pretraining DOEBE")
doebe.pretrain(X[:1000], y[:1000])

print("Fitting DOEBE")
y_mean, y_var, ws = doebe.fit(X[1000:], y[1000:], return_ws=True)

```

To use a SDOEBE (for example, the E-DOEBE discussed in the paper), the API is overall similar besides the inclusion of a Q matrix:
```
# Create SDOEBE/E-DOEBE Model
delta = 1e-2
dogp_list = [
    DOAddHSGP(L, M, d, ls_guess, 1.0, var_eps, 0.25) for var_eps in var_eps_vals
]
Q = jnp.array([[1 - delta, delta], [delta, 1 - delta]])
sdoebe = SDOEBE(dogp_list, Q)

print("Pretraining E-DOEBE")
sdoebe.pretrain(X[:1000], y[:1000])

print("Fitting E-DOEBE")
y_mean, y_var, ws = sdoebe.fit(X[1000:], y[1000:], return_ws=True)
``` 