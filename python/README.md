# pydme

**Python Dynamic Mode Extraction** — a PyTorch implementation of Temporal Matrix Integration (TMI) with sparse SINDy-style dynamics and feature models.

## Overview

`pydme` decomposes a non-negative data matrix **X** (e.g. neural firing rates, counts) into interpretable latent factors using the model:

```
q = softmax_col( -Y @ z.T )
```

and minimises the column-normalised KL divergence D_KL(**X** ‖ **q**).

Three model variants are provided:

| Class | Description |
|---|---|
| `TMI` | Core factorisation: learn `z` (time-domain) and `Y` (feature-domain) latents |
| `TMIWithDynamics` | Adds a sparse polynomial dynamics constraint `DW @ z ≈ W @ θ(z) @ ξz` |
| `TMIWithFeatureModel` | Also constrains `Y ≈ θy @ ξy` using a provided feature library |

Sparse models (`ξz`, `ξy`) are discovered automatically via **EnAdSR** — an ensemble sparse regression method with SLIC / AICc / BIC model selection.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0, NumPy ≥ 1.24.

## Quick start

### Core TMI (2-D data)

```python
import numpy as np
from pydme import TMI

X = np.random.exponential(1.0, (50, 100))   # (n_features, n_timepoints)
model = TMI(K=5, normalize=True, lr=1e-3, max_iters=10_000)
model.fit(X)

z  = model.z_   # (n_timepoints, K) — temporal latents
Y  = model.Y_   # (n_features,  K) — spatial  latents
q  = model.q_   # (n_features, n_timepoints) — reconstruction
```

### TMI with sparse dynamics

```python
from pydme import TMIWithDynamics
import numpy as np

# Galerkin matrices (user-supplied; here W=I, DW=finite-difference)
T = 100
W  = np.eye(T)
DW = np.diag(-np.ones(T)) + np.diag(np.ones(T - 1), k=1)

# Polynomial exponent matrix: constant + linear + quadratic in each dim
N = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [0, 2]])   # (L, K)

model = TMIWithDynamics(
    K=2, W=W, DW=DW, N=N,
    lambda_z=10.0,          # dynamics constraint weight
    sparse_iter=5_000,      # re-run EnAdSR every 5000 iters
    lr=1e-3,
    max_iters=50_000,
)
model.fit(X)

xi_z = model.xi_z_   # (L, K) sparse dynamics coefficients
```

### TMI with dynamics + feature model

```python
from pydme import TMIWithFeatureModel

theta_y = ...   # (n_timepoints, n_features_in_library) — user-provided

model = TMIWithFeatureModel(
    K=2, W=W, DW=DW, N=N, theta_y=theta_y,
    lambda_z=10.0, lambda_y=10.0,
    lr=1e-3, max_iters=50_000,
)
model.fit(X)

xi_y = model.xi_y_   # (n_features_in_library, K)
```

### Hyperparameter search

```python
from pydme.models import scan_hyperparams, score_models

results = scan_hyperparams(
    X, TMIWithFeatureModel,
    param_grid={"lambda_z": [1., 10., 100.], "lambda_y": [1., 10.]},
    K=3, W=W, DW=DW, N=N, theta_y=theta_y,
    max_iters=20_000, verbose=False,
)

scores = score_models(results, N=N, W=W, DW=DW, theta_y=theta_y)
best = results[scores.index(min(scores))]
```

### Sparse regression (standalone)

```python
from pydme import AdSR, EnAdSR
import torch

theta = torch.randn(200, 10)
y     = torch.randn(200,  3)

# Single run
reg = AdSR(ic="slic", max_iter=10).fit(theta, y)
print(reg.coef_)     # (10, 3)

# Ensemble (more robust)
ereg = EnAdSR(ic="slic", num_batches=20, tol=0.7).fit(theta, y)
print(ereg.coef_)           # (10, 3) — sparse
print(ereg.inclusion_probs_)
```

## 3-D data (multi-trial)

All models accept 3-D data of shape `(n_features, n_timepoints, n_trials)`.
`Y` is shared across trials; `z` is trial-specific.

```python
X3 = np.random.exponential(1.0, (50, 100, 20))   # 20 trials
model = TMI(K=3).fit(X3)
# model.z_.shape == (n_timepoints, K, n_trials)
# model.Y_.shape == (n_features, K)
```

## Design notes

- **Gradients via autograd**: loss gradients for `z` and `Y` are computed by PyTorch autograd, making it easy to use GPU acceleration (`X = torch.tensor(X).cuda()`).
- **Sparse coefficients via EnAdSR**: `ξz` and `ξy` are updated by ensemble sparse regression (not autograd), which provides automatic model-order selection.
- **scikit-learn API**: hyperparameters in `__init__`, `fit()` returns `self`, fitted attributes end in `_`.
