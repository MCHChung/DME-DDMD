"""Basic smoke tests for pydme."""

import math

import numpy as np
import pytest
import torch

from pydme import TMI, TMIWithDynamics, TMIWithFeatureModel
from pydme.library import poly_library
from pydme.losses import kld_loss
from pydme.sparse import AdSR, EnAdSR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_2d_data(a=20, alpha=30, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.exponential(1.0, (a, alpha)) + 1e-3
    return X.astype(np.float64)


def _make_3d_data(a=20, alpha=30, T=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.exponential(1.0, (a, alpha, T)) + 1e-3
    return X.astype(np.float64)


def _make_galerkin(T=30, seed=1):
    """Toy Galerkin matrices W (identity) and DW (finite differences)."""
    rng = np.random.default_rng(seed)
    W = np.eye(T, dtype=np.float64)
    # Forward difference as a crude derivative matrix
    DW = np.zeros((T, T), dtype=np.float64)
    for i in range(T - 1):
        DW[i, i] = -1.0
        DW[i, i + 1] = 1.0
    return W, DW


def _poly_exponents(K=2, order=1):
    """Exponent matrix for a K-dim polynomial library up to total degree ``order``.

    Returns an (L, K) array where each row is a multi-index and the sum of
    each row is <= ``order``.
    """
    from itertools import product as iproduct

    terms = [
        combo
        for combo in iproduct(range(order + 1), repeat=K)
        if sum(combo) <= order
    ]
    return np.array(terms, dtype=np.float64)


# ---------------------------------------------------------------------------
# Library tests
# ---------------------------------------------------------------------------


class TestPolyLibrary:
    def test_2d_constant_term(self):
        z = torch.ones(10, 2, dtype=torch.float64)
        N = torch.zeros(1, 2, dtype=torch.float64)  # constant
        theta = poly_library(z, N)
        assert theta.shape == (10, 1)
        assert (theta == 1.0).all()

    def test_2d_linear_terms(self):
        z = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
        N = torch.tensor([[1, 0], [0, 1]], dtype=torch.float64)
        theta = poly_library(z, N)
        assert theta.shape == (1, 2)
        assert math.isclose(theta[0, 0].item(), 2.0)
        assert math.isclose(theta[0, 1].item(), 3.0)

    def test_3d_shape(self):
        T, K, trials = 10, 2, 4
        z = torch.randn(T, K, trials, dtype=torch.float64)
        N = torch.tensor([[1, 0], [0, 1], [2, 0]], dtype=torch.float64)
        theta = poly_library(z, N)
        assert theta.shape == (T, 3, trials)


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------


class TestKLDLoss:
    def test_2d_same_distribution(self):
        p = torch.ones(5, 10, dtype=torch.float64)
        # KLD of identical distributions should be ~0
        loss = kld_loss(p, p.clone())
        assert loss.item() < 1e-8

    def test_2d_positive(self):
        p = torch.rand(5, 10, dtype=torch.float64) + 0.1
        q = torch.rand(5, 10, dtype=torch.float64) + 0.1
        assert kld_loss(p, q).item() >= 0

    def test_3d(self):
        p = torch.rand(5, 10, 3, dtype=torch.float64) + 0.1
        q = torch.rand(5, 10, 3, dtype=torch.float64) + 0.1
        loss = kld_loss(p, q)
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# Sparse regression tests
# ---------------------------------------------------------------------------


class TestAdSR:
    def test_recovers_dense_solution(self):
        """AdSR with no threshold should recover the OLS solution."""
        rng = torch.Generator()
        rng.manual_seed(42)
        n, p = 100, 5
        theta = torch.randn(n, p, generator=rng, dtype=torch.float64)
        xi_true = torch.randn(p, 2, generator=rng, dtype=torch.float64)
        y = theta @ xi_true + 0.01 * torch.randn(n, 2, generator=rng, dtype=torch.float64)

        reg = AdSR(ic="slic", max_iter=1, c=0.0, use_cond=False)
        reg.fit(theta, y)
        assert reg.coef_.shape == (p, 2)

    def test_ic_options(self):
        rng = torch.Generator()
        rng.manual_seed(7)
        theta = torch.randn(80, 4, generator=rng, dtype=torch.float64)
        y = torch.randn(80, 2, generator=rng, dtype=torch.float64)
        for ic in ("slic", "aicc", "bic"):
            reg = AdSR(ic=ic, max_iter=3).fit(theta, y)
            assert reg.coef_.shape == (4, 2)


class TestEnAdSR:
    def test_output_shapes(self):
        rng = torch.Generator()
        rng.manual_seed(0)
        theta = torch.randn(100, 5, generator=rng, dtype=torch.float64)
        y = torch.randn(100, 3, generator=rng, dtype=torch.float64)
        reg = EnAdSR(num_batches=5).fit(theta, y)
        assert reg.coef_.shape == (5, 3)
        assert reg.scores_.shape == (5,)
        assert reg.inclusion_probs_.shape == (5, 3)

    def test_inclusion_probs_in_range(self):
        rng = torch.Generator()
        rng.manual_seed(1)
        theta = torch.randn(100, 4, generator=rng, dtype=torch.float64)
        y = torch.randn(100, 2, generator=rng, dtype=torch.float64)
        reg = EnAdSR(num_batches=8).fit(theta, y)
        assert (reg.inclusion_probs_ >= 0).all()
        assert (reg.inclusion_probs_ <= 1).all()


# ---------------------------------------------------------------------------
# TMI tests
# ---------------------------------------------------------------------------


class TestTMI:
    def test_2d_fit_shapes(self):
        X = _make_2d_data(a=15, alpha=20)
        model = TMI(K=3, max_iters=200, tol=1e-2, verbose=False)
        model.fit(X)
        a, alpha = X.shape
        assert model.z_.shape == (alpha, 3)
        assert model.Y_.shape == (a, 3)
        assert model.q_.shape == (a, alpha)
        assert model.E_ is None

    def test_2d_with_E(self):
        X = _make_2d_data(a=12, alpha=18)
        model = TMI(K=2, use_E=True, max_iters=100, tol=1e-2, verbose=False)
        model.fit(X)
        assert model.E_ is not None
        assert model.E_.shape == (12, 1)

    def test_3d_fit_shapes(self):
        # X: (a=10, alpha=15, T=4) — (n_features, n_timepoints, n_trials)
        X = _make_3d_data(a=10, alpha=15, T=4)
        model = TMI(K=2, max_iters=100, tol=1e-2, verbose=False)
        model.fit(X)
        assert model.z_.shape == (15, 2, 4)   # (alpha, K, n_trials)
        assert model.Y_.shape == (10, 2)       # (a, K)
        assert model.q_.shape == (10, 15, 4)  # (a, alpha, n_trials)

    def test_transform_matches_q(self):
        X = _make_2d_data()
        model = TMI(K=2, max_iters=100, tol=1e-2, verbose=False).fit(X)
        q_transformed = model.transform()
        assert torch.allclose(q_transformed, model.q_, atol=1e-6)

    def test_loss_decreases(self):
        X = _make_2d_data(a=20, alpha=25)
        model = TMI(K=3, max_iters=2000, tol=1e-6, verbose=False)
        model.fit(X)
        # Loss should be lower at the end than at the beginning
        assert model.losses_[-1].item() <= model.losses_[0].item() + 1.0

    def test_rand_init(self):
        X = _make_2d_data()
        model = TMI(K=2, use_rand_init=True, max_iters=50, tol=0.5, verbose=False)
        model.fit(X)  # just check it runs without error

    def test_q_column_normalized(self):
        X = _make_2d_data(a=10, alpha=15)
        model = TMI(K=2, max_iters=50, tol=0.5, normalize=True, verbose=False).fit(X)
        col_sums = model.q_.sum(dim=0)
        assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-5)


# ---------------------------------------------------------------------------
# TMIWithDynamics tests
# ---------------------------------------------------------------------------


class TestTMIWithDynamics:
    def test_fit_2d(self):
        a, alpha = 15, 20
        K = 2
        X = _make_2d_data(a=a, alpha=alpha)
        W, DW = _make_galerkin(T=alpha)
        N = _poly_exponents(K=K, order=1)  # shape (L, K)

        model = TMIWithDynamics(
            K=K, W=W, DW=DW, N=N,
            lambda_z=1.0,
            max_iters=200,
            tol=1e-2,
            sparse_iter=500,
            verbose=False,
        )
        model.fit(X)

        L = N.shape[0]
        assert model.z_.shape == (alpha, K)
        assert model.Y_.shape == (a, K)
        assert model.xi_z_.shape == (L, K)

    def test_xi_z_initial_provided(self):
        a, alpha = 10, 15
        K = 2
        X = _make_2d_data(a=a, alpha=alpha)
        W, DW = _make_galerkin(T=alpha)
        N = _poly_exponents(K=K, order=1)
        L = N.shape[0]
        xi_z_init = np.zeros((L, K), dtype=np.float64)

        model = TMIWithDynamics(
            K=K, W=W, DW=DW, N=N,
            lambda_z=0.0,
            max_iters=50,
            tol=0.5,
            verbose=False,
        )
        model.fit(X, xi_z_init=xi_z_init)
        assert model.xi_z_.shape == (L, K)


# ---------------------------------------------------------------------------
# TMIWithFeatureModel tests
# ---------------------------------------------------------------------------


class TestTMIWithFeatureModel:
    def test_fit_2d(self):
        a, alpha = 15, 20
        K = 2
        X = _make_2d_data(a=a, alpha=alpha)
        W, DW = _make_galerkin(T=alpha)
        N = _poly_exponents(K=K, order=1)
        L = N.shape[0]
        # Feature library: shape (a, P) — one row per feature/neuron
        rng = np.random.default_rng(99)
        theta_y = rng.standard_normal((a, 5)).astype(np.float64)

        model = TMIWithFeatureModel(
            K=K, W=W, DW=DW, N=N, theta_y=theta_y,
            lambda_z=1.0, lambda_y=1.0,
            max_iters=200, tol=1e-2,
            sparse_iter=500,
            verbose=False,
        )
        model.fit(X)

        assert model.xi_z_.shape == (L, K)
        assert model.xi_y_.shape == (5, K)
        assert model.Y_.shape == (a, K)
