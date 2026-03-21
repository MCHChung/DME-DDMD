"""Sparse regression with information-criterion model selection.

Implements Adaptive Sparse Regression (AdSR) and its ensemble variant
(EnAdSR), based on iterative thresholding with least-squares regression
and SLIC / AICc / BIC scoring.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lstsq_col(A: torch.Tensor, b: torch.Tensor, ridge: float) -> torch.Tensor:
    """Solve (A^T A + ridge * I) x = A^T b for x."""
    if ridge == 0.0:
        return torch.linalg.lstsq(A, b).solution
    lhs = A.T @ A + ridge * torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
    rhs = A.T @ b
    return torch.linalg.solve(lhs, rhs)


def _update_active_coefs(
    theta: torch.Tensor,
    dz: torch.Tensor,
    xi: torch.Tensor,
    active: torch.Tensor,
    ridge: float = 0.0,
) -> torch.Tensor:
    """Re-fit coefficients for the active (nonzero) terms only.

    Args:
        theta: Library matrix (n_obs, n_terms).
        dz: Target matrix (n_obs, n_states).
        xi: Current coefficient matrix (n_terms, n_states).
        active: Boolean mask of active terms (n_terms, n_states).
        ridge: Ridge penalty.

    Returns:
        Updated coefficient matrix with inactive terms zeroed out.
    """
    xi = xi * active.to(xi.dtype)
    n_state = dz.shape[1]
    for col in range(n_state):
        big = active[:, col]
        if big.any():
            xi[big, col] = _lstsq_col(
                theta[:, big], dz[:, col : col + 1], ridge
            ).squeeze(-1)
    return xi


def _is_converged(X: torch.Tensor, X_prev: torch.Tensor, abstol: float, reltol: float) -> bool:
    delta = torch.norm(X - X_prev)
    if delta < abstol:
        return True
    if delta / (torch.norm(X) + 1e-12) < reltol:
        return True
    return False


# ---------------------------------------------------------------------------
# AdSR
# ---------------------------------------------------------------------------


class AdSR:
    """Adaptive Sparse Regression (AdSR).

    Iteratively thresholds a least-squares solution and re-fits the
    remaining terms, selecting the sparsity level via an information
    criterion evaluated on a held-out validation split.

    Parameters
    ----------
    ic : {'slic', 'aicc', 'bic'}
        Information criterion used for model selection.
    max_iter : int
        Number of outer threshold-sweep iterations.
    c : float
        Optional penalty that scales with the condition number of the
        training data (when ``use_cond=True``).
    train_pct : float
        Percentage of observations used for training; the rest form the
        validation set.
    abs_tol : float
        Absolute convergence tolerance on the training prediction.
    rel_tol : float
        Relative convergence tolerance on the training prediction.
    ridge : float
        Ridge (L2) penalty for the least-squares sub-problems.
    use_cond : bool
        If True, scale ``c`` by the condition number of the training
        library matrix.
    active_set : torch.Tensor or None
        Boolean mask (n_terms, n_states) of terms that are allowed to be
        nonzero.  If None, all terms are active.

    Attributes
    ----------
    coef_ : torch.Tensor of shape (n_terms, n_states)
        Fitted sparse coefficient matrix.
    score_ : float
        Best information-criterion score achieved.
    """

    def __init__(
        self,
        ic: Literal["slic", "aicc", "bic"] = "slic",
        max_iter: int = 10,
        c: float = 0.0,
        train_pct: float = 80.0,
        abs_tol: float = 1e-7,
        rel_tol: float = 1e-7,
        ridge: float = 0.0,
        use_cond: bool = True,
        active_set: torch.Tensor | None = None,
    ):
        if ic not in ("slic", "aicc", "bic"):
            raise ValueError(f"ic must be 'slic', 'aicc', or 'bic', got {ic!r}")
        self.ic = ic
        self.max_iter = max_iter
        self.c = c
        self.train_pct = train_pct
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.ridge = ridge
        self.use_cond = use_cond
        self.active_set = active_set

    def fit(self, theta: torch.Tensor, y: torch.Tensor) -> "AdSR":
        """Fit the sparse regression model.

        Args:
            theta: Library matrix of shape (n_obs, n_terms).
            y: Target matrix of shape (n_obs, n_states).

        Returns:
            self
        """
        n_obs, n_state = y.shape
        bag_size = int(math.floor(self.train_pct * n_obs / 100))

        perm = torch.randperm(n_obs, device=y.device)
        train_idx = perm[:bag_size]
        test_idx = perm[bag_size:]

        theta_train, theta_test = theta[train_idx], theta[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        n_test = y_test.shape[0]

        cond = torch.linalg.cond(theta_train).item() if self.use_cond else 1.0
        eta = self.c * cond

        def _score(xi: torch.Tensor) -> float:
            k = int((xi.abs() > 0).sum().item()) + 1
            rss = torch.sum((y_test - theta_test @ xi) ** 2).item()
            adj = rss + eta
            if self.ic == "slic":
                return n_test * math.log(k * adj / n_test)
            elif self.ic == "aicc":
                denom = max(n_test - k - 1, 1)
                return n_test * math.log(adj / n_test) + 2 * k * n_test / denom
            else:  # bic
                return n_test * math.log(adj / n_test) + k * math.log(n_test)

        # Initial least-squares estimate
        xi = _lstsq_col(theta_train, y_train, self.ridge)
        active = (
            self.active_set.to(theta.device)
            if self.active_set is not None
            else torch.ones_like(xi, dtype=torch.bool)
        )
        xi = _update_active_coefs(theta_train, y_train, xi, active, self.ridge)

        min_score = float("inf")
        prev_small: torch.Tensor | None = None
        X_prev = theta_train @ xi

        for _ in range(self.max_iter):
            nonzero_vals = xi[xi != 0].abs()
            if nonzero_vals.numel() == 0:
                break

            lam_min = nonzero_vals.min().item()
            lam_max = nonzero_vals.max().item()
            ratio = lam_max / (lam_min + 1e-12)
            n_lambdas = max(2, abs(1000 * int(math.ceil(math.log10(ratio + 1e-10)))))
            lambdas = torch.linspace(lam_min, lam_max, n_lambdas, device=y.device)

            for lam in lambdas:
                temp_xi = xi.clone()
                small = temp_xi.abs() < lam.item()

                if prev_small is not None and torch.equal(small, prev_small):
                    continue

                temp_xi[small] = 0.0
                # All-zero column → skip this threshold
                if (temp_xi.abs() == 0).all(dim=0).any():
                    break

                temp_xi = _update_active_coefs(
                    theta_train, y_train, temp_xi, ~small, self.ridge
                )
                prev_small = small

                s = _score(temp_xi)
                if s < min_score:
                    xi = temp_xi.clone()
                    min_score = s

            X_curr = theta_train @ xi
            if _is_converged(X_curr, X_prev, self.abs_tol, self.rel_tol):
                break
            X_prev = X_curr

        self.coef_ = xi
        self.score_ = min_score
        return self


# ---------------------------------------------------------------------------
# EnAdSR
# ---------------------------------------------------------------------------


class EnAdSR:
    """Ensemble Adaptive Sparse Regression (EnAdSR).

    Runs :class:`AdSR` on multiple random train/validation splits and
    aggregates the results via inclusion probabilities.  Terms whose
    inclusion probability falls below ``tol`` are pruned; the remaining
    terms are re-fitted on all data.

    Parameters
    ----------
    ic : {'slic', 'aicc', 'bic'}
        Information criterion forwarded to each :class:`AdSR` run.
    num_batches : int
        Number of ensemble members (independent AdSR runs).
    tol : float
        Inclusion probability threshold; terms below this are zeroed.
    **kwargs
        Additional keyword arguments forwarded to :class:`AdSR`.

    Attributes
    ----------
    coef_ : torch.Tensor of shape (n_terms, n_states)
        Fitted sparse coefficient matrix.
    scores_ : torch.Tensor of shape (num_batches,)
        Per-batch information-criterion scores.
    inclusion_probs_ : torch.Tensor of shape (n_terms, n_states)
        Fraction of ensemble members in which each coefficient was nonzero.
    """

    def __init__(
        self,
        ic: Literal["slic", "aicc", "bic"] = "slic",
        num_batches: int = 10,
        tol: float = 0.7,
        **kwargs,
    ):
        self.ic = ic
        self.num_batches = num_batches
        self.tol = tol
        self.kwargs = kwargs

    def fit(self, theta: torch.Tensor, y: torch.Tensor) -> "EnAdSR":
        """Fit the ensemble sparse regression model.

        Args:
            theta: Library matrix of shape (n_obs, n_terms).
            y: Target matrix of shape (n_obs, n_states).

        Returns:
            self
        """
        n_terms = theta.shape[1]
        n_states = y.shape[1]

        xi_stack = torch.zeros(n_terms, n_states, self.num_batches, dtype=theta.dtype, device=theta.device)
        scores = torch.zeros(self.num_batches, dtype=theta.dtype, device=theta.device)

        for i in range(self.num_batches):
            reg = AdSR(ic=self.ic, **self.kwargs)
            reg.fit(theta, y)
            xi_stack[:, :, i] = reg.coef_
            scores[i] = reg.score_

        # Inclusion probabilities
        active = xi_stack.abs() > 0
        ips = active.float().mean(dim=2)  # (n_terms, n_states)

        # Ensemble mean over active entries
        n_active = active.float().sum(dim=2).clamp(min=1.0)
        xi_mean = xi_stack.sum(dim=2) / n_active
        xi_mean[ips < self.tol] = 0.0

        # Final re-fit on selected (active) terms using all data
        final_active = xi_mean.abs() > 0
        xi_final = _update_active_coefs(theta, y, xi_mean, final_active, ridge=self.kwargs.get("ridge", 0.0))

        self.coef_ = xi_final
        self.scores_ = scores
        self.inclusion_probs_ = ips
        return self
