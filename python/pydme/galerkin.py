"""Galerkin projection weight matrices for derivative-free dynamics estimation.

Rather than computing dz/dt by finite differences (which amplifies noise),
the Galerkin approach uses integration by parts on smooth local test functions
that vanish at the endpoints of each window:

    ∫ (dz/dt) w_i dt  =  -∫ z (dw_i/dt) dt

This yields the matrix equation:

    DW @ z  ≈  W @ theta(z) @ xi_z

where

  - W  (shape: n_windows × n_time) integrates z against each bump w_i
  - DW (shape: n_windows × n_time) integrates z against -dw_i/dt

Estimating xi_z via this system avoids differentiating z entirely, giving
a noise-robust identification of the latent dynamics.

Python port of ``src/build_weight_mats.jl``.
"""

from __future__ import annotations

from itertools import product as _iproduct

import numpy as np


# ---------------------------------------------------------------------------
# Weight functions
# ---------------------------------------------------------------------------


def _w(t: np.ndarray, p: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """Smooth bump function supported on (a, b), vanishing at both ends.

    w(t; p, a, b) = (2/(b-a))^(2p) * ((t-a)(b-t))^p
    """
    c = (2.0 / (b - a)) ** (2 * p)
    return c * ((t - a) * (b - t)) ** p


def _dw(t: np.ndarray, p: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """First derivative of the bump function with respect to t."""
    c = (2.0 / (b - a)) ** (2 * p)
    inner = (t - a) * (b - t)
    # d/dt [(t-a)(b-t)]^p = p * [(t-a)(b-t)]^(p-1) * (a+b-2t)
    return c * p * inner ** (p - 1) * (a + b - 2.0 * t)


def _d2w(t: np.ndarray, p: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """Second derivative of the bump function with respect to t."""
    c = (2.0 / (b - a)) ** (2 * p)
    inner = (t - a) * (b - t)
    d_inner = a + b - 2.0 * t  # d/dt [(t-a)(b-t)]
    # d/dt [p * inner^(p-1) * d_inner]
    return c * p * ((p - 1) * inner ** (p - 2) * d_inner ** 2 - 2.0 * inner ** (p - 1))


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------


def build_weight_matrix(
    ts: np.ndarray,
    window: int,
    d: int,
    p: int = 10,
) -> np.ndarray:
    """Build a Galerkin projection weight matrix from a 1-D time grid.

    Each row of the output corresponds to one overlapping local window
    ``ts[i : i+window+1]``.  The bump function ``w`` (or its derivative)
    is evaluated on the scaled window coordinates and assembled into a full
    (n_windows × n_time) dense matrix multiplied by the trapezoidal weight
    ``dt / 2``.

    Parameters
    ----------
    ts : ndarray of shape (n,)
        Uniformly spaced time points.
    window : int
        Number of *gaps* in each local window, so each window spans
        ``window + 1`` consecutive time points.
    d : {0, 1, 2}
        Derivative order:

        - 0 → W  (smooth integration matrix; corresponds to ∫ z w dt)
        - 1 → DW (derivative matrix; corresponds to -∫ z dw/dt dt)
        - 2 → DDW (second-derivative matrix)

    p : int
        Smoothness parameter of the bump function.  Higher values
        concentrate weight toward the window centre.

    Returns
    -------
    ndarray of shape (n - window, n)
        Dense Galerkin weight matrix.

    Notes
    -----
    The convention matches ``BuildWeightMat`` in ``src/build_weight_mats.jl``:
    the matrix is returned pre-multiplied by ``dt / 2`` so that a simple
    matrix–vector product gives the discretised integral.
    """
    if d not in (0, 1, 2):
        raise ValueError(f"d must be 0, 1, or 2; got {d}")

    n = len(ts)
    dt = float(ts[1] - ts[0])
    n_rows = n - window
    W = np.zeros((n_rows, n))

    for i in range(n_rows):
        ts_i = ts[i : i + window + 1]
        a_i, b_i = float(ts_i[0]), float(ts_i[-1])
        span = b_i - a_i

        # Map ts_i linearly to [-1, 1]
        sc = ts_i * (2.0 / span) - (b_i + a_i) / span

        if d == 0:
            # W row: 2 * w(sc) / (2/span) = span * w(sc)
            W[i, i : i + window + 1] = 2.0 * _w(sc, p) / (2.0 / span)
        elif d == 1:
            # DW row: -2 * dw(sc)   (chain-rule factor from the scaling cancels)
            W[i, i : i + window + 1] = -2.0 * _dw(sc, p)
        else:  # d == 2
            # DDW row: -2 * d2w(sc) * (2/span)
            W[i, i : i + window + 1] = -2.0 * _d2w(sc, p) * (2.0 / span)

    return W * (dt / 2.0)


def build_disc_weight_matrix(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build simple discrete (finite-difference) weight matrices.

    A lightweight alternative to :func:`build_weight_matrix` for clean,
    noise-free data.  Uses forward differences to approximate dz/dt.

    Parameters
    ----------
    ts : ndarray of shape (n,)
        Time points (only length is used; values are ignored).

    Returns
    -------
    DW : ndarray of shape (n, n)
        Forward-difference matrix: ``(DW @ z)[i] = z[i+1] - z[i]``.
    W : ndarray of shape (n, n)
        Identity matrix (no smoothing).

    Notes
    -----
    Equivalent to ``BuildDiscWeightMat`` in ``src/build_weight_mats.jl``.
    The last row of DW is all zeros (no forward neighbour).
    """
    n = len(ts)
    DW = np.zeros((n, n))
    for i in range(n - 1):
        DW[i, i] = -1.0
        DW[i, i + 1] = 1.0
    W = np.eye(n)
    return DW, W


def build_poly_degree_matrix(K: int, order: int) -> np.ndarray:
    """Build the polynomial exponent matrix for a K-dimensional library.

    Returns an ``(L, K)`` integer array where each row is a multi-index
    ``(e_1, ..., e_K)`` with ``sum(e_k) <= order``.  Library term ``l``
    corresponds to ``z[:,0]**e[l,0] * z[:,1]**e[l,1] * ...``.

    Parameters
    ----------
    K : int
        Number of latent dimensions (columns of the exponent matrix).
    order : int
        Maximum total polynomial degree.  For example, with K=2 and
        order=2 the terms are: 1, z0, z1, z0², z0·z1, z1².

    Returns
    -------
    ndarray of shape (L, K), dtype float64
        Exponent matrix.  The number of terms is
        ``L = binomial(K + order, order)``.

    Examples
    --------
    >>> build_poly_degree_matrix(K=2, order=2)
    array([[0., 0.],
           [1., 0.],
           [0., 1.],
           [2., 0.],
           [1., 1.],
           [0., 2.]])
    """
    if order < 0:
        raise ValueError("order must be >= 0")
    terms = [
        combo
        for combo in _iproduct(range(order + 1), repeat=K)
        if sum(combo) <= order
    ]
    return np.array(terms, dtype=np.float64)
