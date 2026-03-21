"""Core pydme model classes.

All models follow a scikit-learn-style API:
  - Hyperparameters are set in ``__init__``.
  - ``fit(X)`` trains the model and stores results as attributes ending in ``_``.
  - ``fit`` returns ``self`` for method chaining.

Three model classes are provided:

``TMI``
    Core Temporal Matrix Integration: decomposes X ≈ softmax(-Y @ z.T)
    using KL divergence.  Supports 2-D data (n_features, n_timepoints) and
    3-D data (n_features, n_timepoints, n_trials).  Optionally learns a
    per-feature bias term E.

``TMIWithDynamics``
    Extends ``TMI`` with a sparse SINDy-style dynamics constraint on z:
        DW @ z ≈ W @ poly_library(z, N) @ xi_z
    xi_z is discovered via ensemble sparse regression (EnAdSR).

``TMIWithFeatureModel``
    Extends ``TMIWithDynamics`` with an additional sparse feature model on Y:
        Y ≈ theta_y @ xi_y
    Both xi_z and xi_y are discovered via EnAdSR.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from .library import poly_library
from .losses import kld_loss
from .sparse import EnAdSR, _update_active_coefs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_tensor(x, dtype=torch.float64, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.to(dtype)
    else:
        t = torch.tensor(x, dtype=dtype)
    return t if device is None else t.to(device)


def _svd_init_2d(X: torch.Tensor, K: int):
    """SVD initialisation for 2-D data (no bias term)."""
    U, _, Vh = torch.linalg.svd(X, full_matrices=False)
    Y = -U[:, :K]       # (a, K)
    z = -Vh.T[:, :K]    # (alpha, K)
    return z, Y


def _svd_init_2d_with_E(X: torch.Tensor, K: int):
    """SVD initialisation for 2-D data with a mean (bias) term."""
    Xavg = X.mean(dim=1, keepdim=True)          # (a, 1)
    U, _, Vh = torch.linalg.svd(X - Xavg, full_matrices=False)
    Y = -U[:, :K]
    z = -Vh.T[:, :K]
    return z, Y, Xavg


# ---------------------------------------------------------------------------
# TMI (core)
# ---------------------------------------------------------------------------


class TMI:
    """Temporal Matrix Integration (TMI).

    Decomposes a non-negative data matrix X as::

        q = softmax_col( -Y @ z.T )          # 2-D
        q[a,t,j] = softmax_col( -Y[a,:] @ z[t,:,j] )  # 3-D

    and minimises the column-normalised KL divergence D_KL(X || q).

    Parameters
    ----------
    K : int
        Latent dimensionality.
    normalize : bool
        Column-normalise X (and q) so that each column is a probability
        distribution.
    max_iters : int
        Maximum number of gradient-descent steps.
    tol : float
        Stop when the RMS gradient magnitude drops below this value.
    lr : float
        SGD learning rate.
    use_E : bool
        If True, learn a per-feature bias term E (adds mean-field offset).
    l2_z : float
        L2 regularisation weight on z.
    l2_y : float
        L2 regularisation weight on Y.
    use_rand_init : bool
        Random initialisation; otherwise use SVD-based initialisation.
    seed : int
        Random seed (used for both initialisation and sparse regression).
    track_latents : bool
        Record z and Y at every iteration (memory-intensive).
    verbose : bool
        Print progress every 1 000 iterations.

    Attributes
    ----------
    z_ : torch.Tensor
        Fitted time-domain latents.
    Y_ : torch.Tensor
        Fitted feature-domain latents.
    E_ : torch.Tensor or None
        Fitted bias (None if ``use_E=False``).
    q_ : torch.Tensor
        Model reconstruction of X.
    grads_ : torch.Tensor
        RMS gradient magnitude per iteration.
    losses_ : torch.Tensor
        Loss value per iteration.
    z_history_, Y_history_ : list
        Per-iteration snapshots (only if ``track_latents=True``).
    """

    def __init__(
        self,
        K: int,
        *,
        normalize: bool = True,
        max_iters: int = 100_000,
        tol: float = 1e-3,
        lr: float = 1e-3,
        use_E: bool = False,
        l2_z: float = 0.0,
        l2_y: float = 0.0,
        use_rand_init: bool = False,
        seed: int = 0,
        track_latents: bool = False,
        verbose: bool = True,
    ):
        self.K = K
        self.normalize = normalize
        self.max_iters = max_iters
        self.tol = tol
        self.lr = lr
        self.use_E = use_E
        self.l2_z = l2_z
        self.l2_y = l2_y
        self.use_rand_init = use_rand_init
        self.seed = seed
        self.track_latents = track_latents
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_q(
        self,
        Y: torch.Tensor,
        z: torch.Tensor,
        E: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the model reconstruction q from current latents."""
        if z.ndim == 2:
            C = Y @ z.T                                    # (a, alpha)
            bias = E if E is not None else 0.0
            q = torch.exp(-C - bias)
        else:
            # z: (alpha, K, trials) — Julia convention keeps time dim first
            # but here z is (T, K, trials); Y is (a, K)
            C = torch.einsum("ak,tkj->atj", Y, z)          # (a, T, trials)
            bias = E[:, None, None] if E is not None else 0.0
            q = torch.exp(-C - bias)

        if self.normalize:
            q = q / q.sum(dim=0, keepdim=True)
        return q

    def _init_params(self, X: torch.Tensor):
        """Initialise z, Y (and optionally E) from X."""
        device, dtype = X.device, X.dtype
        a = X.shape[0]
        K = self.K
        is_3d = X.ndim == 3

        torch.manual_seed(self.seed)

        if self.use_rand_init:
            if is_3d:
                alpha, n_trials = X.shape[1], X.shape[2]
                z = torch.randn(alpha, K, n_trials, dtype=dtype, device=device)
                Y = torch.randn(a, K, dtype=dtype, device=device)
            else:
                alpha = X.shape[1]
                z = torch.randn(alpha, K, dtype=dtype, device=device)
                Y = torch.randn(a, K, dtype=dtype, device=device)
            E = torch.randn(a, 1, dtype=dtype, device=device) if self.use_E else None

        else:
            if is_3d:
                alpha, n_trials = X.shape[1], X.shape[2]
                Xavg = X.mean(dim=(1, 2), keepdim=False)   # (a,)
                z = torch.zeros(alpha, K, n_trials, dtype=dtype, device=device)
                Y_stack = torch.zeros(a, K, n_trials, dtype=dtype, device=device)
                dX = X - Xavg[:, None, None]
                for j in range(n_trials):
                    U, _, Vh = torch.linalg.svd(dX[:, :, j], full_matrices=False)
                    z[:, :, j] = -Vh.T[:, :K]
                    Y_stack[:, :, j] = -U[:, :K]
                Y = Y_stack.mean(dim=2)
                E = Xavg if self.use_E else None
            else:
                if self.use_E:
                    z, Y, E = _svd_init_2d_with_E(X, K)
                else:
                    z, Y = _svd_init_2d(X, K)
                    E = None

        return z, Y, E

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        *,
        z_init=None,
        Y_init=None,
        E_init=None,
    ) -> "TMI":
        """Fit the TMI model to data.

        Parameters
        ----------
        X : array-like of shape (a, alpha) or (a, alpha, T)
            Input data matrix (or tensor).
        z_init : array-like, optional
            Initial value for z.
        Y_init : array-like, optional
            Initial value for Y.
        E_init : array-like, optional
            Initial value for E (only used when ``use_E=True``).

        Returns
        -------
        self
        """
        X = _to_tensor(X)
        device, dtype = X.device, X.dtype

        if self.normalize:
            X = X / (X.sum(dim=0, keepdim=True) + 1e-12)

        # Initialise latents
        if z_init is None or Y_init is None:
            z0, Y0, E0 = self._init_params(X)
        else:
            z0 = _to_tensor(z_init, dtype=dtype, device=device)
            Y0 = _to_tensor(Y_init, dtype=dtype, device=device)
            E0 = _to_tensor(E_init, dtype=dtype, device=device) if E_init is not None else None

        z = nn.Parameter(z0)
        Y = nn.Parameter(Y0)
        params = [z, Y]

        if self.use_E:
            E_init_val = E0 if E0 is not None else torch.zeros(X.shape[0], 1, dtype=dtype, device=device)
            E = nn.Parameter(E_init_val)
            params.append(E)
        else:
            E = None

        optimizer = torch.optim.SGD(params, lr=self.lr)

        grads_hist: list[float] = []
        losses_hist: list[float] = []
        z_hist = [] if self.track_latents else None
        Y_hist = [] if self.track_latents else None

        for i in range(1, self.max_iters + 1):
            if self.track_latents:
                z_hist.append(z.detach().clone())
                Y_hist.append(Y.detach().clone())

            optimizer.zero_grad()
            q = self._compute_q(Y, z, E)
            loss = kld_loss(X, q)
            if self.l2_z > 0:
                loss = loss + 0.5 * self.l2_z * z.pow(2).sum()
            if self.l2_y > 0:
                loss = loss + 0.5 * self.l2_y * Y.pow(2).sum()
            loss.backward()

            # RMS gradient magnitude as convergence signal
            mse_grad = math.sqrt(
                sum(p.grad.abs().mean().item() ** 2 for p in params if p.grad is not None)
            )
            grads_hist.append(mse_grad)
            losses_hist.append(loss.item())

            optimizer.step()

            if i % 1000 == 0 and self.verbose:
                print(f"iter => {i:>6d}  mse_grad => {mse_grad:.6e}  loss => {loss.item():.6f}")

            if mse_grad < self.tol:
                if self.verbose:
                    print(f"Converged at iter {i}  mse_grad => {mse_grad:.6e}")
                break

        self.z_ = z.detach()
        self.Y_ = Y.detach()
        self.E_ = E.detach() if E is not None else None
        with torch.no_grad():
            self.q_ = self._compute_q(self.Y_, self.z_, self.E_)
        self.grads_ = torch.tensor(grads_hist)
        self.losses_ = torch.tensor(losses_hist)
        if self.track_latents:
            self.z_history_ = z_hist
            self.Y_history_ = Y_hist

        return self

    def transform(
        self,
        z=None,
        Y=None,
        E=None,
    ) -> torch.Tensor:
        """Compute the model reconstruction from latents.

        Uses the fitted latents by default; pass explicit values to override.
        """
        z = self.z_ if z is None else _to_tensor(z)
        Y = self.Y_ if Y is None else _to_tensor(Y)
        E = self.E_ if E is None else _to_tensor(E)
        with torch.no_grad():
            return self._compute_q(Y, z, E)


# ---------------------------------------------------------------------------
# TMIWithDynamics
# ---------------------------------------------------------------------------


class TMIWithDynamics(TMI):
    """TMI with a sparse SINDy-like dynamics constraint on z.

    In addition to the KLD reconstruction loss, the model penalises
    violations of::

        DW @ z ≈ W @ poly_library(z, N) @ xi_z

    where ``xi_z`` is a sparse coefficient matrix discovered via
    :class:`~pydme.sparse.EnAdSR`.

    Parameters
    ----------
    K : int
        Latent dimensionality.
    W : array-like of shape (T, T)
        Galerkin weight matrix applied to the library.
    DW : array-like of shape (T, T)
        Galerkin weight matrix applied to the derivative of z.
    N : array-like of shape (L, K)
        Polynomial exponent matrix defining the library terms.
    lambda_z : float
        Weight of the dynamics constraint loss.
    l2_smooth : float
        Weight of a smoothness penalty ``||DW @ z||^2``.
    sparse_iter : int
        Run :class:`~pydme.sparse.EnAdSR` every this many iterations to
        re-discover the active support of xi_z; between sparse regression
        steps, the active coefficients are updated by least-squares.
    inclusion_tol : float
        EnAdSR inclusion-probability threshold.
    ridge : float
        Ridge penalty for the least-squares sub-problems inside EnAdSR.
    c_z : float
        EnAdSR condition-number scaling parameter for xi_z.
    **kwargs
        Additional keyword arguments forwarded to :class:`TMI`.

    Attributes
    ----------
    xi_z_ : torch.Tensor of shape (L, K)
        Fitted sparse dynamics model coefficients.
    xi_z_history_ : list
        Per-iteration snapshots of xi_z (only if ``track_latents=True``).
    """

    def __init__(
        self,
        K: int,
        W,
        DW,
        N,
        *,
        lambda_z: float = 10.0,
        l2_smooth: float = 0.0,
        sparse_iter: int = 10_000,
        inclusion_tol: float = 0.7,
        ridge: float = 0.0,
        c_z: float = 1e-4,
        **kwargs,
    ):
        super().__init__(K, **kwargs)
        self.W = W
        self.DW = DW
        self.N = N
        self.lambda_z = lambda_z
        self.l2_smooth = l2_smooth
        self.sparse_iter = sparse_iter
        self.inclusion_tol = inclusion_tol
        self.ridge = ridge
        self.c_z = c_z

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dynamics_loss(
        self,
        z: torch.Tensor,
        xi_z: torch.Tensor,
        W: torch.Tensor,
        DW: torch.Tensor,
        N: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the dynamics constraint loss."""
        if z.ndim == 2:
            theta = poly_library(z, N)          # (T, L)
            dzdt = DW @ z                        # (T, K)
            theta_w = W @ theta                  # (T, L)
            residual = dzdt - theta_w @ xi_z     # (T, K)
            loss = 0.5 * self.lambda_z * residual.pow(2).sum()
            if self.l2_smooth > 0:
                loss = loss + 0.5 * self.l2_smooth * dzdt.pow(2).sum()
        else:
            # z: (T, K, trials)
            theta = poly_library(z, N)                                     # (T, L, trials)
            dzdt = torch.einsum("tl,lkj->tkj", DW, z)                     # (T, K, trials)
            theta_w = torch.einsum("tl,lkj->tkj", W, theta)               # (T, L, trials)
            pred = torch.einsum("tlj,lk->tkj", theta_w, xi_z)             # (T, K, trials)
            residual = dzdt - pred
            loss = 0.5 * self.lambda_z * residual.pow(2).sum()
            if self.l2_smooth > 0:
                loss = loss + 0.5 * self.l2_smooth * dzdt.pow(2).sum()
        return loss

    @torch.no_grad()
    def _get_theta_w_dzdt(
        self,
        z: torch.Tensor,
        W: torch.Tensor,
        DW: torch.Tensor,
        N: torch.Tensor,
    ):
        """Return the (possibly reshaped) library and derivative arrays
        suitable for passing to sparse regression."""
        if z.ndim == 2:
            theta = poly_library(z, N)
            return W @ theta, DW @ z           # (T, L), (T, K)
        else:
            theta = poly_library(z, N)                          # (T, L, trials)
            theta_w = torch.einsum("tl,lkj->tkj", W, theta)    # (T, L, trials)
            dzdt = torch.einsum("tl,lkj->tkj", DW, z)          # (T, K, trials)
            T, L, trials = theta_w.shape
            K = dzdt.shape[1]
            theta_w_2d = theta_w.permute(0, 2, 1).reshape(T * trials, L)
            dzdt_2d = dzdt.permute(0, 2, 1).reshape(T * trials, K)
            return theta_w_2d, dzdt_2d

    def _run_enadsr_z(self, theta_w, dzdt, active):
        return EnAdSR(
            ic="slic",
            num_batches=10,
            tol=self.inclusion_tol,
            max_iter=5,
            c=self.c_z,
            ridge=self.ridge,
            active_set=active,
        ).fit(theta_w, dzdt).coef_

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        *,
        z_init=None,
        Y_init=None,
        E_init=None,
        xi_z_init=None,
    ) -> "TMIWithDynamics":
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (a, alpha) or (a, alpha, T)
        z_init, Y_init, E_init : optional initialisations.
        xi_z_init : array-like of shape (L, K), optional
            Initial sparse dynamics coefficients.

        Returns
        -------
        self
        """
        X = _to_tensor(X)
        device, dtype = X.device, X.dtype

        W = _to_tensor(self.W, dtype=dtype, device=device)
        DW = _to_tensor(self.DW, dtype=dtype, device=device)
        N = _to_tensor(self.N, dtype=dtype, device=device)

        if self.normalize:
            X = X / (X.sum(dim=0, keepdim=True) + 1e-12)

        if z_init is None or Y_init is None:
            z0, Y0, E0 = self._init_params(X)
        else:
            z0 = _to_tensor(z_init, dtype=dtype, device=device)
            Y0 = _to_tensor(Y_init, dtype=dtype, device=device)
            E0 = _to_tensor(E_init, dtype=dtype, device=device) if E_init is not None else None

        z = nn.Parameter(z0)
        Y = nn.Parameter(Y0)
        params = [z, Y]
        if self.use_E:
            E_val = E0 if E0 is not None else torch.zeros(X.shape[0], 1, dtype=dtype, device=device)
            E = nn.Parameter(E_val)
            params.append(E)
        else:
            E = None

        # Initialise xi_z
        with torch.no_grad():
            theta_w, dzdt = self._get_theta_w_dzdt(z.data, W, DW, N)
            if xi_z_init is not None:
                xi_z = _to_tensor(xi_z_init, dtype=dtype, device=device)
            else:
                xi_z = torch.linalg.lstsq(theta_w, dzdt).solution
            xi_z_active = xi_z.abs() > 0

        optimizer = torch.optim.SGD(params, lr=self.lr)

        grads_hist: list[float] = []
        losses_hist: list[float] = []
        z_hist = [] if self.track_latents else None
        Y_hist = [] if self.track_latents else None
        xi_z_hist = [] if self.track_latents else None

        for i in range(1, self.max_iters + 1):
            if self.track_latents:
                z_hist.append(z.detach().clone())
                Y_hist.append(Y.detach().clone())
                xi_z_hist.append(xi_z.clone())

            optimizer.zero_grad()
            q = self._compute_q(Y, z, E)
            loss = kld_loss(X, q)
            if self.lambda_z > 0:
                loss = loss + self._dynamics_loss(z, xi_z, W, DW, N)
            if self.l2_z > 0:
                loss = loss + 0.5 * self.l2_z * z.pow(2).sum()
            if self.l2_y > 0:
                loss = loss + 0.5 * self.l2_y * Y.pow(2).sum()
            loss.backward()

            mse_grad = math.sqrt(
                sum(p.grad.abs().mean().item() ** 2 for p in params if p.grad is not None)
            )
            grads_hist.append(mse_grad)
            losses_hist.append(loss.item())

            optimizer.step()

            # Update xi_z
            with torch.no_grad():
                theta_w, dzdt = self._get_theta_w_dzdt(z.data, W, DW, N)
                if self.lambda_z > 0 and i % self.sparse_iter == 0:
                    xi_z = self._run_enadsr_z(theta_w, dzdt, xi_z_active)
                    xi_z_active = xi_z.abs() > 0
                else:
                    xi_z = _update_active_coefs(theta_w, dzdt, xi_z, xi_z_active, self.ridge)

            if i % 1000 == 0 and self.verbose:
                print(f"iter => {i:>6d}  mse_grad => {mse_grad:.6e}  loss => {loss.item():.6f}")

            if mse_grad < self.tol:
                if self.verbose:
                    print(f"Converged at iter {i}  mse_grad => {mse_grad:.6e}")
                break

        self.z_ = z.detach()
        self.Y_ = Y.detach()
        self.E_ = E.detach() if E is not None else None
        with torch.no_grad():
            self.q_ = self._compute_q(self.Y_, self.z_, self.E_)
        self.xi_z_ = xi_z
        self.grads_ = torch.tensor(grads_hist)
        self.losses_ = torch.tensor(losses_hist)
        if self.track_latents:
            self.z_history_ = z_hist
            self.Y_history_ = Y_hist
            self.xi_z_history_ = xi_z_hist

        return self


# ---------------------------------------------------------------------------
# TMIWithFeatureModel
# ---------------------------------------------------------------------------


class TMIWithFeatureModel(TMIWithDynamics):
    """TMI with sparse dynamics on z and a sparse feature model on Y.

    Extends :class:`TMIWithDynamics` by penalising deviations of Y from a
    sparse linear model over a given feature library ``theta_y``::

        Y ≈ theta_y @ xi_y

    Both ``xi_z`` and ``xi_y`` are discovered via
    :class:`~pydme.sparse.EnAdSR`.

    Parameters
    ----------
    K : int
        Latent dimensionality.
    W, DW, N : array-like
        Galerkin matrices and polynomial exponent matrix (see
        :class:`TMIWithDynamics`).
    theta_y : array-like of shape (alpha, P)
        Feature library matrix (not learned; provided by the user).
    lambda_y : float
        Weight of the feature model loss ``||Y - theta_y @ xi_y||^2``.
    c_y : float
        EnAdSR condition-number scaling for xi_y.
    **kwargs
        Additional keyword arguments forwarded to :class:`TMIWithDynamics`.

    Attributes
    ----------
    xi_y_ : torch.Tensor of shape (P, K)
        Fitted sparse feature model coefficients.
    xi_y_history_ : list
        Per-iteration snapshots of xi_y (only if ``track_latents=True``).
    """

    def __init__(
        self,
        K: int,
        W,
        DW,
        N,
        theta_y,
        *,
        lambda_y: float = 10.0,
        c_y: float = 1e-4,
        **kwargs,
    ):
        super().__init__(K, W, DW, N, **kwargs)
        self.theta_y = theta_y
        self.lambda_y = lambda_y
        self.c_y = c_y

    def _feature_loss(
        self,
        Y: torch.Tensor,
        xi_y: torch.Tensor,
        theta_y: torch.Tensor,
    ) -> torch.Tensor:
        residual = Y - theta_y @ xi_y
        return 0.5 * self.lambda_y * residual.pow(2).sum()

    def _run_enadsr_y(self, theta_y, Y, active):
        return EnAdSR(
            ic="slic",
            num_batches=10,
            tol=self.inclusion_tol,
            max_iter=5,
            c=self.c_y,
            ridge=self.ridge,
            active_set=active,
        ).fit(theta_y, Y).coef_

    def fit(
        self,
        X,
        *,
        z_init=None,
        Y_init=None,
        E_init=None,
        xi_z_init=None,
        xi_y_init=None,
    ) -> "TMIWithFeatureModel":
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (a, alpha) or (a, alpha, T)
        z_init, Y_init, E_init : optional initialisations.
        xi_z_init : array-like of shape (L, K), optional
        xi_y_init : array-like of shape (P, K), optional

        Returns
        -------
        self
        """
        X = _to_tensor(X)
        device, dtype = X.device, X.dtype

        W = _to_tensor(self.W, dtype=dtype, device=device)
        DW = _to_tensor(self.DW, dtype=dtype, device=device)
        N = _to_tensor(self.N, dtype=dtype, device=device)
        theta_y = _to_tensor(self.theta_y, dtype=dtype, device=device)

        if self.normalize:
            X = X / (X.sum(dim=0, keepdim=True) + 1e-12)

        if z_init is None or Y_init is None:
            z0, Y0, E0 = self._init_params(X)
        else:
            z0 = _to_tensor(z_init, dtype=dtype, device=device)
            Y0 = _to_tensor(Y_init, dtype=dtype, device=device)
            E0 = _to_tensor(E_init, dtype=dtype, device=device) if E_init is not None else None

        z = nn.Parameter(z0)
        Y = nn.Parameter(Y0)
        params = [z, Y]
        if self.use_E:
            E_val = E0 if E0 is not None else torch.zeros(X.shape[0], 1, dtype=dtype, device=device)
            E = nn.Parameter(E_val)
            params.append(E)
        else:
            E = None

        # Initialise xi_z and xi_y
        with torch.no_grad():
            theta_w, dzdt = self._get_theta_w_dzdt(z.data, W, DW, N)
            xi_z = (
                _to_tensor(xi_z_init, dtype=dtype, device=device)
                if xi_z_init is not None
                else torch.linalg.lstsq(theta_w, dzdt).solution
            )
            xi_z_active = xi_z.abs() > 0

            xi_y = (
                _to_tensor(xi_y_init, dtype=dtype, device=device)
                if xi_y_init is not None
                else torch.linalg.lstsq(theta_y, Y.data).solution
            )
            xi_y_active = xi_y.abs() > 0

        optimizer = torch.optim.SGD(params, lr=self.lr)

        grads_hist: list[float] = []
        losses_hist: list[float] = []
        z_hist = [] if self.track_latents else None
        Y_hist = [] if self.track_latents else None
        xi_z_hist = [] if self.track_latents else None
        xi_y_hist = [] if self.track_latents else None

        for i in range(1, self.max_iters + 1):
            if self.track_latents:
                z_hist.append(z.detach().clone())
                Y_hist.append(Y.detach().clone())
                xi_z_hist.append(xi_z.clone())
                xi_y_hist.append(xi_y.clone())

            optimizer.zero_grad()
            q = self._compute_q(Y, z, E)
            loss = kld_loss(X, q)
            if self.lambda_z > 0:
                loss = loss + self._dynamics_loss(z, xi_z, W, DW, N)
            if self.lambda_y > 0:
                loss = loss + self._feature_loss(Y, xi_y, theta_y)
            if self.l2_z > 0:
                loss = loss + 0.5 * self.l2_z * z.pow(2).sum()
            if self.l2_y > 0:
                loss = loss + 0.5 * self.l2_y * Y.pow(2).sum()
            loss.backward()

            mse_grad = math.sqrt(
                sum(p.grad.abs().mean().item() ** 2 for p in params if p.grad is not None)
            )
            grads_hist.append(mse_grad)
            losses_hist.append(loss.item())

            optimizer.step()

            # Update xi_z and xi_y
            with torch.no_grad():
                theta_w, dzdt = self._get_theta_w_dzdt(z.data, W, DW, N)
                if self.lambda_z > 0 and i % self.sparse_iter == 0:
                    xi_z = self._run_enadsr_z(theta_w, dzdt, xi_z_active)
                    xi_z_active = xi_z.abs() > 0
                    xi_y = self._run_enadsr_y(theta_y, Y.detach(), xi_y_active)
                    xi_y_active = xi_y.abs() > 0
                else:
                    xi_z = _update_active_coefs(theta_w, dzdt, xi_z, xi_z_active, self.ridge)
                    xi_y = _update_active_coefs(theta_y, Y.detach(), xi_y, xi_y_active, self.ridge)

            if i % 1000 == 0 and self.verbose:
                print(f"iter => {i:>6d}  mse_grad => {mse_grad:.6e}  loss => {loss.item():.6f}")

            if mse_grad < self.tol:
                if self.verbose:
                    print(f"Converged at iter {i}  mse_grad => {mse_grad:.6e}")
                break

        self.z_ = z.detach()
        self.Y_ = Y.detach()
        self.E_ = E.detach() if E is not None else None
        with torch.no_grad():
            self.q_ = self._compute_q(self.Y_, self.z_, self.E_)
        self.xi_z_ = xi_z
        self.xi_y_ = xi_y
        self.grads_ = torch.tensor(grads_hist)
        self.losses_ = torch.tensor(losses_hist)
        if self.track_latents:
            self.z_history_ = z_hist
            self.Y_history_ = Y_hist
            self.xi_z_history_ = xi_z_hist
            self.xi_y_history_ = xi_y_hist

        return self


# ---------------------------------------------------------------------------
# Hyperparameter utilities
# ---------------------------------------------------------------------------


def scan_hyperparams(
    X,
    model_cls,
    param_grid: dict[str, Sequence],
    **model_kwargs,
) -> list[dict]:
    """Grid search over hyperparameters.

    Parameters
    ----------
    X : array-like
        Input data.
    model_cls : type
        One of :class:`TMI`, :class:`TMIWithDynamics`,
        :class:`TMIWithFeatureModel`.
    param_grid : dict
        Mapping from constructor parameter names to lists of values to try.
        All combinations are evaluated.
    **model_kwargs
        Fixed keyword arguments forwarded to ``model_cls``.

    Returns
    -------
    list of dict
        Each entry contains ``'model'`` (fitted) and one key per grid
        parameter with the value used.

    Example
    -------
    >>> results = scan_hyperparams(
    ...     X, TMIWithFeatureModel,
    ...     param_grid={"lambda_z": [1., 10.], "lambda_y": [1., 10.]},
    ...     K=3, W=W, DW=DW, N=N, theta_y=theta_y,
    ... )
    """
    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    results = []

    for combo in itertools.product(*values):
        kw = dict(zip(keys, combo))
        print("  ".join(f"{k} => {v}" for k, v in kw.items()))
        m = model_cls(**kw, **model_kwargs)
        m.fit(X)
        results.append({"model": m, **kw})

    return results


def score_models(
    results: list[dict],
    N,
    W,
    DW,
    theta_y=None,
) -> list[float]:
    """Score a list of fitted models using the SLIC information criterion.

    Parameters
    ----------
    results : list of dict
        Output from :func:`scan_hyperparams`.
    N : array-like
        Polynomial exponent matrix (same as used during fitting).
    W, DW : array-like
        Galerkin matrices (same as used during fitting).
    theta_y : array-like or None
        Feature library (required when models have ``xi_y_``).

    Returns
    -------
    list of float
        SLIC score per model (lower is better).
    """
    scores = []
    for r in results:
        model = r["model"]
        z = model.z_
        dtype, device = z.dtype, z.device
        xi_z = model.xi_z_

        N_t = _to_tensor(N, dtype=dtype, device=device)
        W_t = _to_tensor(W, dtype=dtype, device=device)
        DW_t = _to_tensor(DW, dtype=dtype, device=device)

        if z.ndim == 2:
            theta = poly_library(z, N_t)
            theta_w = W_t @ theta
            dzdt = DW_t @ z
            n = z.shape[0]
        else:
            theta = poly_library(z, N_t)
            theta_w = torch.einsum("tl,lkj->tkj", W_t, theta)
            dzdt = torch.einsum("tl,lkj->tkj", DW_t, z)
            T, L, trials = theta_w.shape
            K = dzdt.shape[1]
            theta_w = theta_w.permute(0, 2, 1).reshape(T * trials, L)
            dzdt = dzdt.permute(0, 2, 1).reshape(T * trials, K)
            n = T * trials

        k_z = int((xi_z.abs() > 0).sum().item()) + 1
        rss_z = (dzdt - theta_w @ xi_z).pow(2).sum().item()
        score_total = n * math.log(max(k_z * rss_z / n, 1e-12))

        if theta_y is not None and hasattr(model, "xi_y_"):
            xi_y = model.xi_y_
            Y = model.Y_
            theta_y_t = _to_tensor(theta_y, dtype=dtype, device=device)
            n_y = Y.shape[0]
            k_y = int((xi_y.abs() > 0).sum().item()) + 1
            rss_y = (Y - theta_y_t @ xi_y).pow(2).sum().item()
            score_total += n_y * math.log(max(k_y * rss_y / n_y, 1e-12))

        scores.append(score_total)

    return scores
