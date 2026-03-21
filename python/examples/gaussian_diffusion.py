"""
1+1-D Gaussian diffusion example for pydme
==========================================

We simulate a probability density that diffuses on a 1-D spatial grid,

    ∂u/∂t = D ∂²u/∂x²,    u(x,0) = δ(x)

with analytical solution

    u(x, t) = exp(-x² / (4Dt)) / √(4πDt).

The data matrix X has shape (n_spatial, n_time): each column is the
(discretised) probability density at one time step.

We then fit three models:

  1. TMI (core): discovers spatial and temporal latents purely from X.
  2. TMIWithDynamics (Galerkin): additionally enforces that the temporal
     latent z(t) follows a polynomial ODE discovered by sparse regression.

Because the TMI model represents

    q(x, t) = softmax_col(-Y @ z(t)ᵀ)

with K=1 and Y(x) ≈ x², the scalar latent z(t) ≈ 1/(4Dt) captures the
growing variance of the Gaussian.  The dynamics of z satisfy

    dz/dt = -z / t = -z²·(1/z·t)    [for z = 1/(4Dt)]

equivalently dz/dt ≈ -z² · (const), making this recoverable by a
degree-2 polynomial SINDy library.

Running
-------
    python examples/gaussian_diffusion.py

Outputs a text summary.  Add ``--plot`` to show matplotlib figures.
"""

import argparse
import sys

import numpy as np

from pydme import TMI, TMIWithDynamics
from pydme.galerkin import (
    build_weight_matrix,
    build_disc_weight_matrix,
    build_poly_degree_matrix,
)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def make_diffusion_data(
    n_spatial: int = 60,
    n_time: int = 150,
    D: float = 0.5,
    x_range: tuple[float, float] = (-6.0, 6.0),
    t_range: tuple[float, float] = (0.2, 3.0),
    noise_std: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the diffusion dataset.

    Parameters
    ----------
    n_spatial : int
        Number of spatial grid points (a dimension of X).
    n_time : int
        Number of time points (alpha dimension of X).
    D : float
        Diffusion coefficient.
    x_range, t_range : tuple
        Spatial and temporal extents.
    noise_std : float
        Standard deviation of additive Gaussian noise on X.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_spatial, n_time)
        Noiseless (or noisy) data matrix, column-normalised.
    x : ndarray of shape (n_spatial,)
        Spatial grid.
    t : ndarray of shape (n_time,)
        Time grid.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(*x_range, n_spatial)
    t = np.linspace(*t_range, n_time)

    X = np.zeros((n_spatial, n_time))
    for j, tj in enumerate(t):
        var = 2.0 * D * tj          # σ²(t) = 2Dt
        X[:, j] = np.exp(-x**2 / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)

    if noise_std > 0:
        X = X + noise_std * rng.standard_normal(X.shape)
        X = np.clip(X, 1e-10, None)

    # Column-normalise so each snapshot is a probability distribution
    X = X / X.sum(axis=0, keepdims=True)
    return X, x, t


# ---------------------------------------------------------------------------
# Model fitting helpers
# ---------------------------------------------------------------------------


def fit_core_tmi(X: np.ndarray, K: int = 1, **kwargs) -> TMI:
    """Fit the basic TMI model (no dynamics)."""
    defaults = dict(max_iters=5_000, lr=5e-4, tol=1e-4, normalize=True, verbose=False)
    defaults.update(kwargs)
    model = TMI(K=K, **defaults)
    model.fit(X)
    return model


def fit_tmi_with_dynamics(
    X: np.ndarray,
    t: np.ndarray,
    K: int = 1,
    window: int = 20,
    poly_order: int = 2,
    lambda_z: float = 5.0,
    use_galerkin: bool = True,
    **kwargs,
) -> TMIWithDynamics:
    """Fit TMI with a polynomial dynamics constraint on z.

    Parameters
    ----------
    X : ndarray of shape (n_spatial, n_time)
    t : ndarray of shape (n_time,)
    K : int
        Latent dimensionality.
    window : int
        Galerkin window size (number of time-step gaps per window).
    poly_order : int
        Maximum polynomial degree of the SINDy library.
    lambda_z : float
        Strength of the dynamics constraint.
    use_galerkin : bool
        If True, use smooth Galerkin matrices; otherwise use discrete
        finite differences.
    """
    if use_galerkin:
        W  = build_weight_matrix(t, window=window, d=0)   # (n_time-window, n_time)
        DW = build_weight_matrix(t, window=window, d=1)   # same shape
    else:
        DW, W = build_disc_weight_matrix(t)               # (n_time, n_time) each

    N = build_poly_degree_matrix(K=K, order=poly_order)   # (L, K)

    defaults = dict(
        max_iters=10_000,
        lr=1e-3,
        tol=1e-4,
        normalize=True,
        sparse_iter=3_000,
        verbose=False,
    )
    defaults.update(kwargs)

    model = TMIWithDynamics(
        K=K, W=W, DW=DW, N=N,
        lambda_z=lambda_z,
        **defaults,
    )
    model.fit(X)
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def kld_nats(X: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Mean per-column KL divergence D_KL(X_norm || q_norm) in nats."""
    pn = X / (X.sum(axis=0, keepdims=True) + eps)
    qn = q / (q.sum(axis=0, keepdims=True) + eps)
    return float(np.mean(np.sum(pn * np.log((pn + eps) / (qn + eps)), axis=0)))


def print_summary(X: np.ndarray, model, name: str) -> None:
    """Print reconstruction quality and (if available) dynamics sparsity."""
    import torch
    q = model.q_.numpy() if hasattr(model.q_, "numpy") else np.array(model.q_)
    kld = kld_nats(X, q)
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    print(f"  KLD (mean per column, nats) : {kld:.6f}")
    z = model.z_.numpy()
    print(f"  z  shape : {z.shape}   min={z.min():.3f}  max={z.max():.3f}")
    print(f"  Y  shape : {model.Y_.numpy().shape}")
    if hasattr(model, "xi_z_"):
        xi = model.xi_z_.numpy()
        nnz = int((np.abs(xi) > 0).sum())
        print(f"  xi_z shape : {xi.shape}   nonzero: {nnz}/{xi.size}")
        print(f"  xi_z =\n{xi}")


# ---------------------------------------------------------------------------
# Plotting (optional)
# ---------------------------------------------------------------------------


def plot_results(X, x, t, model_basic, model_dyn=None) -> None:
    """Plot data, TMI reconstruction, and latent z trajectory."""
    import matplotlib.pyplot as plt

    q_basic = model_basic.q_.numpy()
    z_basic = model_basic.z_.numpy().squeeze()  # (n_time,) for K=1

    n_panels = 3 if model_dyn is None else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3.5))

    # ── Data ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.imshow(X, aspect="auto", origin="lower",
              extent=[t[0], t[-1], x[0], x[-1]])
    ax.set_title("Data X")
    ax.set_xlabel("time")
    ax.set_ylabel("space x")

    # ── Basic TMI reconstruction ──────────────────────────────────────────
    ax = axes[1]
    ax.imshow(q_basic, aspect="auto", origin="lower",
              extent=[t[0], t[-1], x[0], x[-1]])
    ax.set_title("TMI reconstruction")
    ax.set_xlabel("time")

    # ── z trajectory ──────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t, z_basic, label="TMI z", lw=2)
    # True z ∝ 1/σ²(t) = 1/(2Dt); sign may be flipped by SVD init
    true_z = 1.0 / (2.0 * 0.5 * t)
    scale = np.abs(z_basic).mean() / np.abs(true_z).mean()
    ax.plot(t, scale * true_z, "--", label="1/σ²(t) (scaled)", lw=1.5)
    ax.set_title("Temporal latent z")
    ax.set_xlabel("time")
    ax.legend(fontsize=8)

    if model_dyn is not None:
        z_dyn = model_dyn.z_.numpy().squeeze()
        ax = axes[3]
        ax.plot(t, z_dyn, label="TMI+dynamics z", lw=2)
        ax.plot(t, scale * true_z, "--", label="1/σ²(t) (scaled)", lw=1.5)
        ax.set_title("z (with dynamics constraint)")
        ax.set_xlabel("time")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("gaussian_diffusion.png", dpi=120)
    print("\n  Figure saved to gaussian_diffusion.png")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--n-spatial", type=int, default=60)
    parser.add_argument("--n-time",    type=int, default=150)
    parser.add_argument("--D",         type=float, default=0.5, help="Diffusion coefficient")
    parser.add_argument("--noise",     type=float, default=0.0, help="Additive noise std")
    parser.add_argument("--K",         type=int,   default=1,   help="Latent dim (1 recommended)")
    parser.add_argument("--window",    type=int,   default=20,  help="Galerkin window size")
    parser.add_argument("--lambda-z",  type=float, default=5.0, help="Dynamics weight")
    parser.add_argument("--no-dynamics", action="store_true",
                        help="Skip TMIWithDynamics (faster)")
    parser.add_argument("--disc", action="store_true",
                        help="Use discrete finite differences instead of Galerkin")
    parser.add_argument("--plot", action="store_true",
                        help="Show matplotlib figures (requires matplotlib)")
    args = parser.parse_args()

    print("Generating 1+1-D Gaussian diffusion data …")
    X, x, t = make_diffusion_data(
        n_spatial=args.n_spatial,
        n_time=args.n_time,
        D=args.D,
        noise_std=args.noise,
    )
    print(f"  X shape: {X.shape}  (n_spatial × n_time)")
    print(f"  Diffusion coefficient D = {args.D}")

    # ── Core TMI ─────────────────────────────────────────────────────────
    print("\nFitting core TMI …")
    model_basic = fit_core_tmi(X, K=args.K)
    print_summary(X, model_basic, "TMI (core, no dynamics)")

    # ── TMI + Galerkin dynamics ───────────────────────────────────────────
    model_dyn = None
    if not args.no_dynamics:
        print("\nFitting TMI with Galerkin dynamics …")
        model_dyn = fit_tmi_with_dynamics(
            X, t,
            K=args.K,
            window=args.window,
            lambda_z=args.lambda_z,
            use_galerkin=not args.disc,
        )
        print_summary(X, model_dyn, "TMI + dynamics (Galerkin)")

        xi = model_dyn.xi_z_.numpy()
        N  = build_poly_degree_matrix(K=args.K, order=2)
        print("\n  Discovered dynamics  dz/dt ≈ Σ xi_z[l] · θ_l(z):")
        for l in range(N.shape[0]):
            c = float(xi[l, 0])
            if abs(c) > 1e-10:
                exps = " · ".join(
                    f"z{k}^{int(N[l,k])}" if N[l,k] > 0 else ""
                    for k in range(N.shape[1])
                    if N[l, k] > 0
                ) or "1"
                print(f"    {c:+.4f} · {exps}")

    if args.plot:
        plot_results(X, x, t, model_basic, model_dyn)

    print("\nDone.")


if __name__ == "__main__":
    main()
