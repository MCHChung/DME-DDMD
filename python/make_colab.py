"""Generate examples/tmi_diffusion_colab.ipynb — self-contained Colab notebook."""

import json
import os
import textwrap

BASE = os.path.dirname(__file__)


def _read(rel):
    with open(os.path.join(BASE, rel)) as f:
        return f.read()


def _lines(text):
    """Split text into a list of JSON-safe notebook source lines."""
    rows = text.split("\n")
    out = [r + "\n" for r in rows[:-1]]
    if rows[-1]:
        out.append(rows[-1])
    return out


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(textwrap.dedent(text).strip())}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(textwrap.dedent(text).strip()),
    }


def writefile(path, content):
    """Cell that writes content to path via %%writefile."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [f"%%writefile {path}\n"] + _lines(content),
    }


# ── Source files to embed ────────────────────────────────────────────────────
losses_src   = _read("pydme/losses.py")
library_src  = _read("pydme/library.py")
sparse_src   = _read("pydme/sparse.py")
models_src   = _read("pydme/models.py")
galerkin_src = _read("pydme/galerkin.py")
init_src     = _read("pydme/__init__.py")

# ── Notebook cells ────────────────────────────────────────────────────────────
cells = []

# --------------------------------------------------------------------------
# Title
# --------------------------------------------------------------------------
cells.append(md("""
# pydme — 1+1-D Gaussian Diffusion (self-contained Colab)

**What this notebook does**

1. Embeds the full `pydme` package source (no external install needed).
2. Generates a 1+1-D Gaussian diffusion dataset:
   `∂u/∂t = D ∂²u/∂x²`, solved analytically.
3. Fits **TMI (core)** — discovers spatial/temporal latents purely from data.
4. Builds **Galerkin weight matrices** (W, DW) for derivative-free dynamics.
5. Fits **TMI + dynamics** — additionally discovers the ODE `dz/dt ≈ ξ·θ(z)`
   using sparse regression, **without ever differentiating z**.

---
"""))

# --------------------------------------------------------------------------
# 0. Setup
# --------------------------------------------------------------------------
cells.append(md("## 0 · Setup\n\nInstall PyTorch + NumPy (already present in Colab), then write the `pydme` source."))

cells.append(code("""
# PyTorch and NumPy are pre-installed in Colab.
# Uncomment the line below only if running outside Colab.
# !pip install torch numpy -q
import sys, os
os.makedirs("pydme", exist_ok=True)
"""))

cells.append(writefile("pydme/losses.py",   losses_src))
cells.append(writefile("pydme/library.py",  library_src))
cells.append(writefile("pydme/sparse.py",   sparse_src))
cells.append(writefile("pydme/models.py",   models_src))
cells.append(writefile("pydme/galerkin.py", galerkin_src))
cells.append(writefile("pydme/__init__.py", init_src))

cells.append(code("""
import sys
sys.path.insert(0, ".")

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pydme import (
    TMI, TMIWithDynamics,
    build_weight_matrix,
    build_disc_weight_matrix,
    build_poly_degree_matrix,
)
print("pydme loaded ✓  torch:", torch.__version__)
"""))

# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
cells.append(md("""
## 1 · Generate 1+1-D Gaussian diffusion data

Analytical solution of the diffusion equation with a point-source initial
condition:

$$u(x, t) = \\frac{1}{\\sqrt{4\\pi D t}}\\exp\\!\\left(-\\frac{x^2}{4Dt}\\right)$$

The data matrix **X** has shape `(n_spatial, n_time)`.  Each column is the
spatial probability distribution at one time step.

With **K = 1** latent dimension, the TMI model represents:

$$q(x, t) = \\text{softmax}\\bigl(-Y(x)\\cdot z(t)\\bigr)$$

The spatial mode $Y(x) \\approx x^2$ and the scalar latent $z(t) \\approx
1/(4Dt)$ capture the decaying precision of the Gaussian.  The latent ODE is
then $\\dot z = -z^2 \\cdot (1/z t) = -z/t$, recoverable by a degree-2
polynomial library.
"""))

cells.append(code("""
# ── Parameters ────────────────────────────────────────────────────────────
N_SPATIAL  = 60      # spatial grid points  (= "a" in the paper)
N_TIME     = 150     # time steps           (= "alpha")
D          = 0.5     # diffusion coefficient
X_RANGE    = (-6., 6.)
T_RANGE    = (0.2, 3.0)
NOISE_STD  = 0.005   # additive noise on X
SEED       = 42

rng  = np.random.default_rng(SEED)
x    = np.linspace(*X_RANGE, N_SPATIAL)
t    = np.linspace(*T_RANGE, N_TIME)

X = np.zeros((N_SPATIAL, N_TIME))
for j, tj in enumerate(t):
    var = 2.0 * D * tj                          # σ²(t) = 2Dt
    X[:, j] = np.exp(-x**2 / (2*var)) / np.sqrt(2*np.pi*var)

if NOISE_STD > 0:
    X = np.clip(X + NOISE_STD * rng.standard_normal(X.shape), 1e-10, None)

# Column-normalise → each snapshot is a probability distribution
X = X / X.sum(axis=0, keepdims=True)

print(f"X shape: {X.shape}  (n_spatial × n_time)")
print(f"D = {D},  t ∈ [{t[0]:.2f}, {t[-1]:.2f}]")
"""))

cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

ax = axes[0]
im = ax.imshow(X, aspect="auto", origin="lower",
               extent=[t[0], t[-1], x[0], x[-1]], cmap="hot")
plt.colorbar(im, ax=ax, label="u(x,t)")
ax.set_title("Data  X  (n_spatial × n_time)", fontsize=11)
ax.set_xlabel("time t"); ax.set_ylabel("space x")

ax = axes[1]
for idx in [0, N_TIME//4, N_TIME//2, 3*N_TIME//4, -1]:
    ax.plot(x, X[:, idx], label=f"t={t[idx]:.2f}")
ax.set_title("Column snapshots of X", fontsize=11)
ax.set_xlabel("space x"); ax.set_ylabel("u(x,t)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
"""))

# --------------------------------------------------------------------------
# 2. TMI core
# --------------------------------------------------------------------------
cells.append(md("""
## 2 · TMI (core) — no dynamics

Fit the basic TMI model to discover spatial mode **Y** and temporal
trajectory **z** purely from reconstruction quality.
"""))

cells.append(code("""
K = 1   # one latent dimension is enough for a 1-D Gaussian family

model_basic = TMI(
    K       = K,
    max_iters = 8_000,
    lr      = 5e-4,
    tol     = 1e-5,
    normalize = True,
    verbose = True,
)
model_basic.fit(X)
"""))

cells.append(code("""
q_basic = model_basic.q_.numpy()
Y_basic = model_basic.Y_.numpy().squeeze()   # (n_spatial,)
z_basic = model_basic.z_.numpy().squeeze()   # (n_time,)

# ── KLD ──────────────────────────────────────────────────────────────────
eps = 1e-12
kld = float(np.mean(np.sum(X * np.log((X + eps) / (q_basic + eps)), axis=0)))
print(f"KLD (mean per column, nats): {kld:.6f}")

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

ax = axes[0]
im = ax.imshow(q_basic, aspect="auto", origin="lower",
               extent=[t[0], t[-1], x[0], x[-1]], cmap="hot")
plt.colorbar(im, ax=ax)
ax.set_title("TMI reconstruction  q", fontsize=11)
ax.set_xlabel("time t"); ax.set_ylabel("space x")

ax = axes[1]
ax.plot(x, Y_basic)
ax.set_title("Spatial mode  Y(x)  (≈ x²)", fontsize=11)
ax.set_xlabel("x"); ax.set_ylabel("Y")

ax = axes[2]
ax.plot(t, z_basic, label="TMI z")
true_z = 1.0 / (4.0 * D * t)
scale  = np.abs(z_basic).mean() / np.abs(true_z).mean()
ax.plot(t, scale * true_z, "--", label="1/(4Dt) scaled", lw=1.5)
ax.set_title("Temporal latent  z(t)  (≈ precision)", fontsize=11)
ax.set_xlabel("time t"); ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
"""))

# --------------------------------------------------------------------------
# 3. Galerkin matrices
# --------------------------------------------------------------------------
cells.append(md("""
## 3 · Galerkin weight matrices

Rather than computing $\\dot z$ by finite differences (noise-amplifying),
the Galerkin projection uses **integration by parts** on smooth local bump
functions $w_i(t)$ that vanish at window endpoints:

$$\\int \\dot z\\, w_i\\, dt = -\\int z\\, \\dot w_i\\, dt$$

This gives the matrix system

$$\\underbrace{\\mathbf{DW}}_{\\text{d=1}}\\, z
  \\;\\approx\\;
  \\underbrace{\\mathbf{W}}_{\\text{d=0}}\\, \\boldsymbol{\\theta}(z)\\,\\boldsymbol{\\xi}_z$$

where **no derivative of z is ever taken explicitly**.
"""))

cells.append(code("""
WINDOW = 20   # number of time-step gaps per local window
P      = 10   # bump smoothness parameter

W  = build_weight_matrix(t, window=WINDOW, d=0, p=P)   # (n_time-WINDOW, n_time)
DW = build_weight_matrix(t, window=WINDOW, d=1, p=P)   # same shape
N  = build_poly_degree_matrix(K=K, order=2)             # {1, z, z²}

print(f"W  shape: {W.shape}")
print(f"DW shape: {DW.shape}")
print(f"N  (polynomial library exponents):\\n{N}")

# Visualise one row of W and DW
fig, axes = plt.subplots(1, 2, figsize=(11, 3))
row = W.shape[0] // 2
axes[0].plot(W[row]);  axes[0].set_title(f"W row {row}  (bump, d=0)");  axes[0].set_xlabel("time index")
axes[1].plot(DW[row]); axes[1].set_title(f"DW row {row}  (−d/dt bump, d=1)"); axes[1].set_xlabel("time index")
plt.tight_layout()
plt.show()
"""))

# --------------------------------------------------------------------------
# 4. TMI + dynamics
# --------------------------------------------------------------------------
cells.append(md("""
## 4 · TMI with Galerkin dynamics

Adds a sparse polynomial ODE constraint on $z$:

$$\\mathbf{DW}\\,z \\;\\approx\\; \\mathbf{W}\\,\\boldsymbol{\\theta}(z)\\,\\boldsymbol{\\xi}_z$$

$\\boldsymbol{\\xi}_z$ is discovered by **EnAdSR** (ensemble adaptive sparse
regression with SLIC model selection) and frozen to its active support
between sparse-regression steps.
"""))

cells.append(code("""
model_dyn = TMIWithDynamics(
    K           = K,
    W           = W,
    DW          = DW,
    N           = N,
    lambda_z    = 8.0,     # dynamics constraint weight
    sparse_iter = 3_000,   # run EnAdSR every 3k iters
    max_iters   = 12_000,
    lr          = 8e-4,
    tol         = 1e-5,
    normalize   = True,
    verbose     = True,
)
model_dyn.fit(X)
"""))

cells.append(code("""
q_dyn  = model_dyn.q_.numpy()
z_dyn  = model_dyn.z_.numpy().squeeze()
xi_z   = model_dyn.xi_z_.numpy()          # (L, K) sparse coefficients
z_dyn_trunc = z_dyn[:W.shape[0]]          # match windowed time axis

kld_dyn = float(np.mean(np.sum(X * np.log((X + 1e-12) / (q_dyn + 1e-12)), axis=0)))
print(f"KLD (TMI+dynamics): {kld_dyn:.6f}")
print(f"\\nDiscovered xi_z (L={N.shape[0]} library terms, K={K}):")
term_labels = []
for row in N.astype(int):
    label = "·".join(f"z{k}^{e}" for k, e in enumerate(row) if e > 0) or "1"
    term_labels.append(label)
for lbl, coeff in zip(term_labels, xi_z[:, 0]):
    marker = " ◀" if abs(coeff) > 1e-10 else ""
    print(f"  {lbl:<10s}: {coeff:+.5f}{marker}")
"""))

cells.append(code("""
# ── Side-by-side comparison ───────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

def show(ax, mat, title):
    im = ax.imshow(mat, aspect="auto", origin="lower",
                   extent=[t[0], t[-1], x[0], x[-1]], cmap="hot",
                   vmin=0, vmax=X.max())
    plt.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time t"); ax.set_ylabel("space x")

show(fig.add_subplot(gs[0, 0]), X,       "Data  X")
show(fig.add_subplot(gs[0, 1]), q_basic, "TMI (core)  q")
show(fig.add_subplot(gs[0, 2]), q_dyn,   "TMI + dynamics  q")

# z trajectories
ax = fig.add_subplot(gs[1, :2])
true_z = 1.0 / (4.0 * D * t)

def align(z_arr):
    s = np.abs(z_arr).mean() / np.abs(true_z).mean()
    return s * true_z if np.corrcoef(z_arr, true_z)[0,1] > 0 else -s * true_z

ax.plot(t, z_basic, lw=2,   label="TMI z (core)")
ax.plot(t, z_dyn,   lw=2,   label="TMI z (dynamics)")
ax.plot(t, align(z_dyn), "--", lw=1.5, label="1/(4Dt) (scaled)", c="k")
ax.set_xlabel("time t"); ax.set_title("Temporal latent z(t)", fontsize=10)
ax.legend(fontsize=8)

# Residual DW @ z vs W @ theta @ xi_z
ax2 = fig.add_subplot(gs[1, 2])
z_t   = torch.tensor(model_dyn.z_.numpy(), dtype=torch.float64)
N_t   = torch.tensor(N,  dtype=torch.float64)
W_t   = torch.tensor(W,  dtype=torch.float64)
DW_t  = torch.tensor(DW, dtype=torch.float64)
xi_t  = model_dyn.xi_z_
from pydme.library import poly_library
with torch.no_grad():
    theta = poly_library(z_t, N_t)
    lhs   = (DW_t @ z_t).numpy()
    rhs   = (W_t  @ theta @ xi_t).numpy()
ax2.plot(lhs[:, 0], label="DW @ z", lw=1.5)
ax2.plot(rhs[:, 0], "--", label="W @ θ(z) @ ξ_z", lw=1.5)
ax2.set_title("Dynamics fit: DW·z ≈ W·θ(z)·ξz", fontsize=9)
ax2.set_xlabel("window index"); ax2.legend(fontsize=8)

plt.suptitle("TMI Gaussian Diffusion Demo", fontsize=12, y=1.01)
plt.show()
"""))

# --------------------------------------------------------------------------
# 5. Convergence
# --------------------------------------------------------------------------
cells.append(md("## 5 · Convergence curves"))

cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

for ax, model, label in [
    (axes[0], model_basic, "TMI core"),
    (axes[1], model_dyn,   "TMI + dynamics"),
]:
    losses = model.losses_.numpy()
    grads  = model.grads_.numpy()
    iters  = np.arange(1, len(losses)+1)
    ax2    = ax.twinx()
    ax.semilogy(iters, losses, c="steelblue", lw=1.5, label="loss")
    ax2.semilogy(iters, grads,  c="tomato",    lw=1.2, alpha=0.7, label="‖grad‖")
    ax.set_xlabel("iteration")
    ax.set_ylabel("KLD loss", color="steelblue")
    ax2.set_ylabel("‖grad‖", color="tomato")
    ax.set_title(label, fontsize=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

plt.tight_layout()
plt.show()
"""))

# ── Assemble notebook ─────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
        "colab": {
            "provenance": [],
            "collapsed_sections": ["0 · Setup"],
        },
    },
    "cells": cells,
}

out = os.path.join(BASE, "examples", "tmi_diffusion_colab.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written → {out}")
print(f"  {len(cells)} cells")
