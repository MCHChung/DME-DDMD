"""Polynomial library construction for SINDy-style dynamics."""

import torch


def poly_library(z: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
    """Build a polynomial feature library from latent variables z.

    Each library term l is the product z[:, 0]^N[l,0] * z[:, 1]^N[l,1] * ...

    Args:
        z: Latent variables of shape (T, K) or (T, K, trials).
        N: Exponent matrix of shape (L, K), where L is the number of library
           terms and K is the latent dimensionality.

    Returns:
        theta: Library matrix of shape (T, L) or (T, L, trials).
    """
    if N.device != z.device:
        N = N.to(z.device)

    if z.ndim == 2:
        T, K = z.shape
        L = N.shape[0]
        # Build each column separately to avoid in-place ops (required for autograd)
        cols = []
        for l in range(L):
            col = z.new_ones(T)
            for s in range(K):
                if N[l, s] != 0:
                    col = col * z[:, s] ** N[l, s]
            cols.append(col)
        return torch.stack(cols, dim=1)

    elif z.ndim == 3:
        T, K, trials = z.shape
        L = N.shape[0]
        cols = []
        for l in range(L):
            slices = []
            for j in range(trials):
                col = z.new_ones(T)
                for s in range(K):
                    if N[l, s] != 0:
                        col = col * z[:, s, j] ** N[l, s]
                slices.append(col)
            cols.append(torch.stack(slices, dim=1))   # (T, trials)
        return torch.stack(cols, dim=1)   # (T, L, trials)

    else:
        raise ValueError(f"z must be 2D or 3D, got {z.ndim}D")
