"""Loss functions for pydme models."""

import torch


def kld_loss(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL divergence D_KL(p_norm || q_norm), summed over all elements.

    Both p and q are column-normalized before computing the divergence so
    that each column is treated as a discrete probability distribution.

    Supports 2D data (a, alpha) and 3D data (a, alpha, T), where for 3D
    the KLD is averaged over the trial dimension.

    Args:
        p: Target matrix.
        q: Model reconstruction, same shape as p.
        eps: Small value for numerical stability.

    Returns:
        Scalar KLD loss.
    """
    if p.ndim == 2:
        pn = p / (p.sum(dim=0, keepdim=True) + eps)
        qn = q / (q.sum(dim=0, keepdim=True) + eps)
        return (pn * torch.log((pn + eps) / (qn + eps))).sum()

    elif p.ndim == 3:
        n = p.shape[2]
        total = p.new_zeros(1).squeeze()
        for i in range(n):
            total = total + kld_loss(p[:, :, i], q[:, :, i], eps=eps)
        return total / n

    else:
        raise ValueError(f"p must be 2D or 3D, got {p.ndim}D")
