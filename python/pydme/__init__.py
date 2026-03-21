"""
pydme — Python Dynamic Mode Extraction

A PyTorch implementation of Temporal Matrix Integration (TMI) with
sparse dynamics and feature models.
"""

from .models import TMI, TMIWithDynamics, TMIWithFeatureModel
from .sparse import AdSR, EnAdSR
from .library import poly_library
from .losses import kld_loss
from .galerkin import build_weight_matrix, build_disc_weight_matrix, build_poly_degree_matrix

__all__ = [
    "TMI",
    "TMIWithDynamics",
    "TMIWithFeatureModel",
    "AdSR",
    "EnAdSR",
    "poly_library",
    "kld_loss",
    "build_weight_matrix",
    "build_disc_weight_matrix",
    "build_poly_degree_matrix",
]
