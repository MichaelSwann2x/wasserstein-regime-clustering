"""Wasserstein regime clustering — improved, modern package.

Faithful re-implementation of the core algorithms from
Horvath, Issa & Muguruza (2021) “Clustering Market Regimes Using the
Wasserstein Distance”, with vectorized numerics, streaming support,
transition statistics, and a clean public API.
"""

from .core import (
    WassersteinKMeans,
    MomentKMeans,
    wasserstein_distance_1d,
    wasserstein_barycenter_1d,
    create_sliding_windows,
    compute_log_returns,
)
from .synthetic import (
    RegimeSwitchingParams,
    GBMParams,
    MertonParams,
    generate_regime_switching_gbm,
    generate_regime_switching_merton,
    compute_accuracy_scores,
)
from .streaming import StreamingRegimeDetector
from .stats import regime_transition_matrix, regime_persistence

__version__ = "0.2.0"
__all__ = [
    "WassersteinKMeans",
    "MomentKMeans",
    "wasserstein_distance_1d",
    "wasserstein_barycenter_1d",
    "create_sliding_windows",
    "compute_log_returns",
    "RegimeSwitchingParams",
    "GBMParams",
    "MertonParams",
    "generate_regime_switching_gbm",
    "generate_regime_switching_merton",
    "compute_accuracy_scores",
    "StreamingRegimeDetector",
    "regime_transition_matrix",
    "regime_persistence",
]
