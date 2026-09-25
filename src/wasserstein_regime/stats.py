"""Post-clustering statistics: transition matrix and regime persistence (new)."""

from __future__ import annotations

import numpy as np


def regime_transition_matrix(labels: np.ndarray, n_regimes: int | None = None) -> np.ndarray:
    """Empirical first-order transition matrix P[i, j] = P(next=j | current=i)."""
    labels = np.asarray(labels, dtype=np.int64)
    if n_regimes is None:
        n_regimes = int(labels.max()) + 1
    counts = np.zeros((n_regimes, n_regimes), dtype=float)
    for a, b in zip(labels[:-1], labels[1:]):
        if 0 <= a < n_regimes and 0 <= b < n_regimes:
            counts[a, b] += 1
    row_sum = counts.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return counts / row_sum


def regime_persistence(labels: np.ndarray) -> dict:
    """Mean and median run lengths per regime label."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return {}
    runs: dict[int, list[int]] = {}
    current = labels[0]
    length = 1
    for lab in labels[1:]:
        if lab == current:
            length += 1
        else:
            runs.setdefault(int(current), []).append(length)
            current = lab
            length = 1
    runs.setdefault(int(current), []).append(length)

    out = {}
    for k, lens in runs.items():
        arr = np.array(lens, dtype=float)
        out[k] = {
            "mean_run": float(arr.mean()),
            "median_run": float(np.median(arr)),
            "n_runs": int(arr.size),
            "max_run": int(arr.max()),
        }
    return out
