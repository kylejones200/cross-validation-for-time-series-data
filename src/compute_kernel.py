"""Rolling-window CV split bounds (train/test index ranges)."""

from __future__ import annotations

import numpy as np


def rolling_window_cv_bounds(n_samples: int, window_size: int, step_size: int) -> np.ndarray:
    out: list[list[int]] = []
    start_idx = 0
    while start_idx + window_size < n_samples:
        end_idx = start_idx + window_size
        if end_idx + step_size <= n_samples:
            out.append([start_idx, end_idx, end_idx, end_idx + step_size])
            start_idx += step_size
        else:
            break
    if not out:
        return np.empty((0, 4), dtype=float)
    return np.asarray(out, dtype=float).ravel()
