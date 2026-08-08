"""Shared input validation."""

import numpy as np
from numpy.typing import NDArray


def normalize_data(X: object) -> NDArray[np.floating]:
    """Return finite numeric data with a leading sample dimension."""

    data = np.asarray(X)
    if data.dtype == np.float16:
        data = data.astype(np.float32)
    elif data.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        data = np.asarray(X, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("X must be a non-empty sample or two-dimensional matrix")
    if not np.isfinite(data).all():
        raise ValueError("X must contain only finite values")
    return np.ascontiguousarray(data)
