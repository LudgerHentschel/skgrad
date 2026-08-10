"""Shared input validation."""

import operator

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse


def normalize_data(X: object) -> NDArray[np.floating]:
    """Return finite numeric data with a leading sample dimension."""

    if issparse(X):
        raise TypeError(
            "X must be a dense numeric array; sparse inputs are not supported"
        )
    data = np.asarray(X)
    if data.dtype == np.float16:
        data = data.astype(np.float32)
    elif data.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        data = np.asarray(X, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("X must be a non-empty one- or two-dimensional array")
    if not np.isfinite(data).all():
        raise ValueError("X must contain only finite values")
    return np.ascontiguousarray(data)


def validate_target(n_outputs: int, target: object) -> int:
    """Return a validated Python index for one scalar model output."""

    if target is None:
        if n_outputs != 1:
            raise ValueError("target is required when the model has multiple outputs")
        return 0
    if isinstance(target, (bool, np.bool_)):
        raise TypeError("target must be an integer or None")
    try:
        target_index = operator.index(target)
    except TypeError as error:
        raise TypeError("target must be an integer or None") from error
    if target_index < 0 or target_index >= n_outputs:
        raise ValueError(
            f"target must be between 0 and {n_outputs - 1}, got {target_index}"
        )
    return target_index
