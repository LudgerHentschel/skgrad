"""Analytic Jacobians for fitted affine estimators."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.utils.validation import check_is_fitted


FloatArray = NDArray[np.floating]
_AFFINE_TYPES = (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
    RidgeClassifier,
)


def affine_supports(model: object) -> bool:
    return isinstance(model, _AFFINE_TYPES)


def affine_value_and_jacobian(
    model: object,
    X: FloatArray,
) -> Tuple[FloatArray, FloatArray]:
    coefficients, intercept = _parameters(model, X)
    values = X @ coefficients.T + intercept
    jacobian = np.broadcast_to(
        coefficients,
        (X.shape[0], coefficients.shape[0], coefficients.shape[1]),
    ).copy()
    return np.asarray(values, dtype=float), jacobian


def affine_model_output(model: object, X: FloatArray) -> FloatArray:
    """Return affine predictions or decision scores without a Jacobian."""

    coefficients, intercept = _parameters(model, X)
    return np.asarray(X @ coefficients.T + intercept, dtype=float)


def _parameters(model: object, X: FloatArray) -> Tuple[FloatArray, FloatArray]:
    check_is_fitted(model, attributes=["coef_", "intercept_"])
    n_features = int(model.n_features_in_)
    if X.shape[1] != n_features:
        raise ValueError(f"X has {X.shape[1]} features; model expects {n_features}")

    coefficients = np.asarray(model.coef_, dtype=float)
    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(1, -1)
    intercept = np.asarray(model.intercept_, dtype=float).reshape(-1)
    if intercept.size == 1 and coefficients.shape[0] > 1:
        intercept = np.repeat(intercept, coefficients.shape[0])
    return coefficients, intercept
