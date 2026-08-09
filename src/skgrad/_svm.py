"""Analytic input gradients for fitted binary and regression kernel SVMs."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.svm import NuSVC, NuSVR, SVC, SVR
from sklearn.utils.validation import check_is_fitted


FloatArray = NDArray[np.floating]
_CLASSIFIERS = (SVC, NuSVC)
_REGRESSORS = (SVR, NuSVR)
_BUILTIN_KERNELS = ("linear", "poly", "rbf", "sigmoid")


def svm_supports(model: object) -> bool:
    """Return whether a LibSVM estimator has a supported scalar output."""

    if not isinstance(model, _CLASSIFIERS + _REGRESSORS):
        return False
    if getattr(model, "kernel", None) not in _BUILTIN_KERNELS:
        return False
    classes = getattr(model, "classes_", None)
    return not isinstance(model, _CLASSIFIERS) or classes is None or len(classes) == 2


def svm_constant_jacobian(model: object) -> bool:
    """Return whether the fitted kernel produces an input-constant Jacobian."""

    return model.kernel == "linear" or (model.kernel == "poly" and model.degree <= 1)


def svm_model_output(model: object, X: FloatArray) -> FloatArray:
    """Return predictions or binary decision scores without a Jacobian."""

    coefficients, intercept = _parameters(model, X)
    values = _kernel_values(model, X)
    return np.asarray(values @ coefficients + intercept).reshape(-1, 1)


def svm_value_and_jacobian(
    model: object,
    X: FloatArray,
) -> Tuple[FloatArray, FloatArray]:
    """Return scalar values and their analytic input Jacobians."""

    coefficients, intercept = _parameters(model, X)
    values, kernel_gradients = _kernel_values_and_gradients(model, X)
    output = np.asarray(values @ coefficients + intercept).reshape(-1, 1)
    gradient = np.einsum("m,nmf->nf", coefficients, kernel_gradients)
    return output, gradient[:, None, :]


def _parameters(model: object, X: FloatArray) -> Tuple[FloatArray, float]:
    check_is_fitted(model, attributes=["support_vectors_", "dual_coef_", "intercept_"])
    n_features = int(model.n_features_in_)
    if X.shape[1] != n_features:
        raise ValueError(f"X has {X.shape[1]} features; model expects {n_features}")
    coefficients = model.dual_coef_
    if hasattr(coefficients, "toarray"):
        coefficients = coefficients.toarray()
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape[0] != 1:
        raise ValueError("multiclass kernel SVMs are not supported")
    return coefficients[0], float(np.asarray(model.intercept_)[0])


def _kernel_values_and_gradients(
    model: object,
    X: FloatArray,
) -> Tuple[FloatArray, FloatArray]:
    support_vectors = _support_vectors(model)
    products = X @ support_vectors.T
    kernel = model.kernel

    if kernel == "linear":
        gradients = np.broadcast_to(
            support_vectors, (X.shape[0],) + support_vectors.shape
        )
        return products, gradients

    gamma = float(model._gamma)
    if kernel == "poly":
        base = gamma * products + float(model.coef0)
        values = base ** int(model.degree)
        if model.degree == 0:
            gradients = np.zeros((X.shape[0],) + support_vectors.shape)
        else:
            scale = gamma * model.degree * base ** (model.degree - 1)
            gradients = scale[:, :, None] * support_vectors[None, :, :]
        return values, gradients

    if kernel == "rbf":
        displacement = support_vectors[None, :, :] - X[:, None, :]
        values = np.exp(-gamma * np.sum(displacement * displacement, axis=2))
        gradients = 2.0 * gamma * values[:, :, None] * displacement
        return values, gradients

    if kernel == "sigmoid":
        values = np.tanh(gamma * products + float(model.coef0))
        scale = gamma * (1.0 - values * values)
        gradients = scale[:, :, None] * support_vectors[None, :, :]
        return values, gradients

    raise TypeError(f"skgrad does not support the {kernel!r} kernel")


def _kernel_values(model: object, X: FloatArray) -> FloatArray:
    support_vectors = _support_vectors(model)
    products = X @ support_vectors.T
    kernel = model.kernel
    if kernel == "linear":
        return products
    if kernel == "poly":
        return (float(model._gamma) * products + float(model.coef0)) ** int(
            model.degree
        )
    if kernel == "rbf":
        squared_distances = (
            np.sum(X * X, axis=1)[:, None]
            + np.sum(support_vectors * support_vectors, axis=1)[None, :]
            - 2.0 * products
        )
        np.maximum(squared_distances, 0.0, out=squared_distances)
        return np.exp(-float(model._gamma) * squared_distances)
    if kernel == "sigmoid":
        return np.tanh(float(model._gamma) * products + float(model.coef0))
    raise TypeError(f"skgrad does not support the {kernel!r} kernel")


def _support_vectors(model: object) -> FloatArray:
    support_vectors = model.support_vectors_
    if hasattr(support_vectors, "toarray"):
        support_vectors = support_vectors.toarray()
    return np.asarray(support_vectors, dtype=float)
