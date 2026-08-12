"""Analytic chain-rule support for selected scikit-learn pipelines."""

from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils.validation import check_is_fitted


FloatArray = NDArray[np.floating]
ValueJacobian = Tuple[FloatArray, FloatArray]


def polynomial_pipeline_exact_quadrature_steps(
    model: object,
    downstream_constant_jacobian: Callable[[object], bool],
) -> Optional[int]:
    """Return the Gauss-Legendre order that exactly integrates pipeline IG."""

    parts = _required_pipeline_parts(model)
    if not downstream_constant_jacobian(parts[-1]):
        return None
    degree = parts[0].degree
    maximum_degree = degree[1] if isinstance(degree, tuple) else degree
    return (int(maximum_degree) + 1) // 2


def polynomial_pipeline_supports(
    model: object,
    downstream_supports: Callable[[object], bool],
) -> bool:
    """Return whether ``model`` is a supported polynomial pipeline."""

    parts = _pipeline_parts(model)
    return parts is not None and downstream_supports(parts[-1])


def polynomial_pipeline_value_and_jacobian(
    model: object,
    X: FloatArray,
    downstream_value_and_jacobian: Callable[[object, FloatArray], ValueJacobian],
) -> ValueJacobian:
    """Differentiate a polynomial pipeline with respect to its original inputs."""

    parts = _required_pipeline_parts(model)
    transformed, transform_jacobian = _polynomial_value_and_jacobian(parts[0], X)
    if len(parts) == 3:
        transformed, transform_jacobian = _scale_values_and_jacobian(
            parts[1], transformed, transform_jacobian
        )
    values, downstream_jacobian = downstream_value_and_jacobian(
        parts[-1], transformed
    )
    jacobian = np.einsum(
        "noq,nqp->nop",
        downstream_jacobian,
        transform_jacobian,
        optimize=True,
    )
    return values, jacobian


def polynomial_pipeline_model_output(
    model: object,
    X: FloatArray,
    downstream_model_output: Callable[[object, FloatArray], FloatArray],
) -> FloatArray:
    """Return downstream predictions or scores for a polynomial pipeline."""

    parts = _required_pipeline_parts(model)
    transformed = np.asarray(parts[0].transform(X))
    if len(parts) == 3:
        check_is_fitted(parts[1])
        transformed = np.asarray(parts[1].transform(transformed))
    return downstream_model_output(parts[-1], transformed)


def _pipeline_parts(model: object) -> Optional[Tuple[object, ...]]:
    if not isinstance(model, Pipeline):
        return None
    parts = tuple(step for _, step in model.steps)
    if len(parts) == 2 and isinstance(parts[0], PolynomialFeatures):
        return parts
    if (
        len(parts) == 3
        and isinstance(parts[0], PolynomialFeatures)
        and isinstance(parts[1], StandardScaler)
    ):
        return parts
    return None


def _required_pipeline_parts(model: object) -> Tuple[object, ...]:
    parts = _pipeline_parts(model)
    if parts is None:
        raise TypeError(f"skgrad does not support {type(model).__name__}")
    return parts


def _polynomial_value_and_jacobian(
    transformer: object,
    X: FloatArray,
) -> ValueJacobian:
    check_is_fitted(transformer, attributes=["powers_"])
    powers = np.asarray(transformer.powers_, dtype=np.intp)
    transformed = np.asarray(transformer.transform(X))
    jacobian = np.zeros(
        (X.shape[0], powers.shape[0], X.shape[1]),
        dtype=np.result_type(X.dtype, transformed.dtype),
    )

    # Construct reduced monomials directly instead of dividing transformed
    # terms by X. The direct form remains correct when an input coordinate is 0.
    for feature in range(X.shape[1]):
        active = powers[:, feature] > 0
        if not np.any(active):
            continue
        reduced = powers[active].copy()
        coefficients = reduced[:, feature].copy()
        reduced[:, feature] -= 1
        monomials = np.prod(
            np.power(X[:, None, :], reduced[None, :, :]),
            axis=2,
        )
        jacobian[:, active, feature] = coefficients[None, :] * monomials
    return transformed, jacobian


def _scale_values_and_jacobian(
    scaler: object,
    values: FloatArray,
    jacobian: FloatArray,
) -> ValueJacobian:
    check_is_fitted(scaler)
    scaled = np.asarray(scaler.transform(values))
    if scaler.with_std:
        scale = np.asarray(scaler.scale_, dtype=jacobian.dtype)
        jacobian = jacobian / scale[None, :, None]
    return scaled, jacobian
