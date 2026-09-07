"""Analytic composition through explicitly supported continuous pipelines."""

from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe,
    GenericUnivariateSelect, VarianceThreshold, SelectFromModel, RFE, RFECV,
    SequentialFeatureSelector,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PolynomialFeatures, StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler,
)
from sklearn.utils.validation import check_is_fitted

FloatArray = NDArray[np.floating]
ValueJacobian = Tuple[FloatArray, FloatArray]
History = List[Tuple[object, FloatArray]]
_SCALERS = (StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler)
_SELECTORS = (
    SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe,
    GenericUnivariateSelect, VarianceThreshold, SelectFromModel, RFE, RFECV,
    SequentialFeatureSelector,
)


def _identity(step: object) -> bool:
    return step is None or (isinstance(step, str) and step == "passthrough")


def _flatten(model: object) -> Tuple[object, ...]:
    if type(model) is Pipeline:
        return tuple(part for _, step in model.steps for part in _flatten(step))
    return () if _identity(model) else (model,)


def _transformer_supports(step: object) -> bool:
    # Exact classes prevent custom transform overrides from inheriting an
    # incorrect derivative. Estimator dispatch is handled separately.
    return type(step) in _SCALERS + _SELECTORS + (PolynomialFeatures, PCA)


def pipeline_supports(
    model: object, downstream_supports: Callable[[object], bool]
) -> bool:
    if type(model) is not Pipeline or not model.steps:
        return False
    # A transformer-only or identity-ending Pipeline is not a predictor.
    tail = model.steps[-1][1]
    while type(tail) is Pipeline:
        if not tail.steps:
            return False
        tail = tail.steps[-1][1]
    if _identity(tail) or not _nonempty_pipelines(model):
        return False
    parts = _flatten(model)
    return (
        bool(parts)
        and all(_transformer_supports(s) for s in parts[:-1])
        and downstream_supports(parts[-1])
    )


def _nonempty_pipelines(model: object) -> bool:
    if type(model) is not Pipeline:
        return True
    return bool(model.steps) and all(
        _nonempty_pipelines(step) for _, step in model.steps
    )


def pipeline_properties(
    model: object, downstream_constant_jacobian: Callable[[object], bool]
) -> Tuple[bool, Optional[int]]:
    parts = _flatten(model)
    if not downstream_constant_jacobian(parts[-1]):
        return False, None
    degree = 1
    for step in parts[:-1]:
        if type(step) is MinMaxScaler and step.clip:
            return False, None
        if type(step) is PolynomialFeatures:
            d = step.degree[1] if isinstance(step.degree, tuple) else step.degree
            degree *= int(d)
    return degree <= 1, max(1, (degree + 1) // 2)


def _forward(
    model: object, X: FloatArray
) -> Tuple[object, FloatArray, History]:
    parts = _flatten(model)
    history: History = []
    for step in parts[:-1]:
        history.append((step, X))
        X = np.asarray(_transform(step, X))
    return parts[-1], X, history


def _transform(step: object, X: object) -> object:
    """Apply one fitted transform without modifying caller-owned input."""
    check_is_fitted(step)
    if type(step) is PCA and step.whiten:
        scale = np.sqrt(step.explained_variance_)
        if np.any(scale <= np.finfo(scale.dtype).eps):
            raise ValueError(
                "PCA whitening requires non-degenerate explained variance"
            )
    return step.transform(X.copy())


def pipeline_value_and_jacobian(
    model: object, X: FloatArray,
    downstream_value_and_jacobian: Callable[[object, FloatArray], ValueJacobian],
) -> ValueJacobian:
    estimator, transformed, history = _forward(model, X)
    values, jacobian = downstream_value_and_jacobian(estimator, transformed)
    return values, _pullback(history, jacobian)


def pipeline_input_gradient(
    model: object, X: FloatArray, target: Optional[int],
    downstream_input_gradient: Callable[
        [object, FloatArray, Optional[int]], FloatArray
    ],
) -> FloatArray:
    estimator, transformed, history = _forward(model, X)
    gradient = downstream_input_gradient(estimator, transformed, target)
    return _pullback(history, gradient[:, None, :])[:, 0, :]


def pipeline_model_output(
    model: object, X: FloatArray,
    downstream_model_output: Callable[[object, FloatArray], FloatArray],
) -> FloatArray:
    estimator, transformed, _ = _forward(model, X)
    return downstream_model_output(estimator, transformed)


def _pullback(history: History, jacobian: FloatArray) -> FloatArray:
    for step, X in reversed(history):
        if type(step) in _SCALERS:
            if type(step) is StandardScaler:
                scale = 1.0 / step.scale_ if step.with_std else 1.0
            elif type(step) is RobustScaler:
                scale = 1.0 / step.scale_ if step.with_scaling else 1.0
            elif type(step) is MaxAbsScaler:
                scale = 1.0 / step.scale_
            else:
                scale = step.scale_
            # Scalers preserve their input dtype even when fitted statistics
            # use float64 (notably StandardScaler and RobustScaler).
            jacobian = jacobian * np.asarray(scale, dtype=jacobian.dtype)
            if type(step) is MinMaxScaler and step.clip:
                before_clip = X * step.scale_ + step.min_
                interior = (
                    (before_clip > step.feature_range[0])
                    & (before_clip < step.feature_range[1])
                )
                jacobian = jacobian * interior[:, None, :]
        elif type(step) is PCA:
            components = step.components_
            if step.whiten:
                components = components / np.sqrt(step.explained_variance_)[:, None]
            jacobian = jacobian @ components
        elif type(step) in _SELECTORS:
            result = np.zeros(
                (X.shape[0], jacobian.shape[1], X.shape[1]), dtype=jacobian.dtype
            )
            result[:, :, step.get_support(indices=True)] = jacobian
            jacobian = result
        else:
            _, local = _polynomial_value_and_jacobian(step, X)
            jacobian = np.einsum("noq,nqp->nop", jacobian, local, optimize=True)
    return jacobian


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
